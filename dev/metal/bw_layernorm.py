import torch


def ln_backward(dout, x, w, mean, rstd):
    # recompute the norm (save memory at the cost of compute)
    norm = (x - mean) * rstd
    # gradients for weights, bias
    db = dout.sum((0, 1))
    dw = (dout * norm).sum((0, 1))
    # gradients for input
    dnorm = dout * w
    dx = dnorm - dnorm.mean(-1, keepdim=True) - norm * (dnorm * norm).mean(-1, keepdim=True)
    dx *= rstd
    return dx, dw, db

def ln_backward_tiled(dout, x, w, mean, rstd):
    B, T, C = x.shape
    dbias = torch.zeros_like(w)
    dweight = torch.zeros_like(w)
    dinp = torch.zeros_like(x)
    for b in range(B):
        for t in range(T//32):
            mean_tile = mean[b, t*32:(t+1)*32]
            rstd_tile = rstd[b, t*32:(t+1)*32]

            dnorm_mean = torch.zeros(32, 1)
            dnorm_norm_mean = torch.zeros(32, 1)

            for c in range(C//32):
                x_tile = x[b, t*32:(t+1)*32, c*32:(c+1)*32]
                dout_tile = dout[b, t*32:(t+1)*32, c*32:(c+1)*32]
                w_tile = w[c*32:(c+1)*32].view(1, 32)

                norm_bti = (x_tile - mean_tile) * rstd_tile
                dnorm_i = dout_tile * w_tile

                dnorm_mean += dnorm_i.sum(1, keepdim=True)
                dnorm_norm_mean += (dnorm_i * norm_bti).sum(1, keepdim=True)

            dnorm_mean /= C
            dnorm_norm_mean /= C

            for c in range(C//32):
                x_tile = x[b, t*32:(t+1)*32, c*32:(c+1)*32]
                dout_tile = dout[b, t*32:(t+1)*32, c*32:(c+1)*32]
                w_tile = w[c*32:(c+1)*32].view(1, 32)

                norm_bti = (x_tile - mean_tile) * rstd_tile
                dnorm_i = dout_tile * w_tile
                dbias[c*32:(c+1)*32] += dout_tile.sum(0)
                dweight[c*32:(c+1)*32] += (norm_bti * dout_tile).sum(0)

                dinp_tile = dnorm_i - dnorm_mean
                dinp_tile -= norm_bti * dnorm_norm_mean
                dinp_tile *= rstd_tile
                dinp[b, t*32:(t+1)*32, c*32:(c+1)*32] += dinp_tile

    return dinp, dweight, dbias


def ln_backward_tiled_tt(dout, x, w, mean, rstd):
    B, T, C = x.shape
    dbiasR = torch.zeros_like(w)
    dweightR = torch.zeros_like(w)
    dinp = torch.zeros_like(x)
    for b in range(B):
        for t in range(T//32):
            mean_tileC = mean[b, t*32:(t+1)*32]
            rstd_tileC = rstd[b, t*32:(t+1)*32]

            for c in range(C//32):
                x_tile = x[b, t*32:(t+1)*32, c*32:(c+1)*32]
                dout_tile = dout[b, t*32:(t+1)*32, c*32:(c+1)*32]
                w_tileR = w[c*32:(c+1)*32].view(1, 32)

                # sub_tiles_bcast_cols
                s0norm_bti = (x_tile - mean_tileC)
                # mul_tiles_bcast_cols
                s0norm_bti = s0norm_bti * rstd_tileC
                # mul_tiles_bcast_rows
                s1dnorm_i = dout_tile * w_tileR
                # mul_tiles
                s2dnorm_times_norm = s1dnorm_i * s0norm_bti

                # reduce_sum
                s3_interm_0C = s1dnorm_i.sum(1, keepdim=True)
                # reduce_sum
                s4_interm_1C = s2dnorm_times_norm.sum(1, keepdim=True)
                if c == 0:
                    s5dnorm_meanC = s3_interm_0C
                    s6dnorm_norm_meanC = s4_interm_1C
                else:
                    # add_tiles
                    s5dnorm_meanC = s5dnorm_meanC + s3_interm_0C
                    # add_tiles
                    s6dnorm_norm_meanC = s6dnorm_norm_meanC + s4_interm_1C

            # mul_tiles
            s5dnorm_meanC = s5dnorm_meanC * (1/C) # Column of 1/C
            s6dnorm_norm_meanC = s6dnorm_norm_meanC * (1/C) # Column of 1/C

            for c in range(C//32):
                x_tile = x[b, t*32:(t+1)*32, c*32:(c+1)*32]
                dout_tile = dout[b, t*32:(t+1)*32, c*32:(c+1)*32]
                w_tileR = w[c*32:(c+1)*32].view(1, 32)

                # sub_tiles_bcast_cols
                s0norm_bti = (x_tile - mean_tileC)
                # mul_tiles_bcast_cols
                s0norm_bti = s0norm_bti * rstd_tileC
                # mul_tiles_bcast_rows
                s1dnorm_i = dout_tile * w_tileR
                # reduce_sum (REDUCE_COL)
                s2_interm_1R = dout_tile.sum(0)
                # add_tiles
                dbiasR[c*32:(c+1)*32] = dbiasR[c*32:(c+1)*32] + s2_interm_1R

                # mul_tiles
                s2_interm_2 = (s0norm_bti * dout_tile)
                # reduce_sum (REDUCE_COL)
                s2_interm_2R = s2_interm_2.sum(0)
                dweightR[c*32:(c+1)*32] = dweightR[c*32:(c+1)*32] + s2_interm_2R

                # sub_tiles_bcast_cols
                dinp_tile = s1dnorm_i - s5dnorm_meanC
                # mul_tiles_bcast_cols
                s3_interm_3 = s0norm_bti * s6dnorm_norm_meanC
                # sub_tiles
                dinp_tile = dinp_tile - s3_interm_3
                # mul_tiles_bcast_cols
                dinp_tile = dinp_tile * rstd_tileC
                # add_tiles
                dinp[b, t*32:(t+1)*32, c*32:(c+1)*32] = dinp[b, t*32:(t+1)*32, c*32:(c+1)*32] + dinp_tile

    return dinp, dweightR, dbiasR


# Test Karpathy's impl
B = 8
T = 1024
C = 1600

x = torch.randn(B, T, C)
w = torch.randn(C)
b = torch.randn(C)
mean = x.mean(-1, keepdim=True)
rstd = 1.0 / (x.var(-1, keepdim=True) + 1e-5).sqrt()
dout = torch.randn(B, T, C)

dx, dw, db = ln_backward(dout, x, w, mean, rstd)

# Test Karpathy impl with torch's autograd
xt = x.clone().detach().requires_grad_(True)
wt = w.clone().detach().requires_grad_(True)
bt = b.clone().detach().requires_grad_(True)

x_norm = (xt - xt.mean(-1, keepdim=True)) / (xt.var(-1, keepdim=True) + 1e-5).sqrt()
y = x_norm * wt + bt

y.backward(gradient=dout)
assert torch.allclose(xt.grad, dx, atol=1e-3, rtol=1e-8)
assert torch.allclose(wt.grad, dw, atol=1e-3, rtol=1e-8)
assert torch.allclose(bt.grad, db, atol=1e-3, rtol=1e-8)

# Test mine against Karpathy's
dx_tiled, dw_tiled, db_tiled = ln_backward_tiled_tt(dout, x, w, mean, rstd)
assert torch.allclose(dx_tiled, dx, atol=1e-3, rtol=1e-8)
assert torch.allclose(dw_tiled, dw, atol=1e-3, rtol=1e-8)
assert torch.allclose(db_tiled, db, atol=1e-3, rtol=1e-8)

# Compare to pytorch too
assert torch.allclose(dw_tiled, wt.grad, atol=1e-3, rtol=1e-8)
assert torch.allclose(db_tiled, bt.grad, atol=1e-3, rtol=1e-8)
assert torch.allclose(dx_tiled, xt.grad, atol=1e-3, rtol=1e-8)
