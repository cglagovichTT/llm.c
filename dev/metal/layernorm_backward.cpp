#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <vector>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string_view>

// metal imports
#include "common/core_coord.h"
#include "tt_metal/host_api.hpp"
#include "common/bfloat16.hpp"
// #include "tensor/tensor.hpp"
#include "common/constants.hpp"
// #include "tt_numpy/functions.hpp"

// project imports
#include "common.h"

// ----------------------------------------------------------------------------
// CPU code reference

void layernorm_forward_cpu1(std::vector<float> &out, std::vector<float> &mean, std::vector<float> &rstd,
                       const std::vector<float> &inp, const std::vector<float> &weight, const std::vector<float> &bias,
                       int B, int T, int C) {
    float eps = 1e-5f;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            std::size_t offset = b * T * C + t * C;
            // calculate the mean
            float m = 0.0f;
            for (int i = 0; i < C; i++) {
                m += inp[offset + i];
            }
            m = m/C;
            // calculate the variance (without any bias correction)
            float v = 0.0f;
            for (int i = 0; i < C; i++) {
                float xshift = inp[offset + i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // calculate the rstd
            float s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            for (int i = 0; i < C; i++) {
                float n = (s * (inp[offset + i] - m)); // normalized output
                float o = n * weight[i] + bias[i]; // scale and shift it
                out[offset + i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

void layernorm_forward_cpu(float* out, float* mean, float* rstd,
                       const float* inp, const float* weight, const float* bias,
                       int B, int T, int C) {
    // reference: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    // both inp and out are (B,T,C) of the activations
    // mean and rstd are (B,T) buffers, to be used later in backward pass
    // at each position (b,t) of the input, the C-dimensional vector
    // of activations gets normalized, then scaled and shifted
    float eps = 1e-5f;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            const float* x = inp + b * T * C + t * C;
            // calculate the mean
            float m = 0.0f;
            for (int i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;
            // calculate the variance (without any bias correction)
            float v = 0.0f;
            for (int i = 0; i < C; i++) {
                float xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // calculate the rstd (reciprocal standard deviation)
            float s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
            for (int i = 0; i < C; i++) {
                float n = (s * (x[i] - m)); // normalize
                float o = n * weight[i] + bias[i]; // scale and shift
                out_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

void layernorm_backward_cpu(float* dinp, float* dweight, float* dbias,
                        const float* dout, const float* inp, const float* weight, const float* mean, const float* rstd,
                        int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            const float* dout_bt = dout + b * T * C + t * C;
            const float* inp_bt = inp + b * T * C + t * C;
            float* dinp_bt = dinp + b * T * C + t * C;
            const float mean_bt = mean[b * T + t];
            const float rstd_bt = rstd[b * T + t];

            // first: two reduce operations
            float dnorm_mean = 0.0f;
            float dnorm_norm_mean = 0.0f;
            for (int i = 0; i < C; i++) {
                float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                float dnorm_i = weight[i] * dout_bt[i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean = dnorm_mean / C;
            dnorm_norm_mean = dnorm_norm_mean / C;

            // now iterate again and accumulate all the gradients
            for (int i = 0; i < C; i++) {
                float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                float dnorm_i = weight[i] * dout_bt[i];
                // gradient contribution to bias
                dbias[i] += dout_bt[i];
                // gradient contribution to weight
                dweight[i] += norm_bti * dout_bt[i];
                // gradient contribution to input
                float dval = 0.0f;
                dval += dnorm_i; // term 1
                dval -= dnorm_mean; // term 2
                dval -= norm_bti * dnorm_norm_mean; // term 3
                dval *= rstd_bt; // final scale
                dinp_bt[i] += dval;
            }
        }
    }
}

void layernorm_backward_cpu1(std::vector<float> &dinp, std::vector<float> &dweight, std::vector<float> &dbias,
                        const std::vector<float> &dout, const std::vector<float> &inp, const std::vector<float> &weight, const std::vector<float> &mean, const std::vector<float> &rstd,
                        int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            const int bt_offset = b * T * C + t * C;
            const float mean_bt = mean[b * T + t];
            const float rstd_bt = rstd[b * T + t];

            // first: two reduce operations
            float dnorm_mean = 0.0f;
            float dnorm_norm_mean = 0.0f;
            for (int i = 0; i < C; i++) {
                float norm_bti = (inp[bt_offset + i] - mean_bt) * rstd_bt;
                float dnorm_i = weight[i] * dout[bt_offset + i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean = dnorm_mean / C;
            dnorm_norm_mean = dnorm_norm_mean / C;

            // now iterate again and accumulate all the gradients
            for (int i = 0; i < C; i++) {
                float norm_bti = (inp[bt_offset + i] - mean_bt) * rstd_bt;
                float dnorm_i = weight[i] * dout[bt_offset + i];
                // gradient contribution to bias
                dbias[i] += dout[bt_offset + i];
                // gradient contribution to weight
                dweight[i] += norm_bti * dout[bt_offset + i];
                // gradient contribution to input
                float dval = 0.0f;
                dval += dnorm_i; // term 1
                dval -= dnorm_mean; // term 2
                dval -= norm_bti * dnorm_norm_mean; // term 3
                dval *= rstd_bt; // final scale
                dinp[bt_offset + i] += dval;
            }
        }
    }
}

void layernorm_backwards_setup_program(
    Program& program, Device* device, uint32_t B, uint32_t T, uint32_t C, 
    std::shared_ptr<Buffer>& dout_buffer, 
    std::shared_ptr<Buffer>& inp_buffer, 
    std::shared_ptr<Buffer>& weight_buffer,
    std::shared_ptr<Buffer>& mean_buffer,
    std::shared_ptr<Buffer>& rstd_buffer,
    std::shared_ptr<Buffer>& dinp_buffer,
    std::shared_ptr<Buffer>& dweight_buffer,
    std::shared_ptr<Buffer>& dbias_buffer
) {

    uint32_t tile_size = 32 * 32;
    const int sizeof_bfloat = 2;
    uint32_t tile_size_b = sizeof_bfloat * tile_size;
    const int stick_size = 32 * sizeof_bfloat;
    const uint32_t log_stick_size = std::log2(stick_size);

    const float one_scalar = 1.0;
    bfloat16 bf_one_scalar = bfloat16(one_scalar);
    uint32_t packed_one_scalar = pack_two_bfloat16_into_uint32({bf_one_scalar, bf_one_scalar});
    union { float f; uint32_t u; } one_packed; one_packed.f = one_scalar;
    const float mean_recip_scalar = 1.0 / C;
    union { float f; uint32_t u; } mean_recip_packed; mean_recip_packed.f = mean_recip_scalar;

    // This first implementation is on a single core. TODO: parallelize, involving 2D reduction?
    constexpr CoreCoord core = {0, 0};

    tt::CB cb_inp = tt::CB::c_in0;
    tt::CB cb_dout = tt::CB::c_in1;
    tt::CB cb_weight = tt::CB::c_in2;
    tt::CB cb_mean = tt::CB::c_in3;
    tt::CB cb_rstd = tt::CB::c_in4;
    tt::CB cb_dinp = tt::CB::c_in5;
    tt::CB cb_dweight = tt::CB::c_in6;
    tt::CB cb_dbias = tt::CB::c_in7;

    tt::CB cb_identity_scalar = tt::CB::dataflow0;
    tt::CB cb_mean_scalar = tt::CB::dataflow1;

    tt::CB cb_scratch_0 = tt::CB::c_intermed0;
    tt::CB cb_scratch_1 = tt::CB::c_intermed1;
    tt::CB cb_scratch_2 = tt::CB::c_intermed2;
    tt::CB cb_scratch_3 = tt::CB::c_intermed3;
    tt::CB cb_scratch_4 = tt::CB::c_intermed4;
    tt::CB cb_scratch_5 = tt::CB::c_intermed5;
    tt::CB cb_scratch_6 = tt::CB::c_intermed6;

    tt::CB cb_out_dinp = tt::CB::c_out0;
    tt::CB cb_out_dweight = tt::CB::c_out1;
    tt::CB cb_out_dbias = tt::CB::c_out2;

    auto inp_num_tiles = C / 32; // buffer 32 in T and full C
    auto stats_num_tiles = 1; // Buffer 1 tile of stats at a time
    auto intermed_num_tiles = 1;
    auto out_num_tiles = C / 32; // buffer 32 in T and full C

    // Make CBs
    // TODO: Double buffering for inputs and outputs
    // Input CBs
    MakeCircularBuffer(program, core, cb_inp, inp_num_tiles * tile_size_b, tile_size_b, tt::DataFormat::Float16_b);
    MakeCircularBuffer(program, core, cb_dout, inp_num_tiles * tile_size_b, tile_size_b, tt::DataFormat::Float16_b);
    MakeCircularBuffer(program, core, cb_weight, inp_num_tiles * tile_size_b, tile_size_b, tt::DataFormat::Float16_b);
    MakeCircularBuffer(program, core, cb_mean, stats_num_tiles * tile_size_b, tile_size_b, tt::DataFormat::Float16_b);
    MakeCircularBuffer(program, core, cb_rstd, stats_num_tiles * tile_size_b, tile_size_b, tt::DataFormat::Float16_b);
    MakeCircularBuffer(program, core, cb_dinp, inp_num_tiles * tile_size_b, tile_size_b, tt::DataFormat::Float16_b);
    MakeCircularBuffer(program, core, cb_dweight, inp_num_tiles * tile_size_b, tile_size_b, tt::DataFormat::Float16_b);
    MakeCircularBuffer(program, core, cb_dbias, inp_num_tiles * tile_size_b, tile_size_b, tt::DataFormat::Float16_b);

    // Dataflow CBs
    MakeCircularBuffer(program, core, cb_identity_scalar, tile_size_b, tile_size_b, tt::DataFormat::Float16_b);
    MakeCircularBuffer(program, core, cb_mean_scalar, tile_size_b, tile_size_b, tt::DataFormat::Float16_b);
    
    // Intermediate CBs
    MakeCircularBuffer(program, core, cb_scratch_0, intermed_num_tiles * tile_size_b, tile_size_b, tt::DataFormat::Float16_b);
    MakeCircularBuffer(program, core, cb_scratch_1, intermed_num_tiles * tile_size_b, tile_size_b, tt::DataFormat::Float16_b);
    MakeCircularBuffer(program, core, cb_scratch_2, intermed_num_tiles * tile_size_b, tile_size_b, tt::DataFormat::Float16_b);
    MakeCircularBuffer(program, core, cb_scratch_3, intermed_num_tiles * tile_size_b, tile_size_b, tt::DataFormat::Float16_b);
    MakeCircularBuffer(program, core, cb_scratch_4, intermed_num_tiles * tile_size_b, tile_size_b, tt::DataFormat::Float16_b);
    MakeCircularBuffer(program, core, cb_scratch_5, intermed_num_tiles * tile_size_b, tile_size_b, tt::DataFormat::Float16_b);
    MakeCircularBuffer(program, core, cb_scratch_6, intermed_num_tiles * tile_size_b, tile_size_b, tt::DataFormat::Float16_b);

    // Out CBs
    MakeCircularBuffer(program, core, cb_out_dinp, out_num_tiles * tile_size_b, tile_size_b, tt::DataFormat::Float16_b);
    MakeCircularBuffer(program, core, cb_out_dweight, out_num_tiles * tile_size_b, tile_size_b, tt::DataFormat::Float16_b);
    MakeCircularBuffer(program, core, cb_out_dbias, out_num_tiles * tile_size_b, tile_size_b, tt::DataFormat::Float16_b);


    // TODO: Pass relevant CBs into compile time args
    std::vector<uint32_t> reader_ct_args = {
        packed_one_scalar,
        mean_recip_packed.u,
        // log of stick_size
        log_stick_size // should be 6 for bf16
        // CBs
        cb_inp,
        cb_dout,
        cb_weight,
        cb_mean,
        cb_rstd,
        cb_dinp,
        cb_dweight,
        cb_dbias
    };
    std::vector<uint32_t> writer_ct_args = {
        log_stick_size,
        cb_indentity_scalar,
        cb_mean_scalar,
        cb_out_dinp,
        cb_out_dweight,
        cb_out_dbias
    };
    std::vector<uint32_t> compile_ct_args = {
        cb_inp,
        cb_dout,
        cb_weight,
        cb_mean,
        cb_rstd,
        cb_dinp,
        cb_dweight,
        cb_dbias,
        cb_indentity_scalar,
        cb_mean_scalar,
        cb_out_dinp,
        cb_out_dweight,
        cb_out_dbias,
        cb_scratch_0,
        cb_scratch_1,
        cb_scratch_2,
        cb_scratch_3,
        cb_scratch_4,
        cb_scratch_5,
        cb_scratch_6
    };
    std::cout << "log of stick_size: " << log_stick_size << std::endl;
    auto reader = CreateKernel(
        program,
        "kernels/layernorm_backward/reader_interleaved.cpp",
        core,
        DataMovementConfig {.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = reader_ct_args}
    );
    auto writer = CreateKernel(
        program,
        "kernels/layernorm_backward/writer_interleaved.cpp",
        core,
        DataMovementConfig {.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = writer_ct_args}
    );
    auto compute = CreateKernel(
        program,
        "kernels/layernorm_backward/compute.cpp",
        core,
        ComputeConfig{
            .fp32_dest_acc_en = true,
            .math_approx_mode = false,
            .compile_args = compile_ct_args,
            .defines = {}
        }
    );

    SetRuntimeArgs(program, reader, core, {
        B, T, C,
        dout_buffer->address(),
        dinp_buffer->address(),
        weight_buffer->address(),
        mean_buffer->address(),
        rstd_buffer->address(),
        dinp_buffer->address(),
        dweight_buffer->address(),
        dbias_buffer->address()
    });

    SetRuntimeArgs(program, writer, core, {
        B, T, C,
        dinp_buffer->address(),
        dweight_buffer->address(),
        dbias_buffer->address()
    });

    SetRuntimeArgs(program, compute, core, {
        B, T, C,
    });

    // uint32_t seq_tiles = T / 32;
    // uint32_t batch_parallel_factor = std::min(B, num_cores);
    // uint32_t seq_parallel_factor = std::min(num_cores / batch_parallel_factor, seq_tiles);
    // uint32_t batch_per_core = std::ceil((float)B / batch_parallel_factor);
    // uint32_t seq_tiles_per_core = std::ceil((float)seq_tiles / seq_parallel_factor);

    // for (uint32_t core_idx = 0; core_idx < num_cores; ++core_idx) {
    //     CoreCoord core = {core_idx % device_grid_size.x, core_idx / device_grid_size.x};

    //     uint32_t start_b = (core_idx / seq_parallel_factor) * batch_per_core;
    //     uint32_t end_b = start_b + batch_per_core;
    //     uint32_t start_t = (core_idx % seq_parallel_factor) * seq_tiles_per_core;
    //     uint32_t end_t = start_t + seq_tiles_per_core;
                
    //     start_b = std::min(start_b, B);
    //     end_b = std::min(end_b, B);
    //     start_t = std::min(start_t, seq_tiles);
    //     end_t = std::min(end_t, seq_tiles);
        
    //     std::cout << "core: " << core_idx << " batch range (" << start_b << ", " << end_b <<  "), sequence range (" << start_t << ", " << end_t << ")" << std::endl;

    //     SetRuntimeArgs(program, reader, core, {
    //         B, T, C,
    //         inp_buffer->address(),
    //         weight_buffer->address(),
    //         bias_buffer->address(),
    //         start_b, start_t, end_b, end_t
    //     });

    //     SetRuntimeArgs(program, writer, core, {
    //         B, T, C,
    //         out_buffer->address(),
    //         mean_buffer->address(),
    //         rstd_buffer->address(),
    //         start_b, start_t, end_b, end_t
    //     });

    //     SetRuntimeArgs(program, compute, core, {
    //         B, T, C,
    //         start_b, start_t, end_b, end_t
    //     });
    // }

}

// ----------------------------------------------------------------------------

int main(int argc, char **argv) {

    srand(0);

    // GPT2-XL shapes
    uint32_t B = 8;
    uint32_t T = 1024;
    uint32_t C = 1600;
    if (argc == 4) {
	std::cout << "Using parameters provided from the command line" << std::endl;

        B = std::stoi(argv[1]);
        T = std::stoi(argv[2]);
        C = std::stoi(argv[3]);
    }

    std::cout << "B="<< B << " T=" << T << " C=" << C << std::endl;

    // create host memory of random numbers
    auto out_cpu = std::vector<float>(B * T * C);
    auto mean_cpu = std::vector<float>(B * T);
    auto rstd_cpu = std::vector<float>(B * T);
    auto inp = make_random_float_vec(B * T * C);
    auto weight = make_random_float_vec(C);
    auto bias = make_random_float_vec(C);
    
    // CPU ground truth
    layernorm_forward_cpu1(out_cpu, mean_cpu, rstd_cpu, inp, weight, bias, B, T, C);

    // now do the backward pass, again on CPU
    // All grad tensors shoud be random to test grad accumulation
    auto dout = make_random_float_vec(B * T * C);
    // Ground truth outputs
    auto dinp_cpu = make_random_float_vec(B * T * C);
    auto dweight_cpu = make_random_float_vec(C);
    auto dbias_cpu = make_random_float_vec(C);

    auto dinp_tt = dinp_cpu;
    auto dweight_tt = dweight_cpu;
    auto dbias_tt = dbias_cpu;

    layernorm_backward_cpu1(dinp_cpu, dweight_cpu, dbias_cpu, dout, inp, weight, mean_cpu, rstd_cpu, B, T, C);

    auto shape = std::vector<uint32_t>{B, T, C};
    auto inp_volume = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<uint32_t>());

    std::cout << "Running on Metal" << std::endl;
    // auto pcie_id = GetPCIeDeviceID(device_id);
    Device *device = CreateDevice(0);
    std::cout << "Created device" << std::endl;
    CommandQueue& cq = device->command_queue();
    std::cout << "Created command queue" << std::endl;
    
 try {

    uint32_t tile_size = 32 * 32;
    const int sizeof_bfloat = 2;
    uint32_t tile_size_b = sizeof_bfloat * tile_size;

    // Host inputs for Device
    std::vector<uint32_t> dout_tilized = fp32_to_bfloat16_packed(tilize(dout, shape));
    std::vector<uint32_t> inp_tilized = fp32_to_bfloat16_packed(tilize(inp, shape));
    std::vector<uint32_t> weight_rowmajor = fp32_to_bfloat16_packed(weight);
    std::vector<uint32_t> mean_rowmajor = fp32_to_bfloat16_packed(mean_cpu);
    std::vector<uint32_t> rstd_rowmajor = fp32_to_bfloat16_packed(rstd_cpu);
    // Gradients outputs to accumulate into
    std::vector<uint32_t> dinp_tilized = fp32_to_bfloat16_packed(tilize(dinp_tt, shape));
    std::vector<uint32_t> dweight_rowmajor = fp32_to_bfloat16_packed(dweight_tt);
    std::vector<uint32_t> dbias_rowmajor = fp32_to_bfloat16_packed(dbias_tt);


    // make dram buffers
    const int stick_size = 32 * sizeof_bfloat;
    // inputs
    auto dout_buffer = MakeBuffer(device, inp_volume * sizeof_bfloat, tile_size_b, false); //  tilized
    auto inp_buffer = MakeBuffer(device, inp_volume * sizeof_bfloat, tile_size_b, false); //  tilized
    // page size of mean_buffer implies how we can parallelize. If page is 32, then one core can write 32 rows at a time
    auto weight_buffer = MakeBuffer(device, C * sizeof_bfloat, stick_size, false); // RM, page_size = 32
    auto mean_buffer = MakeBuffer(device, B * T * sizeof_bfloat, stick_size, false); // RM, page_size = 32
    auto rstd_buffer = MakeBuffer(device, B * T * sizeof_bfloat, stick_size, false); // RM, page_size = 32
    // outputs
    auto dinp_buffer = MakeBuffer(device, inp_volume * sizeof_bfloat, tile_size_b, false); // tilized
    auto dweight_buffer = MakeBuffer(device, C * sizeof_bfloat, stick_size, false); // RM, page_size = 32
    auto dbias_buffer = MakeBuffer(device, C * sizeof_bfloat, stick_size, false); // RM, page_size = 32


    // For backwards we write all inputs/outputs to device because of accumulation
    EnqueueWriteBuffer(cq, dout_buffer, dout_tilized, false);
    EnqueueWriteBuffer(cq, inp_buffer, inp_tilized, false);
    EnqueueWriteBuffer(cq, weight_buffer, weight_rowmajor, false);
    EnqueueWriteBuffer(cq, mean_buffer, mean_rowmajor, false);
    EnqueueWriteBuffer(cq, rstd_buffer, rstd_rowmajor, false);
    EnqueueWriteBuffer(cq, dinp_buffer, dinp_tilized, false);
    EnqueueWriteBuffer(cq, dweight_buffer, dweight_rowmajor, false);
    EnqueueWriteBuffer(cq, dbias_buffer, dbias_rowmajor, false);

    Program layernorm_program = CreateProgram();
    layernorm_backwards_setup_program(layernorm_program, device, B, T, C, dout_buffer, inp_buffer, weight_buffer, mean_buffer, rstd_buffer, dinp_buffer, dweight_buffer, dbias_buffer);

    // launch program
    // EnqueueProgram(cq, layernorm_program, true);
    std::cout << "Kernel execution finished" << std::endl;

    // Read the output buffer.
    std::vector<uint32_t> dinp_tt;
    std::vector<uint32_t> dweight_tt;
    std::vector<uint32_t> dbias_tt;
    EnqueueReadBuffer(cq, dinp_buffer, dinp_tt, true);
    EnqueueReadBuffer(cq, dweight_buffer, dweight_tt, true);
    EnqueueReadBuffer(cq, dbias_buffer, dbias_tt, true);


    auto dinp_tt_back = untilize(bfloat16_packed_to_fp32(dinp_tt), shape);
    auto dweight_tt_back = bfloat16_packed_to_fp32(dweight_tt);
    auto dbias_tt_back = bfloat16_packed_to_fp32(dbias_tt);

    validate_result_vec(dinp_cpu, dinp_tt_back, "dinp", B * T * C, 1e-1f);
    validate_result_vec(dweight_cpu, dweight_tt_back, "dweight", C, 1e-1f);
    validate_result_vec(dbias_cpu, dbias_tt_back, "dbias", C, 1e-1f);

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        CloseDevice(device);
        return -1;
    }

    CloseDevice(device);
    return 0;
}
