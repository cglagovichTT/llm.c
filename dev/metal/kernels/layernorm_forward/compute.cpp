#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#include <cstdint>
#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/eltwise_unary/sqrt.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api/bcast.h"

#include "debug/dprint.h"



namespace NAMESPACE {
void MAIN {
    const uint32_t B = get_arg_val<uint32_t>(0);
    const uint32_t T = get_arg_val<uint32_t>(1);
    const uint32_t C = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_inp = tt::CB::c_in0;
    constexpr uint32_t cb_weight = tt::CB::c_in1;
    constexpr uint32_t cb_bias = tt::CB::c_in2;
    constexpr uint32_t cb_one_scalar = tt::CB::c_in3;
    constexpr uint32_t cb_mean_recip_scalar = tt::CB::c_in4;
    constexpr uint32_t cb_epsilon_scalar = tt::CB::c_in5;

    constexpr uint32_t cb_out = tt::CB::c_out0;
    constexpr uint32_t cb_mean = tt::CB::c_out1;
    constexpr uint32_t cb_rstd = tt::CB::c_out2;

    constexpr uint32_t cb_xmm = tt::CB::c_intermed0;
    constexpr uint32_t cb_xmm2 = tt::CB::c_intermed1;
    constexpr uint32_t cb_intermed_mean = tt::CB::c_intermed2;
    constexpr uint32_t cb_intermed_rstd = tt::CB::c_intermed3;
    constexpr uint32_t cb_xmm_rstd = tt::CB::c_intermed4;
    constexpr uint32_t cb_xmm_rstd_scaled = tt::CB::c_intermed5;

    const uint32_t num_weight_pages = C / 32;
    // PACK( DPRINT << "num_weight_pages: " << num_weight_pages << ENDL() );

    binary_op_init_common(cb_inp, cb_inp);


    cb_wait_front(cb_one_scalar, 1);
    cb_wait_front(cb_mean_recip_scalar, 1);
    cb_wait_front(cb_epsilon_scalar, 1);
    cb_wait_front(cb_weight, num_weight_pages);
    cb_wait_front(cb_bias, num_weight_pages);

    /* Enter compute loop */
    const uint32_t c_tiles = C / 32;
    for (uint32_t b = 0; b < B; ++b) {
        for (uint32_t t_tile = 0; t_tile < T / 32; ++t_tile) {
            PACK( DPRINT << "compute: b=" << b << " t_tile=" << t_tile << ENDL() );
            cb_wait_front(cb_inp, c_tiles);
            // cb_reserve_back(cb_out, c_tiles);

            // Sum inp
            reduce_init_delta<false, PoolType::SUM, ReduceDim::REDUCE_ROW>(PoolType::SUM, ReduceDim::REDUCE_ROW, cb_intermed_mean, cb_inp, cb_one_scalar);

            cb_reserve_back(cb_intermed_mean, 1);
            tile_regs_acquire();
            for (uint32_t c_tile = 0; c_tile < c_tiles; ++c_tile) {
                reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_inp, cb_one_scalar, c_tile, 0, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_intermed_mean);
            tile_regs_release();
            cb_push_back(cb_intermed_mean, 1);
            reduce_revert_delta< ReduceDim::REDUCE_ROW>();

            // Get mean
            cb_wait_front(cb_intermed_mean, 1);
            mul_tiles_init(cb_intermed_mean, cb_mean_recip_scalar);
            tile_regs_acquire();
            mul_tiles(cb_intermed_mean, cb_mean_recip_scalar, 0, 0, 0);
            tile_regs_commit();
            cb_pop_front(cb_intermed_mean, 1);
            cb_reserve_back(cb_intermed_mean, 1);
            tile_regs_wait();
            pack_tile(0, cb_intermed_mean);
            tile_regs_release();
            cb_push_back(cb_intermed_mean, 1);

            // x - mean
            cb_wait_front(cb_intermed_mean, 1);
            sub_bcast_cols_init_short(cb_inp, cb_intermed_mean);
            for (uint32_t c_tile = 0; c_tile < c_tiles; ++c_tile) {
                cb_reserve_back(cb_xmm, 1);
                tile_regs_acquire();
                sub_tiles_bcast_cols(cb_inp, cb_intermed_mean, c_tile, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_xmm);
                tile_regs_release();
                cb_push_back(cb_xmm, 1);
            }
            // Clear out cb_pop_front since this is its last use
            cb_pop_front(cb_inp, c_tiles);
            cb_pop_front(cb_intermed_mean, 1);

            // (x - mean) **2
            mul_tiles_init(cb_xmm, cb_xmm);
            cb_wait_front(cb_xmm, c_tiles);
            for (uint32_t c_tile = 0; c_tile < c_tiles; ++c_tile) {
                cb_reserve_back(cb_xmm2, 1);
                tile_regs_acquire();
                mul_tiles(cb_xmm, cb_xmm, c_tile, c_tile, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_xmm2);
                tile_regs_release();
                cb_push_back(cb_xmm2, 1);
            }

            // Sum (x - mean) **2
            reduce_init_delta<false, PoolType::SUM, ReduceDim::REDUCE_ROW>(PoolType::SUM, ReduceDim::REDUCE_ROW, cb_intermed_rstd, cb_xmm2, cb_one_scalar);
            cb_reserve_back(cb_intermed_rstd, 1);
            cb_wait_front(cb_xmm2, c_tiles);
            tile_regs_acquire();
            for (uint32_t c_tile = 0; c_tile < c_tiles; ++c_tile) {
                reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_xmm2, cb_one_scalar, c_tile, 0, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_intermed_rstd);
            tile_regs_release();
            cb_push_back(cb_intermed_rstd, 1);
            reduce_revert_delta<ReduceDim::REDUCE_ROW>();
            // last use of cb_xmm2
            cb_pop_front(cb_xmm2, c_tiles);

            // Get rstd
            // TODO: Add epsilon!
            // TODO: Get rid of copy tile, and just sqrt+recip while tile in DST
            cb_wait_front(cb_intermed_rstd, 1);
            // copy_tile_to_dst_init_short(cb_intermed_rstd);
            add_bcast_scalar_init_short();
            tile_regs_acquire();
            add_tiles_bcast_scalar(cb_intermed_rstd, cb_epsilon_scalar, 0, 0, 0);
            sqrt_tile_init();
            sqrt_tile(0);
            recip_tile_init();
            recip_tile(0);
            tile_regs_commit();
            cb_pop_front(cb_intermed_rstd, 1);
            cb_reserve_back(cb_intermed_rstd, 1);
            tile_regs_wait();
            pack_tile(0, cb_intermed_rstd);
            tile_regs_release();
            cb_push_back(cb_intermed_rstd, 1);


            // (x - mean) * rstd
            mul_bcast_cols_init_short(cb_xmm, cb_intermed_rstd);
            cb_wait_front(cb_xmm, c_tiles);
            cb_wait_front(cb_intermed_rstd, 1);
            for (uint32_t c_tile = 0; c_tile < c_tiles; ++c_tile) {
                cb_reserve_back(cb_xmm_rstd, 1);
                tile_regs_acquire();
                mul_tiles_bcast_cols(cb_xmm, cb_intermed_rstd, c_tile, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_xmm_rstd);
                tile_regs_release();
                cb_push_back(cb_xmm_rstd, 1);
            }
            // last use, pop cb_xmm
            cb_pop_front(cb_xmm, c_tiles);
            cb_pop_front(cb_intermed_rstd, 1);

            // Scale by weight
            mul_bcast_rows_init_short(cb_xmm_rstd, cb_weight);
            cb_wait_front(cb_xmm_rstd, c_tiles);
            for (uint32_t c_tile = 0; c_tile < c_tiles; ++c_tile) {
                cb_reserve_back(cb_xmm_rstd_scaled, 1);
                tile_regs_acquire();
                mul_tiles_bcast_rows(cb_xmm_rstd, cb_weight, c_tile, c_tile, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_xmm_rstd_scaled);
                tile_regs_release();
                cb_push_back(cb_xmm_rstd_scaled, 1);
            }
            cb_pop_front(cb_xmm_rstd, c_tiles);

            // Add bias
            add_bcast_rows_init_short(cb_xmm_rstd_scaled, cb_bias);
            cb_wait_front(cb_xmm_rstd_scaled, c_tiles);
            for (uint32_t c_tile = 0; c_tile < c_tiles; ++c_tile) {
                cb_reserve_back(cb_out, 1);
                tile_regs_acquire();
                add_tiles_bcast_rows(cb_xmm_rstd_scaled, cb_bias, c_tile, c_tile, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_out);
                tile_regs_release();
                cb_push_back(cb_out, 1);
            }
            cb_pop_front(cb_xmm_rstd_scaled, c_tiles);


            // cb_push_back(cb_out, c_tiles);
            // cb_pop_front(cb_inp, c_tiles);

            // We should now have a page of mean and rstd to give to writer
            cb_reserve_back(cb_mean, 1);
            cb_reserve_back(cb_rstd, 1);
            // Do compute

            cb_push_back(cb_mean, 1);
            cb_push_back(cb_rstd, 1);

        }
    }

    cb_pop_front(cb_weight, num_weight_pages);
    cb_pop_front(cb_bias, num_weight_pages);
}
}