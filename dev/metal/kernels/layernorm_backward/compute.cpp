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
#include "compute_kernel_api/transpose_wh.h"

#include "debug/dprint.h"



namespace NAMESPACE {
void MAIN {
    const uint32_t B = get_arg_val<uint32_t>(0);
    const uint32_t T = get_arg_val<uint32_t>(1);
    const uint32_t C = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_inp = get_compile_time_arg_val(0);
    constexpr uint32_t cb_dout = get_compile_time_arg_val(1);
    constexpr uint32_t cb_weight = get_compile_time_arg_val(2);
    constexpr uint32_t cb_mean = get_compile_time_arg_val(3);
    constexpr uint32_t cb_rstd = get_compile_time_arg_val(4);
    constexpr uint32_t cb_dinp = get_compile_time_arg_val(5);
    constexpr uint32_t cb_dweight = get_compile_time_arg_val(6);
    constexpr uint32_t cb_dbias = get_compile_time_arg_val(7);

    constexpr uint32_t cb_identity_scalar = get_compile_time_arg_val(8);
    constexpr uint32_t cb_mean_scalar = get_compile_time_arg_val(9);

    constexpr uint32_t cb_out_dinp = get_compile_time_arg_val(10);
    constexpr uint32_t cb_out_dweight = get_compile_time_arg_val(11);
    constexpr uint32_t cb_out_dbias = get_compile_time_arg_val(12);

    constexpr uint32_t cb_scratch_0 = get_compile_time_arg_val(13);
    constexpr uint32_t cb_scratch_1 = get_compile_time_arg_val(14);
    constexpr uint32_t cb_scratch_2 = get_compile_time_arg_val(15);
    constexpr uint32_t cb_scratch_3 = get_compile_time_arg_val(16);
    constexpr uint32_t cb_scratch_4 = get_compile_time_arg_val(17);
    constexpr uint32_t cb_scratch_5 = get_compile_time_arg_val(18);
    constexpr uint32_t cb_scratch_6 = get_compile_time_arg_val(19);

    const uint32_t num_weight_pages = C / 32;
    // PACK( DPRINT << "num_weight_pages: " << num_weight_pages << ENDL() );

    binary_op_init_common(cb_inp, cb_inp);


    cb_wait_front(cb_identity_scalar, 1);
    cb_wait_front(cb_mean_scalar, 1);
    cb_wait_front(cb_dweight, num_weight_pages);
    cb_wait_front(cb_dbias, num_weight_pages);
    cb_wait_front(cb_weight, num_weight_pages);

    /* Enter compute loop */
    const uint32_t c_tiles = C / 32;
    const uint32_t t_tiles = T / 32;
    for (uint32_t b = 0; b < B; ++b) {
        for (uint32_t t_tile = 0; t_tile < t_tiles; ++t_tile) {
            // PACK( DPRINT << "compute: b=" << b << " t_tile=" << t_tile << ENDL() );

            cb_wait_front(cb_mean, 1);
            cb_wait_front(cb_rstd, 1);

            cb_wait_front(cb_inp, c_tiles);
            cb_wait_front(cb_dout, c_tiles);

            // First pass through channels to calculate two intermediate mean values
            for (uint32_t c_tile = 0; c_tile < c_tiles; ++c_tile){
                // s0 = mean_tile.transpose()
                cb_reserve_back(cb_scratch_0, 1);
                transpose_wh_init(cb_mean, cb_scratch_0);
                acquire_dst(tt::DstMode::Half);
                transpose_wh_tile(cb_mean, 0, 0);
                pack_tile(0, cb_scratch_0);
                release_dst(tt::DstMode::Half);
                cb_push_back(cb_scratch_0, 1);

                // s0 = x_tile - s0 (sub bcast cols)
                cb_wait_front(cb_scratch_0, 1);
                sub_bcast_cols_init_short(cb_inp, cb_scratch_0);
                acquire_dst(tt::DstMode::Half);
                sub_tiles_bcast_cols(cb_inp, cb_scratch_0, c_tile, 0, 0);
                cb_pop_front(cb_scratch_0, 1);
                cb_reserve_back(cb_scratch_0, 1);
                pack_tile(0, cb_scratch_0);
                release_dst(tt::DstMode::Half);
                cb_push_back(cb_scratch_0, 1);

                // s1 = rstd_tile.transpose()
                cb_reserve_back(cb_scratch_1, 1);
                transpose_wh_init(cb_rstd, cb_scratch_1);
                acquire_dst(tt::DstMode::Half);
                transpose_wh_tile(cb_rstd, 0, 0);
                pack_tile(0, cb_scratch_1);
                release_dst(tt::DstMode::Half);
                cb_push_back(cb_scratch_1, 1);

                // s0 = s0 * s1 (mul bcast cols)
                cb_wait_front(cb_scratch_1, 1);
                cb_wait_front(cb_scratch_0, 1);
                mul_bcast_cols_init_short(cb_scratch_0, cb_scratch_1);
                acquire_dst(tt::DstMode::Half);
                mul_tiles_bcast_cols(cb_scratch_0, cb_scratch_1, 0, 0, 0);
                cb_pop_front(cb_scratch_0, 1);
                cb_reserve_back(cb_scratch_0, 1);
                pack_tile(0, cb_scratch_0);
                release_dst(tt::DstMode::Half);
                cb_push_back(cb_scratch_0, 1);
                cb_pop_front(cb_scratch_1, 1);

                // s1 = dout_tile * w_tile (mul bcast rows)
                cb_reserve_back(cb_scratch_1, 1);
                mul_bcast_rows_init_short(cb_dout, cb_weight);
                acquire_dst(tt::DstMode::Half);
                mul_tiles_bcast_rows(cb_dout, cb_weight, c_tile, c_tile, 0);
                pack_tile(0, cb_scratch_1);
                release_dst(tt::DstMode::Half);
                cb_push_back(cb_scratch_1, 1);
                
                // s2 = s1 * s0 (mul)
                cb_wait_front(cb_scratch_0, 1);
                cb_wait_front(cb_scratch_1, 1);
                cb_reserve_back(cb_scratch_2, 1);
                mul_tiles_init(cb_scratch_1, cb_scratch_0);
                acquire_dst(tt::DstMode::Half);
                mul_tiles(cb_scratch_1, cb_scratch_0, 0, 0, 0);
                pack_tile(0, cb_scratch_2);
                release_dst(tt::DstMode::Half);
                cb_push_back(cb_scratch_2, 1);
                cb_pop_front(cb_scratch_0, 1);

                // s3 = reduce_sum_c(s1)
                cb_reserve_back(cb_scratch_3, 1);
                reduce_init_delta<false, REDUCE_OP, ReduceDim::REDUCE_ROW>(cb_scratch_3, cb_scratch_1, cb_identity_scalar);
                acquire_dst(tt::DstMode::Half);
                reduce_tile<REDUCE_OP, ReduceDim::REDUCE_ROW>(cb_scratch_1, cb_identity_scalar, 0, 0, 0);
                pack_tile(0, cb_scratch_3);
                release_dst(tt::DstMode::Half);
                cb_push_back(cb_scratch_3, 1);
                reduce_revert_delta<ReduceDim::REDUCE_ROW>(cb_scratch_3);
                cb_pop_front(cb_scratch_1, 1);

                // s4 = reduce_sum_c(s2)
                cb_reserve_back(cb_scratch_4, 1);
                cb_wait_front(cb_scratch_2, 1);
                reduce_init_delta<false, REDUCE_OP, ReduceDim::REDUCE_ROW>(cb_scratch_4, cb_scratch_2, cb_identity_scalar);
                acquire_dst(tt::DstMode::Half);
                reduce_tile<REDUCE_OP, ReduceDim::REDUCE_ROW>(cb_scratch_2, cb_identity_scalar, 0, 0, 0);
                pack_tile(0, cb_scratch_4);
                release_dst(tt::DstMode::Half);
                cb_push_back(cb_scratch_4, 1);
                reduce_revert_delta<ReduceDim::REDUCE_ROW>(cb_scratch_4);
                cb_pop_front(cb_scratch_2, 1);

                if (c_tile == 0) {
                    //     s5 = s3
                    //     s6 = s4
                    copy_tile_to_dst_init_short(cb_scratch_3);

                    cb_reserve_back(cb_scratch_5, 1);
                    cb_wait_front(cb_scratch_3, 1);
                    copy_tile(cb_scratch_3, 0, 0);
                    pack_tile(0, cb_scratch_5);
                    cb_push_back(cb_scratch_5, 1);
                    cb_pop_front(cb_scratch_3, 1);

                    cb_reserve_back(cb_scratch_6, 1);
                    cb_wait_front(cb_scratch_4, 1);
                    copy_tile(cb_scratch_4, 0, 0);
                    pack_tile(0, cb_scratch_6);
                    cb_push_back(cb_scratch_6, 1);
                    cb_pop_front(cb_scratch_4, 1);
                } else {
                    //     s5 += s3 (add tiles)
                    //     s6 += s4 (add tiles)
                    add_tiles_init(cb_scratch_5, cb_scratch_3);

                    cb_wait_front(cb_scratch_5, 1);
                    cb_wait_front(cb_scratch_3, 1);
                    acquire_dst(tt::DstMode::Half);
                    add_tiles(cb_scratch_5, cb_scratch_3, 0, 0, 0);
                    cb_pop_front(cb_scratch_5, 1);
                    cb_reserve_back(cb_scratch_5, 1);
                    pack_tile(0, cb_scratch_5);
                    release_dst(tt::DstMode::Half);
                    cb_push_back(cb_scratch_5, 1);
                    cb_pop_front(cb_scratch_3, 1);

                    cb_wait_front(cb_scratch_6, 1);
                    cb_wait_front(cb_scratch_4, 1);
                    acquire_dst(tt::DstMode::Half);
                    add_tiles(cb_scratch_6, cb_scratch_4, 0, 0, 0);
                    cb_pop_front(cb_scratch_6, 1);
                    cb_reserve_back(cb_scratch_6, 1);
                    pack_tile(0, cb_scratch_6);
                    release_dst(tt::DstMode::Half);
                    cb_push_back(cb_scratch_6, 1);
                    cb_pop_front(cb_scratch_4, 1);
                }

            }

            mul_tiles_init(cb_scratch_5, cb_mean_scalar);
            
            // s5 = s5 * mean_scalar
            cb_wait_front(cb_scratch_5, 1);
            acquire_dst(tt::DstMode::Half);
            mul_tiles(cb_scratch_5, cb_mean_scalar, 0, 0, 0);
            cb_pop_front(cb_scratch_5, 1);
            cb_reserve_back(cb_scratch_5, 1);
            pack_tile(0, cb_scratch_5);
            release_dst(tt::DstMode::Half);
            cb_push_back(cb_scratch_5, 1);

            // s6 = s6 * mean_scalar
            cb_wait_front(cb_scratch_6, 1);
            acquire_dst(tt::DstMode::Half);
            mul_tiles(cb_scratch_6, cb_mean_scalar, 0, 0, 0);
            cb_pop_front(cb_scratch_6, 1);
            cb_reserve_back(cb_scratch_6, 1);
            pack_tile(0, cb_scratch_6);
            release_dst(tt::DstMode::Half);
            cb_push_back(cb_scratch_6, 1);
            
            
            cb_wait_front(cb_dinp, c_tiles);

            for (uint32_t c_tile = 0; c_tile < c_tiles; ++c_tile){
                // s0 = mean_tile.transpose()
                cb_reserve_back(cb_scratch_0, 1);
                transpose_wh_init(cb_mean, cb_scratch_0);
                acquire_dst(tt::DstMode::Half);
                transpose_wh_tile(cb_mean, 0, 0);
                pack_tile(0, cb_scratch_0);
                release_dst(tt::DstMode::Half);
                cb_push_back(cb_scratch_0, 1);

                // s0 = x_tile - s0 (sub bcast cols)
                cb_wait_front(cb_scratch_0, 1);
                sub_bcast_cols_init_short(cb_inp, cb_scratch_0);
                acquire_dst(tt::DstMode::Half);
                sub_tiles_bcast_cols(cb_inp, cb_scratch_0, c_tile, 0, 0);
                cb_pop_front(cb_scratch_0, 1);
                cb_reserve_back(cb_scratch_0, 1);
                pack_tile(0, cb_scratch_0);
                release_dst(tt::DstMode::Half);
                cb_push_back(cb_scratch_0, 1);

                // s1 = rstd_tile.transpose()
                cb_reserve_back(cb_scratch_1, 1);
                transpose_wh_init(cb_rstd, cb_scratch_1);
                acquire_dst(tt::DstMode::Half);
                transpose_wh_tile(cb_rstd, 0, 0);
                pack_tile(0, cb_scratch_1);
                release_dst(tt::DstMode::Half);
                cb_push_back(cb_scratch_1, 1);

                // s0 = s0 * s1 (mul bcast cols)
                cb_wait_front(cb_scratch_1, 1);
                cb_wait_front(cb_scratch_0, 1);
                mul_bcast_cols_init_short(cb_scratch_0, cb_scratch_1);
                acquire_dst(tt::DstMode::Half);
                mul_tiles_bcast_cols(cb_scratch_0, cb_scratch_1, 0, 0, 0);
                cb_pop_front(cb_scratch_0, 1);
                cb_reserve_back(cb_scratch_0, 1);
                pack_tile(0, cb_scratch_0);
                release_dst(tt::DstMode::Half);
                cb_push_back(cb_scratch_0, 1);
                cb_pop_front(cb_scratch_1, 1);

                // s1 = dout_tile * w_tile (mul bcast rows)
                cb_reserve_back(cb_scratch_1, 1);
                mul_bcast_rows_init_short(cb_dout, cb_weight);
                acquire_dst(tt::DstMode::Half);
                mul_tiles_bcast_rows(cb_dout, cb_weight, c_tile, c_tile, 0);
                pack_tile(0, cb_scratch_1);
                release_dst(tt::DstMode::Half);
                cb_push_back(cb_scratch_1, 1);

                // s2 = reduce_sum_r(dout_tile) (reduce rows into top row)
                cb_reserve_back(cb_scratch_2, 1);
                reduce_init_delta<false, REDUCE_OP, ReduceDim::REDUCE_COL>(cb_scratch_2, cb_dout, cb_identity_scalar);
                acquire_dst(tt::DstMode::Half);
                reduce_tile<REDUCE_OP, ReduceDim::REDUCE_COL>(cb_dout, cb_identity_scalar, c_tile, 0, 0);
                pack_tile(0, cb_scratch_2);
                release_dst(tt::DstMode::Half);
                cb_push_back(cb_scratch_2, 1);

                // dbias_tile = dbias_tile + s2 (add, no bcast)
                // in-place add to cb_dbias because we own it after reader pushes it
                add_tiles_init(cb_dbias, cb_scratch_2);
                cb_wait_front(cb_dbias, 1);
                cb_wait_front(cb_scratch_2, 1);
                acquire_dst(tt::DstMode::Half);
                add_tiles(cb_dbias, cb_scratch_2, 0, 0, 0);
                cb_pop_front(cb_dbias, 1);
                cb_reserve_back(cb_dbias, 1);
                pack_tile(0, cb_dbias);
                release_dst(tt::DstMode::Half);
                cb_push_back(cb_dbias, 1);
                cb_pop_front(cb_scratch_2, 1);

                // s2 = s0 * dout_tile
                cb_wait_front(cb_scratch_0, 1);
                cb_wait_front(cb_dout, 1);
                cb_reserve_back(cb_scratch_2, 1);
                mul_tiles_init(cb_scratch_0, cb_dout);
                acquire_dst(tt::DstMode::Half);
                mul_tiles(cb_scratch_0, cb_dout, 0, c_tile, 0);
                pack_tile(0, cb_scratch_2);
                release_dst(tt::DstMode::Half);
                cb_push_back(cb_scratch_2, 1);
                cb_pop_front(cb_scratch_0, 1);

                // s2 = reduce_sum_r(s2) (reduce rows into top row)
                cb_wait_front(cb_scratch_2, 1);
                reduce_init_delta<false, REDUCE_OP, ReduceDim::REDUCE_COL>(cb_scratch_2, cb_scratch_2, cb_identity_scalar);
                acquire_dst(tt::DstMode::Half);
                reduce_tile<REDUCE_OP, ReduceDim::REDUCE_COL>(cb_scratch_2, cb_identity_scalar, 0, 0, 0);
                cb_pop_front(cb_scratch_2, 1);
                cb_reserve_back(cb_scratch_2, 1);
                pack_tile(0, cb_scratch_2);
                release_dst(tt::DstMode::Half);
                cb_push_back(cb_scratch_2, 1);
                reduce_revert_delta<ReduceDim::REDUCE_COL>(cb_scratch_2);

                // dweight_tile = dweight_tile + s2 (add, no bcast)
                // in-place add to cb_dweight because we own it after reader pushes it
                add_tiles_init(cb_dweight, cb_scratch_2);
                cb_wait_front(cb_dweight, 1);
                cb_wait_front(cb_scratch_2, 1);
                acquire_dst(tt::DstMode::Half);
                add_tiles(cb_dweight, cb_scratch_2, 0, 0, 0);
                cb_pop_front(cb_dweight, 1);
                cb_reserve_back(cb_dweight, 1);
                pack_tile(0, cb_dweight);
                release_dst(tt::DstMode::Half);
                cb_push_back(cb_dweight, 1);
                cb_pop_front(cb_scratch_2, 1);

                // s4 = s1 - s5 (sub bast cols)
                cb_wait_front(cb_scratch_1, 1);
                cb_wait_front(cb_scratch_5, 1);
                sub_bcast_cols_init_short(cb_scratch_1, cb_scratch_5);
                acquire_dst(tt::DstMode::Half);
                sub_tiles_bcast_cols(cb_scratch_1, cb_scratch_5, 0, 0, 0);
                pack_tile(0, cb_scratch_4);
                release_dst(tt::DstMode::Half);
                cb_push_back(cb_scratch_4, 1);
                cb_pop_front(cb_scratch_1, 1);

                // s3 = s0 * s6 (mul bcast cols)
                cb_wait_front(cb_scratch_0, 1);
                cb_wait_front(cb_scratch_6, 1);
                mul_bcast_cols_init_short(cb_scratch_0, cb_scratch_6);
                acquire_dst(tt::DstMode::Half);
                mul_tiles_bcast_cols(cb_scratch_0, cb_scratch_6, 0, 0, 0);
                pack_tile(0, cb_scratch_3);
                release_dst(tt::DstMode::Half);
                cb_push_back(cb_scratch_3, 1);
                cb_pop_front(cb_scratch_0, 1);

                // s4 = s4 - s3
                cb_wait_front(cb_scratch_4, 1);
                cb_wait_front(cb_scratch_3, 1);
                sub_tiles_init(cb_scratch_4, cb_scratch_3);
                acquire_dst(tt::DstMode::Half);
                sub_tiles(cb_scratch_4, cb_scratch_3, 0, 0, 0);
                cb_pop_front(cb_scratch_4, 1);
                cb_reserve_back(cb_scratch_4, 1);
                pack_tile(0, cb_scratch_4);
                release_dst(tt::DstMode::Half);
                cb_push_back(cb_scratch_4, 1);
                cb_pop_front(cb_scratch_3, 1);

                // s1 = rstd_tile.transpose()
                cb_reserve_back(cb_scratch_1, 1);
                transpose_wh_init(cb_rstd, cb_scratch_1);
                acquire_dst(tt::DstMode::Half);
                transpose_wh_tile(cb_rstd, 0, 0);
                pack_tile(0, cb_scratch_1);
                release_dst(tt::DstMode::Half);
                cb_push_back(cb_scratch_1, 1);

                // s4 = s4 * s1 (mul bcast cols)
                cb_wait_front(cb_scratch_1, 1);
                cb_wait_front(cb_scratch_4, 1);
                mul_bcast_cols_init_short(cb_scratch_4, cb_scratch_1);
                acquire_dst(tt::DstMode::Half);
                mul_tiles_bcast_cols(cb_scratch_4, cb_scratch_1, 0, 0, 0);
                cb_pop_front(cb_scratch_4, 1);
                cb_reserve_back(cb_scratch_4, 1);
                pack_tile(0, cb_scratch_4);
                release_dst(tt::DstMode::Half);
                cb_push_back(cb_scratch_4, 1);
                cb_pop_front(cb_scratch_1, 1);

                // dinp_out_tile = dinp_tile + s4 (add, no bcast)
                add_tiles_init(cb_dinp, cb_scratch_4);
                cb_wait_front(cb_dinp, 1);
                cb_wait_front(cb_scratch_4, 1);
                cb_reserve_back(cb_out_dinp, 1);
                acquire_dst(tt::DstMode::Half);
                add_tiles(cb_dinp, cb_scratch_4, c_tile, 0, 0);
                pack_tile(0, cb_out_dinp);
                release_dst(tt::DstMode::Half);
                cb_push_back(cb_out_dinp, 1);
                cb_pop_front(cb_scratch_4, 1);
            }
            cb_pop_front(cb_scratch_5, 1);
            cb_pop_front(cb_scratch_6, 1);

            cb_pop_front(cb_dinp, c_tiles);
            cb_pop_front(cb_dout, c_tiles);
            cb_pop_front(cb_inp, c_tiles);

            cb_pop_front(cb_rstd, 1);
            cb_pop_front(cb_mean, 1);


            // We write to out_dinp inside of c_tiles loop
            // cb_reserve_back(cb_out_dinp, c_tiles);
            // cb_push_back(cb_out_dinp, c_tiles);

        }
    }
    // Write cb_dweight and cb_dbias to out_dweight and out_dbias
    copy_tile_to_dst_init_short(cb_dweight);
    for (uint32_t c_tile = 0; c_tile < c_tiles; ++c_tile) {
        cb_wait_front(cb_dweight, 1);
        cb_reserve_back(cb_out_dweight, 1);
        acquire_dst(tt::DstMode::Half);
        copy_tile(cb_dweight, 0, 0);
        pack_tile(0, cb_out_dweight);
        release_dst(tt::DstMode::Half);
        cb_push_back(cb_out_dweight, 1);
        
        cb_wait_front(cb_dbias, 1);
        cb_reserve_back(cb_out_dbias, 1);
        acquire_dst(tt::DstMode::Half);
        copy_tile(cb_dbias, 0, 0);
        pack_tile(0, cb_out_dbias);
        release_dst(tt::DstMode::Half);
        cb_push_back(cb_out_dbias, 1);

    }
    // cb_wait_front()
    // cb_reserve_back(cb_out_dweight, c_tiles);
    // cb_push_back(cb_out_dweight, c_tiles);
    
    // cb_reserve_back(cb_out_dbias, c_tiles);
    // cb_push_back(cb_out_dbias, c_tiles);

    cb_pop_front(cb_identity_scalar, 1);
    cb_pop_front(cb_mean_scalar, 1);
    cb_pop_front(cb_dweight, num_weight_pages);
    cb_pop_front(cb_dbias, num_weight_pages);
    cb_pop_front(cb_weight, num_weight_pages);

            // cb_wait_front(cb_inp, c_tiles);
            // // cb_reserve_back(cb_out, c_tiles);

            // // Sum inp
            // reduce_init_delta<false, PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_inp, cb_inp, cb_one_scalar);

            // cb_reserve_back(cb_intermed_mean, 1);
            // tile_regs_acquire();
            // for (uint32_t c_tile = 0; c_tile < c_tiles; ++c_tile) {
            //     reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_inp, cb_one_scalar, c_tile, 0, 0);
            // }
            // tile_regs_commit();
            // tile_regs_wait();
            // pack_tile(0, cb_intermed_mean);
            // tile_regs_release();
            // cb_push_back(cb_intermed_mean, 1);
            // reduce_revert_delta< ReduceDim::REDUCE_ROW>();

            // // Get mean
            // cb_wait_front(cb_intermed_mean, 1);
            // mul_tiles_init(cb_intermed_mean, cb_mean_recip_scalar);
            // tile_regs_acquire();
            // mul_tiles(cb_intermed_mean, cb_mean_recip_scalar, 0, 0, 0);
            // tile_regs_commit();
            // cb_pop_front(cb_intermed_mean, 1);
            // cb_reserve_back(cb_intermed_mean, 1);
            // tile_regs_wait();
            // pack_tile(0, cb_intermed_mean);
            // tile_regs_release();
            // cb_push_back(cb_intermed_mean, 1);

            // // x - mean
            // cb_wait_front(cb_intermed_mean, 1);
            // sub_bcast_cols_init_short(cb_inp, cb_intermed_mean);
            // for (uint32_t c_tile = 0; c_tile < c_tiles; ++c_tile) {
            //     cb_reserve_back(cb_xmm, 1);
            //     tile_regs_acquire();
            //     sub_tiles_bcast_cols(cb_inp, cb_intermed_mean, c_tile, 0, 0);
            //     tile_regs_commit();
            //     tile_regs_wait();
            //     pack_tile(0, cb_xmm);
            //     tile_regs_release();
            //     cb_push_back(cb_xmm, 1);
            // }
            // // Clear out cb_pop_front since this is its last use
            // cb_pop_front(cb_inp, c_tiles);
            // // cb_pop_front(cb_intermed_mean, 1);

            // // (x - mean) **2
            // mul_tiles_init(cb_xmm, cb_xmm);
            // cb_wait_front(cb_xmm, c_tiles);
            // for (uint32_t c_tile = 0; c_tile < c_tiles; ++c_tile) {
            //     cb_reserve_back(cb_xmm2, 1);
            //     tile_regs_acquire();
            //     mul_tiles(cb_xmm, cb_xmm, c_tile, c_tile, 0);
            //     tile_regs_commit();
            //     tile_regs_wait();
            //     pack_tile(0, cb_xmm2);
            //     tile_regs_release();
            //     cb_push_back(cb_xmm2, 1);
            // }

            // // Sum (x - mean) **2
            // reduce_init_delta<false, PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_intermed_rstd, cb_xmm2, cb_one_scalar);
            // cb_reserve_back(cb_intermed_rstd, 1);
            // cb_wait_front(cb_xmm2, c_tiles);
            // tile_regs_acquire();
            // for (uint32_t c_tile = 0; c_tile < c_tiles; ++c_tile) {
            //     reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_xmm2, cb_one_scalar, c_tile, 0, 0);
            // }
            // tile_regs_commit();
            // tile_regs_wait();
            // pack_tile(0, cb_intermed_rstd);
            // tile_regs_release();
            // cb_push_back(cb_intermed_rstd, 1);
            // reduce_revert_delta<ReduceDim::REDUCE_ROW>();
            // // last use of cb_xmm2
            // cb_pop_front(cb_xmm2, c_tiles);

            // // Get mean(x - mean) ** 2
            // cb_wait_front(cb_intermed_rstd, 1);
            // mul_tiles_init(cb_intermed_rstd, cb_mean_recip_scalar);
            // tile_regs_acquire();
            // mul_tiles(cb_intermed_rstd, cb_mean_recip_scalar, 0, 0, 0);
            // tile_regs_commit();
            // cb_pop_front(cb_intermed_rstd, 1);
            // cb_reserve_back(cb_intermed_rstd, 1);
            // tile_regs_wait();
            // pack_tile(0, cb_intermed_rstd);
            // tile_regs_release();
            // cb_push_back(cb_intermed_rstd, 1);

            // // Get rstd
            // // TODO: Add epsilon!
            // // TODO: Get rid of copy tile, and just sqrt+recip while tile in DST
            // cb_wait_front(cb_intermed_rstd, 1);
            // // copy_tile_to_dst_init_short(cb_intermed_rstd);
            // add_tiles_init();
            // tile_regs_acquire();
            // add_tiles(cb_intermed_rstd, cb_epsilon_scalar, 0, 0, 0);
            // sqrt_tile_init();
            // sqrt_tile(0);
            // recip_tile_init();
            // recip_tile(0);
            // tile_regs_commit();
            // cb_pop_front(cb_intermed_rstd, 1);
            // cb_reserve_back(cb_intermed_rstd, 1);
            // tile_regs_wait();
            // pack_tile(0, cb_intermed_rstd);
            // tile_regs_release();
            // cb_push_back(cb_intermed_rstd, 1);


            // // (x - mean) * rstd
            // mul_bcast_cols_init_short(cb_xmm, cb_intermed_rstd);
            // cb_wait_front(cb_xmm, c_tiles);
            // cb_wait_front(cb_intermed_rstd, 1);
            // for (uint32_t c_tile = 0; c_tile < c_tiles; ++c_tile) {
            //     cb_reserve_back(cb_xmm_rstd, 1);
            //     tile_regs_acquire();
            //     mul_tiles_bcast_cols(cb_xmm, cb_intermed_rstd, c_tile, 0, 0);
            //     tile_regs_commit();
            //     tile_regs_wait();
            //     pack_tile(0, cb_xmm_rstd);
            //     tile_regs_release();
            //     cb_push_back(cb_xmm_rstd, 1);
            // }
            // // last use, pop cb_xmm
            // cb_pop_front(cb_xmm, c_tiles);
            // // cb_pop_front(cb_intermed_rstd, 1);

            // // Scale by weight
            // mul_bcast_rows_init_short(cb_xmm_rstd, cb_weight);
            // cb_wait_front(cb_xmm_rstd, c_tiles);
            // for (uint32_t c_tile = 0; c_tile < c_tiles; ++c_tile) {
            //     cb_reserve_back(cb_xmm_rstd_scaled, 1);
            //     tile_regs_acquire();
            //     mul_tiles_bcast_rows(cb_xmm_rstd, cb_weight, c_tile, c_tile, 0);
            //     tile_regs_commit();
            //     tile_regs_wait();
            //     pack_tile(0, cb_xmm_rstd_scaled);
            //     tile_regs_release();
            //     cb_push_back(cb_xmm_rstd_scaled, 1);
            // }
            // cb_pop_front(cb_xmm_rstd, c_tiles);

            // // Add bias
            // add_bcast_rows_init_short(cb_xmm_rstd_scaled, cb_bias);
            // cb_wait_front(cb_xmm_rstd_scaled, c_tiles);
            // for (uint32_t c_tile = 0; c_tile < c_tiles; ++c_tile) {
            //     cb_reserve_back(cb_out, 1);
            //     tile_regs_acquire();
            //     add_tiles_bcast_rows(cb_xmm_rstd_scaled, cb_bias, c_tile, c_tile, 0);
            //     tile_regs_commit();
            //     tile_regs_wait();
            //     pack_tile(0, cb_out);
            //     tile_regs_release();
            //     cb_push_back(cb_out, 1);
            // }
            // cb_pop_front(cb_xmm_rstd_scaled, c_tiles);

            // // We should now have a page of mean and rstd to give to writer
            // cb_reserve_back(cb_mean, 1);
            // cb_reserve_back(cb_rstd, 1);

            // // Copy Mean
            // transpose_wh_init_short(cb_intermed_mean);
            // acquire_dst(tt::DstMode::Half);
            // transpose_wh_tile(cb_intermed_mean, 0, 0);
            // pack_tile(0, cb_mean);
            // release_dst(tt::DstMode::Half);
            // cb_push_back(cb_mean, 1);
            // cb_pop_front(cb_intermed_mean, 1);
            // // Copy Rstd
            // acquire_dst(tt::DstMode::Half);
            // transpose_wh_tile(cb_intermed_rstd, 0, 0);
            // pack_tile(0, cb_rstd);
            // release_dst(tt::DstMode::Half);
            // cb_push_back(cb_rstd, 1);
            // cb_pop_front(cb_intermed_rstd, 1);
        // }
    // }

    // cb_pop_front(cb_weight, num_weight_pages);
    // cb_pop_front(cb_bias, num_weight_pages);
}
}
