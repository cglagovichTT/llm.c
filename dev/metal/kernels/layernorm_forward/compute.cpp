#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api.h"
#include "debug/dprint.h"

namespace NAMESPACE {
void MAIN {
    const uint32_t B = get_arg_val<uint32_t>(0);
    const uint32_t T = get_arg_val<uint32_t>(1);
    const uint32_t C = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_inp = tt::CB::c_in0;
    constexpr uint32_t cb_weight = tt::CB::c_in1;
    constexpr uint32_t cb_bias = tt::CB::c_in2;

    constexpr uint32_t cb_out = tt::CB::c_out0;
    constexpr uint32_t cb_mean = tt::CB::c_out1;
    constexpr uint32_t cb_rstd = tt::CB::c_out2;

    const uint32_t num_weight_pages = C / 32;
    PACK( DPRINT << "num_weight_pages: " << num_weight_pages << ENDL() );

    cb_wait_front(cb_weight, num_weight_pages);
    cb_wait_front(cb_bias, num_weight_pages);

    /* Enter compute loop */
    const uint32_t c_tiles = C / 32;
    for (uint32_t b = 0; b < B; ++b) {
        for (uint32_t t_tile = 0; t_tile < T / 32; ++t_tile) {
            PACK( DPRINT << "compute: b=" << b << " t_tile=" << t_tile << ENDL() );
            cb_wait_front(cb_inp, c_tiles);
            cb_reserve_back(cb_out, c_tiles);

            for (uint32_t c_tile = 0; c_tile < c_tiles; ++c_tile) {
                // Do compute
            }

            cb_push_back(cb_out, c_tiles);
            cb_pop_front(cb_inp, c_tiles);

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