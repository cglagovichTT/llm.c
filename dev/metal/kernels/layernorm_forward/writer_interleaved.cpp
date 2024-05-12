#include <cstdint>
#include "dataflow_api.h"
#include "debug/dprint.h"
void kernel_main()
{
    const uint32_t B = get_arg_val<uint32_t>(0);
    const uint32_t T = get_arg_val<uint32_t>(1);
    const uint32_t C = get_arg_val<uint32_t>(2);
    const uint32_t out_addr = get_arg_val<uint32_t>(3);
    const uint32_t mean_addr = get_arg_val<uint32_t>(4);
    const uint32_t rstd_addr = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_out = tt::CB::c_out0;
    constexpr uint32_t cb_mean = tt::CB::c_out1;
    constexpr uint32_t cb_rstd = tt::CB::c_out2;

    const uint32_t out_tile_size_bytes = get_tile_size(cb_out);
    constexpr DataFormat out_data_format = get_dataformat(cb_out);
    DPRINT << "out_tile_size_bytes: " << out_tile_size_bytes << ENDL();

    const InterleavedAddrGenFast<true> out_gen = {
        .bank_base_address = out_addr,
        .page_size = out_tile_size_bytes,
        .data_format = out_data_format, // The data format of the buffer
    };

    const InterleavedPow2AddrGen<true> mean_gen = {
        .bank_base_address = mean_addr,
        .log_base_2_of_page_size = 5, // log(32)
        // .data_format = mean_data_format, // The data format of the buffer
    };

    // rstd has same page size and dataformat as mean
    const InterleavedPow2AddrGen<true> rstd_gen = {
        .bank_base_address = rstd_addr,
        .log_base_2_of_page_size = 5, // log(32)
        // .data_format = mean_data_format, // The data format of the buffer
    };

    /* Enter writing loop */
    const uint32_t c_tiles = C / 32;
    uint32_t out_tile_idx = 0;
    uint32_t stats_tile_idx = 0;
    for (uint32_t b = 0; b < B; ++b) {
        for (uint32_t t_tile = 0; t_tile < T / 32; ++t_tile) {
            DPRINT << "writer: b=" << b << " t_tile=" << t_tile << ENDL();

            cb_wait_front(cb_out, c_tiles);
            uint32_t out_rd_ptr = get_read_ptr(cb_out);
            for (uint32_t c_tile = 0; c_tile < c_tiles; ++c_tile) {
                noc_async_write_tile(out_tile_idx, out_gen, out_rd_ptr);
                ++out_tile_idx;
                out_rd_ptr += out_tile_size_bytes;
            }
            noc_async_write_barrier();
            cb_pop_front(cb_out, c_tiles);

            // We should now have a page of mean and rstd to write
            cb_wait_front(cb_mean, 1);
            cb_wait_front(cb_rstd, 1);
            uint32_t mean_rd_ptr = get_read_ptr(cb_mean);
            uint32_t rstd_rd_ptr = get_read_ptr(cb_rstd);
            noc_async_write_tile(stats_tile_idx, mean_gen, mean_rd_ptr);
            noc_async_write_tile(stats_tile_idx, rstd_gen, rstd_rd_ptr);
            noc_async_write_barrier();
            cb_pop_front(cb_mean, 1);
            cb_pop_front(cb_rstd, 1);

            ++stats_tile_idx;
        }
    }
}