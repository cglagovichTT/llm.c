#include <cstdint>
#include "dataflow_api.h"

void kernel_main()
{
    uint32_t out_addr = get_arg_val<uint32_t>(0);
    uint32_t n_tiles = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_in0 = tt::CB::c_in0;

    const uint32_t tile_size_bytes = get_tile_size(cb_in0);
    constexpr DataFormat data_format = get_dataformat(cb_in0);

    const InterleavedAddrGenFast<true> s = {
        .bank_base_address = out_addr,
        .page_size = tile_size_bytes,
        .data_format = data_format, // The data format of the buffer
    };


    for(uint32_t i = 0; i < n_tiles; i++) {
        cb_wait_front(cb_in0, 1);
        uint32_t read_ptr = get_read_ptr(cb_in0);
        noc_async_write_tile(i, s, read_ptr);

        noc_async_write_barrier(); // Wait until tile reads are done
        cb_pop_front(cb_in0, 1);
    }
}