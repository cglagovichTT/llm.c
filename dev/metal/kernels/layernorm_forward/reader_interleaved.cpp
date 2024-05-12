#include <cstdint>
#include "dataflow_api.h"
#include "debug/dprint.h"
void kernel_main()
{
    const uint32_t B = get_arg_val<uint32_t>(0);
    const uint32_t T = get_arg_val<uint32_t>(1);
    const uint32_t C = get_arg_val<uint32_t>(2);
    const uint32_t inp_addr = get_arg_val<uint32_t>(3);
    const uint32_t weight_addr = get_arg_val<uint32_t>(4);
    const uint32_t bias_addr = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_inp = tt::CB::c_in0;
    constexpr uint32_t cb_weight = tt::CB::c_in1;
    constexpr uint32_t cb_bias = tt::CB::c_in2;

    const uint32_t inp_tile_size_bytes = get_tile_size(cb_inp);
    constexpr DataFormat inp_data_format = get_dataformat(cb_inp);
    DPRINT << "inp_tile_size_bytes: " << inp_tile_size_bytes << ENDL();

    const InterleavedAddrGenFast<true> inp_gen = {
        .bank_base_address = inp_addr,
        .page_size = inp_tile_size_bytes,
        .data_format = inp_data_format, // The data format of the buffer
    };

    const InterleavedPow2AddrGen<true> weight_gen = {
        .bank_base_address = weight_addr,
        .log_base_2_of_page_size = 5, // log(32)
    };

    // Bias has same page size and dataformat as weight
    const InterleavedAddrGenFast<true> bias_gen = {
        .bank_base_address = bias_addr,
        .log_base_2_of_page_size = 5, // log(32)    
    };

    const uint32_t num_weight_pages = C / 32; // page size should be 32 datums

    /* NEED something like this so we can mul bcast rows in compute
    
                        cb_reserve_back(cb_id_gamma, blk);
                    uint32_t l1_write_addr = get_write_ptr(cb_id_gamma);
                    for (uint32_t r = 0; r<blk; r++) {
                        uint64_t gamma_noc_addr = get_noc_addr(wt + r, addrg);
                        noc_async_read(gamma_noc_addr, l1_write_addr, 32);
                        gamma_noc_addr += 32;
                        noc_async_read(gamma_noc_addr, l1_write_addr + 512, 32);
                        l1_write_addr += gamma_tile_bytes;
                    }
                    noc_async_read_barrier();
                    cb_push_back(cb_id_gamma, blk);
                    */

    /* Read weight and bias once */
    cb_reserve_back(cb_weight, num_weight_pages);
    cb_reserve_back(cb_bias, num_weight_pages);
    uint32_t weight_wr_ptr = get_read_ptr(cb_weight);
    uint32_t bias_wr_ptr = get_read_ptr(cb_bias);
    for (uint32_t i = 0; i < num_weight_pages; i++) {
        DPRINT << "reader: weight=" << i << ENDL();
        noc_async_read_tile(i, weight_gen, weight_wr_ptr);
        noc_async_read_tile(i, bias_gen, bias_wr_ptr);
        weight_wr_ptr += weight_tile_size_bytes;
        bias_wr_ptr += weight_tile_size_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_weight, num_weight_pages);
    cb_push_back(cb_bias, num_weight_pages);

    /* Enter input reading loop */
    const uint32_t c_tiles = C / 32;
    uint32_t inp_tile_idx = 0;
    for (uint32_t b = 0; b < B; ++b) {
        for (uint32_t t_tile = 0; t_tile < T / 32; ++t_tile) {
            DPRINT << "reader: b=" << b << " t_tile=" << t_tile << ENDL();

            cb_reserve_back(cb_inp, c_tiles);
            uint32_t inp_wr_ptr = get_write_ptr(cb_inp);
            for (uint32_t c_tile = 0; c_tile < c_tiles; ++c_tile) {
                noc_async_read_tile(inp_tile_idx, inp_gen, inp_wr_ptr);
                inp_wr_ptr += inp_tile_size_bytes;
                ++inp_tile_idx;
            }
            noc_async_read_barrier();
            cb_push_back(cb_inp, c_tiles);
        
        }
    }
}