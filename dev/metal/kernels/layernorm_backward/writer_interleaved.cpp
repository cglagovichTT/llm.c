#include <cstdint>
#include "dataflow_api.h"
#include "debug/dprint.h"

FORCE_INLINE void generate_reduce_scaler(const uint32_t cb_id, const uint32_t scaler) {
    cb_reserve_back(cb_id, 1);

    constexpr uint32_t num_zeros_reads = 2048 / MEM_ZEROS_SIZE;
    uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
    uint32_t write_addr = get_write_ptr(cb_id);
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(write_addr);

    // Fill tile with zeros
    for (uint32_t i = 0; i < num_zeros_reads; ++i) {
        noc_async_read(zeros_noc_addr, write_addr, MEM_ZEROS_SIZE);
        write_addr += MEM_ZEROS_SIZE;
    }
    noc_async_read_barrier();

    if (scaler != 0) {
        for (int k = 0; k < 4; ++k) {
            uint32_t idx = k << 7;
            for (int j = 0; j < 8; ++j) {
                ptr[idx + j] = scaler;
            }
        }
    }
    cb_push_back(cb_id, 1);
}

FORCE_INLINE void generate_bcast_col_scalar(const uint32_t cb_id, const uint32_t scalar) {
    const uint16_t scalar_val = scalar>>16;
    cb_reserve_back(cb_id, 1);
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_id));
    for (int k = 0; k < 4; k+=2) {
        uint32_t idx = k << 8;
        for (int j = 0; j < 256; j+=16) {
            ptr[idx + j] = scalar_val;
        }
    }
    cb_push_back(cb_id, 1);
}

void kernel_main()
{
    const uint32_t B = get_arg_val<uint32_t>(0);
    const uint32_t T = get_arg_val<uint32_t>(1);
    const uint32_t C = get_arg_val<uint32_t>(2);
    const uint32_t dinp_buffer = get_arg_val<uint32_t>(3);
    const uint32_t dweight_buffer = get_arg_val<uint32_t>(4);
    const uint32_t dbias_buffer = get_arg_val<uint32_t>(5);

    constexpr uint32_t one_scalar = get_compile_time_arg_val(0);
    constexpr uint32_t mean_scalar = get_compile_time_arg_val(1);
    constexpr uint32_t log_page_size = get_compile_time_arg_val(2);
    constexpr uint32_t cb_identity_scalar = get_compile_time_arg_val(3);
    constexpr uint32_t cb_mean_scalar = get_compile_time_arg_val(4);
    constexpr uint32_t cb_out_dinp = get_compile_time_arg_val(5);
    constexpr uint32_t cb_out_dweight = get_compile_time_arg_val(6);
    constexpr uint32_t cb_out_dbias = get_compile_time_arg_val(7);

    const uint32_t tile_size_bytes = get_tile_size(cb_out_dinp);
    constexpr DataFormat tile_data_format = get_dataformat(cb_out_dinp);
    // DPRINT << "out_tile_size_bytes: " << out_tile_size_bytes << ENDL();

    // Generate constant tiles for layernorm compute
    generate_reduce_scaler(cb_identity_scalar, one_scalar);
    generate_bcast_col_scalar(cb_mean_scalar, mean_scalar);

    const InterleavedAddrGenFast<true> dinp_gen = {
        .bank_base_address = dinp_buffer,
        .page_size = tile_size_bytes,
        .data_format = tile_data_format, // The data format of the buffer
    };

    const InterleavedPow2AddrGen<true> dweight_gen = {
        .bank_base_address = dweight_buffer,
        .log_base_2_of_page_size = log_page_size,
    };

    // rstd has same page size and dataformat as mean
    const InterleavedPow2AddrGen<true> dbias_gen = {
        .bank_base_address = dbias_buffer,
        .log_base_2_of_page_size = log_page_size,
    };

    // How many bytes is one vector of 16 weight datums
    constexpr uint32_t face_row_bf16_byte = 16 * 2;
    // How many bytes do you stride to get to the next face
    constexpr uint32_t face_bf16_bytes = 16 * 16 * 2;

    /* Enter writing loop */
    const uint32_t c_tiles = C / 32;
    const uint32_t t_tiles = T / 32;
    for (uint32_t b = start_b; b < end_b; ++b) {
	uint32_t batch_tile_offset = b * t_tiles * c_tiles;
        for (uint32_t t_tile = start_t; t_tile < end_t; ++t_tile) {
            // DPRINT << "writer: b=" << b << " t_tile=" << t_tile << ENDL();
	    uint32_t seq_start_tile = batch_tile_offset + t_tile * c_tiles; 

            cb_wait_front(cb_out_dinp, c_tiles);
            uint32_t out_rd_ptr = get_read_ptr(cb_out_dinp);
            for (uint32_t c_tile = 0; c_tile < c_tiles; ++c_tile) {
                noc_async_write_tile(seq_start_tile, dinp_gen, out_rd_ptr);
                ++seq_start_tile;
                out_rd_ptr += out_tile_size_bytes;
            }
            noc_async_write_barrier();
            cb_pop_front(cb_out_dinp, c_tiles);

        }
    }

    // We should now have a page of mean and rstd to write
    cb_wait_front(cb_out_dweight, c_tiles);
    cb_wait_front(cb_out_dbias, c_tiles);
    uint32_t dweight_rd_ptr = get_read_ptr(cb_out_dweight);
    uint32_t dbias_rd_ptr = get_read_ptr(cb_out_dbias);

    for (uint32_t c_tile = 0; c < c_tiles; ++c_tile) {
        uint64_t dweight_dram_noc_addr = get_noc_addr(c_tile, dweight_gen);
        noc_async_write(dweight_rd_ptr, dweight_dram_noc_addr, face_row_bf16_byte);
        dweight_dram_noc_addr += face_row_bf16_byte;
        noc_async_write(dweight_rd_ptr + face_bf16_bytes, dweight_dram_noc_addr, face_row_bf16_byte);
        dweight_rd_ptr += tile_size_bytes;

        uint64_t dbias_dram_noc_addr = get_noc_addr(c_tile, dbias_gen);
        noc_async_write(dbias_rd_ptr, dbias_dram_noc_addr, face_row_bf16_byte);
        dbias_dram_noc_addr += face_row_bf16_byte;
        noc_async_write(dbias_rd_ptr + face_bf16_bytes, dbias_dram_noc_addr, face_row_bf16_byte);
        dbias_rd_ptr += tile_size_bytes;
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out_dweight, c_tiles);
    cb_pop_front(cb_out_dbias, c_tiles);

}
