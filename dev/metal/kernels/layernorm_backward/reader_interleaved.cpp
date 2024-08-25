#include <cstdint>
#include "dataflow_api.h"
#include "debug/dprint.h"

union FloatConverter {
    uint32_t intValue;
    float floatValue;
};

float convertToFloat(uint32_t intValue) {
    FloatConverter converter;
    converter.intValue = intValue;
    return converter.floatValue;
}

void print_tile_contents(const uint32_t cb_id, const uint32_t tile_idx) {
    const uint32_t tile_size_bytes = get_tile_size(cb_id);
    volatile tt_l1_ptr uint16_t* tile = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(cb_id) + tile_idx * tile_size_bytes);
    for (int k = 0; k < 4; ++k) {
        DPRINT << "Face " << k << ENDL();
        uint32_t idx = k << 8; // Start index of the face
        for (int i = 0; i < 16; ++i) {
            for (int j = 0; j < 16; ++j) {
                uint32_t val = tile[idx + i * 16 + j];
                float f1 = convertToFloat(((uint32_t)val)<<16);
                // float f1 = convertToFloat((val & 0xFFFF0000));
                // float f2 = convertToFloat((val & 0x0000FFFF)<<16);
                DPRINT << f1 << " ";
            }
            DPRINT << ENDL();
        }
        DPRINT << ENDL();
    }
}

void read_row_major_stick_to_tile_row(uint64_t noc_addr, uint32_t dst_addr, uint32_t face_row_bytes, uint32_t face_size_bytes) {
    noc_async_read(noc_addr, dst_addr, face_row_bytes);
    noc_addr += face_row_bytes;
    dst_addr += face_size_bytes;
    noc_async_read(noc_addr, dst_addr, face_row_bytes);
}

void kernel_main()
{
    const uint32_t B = get_arg_val<uint32_t>(0);
    const uint32_t T = get_arg_val<uint32_t>(1);
    const uint32_t C = get_arg_val<uint32_t>(2);
    const uint32_t dout_buffer = get_arg_val<uint32_t>(3);
    const uint32_t dinp_buffer = get_arg_val<uint32_t>(4);
    const uint32_t weight_buffer = get_arg_val<uint32_t>(5);
    const uint32_t mean_buffer = get_arg_val<uint32_t>(6);
    const uint32_t rstd_buffer = get_arg_val<uint32_t>(7);
    const uint32_t inp_buffer = get_arg_val<uint32_t>(8);
    const uint32_t dweight_buffer = get_arg_val<uint32_t>(9);
    const uint32_t dbias_buffer = get_arg_val<uint32_t>(10);

    constexpr uint32_t log_page_size = get_compile_time_arg_val(0);

    constexpr uint32_t cb_inp = get_compile_time_arg_val(1);
    constexpr uint32_t cb_dout = get_compile_time_arg_val(2);
    constexpr uint32_t cb_weight = get_compile_time_arg_val(3);
    constexpr uint32_t cb_mean = get_compile_time_arg_val(4);
    constexpr uint32_t cb_rstd = get_compile_time_arg_val(5);
    constexpr uint32_t cb_dinp = get_compile_time_arg_val(6);
    constexpr uint32_t cb_dweight = get_compile_time_arg_val(7);
    constexpr uint32_t cb_dbias = get_compile_time_arg_val(8);

    const uint32_t inp_tile_size_bytes = get_tile_size(cb_inp);
    constexpr DataFormat inp_data_format = get_dataformat(cb_inp);

    const uint32_t weight_tile_size_bytes = get_tile_size(cb_weight);


    const InterleavedAddrGenFast<true> dout_gen = {
        .bank_base_address = dout_buffer,
        .page_size = inp_tile_size_bytes,
        .data_format = inp_data_format, // The data format of the buffer
    };

    const InterleavedAddrGenFast<true> dinp_gen = {
        .bank_base_address = dinp_buffer,
        .page_size = inp_tile_size_bytes,
        .data_format = inp_data_format, // The data format of the buffer
    };

    const InterleavedPow2AddrGen<true> weight_gen = {
        .bank_base_address = weight_buffer,
        .log_base_2_of_page_size = log_page_size,
    };

    const InterleavedPow2AddrGen<true> mean_gen = {
        .bank_base_address = mean_buffer,
        .log_base_2_of_page_size = log_page_size,
    };
    
    const InterleavedPow2AddrGen<true> rstd_gen = {
        .bank_base_address = rstd_buffer,
        .log_base_2_of_page_size = log_page_size,
    };

    const InterleavedAddrGenFast<true> inp_gen = {
        .bank_base_address = inp_buffer,
        .page_size = inp_tile_size_bytes,
        .data_format = inp_data_format, // The data format of the buffer
    };

    const InterleavedPow2AddrGen<true> dweight_gen = {
        .bank_base_address = dweight_buffer,
        .log_base_2_of_page_size = log_page_size,
    };
    
    const InterleavedPow2AddrGen<true> dbias_gen = {
        .bank_base_address = dbias_buffer,
        .log_base_2_of_page_size = log_page_size,
    };


    const uint32_t num_weight_pages = C / 32; // page size should be 32 datums

    /* Read weight and bias once */
    // How many bytes is one vector of 16 weight datums
    constexpr uint32_t face_row_bf16_byte = 16 * 2;
    // How many bytes do you stride to get to the next face
    constexpr uint32_t face_bf16_bytes = 16 * 16 * 2;
    

    // Step 1: Read dweight, dbias, and weight into CBs
    cb_reserve_back(cb_dweight, num_weight_pages);
    cb_reserve_back(cb_dbias, num_weight_pages);
    cb_reserve_back(cb_weight, num_weight_pages);
    uint32_t dweight_wr_ptr = get_read_ptr(cb_dweight);
    uint32_t dbias_wr_ptr = get_read_ptr(cb_dbias);
    uint32_t weight_wr_ptr = get_read_ptr(cb_weight);
    for (uint32_t i = 0; i < num_weight_pages; i++) {
        // Weights are RM sticks, 1xC with page size 32. Bias is the same.
        // For use in the compute kernel, we need each 32-long vector to occupy
        // the top row of a tile for use in `mul_tiles_bcast_rows` and
        // `add_tiles_bcast_rows`. This reading must be aware of 16x16 faces.

        uint64_t dweight_dram_noc_addr = get_noc_addr(i, dweight_gen);
        read_row_major_stick_to_tile_row(dweight_dram_noc_addr, dbias_wr_ptr, face_row_bf16_byte, face_bf16_bytes)
        dweight_wr_ptr += weight_tile_size_bytes;

        uint64_t dbias_dram_noc_addr = get_noc_addr(i, dbias_gen);
        read_row_major_stick_to_tile_row(dbias_dram_noc_addr, dbias_wr_ptr, face_row_bf16_byte, face_bf16_bytes)
        dbias_wr_ptr += weight_tile_size_bytes;

        uint64_t weight_dram_noc_addr = get_noc_addr(i, weight_gen);
        read_row_major_stick_to_tile_row(weight_dram_noc_addr, weight_wr_ptr, face_row_bf16_byte, face_bf16_bytes)
        weight_wr_ptr += weight_tile_size_bytes;

    }
    noc_async_read_barrier();
    cb_push_back(cb_dweight, num_weight_pages);
    cb_push_back(cb_dbias, num_weight_pages);
    cb_push_back(cb_weight, num_weight_pages);

    /* Enter input reading loop */
    const uint32_t c_tiles = C / 32;
    const uint32_t t_tiles = T / 32;

    for (uint32_t b = start_b; b < end_b; ++b) {
	    uint32_t batch_tile_offset = b * t_tiles * c_tiles;
        for (uint32_t t_tile = start_t; t_tile < end_t; ++t_tile) {
	        uint32_t seq_start_tile = batch_tile_offset + t_tile * c_tiles; 
            // DPRINT << "reader: b=" << b << " t_tile=" << t_tile << ENDL();

            // read mean and rstd for this row of tiles
            uint32_t stats_tile_id = b * t_tiles + t_tile;
            cb_reserve_back(cb_mean, 1);
            uint32_t mean_wr_ptr = get_write_ptr(cb_mean);
            uint64_t mean_dram_noc_addr = get_noc_addr(stats_tile_id, mean_gen);
            read_row_major_stick_to_tile_row(mean_dram_noc_addr, mean_wr_ptr, face_row_bf16_byte, face_bf16_bytes);
            cb_push_back(cb_mean, 1);
            cb_reserve_back(cb_rstd, 1);
            uint32_t rstd_wr_ptr = get_write_ptr(cb_rstd);
            uint64_t rstd_dram_noc_addr = get_noc_addr(stats_tile_id, rstd_gen);
            read_row_major_stick_to_tile_row(rstd_dram_noc_addr, rstd_wr_ptr, face_row_bf16_byte, face_bf16_bytes);
            cb_push_back(cb_rstd, 1);



            // Read c_tiles of input and dout
            cb_reserve_back(cb_inp, c_tiles);
            uint32_t inp_wr_ptr = get_write_ptr(cb_inp);
            for (uint32_t c_tile = 0; c_tile < c_tiles; ++c_tile) {
                noc_async_read_tile(seq_start_tile, inp_gen, inp_wr_ptr);
                inp_wr_ptr += inp_tile_size_bytes;
                ++seq_start_tile;
                noc_async_read_barrier();
            }
            cb_push_back(cb_inp, c_tiles);

            cb_reserve_back(cb_dout, c_tiles);
            uint32_t dout_wr_ptr = get_write_ptr(cb_dout);
            for (uint32_t c_tile = 0; c_tile < c_tiles; ++c_tile) {
                noc_async_read_tile(seq_start_tile, dout_gen, dout_wr_ptr);
                dout_wr_ptr += inp_tile_size_bytes;
                ++seq_start_tile;
                noc_async_read_barrier();
            }
        
        }
    }
}
