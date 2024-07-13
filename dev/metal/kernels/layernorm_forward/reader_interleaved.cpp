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

FORCE_INLINE void generate_reduce_scaler_fp32(const uint32_t cb_id, const uint32_t scaler) {
    cb_reserve_back(cb_id, 1);

    const uint32_t num_zeros_reads = get_tile_size(cb_id) / MEM_ZEROS_SIZE;
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
            // Iterate over faces. 2**8 == 16*16
            uint32_t idx = k << 8; 
            for (int j = 0; j < 16; ++j) { 
                // Iterate over first row of a face
                ptr[idx + j] = scaler;
            }
        }
    }
    cb_push_back(cb_id, 1);
}

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

FORCE_INLINE void generate_bcast_col_scalar_fp32(const uint32_t cb_id, const uint32_t scalar) {
    // const uint16_t scalar_val = scalar>>16;
    cb_reserve_back(cb_id, 1);
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_id));
    for (int k = 0; k < 4; k+=2) {
        // Iterate over left two faces
        uint32_t idx = k << 8;
        for (int j = 0; j < 256; j+=16) {
            // Iterate over fist col of a face
            ptr[idx + j] = scalar;
        }
    }
    cb_push_back(cb_id, 1);
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

void kernel_main()
{
    const uint32_t B = get_arg_val<uint32_t>(0);
    const uint32_t T = get_arg_val<uint32_t>(1);
    const uint32_t C = get_arg_val<uint32_t>(2);
    const uint32_t inp_addr = get_arg_val<uint32_t>(3);
    const uint32_t weight_addr = get_arg_val<uint32_t>(4);
    const uint32_t bias_addr = get_arg_val<uint32_t>(5);
    const uint32_t start_b = get_arg_val<uint32_t>(6);
    const uint32_t start_t = get_arg_val<uint32_t>(7);
    const uint32_t end_b = get_arg_val<uint32_t>(8);
    const uint32_t end_t = get_arg_val<uint32_t>(9);

    constexpr uint32_t one_scalar = get_compile_time_arg_val(0);
    constexpr uint32_t mean_scalar = get_compile_time_arg_val(1);
    constexpr uint32_t epsilon_scalar = get_compile_time_arg_val(2);
    constexpr uint32_t log_page_size = get_compile_time_arg_val(3);

    constexpr uint32_t cb_inp = tt::CB::c_in0;
    constexpr uint32_t cb_weight = tt::CB::c_in1;
    constexpr uint32_t cb_bias = tt::CB::c_in2;
    constexpr uint32_t cb_one_scalar = tt::CB::c_in3;
    constexpr uint32_t cb_mean_recip_scalar = tt::CB::c_in4;
    constexpr uint32_t cb_epsilon_scalar = tt::CB::c_in5;

    const uint32_t inp_tile_size_bytes = get_tile_size(cb_inp);
    constexpr DataFormat inp_data_format = get_dataformat(cb_inp);

    const uint32_t weight_tile_size_bytes = get_tile_size(cb_weight);

    const InterleavedAddrGenFast<true> inp_gen = {
        .bank_base_address = inp_addr,
        .page_size = inp_tile_size_bytes,
        .data_format = inp_data_format, // The data format of the buffer
    };

    const InterleavedPow2AddrGen<true> weight_gen = {
        .bank_base_address = weight_addr,
        .log_base_2_of_page_size = log_page_size,
    };

    // Bias has same page size and dataformat as weight
    const InterleavedPow2AddrGen<true> bias_gen = {
        .bank_base_address = bias_addr,
        .log_base_2_of_page_size = log_page_size,
    };

    // Generate constant tiles for layernorm compute
    generate_reduce_scaler(cb_one_scalar, one_scalar);
    generate_bcast_col_scalar(cb_mean_recip_scalar, mean_scalar);
    generate_bcast_col_scalar(cb_epsilon_scalar, epsilon_scalar);

    const uint32_t num_weight_pages = C / 32; // page size should be 32 datums

    /* Read weight and bias once */
    // How many bytes is one vector of 16 weight datums
    constexpr uint32_t face_row_bf16_byte = 16 * 2;
    // How many bytes do you stride to get to the next face
    constexpr uint32_t face_bf16_bytes = 16 * 16 * 2;
    
    cb_reserve_back(cb_weight, num_weight_pages);
    cb_reserve_back(cb_bias, num_weight_pages);
    uint32_t weight_wr_ptr = get_read_ptr(cb_weight);
    uint32_t bias_wr_ptr = get_read_ptr(cb_bias);
    for (uint32_t i = 0; i < num_weight_pages; i++) {
        // DPRINT << "reader: weight=" << i << ENDL();

        // Weights are RM sticks, 1xC with page size 32. Bias is the same.
        // For use in the compute kernel, we need each 32-long vector to occupy
        // the top row of a tile for use in `mul_tiles_bcast_rows` and
        // `add_tiles_bcast_rows`. This reading must be aware of 16x16 faces.

        uint64_t weight_dram_noc_addr = get_noc_addr(i, weight_gen);
        noc_async_read(weight_dram_noc_addr, weight_wr_ptr, face_row_bf16_byte);
        weight_dram_noc_addr += face_row_bf16_byte;
        noc_async_read(weight_dram_noc_addr, weight_wr_ptr + face_bf16_bytes, face_row_bf16_byte);
        weight_wr_ptr += weight_tile_size_bytes;

        uint64_t bias_dram_noc_addr = get_noc_addr(i, bias_gen);
        noc_async_read(bias_dram_noc_addr, bias_wr_ptr, face_row_bf16_byte);
        bias_dram_noc_addr += face_row_bf16_byte;
        noc_async_read(bias_dram_noc_addr, bias_wr_ptr + face_bf16_bytes, face_row_bf16_byte);
        bias_wr_ptr += weight_tile_size_bytes;

    }
    noc_async_read_barrier();
    cb_push_back(cb_weight, num_weight_pages);
    cb_push_back(cb_bias, num_weight_pages);

    /* Enter input reading loop */
    const uint32_t c_tiles = C / 32;
    const uint32_t t_tiles = T / 32;

    for (uint32_t b = start_b; b < end_b; ++b) {
	uint32_t batch_tile_offset = b * t_tiles * c_tiles;
        for (uint32_t t_tile = start_t; t_tile < end_t; ++t_tile) {
	    uint32_t seq_start_tile = batch_tile_offset + t_tile * c_tiles; 
            // DPRINT << "reader: b=" << b << " t_tile=" << t_tile << ENDL();

            cb_reserve_back(cb_inp, c_tiles);
            uint32_t inp_wr_ptr = get_write_ptr(cb_inp);
            for (uint32_t c_tile = 0; c_tile < c_tiles; ++c_tile) {
                noc_async_read_tile(seq_start_tile, inp_gen, inp_wr_ptr);
                inp_wr_ptr += inp_tile_size_bytes;
                ++seq_start_tile;
                noc_async_read_barrier();
            }
            cb_push_back(cb_inp, c_tiles);
        
        }
    }
}
