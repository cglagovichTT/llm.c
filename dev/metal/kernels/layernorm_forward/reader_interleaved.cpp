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
    volatile tt_l1_ptr uint32_t* tile = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_id) + tile_idx * tile_size_bytes);
    for (int k = 0; k < 4; ++k) {
        DPRINT << "Face " << k << ENDL();
        uint32_t idx = k << 8; // Start index of the face
        for (int i = 0; i < 16; ++i) {
            for (int j = 0; j < 16; ++j) {
                DPRINT << convertToFloat(tile[idx + i * 16 + j]) << " ";
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

    constexpr uint32_t one_scalar = get_compile_time_arg_val(0);
    constexpr uint32_t mean_scalar = get_compile_time_arg_val(1);
    constexpr uint32_t epsilon_scalar = get_compile_time_arg_val(2);

    constexpr uint32_t cb_inp = tt::CB::c_in0;
    constexpr uint32_t cb_weight = tt::CB::c_in1;
    constexpr uint32_t cb_bias = tt::CB::c_in2;
    constexpr uint32_t cb_one_scalar = tt::CB::c_in3;
    constexpr uint32_t cb_mean_recip_scalar = tt::CB::c_in4;
    constexpr uint32_t cb_epsilon_scalar = tt::CB::c_in5;

    const uint32_t inp_tile_size_bytes = get_tile_size(cb_inp);
    constexpr DataFormat inp_data_format = get_dataformat(cb_inp);
    DPRINT << "inp_tile_size_bytes: " << inp_tile_size_bytes << ENDL();

    const uint32_t weight_tile_size_bytes = get_tile_size(cb_weight);

    const InterleavedAddrGenFast<true> inp_gen = {
        .bank_base_address = inp_addr,
        .page_size = inp_tile_size_bytes,
        .data_format = inp_data_format, // The data format of the buffer
    };

    const InterleavedPow2AddrGen<true> weight_gen = {
        .bank_base_address = weight_addr,
        .log_base_2_of_page_size = 7, // log(32 * 4)
    };
    DPRINT << "weight_gen base_address=" << weight_gen.bank_base_address << ENDL();

    // Bias has same page size and dataformat as weight
    const InterleavedPow2AddrGen<true> bias_gen = {
        .bank_base_address = bias_addr,
        .log_base_2_of_page_size = 7, // log(32 * 4)
    };
    DPRINT << "bias_gen base_address=" << bias_gen.bank_base_address << ENDL();

    // Generate constant tiles for layernorm compute
    generate_reduce_scaler_fp32(cb_one_scalar, one_scalar);
    // DPRINT << "cb_one_scalar:" << ENDL();
    // print_tile_contents(cb_one_scalar, 0);
    // Bcast scalars
    generate_bcast_col_scalar_fp32(cb_mean_recip_scalar, mean_scalar);
    // DPRINT << "cb_mean_recip_scalar:" << ENDL();
    // print_tile_contents(cb_mean_recip_scalar, 0);
    generate_bcast_col_scalar_fp32(cb_epsilon_scalar, epsilon_scalar);
    // DPRINT << "cb_epsilon_scalar:" << ENDL();
    // print_tile_contents(cb_epsilon_scalar, 0);

    const uint32_t num_weight_pages = C / 32; // page size should be 32 datums

    /* Read weight and bias once */
    // How many bytes is one vector of 16 weight datums
    constexpr uint32_t weight_face_fp32_bytes = 16 * 4;
    // How many bytes do you stride to get to the next face
    constexpr uint32_t face_fp32_bytes = 16 * 16 * 4;
    
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
        // DPRINT << "reader: FACE 0 weight_dram_noc_addr=" << weight_dram_noc_addr << ENDL();
        // DPRINT << "reader: FACE 0 weight_wr_ptr=" << weight_wr_ptr << ENDL();
        noc_async_read(weight_dram_noc_addr, weight_wr_ptr, weight_face_fp32_bytes);
        weight_dram_noc_addr += weight_face_fp32_bytes;
        // DPRINT << "reader: FACE 1 weight_dram_noc_addr=" << weight_dram_noc_addr << ENDL();
        // DPRINT << "reader: FACE 1 weight_wr_ptr=" << weight_wr_ptr + face_fp32_bytes << ENDL();
        noc_async_read(weight_dram_noc_addr, weight_wr_ptr + face_fp32_bytes, weight_face_fp32_bytes);
        weight_wr_ptr += weight_tile_size_bytes;

        uint64_t bias_dram_noc_addr = get_noc_addr(i, bias_gen);
        noc_async_read(bias_dram_noc_addr, bias_wr_ptr, weight_face_fp32_bytes);
        bias_dram_noc_addr += weight_face_fp32_bytes;
        noc_async_read(bias_dram_noc_addr, bias_wr_ptr + face_fp32_bytes, weight_face_fp32_bytes);
        bias_wr_ptr += weight_tile_size_bytes;

        // DPRINT << "cb_weight:" << ENDL();
        // print_tile_contents(cb_weight, i);
        // DPRINT << "cb_bias:" << ENDL();
        // print_tile_contents(cb_bias, i);
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