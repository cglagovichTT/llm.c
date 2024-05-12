/*
Kernels for layernorm forward pass.

Compile:
```bash
cmake .
make
./layernorm_forward
```
*/
// std imports
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <vector>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string_view>

// metal imports
#include "common/core_coord.h"
#include "tt_metal/host_api.hpp"
#include "common/bfloat16.hpp"
// #include "tensor/tensor.hpp"
#include "common/constants.hpp"
// #include "tt_numpy/functions.hpp"

// project imports
#include "common.h"

using namespace tt::tt_metal;

int main(int argc, char **argv) {
    srand(0);
    int device_id = 0;
    Device *device = CreateDevice(device_id);
    CommandQueue& cq = device->command_queue();


    uint32_t B = 8;
    uint32_t T = 1024;
    uint32_t C = 768;
    auto shape = std::vector<uint32_t>{B, T, C};
    auto volume = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<uint32_t>());
    uint32_t n_tiles = volume / (32*32);

    // create host memory of random numbers
    auto inp = arange_float_vec(volume);
    std::vector<uint32_t> inp_raw = {inp.begin(), inp.end()};
    auto inp_tilized = tilize(inp_raw, shape);


    // make dram buffers
    auto inp_buffer = MakeBufferFP32(device, n_tiles, false);
    auto out_buffer = MakeBufferFP32(device, n_tiles, false);
    try {
    // write buffer to dram
    EnqueueWriteBuffer(cq, inp_buffer, inp_tilized, false);

    // make program
    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};
    auto reader = CreateKernel(
        program,
        "kernels/loopback/dataflow/reader_interleaved.cpp",
        core,
        DataMovementConfig {.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
    );
    auto writer = CreateKernel(
        program,
        "kernels/loopback/dataflow/writer_interleaved.cpp",
        core,
        DataMovementConfig {.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default}
    );

    // Make CBs
    MakeCircularBufferFP32(program, core, tt::CB::c_in0, 1);

    SetRuntimeArgs(program, reader, core, {
        inp_buffer->address(),
        n_tiles
    });
    SetRuntimeArgs(program, writer, core, {
        out_buffer->address(),
        n_tiles
    });

    // launch program
    EnqueueProgram(cq, program, true);
    std::cout << "Kernel execution finished" << std::endl;

    // Read the output buffer.
    std::vector<uint32_t> out;
    EnqueueReadBuffer(cq, out_buffer, out, true);


    std::vector<uint32_t> out_untilized_raw = untilize(out, shape);
    std::vector<float> out_untilized(out_untilized_raw.begin(), out_untilized_raw.end());

    // check equality
    for (int i = 0; i < volume; i++) {
        if (inp[i] != out_untilized[i]) {
            std::cerr << "Mismatch at index " << i << ": " << inp[i] << " != " << out_untilized[i] << std::endl;
        }
        TT_FATAL(inp[i] == out_untilized[i]);
    }
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        CloseDevice(device);
        return -1;
    }

    CloseDevice(device);
    return 0;
}