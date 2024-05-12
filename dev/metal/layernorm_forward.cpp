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

// ----------------------------------------------------------------------------
// CPU code reference

// GPT-2 layernorm forward pass
void layernorm_forward_cpu1(std::vector<float> &out, std::vector<float> &mean, std::vector<float> &rstd,
                       const std::vector<float> &inp, const std::vector<float> &weight, const std::vector<float> &bias,
                       int B, int T, int C) {
    float eps = 1e-5f;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            std::size_t offset = b * T * C + t * C;
            // calculate the mean
            float m = 0.0f;
            for (int i = 0; i < C; i++) {
                m += inp[offset + i];
            }
            m = m/C;
            // calculate the variance (without any bias correction)
            float v = 0.0f;
            for (int i = 0; i < C; i++) {
                float xshift = inp[offset + i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // calculate the rstd
            float s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            for (int i = 0; i < C; i++) {
                float n = (s * (inp[offset + i] - m)); // normalized output
                float o = n * weight[i] + bias[i]; // scale and shift it
                out[offset + i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}


// GPT-2 layernorm forward pass
void layernorm_forward_cpu2(float* out, float* mean, float* rstd,
                       const float* inp, const float* weight, const float* bias,
                       int B, int T, int C) {
    float eps = 1e-5f;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            const float* x = inp + b * T * C + t * C;
            // calculate the mean
            float m = 0.0f;
            for (int i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;
            // calculate the variance (without any bias correction)
            float v = 0.0f;
            for (int i = 0; i < C; i++) {
                float xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // calculate the rstd
            float s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
            for (int i = 0; i < C; i++) {
                float n = (s * (x[i] - m)); // normalized output
                float o = n * weight[i] + bias[i]; // scale and shift it
                out_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}
// ----------------------------------------------------------------------------
// kernel launcher

// void layernorm_forward1(float* out, float* mean, float* rstd,
//                            const float* inp, const float* weight, const float* bias,
//                            int B, int T, int C,
//                            const int block_size) {
//     const int N = B * T;
//     const int grid_size = ceil_div(N, block_size);
//     layernorm_forward_kernel1<<<grid_size, block_size>>>(out, mean, rstd, inp, weight, bias, N, C);
//     cudaCheck(cudaGetLastError());
// }

// // kernel version dispatch
// void layernorm_forward(int kernel_num,
//                     float* out, float* mean, float* rstd,
//                     const float* inp, const float* weight, const float* bias,
//                     int B, int T, int C,
//                     const int block_size) {
//     switch (kernel_num) {
//         case 1:
//             layernorm_forward1(out, mean, rstd, inp, weight, bias, B, T, C, block_size);
//             break;
//         default:
//             printf("Invalid kernel number\n");
//             exit(1);
//     }
// }

// ----------------------------------------------------------------------------

int main(int argc, char **argv) {



    // layernorm_forward_cpu1(out1, mean1, rstd1, inp, weight, bias, B, T, C);
    // layernorm_forward_cpu2(out2.data(), mean2.data(), rstd2.data(), inp.data(), weight.data(), bias.data(), B, T, C);

    // validate_result_vec(out1.data(), out2.data(), "out", B * T * C);
    // validate_result_vec(mean1.data(), mean2.data(), "mean", B * T);
    // validate_result_vec(rstd1.data(), rstd2.data(), "rstd", B * T);

    srand(0);
    int device_id = 0;
    Device *device = CreateDevice(device_id);
    CommandQueue& cq = device->command_queue();

    try {
    uint32_t B = 8;
    uint32_t T = 1024;
    uint32_t C = 768;
    auto shape = std::vector<uint32_t>{B, T, C};
    auto inp_volume = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<uint32_t>());
    uint32_t tile_size = 32 * 32;
    uint32_t n_tiles = inp_volume / (tile_size);
    uint32_t tile_size_b = sizeof(float) * tile_size;

    // create host memory of random numbers
    auto out_cpu = std::vector<float>(B * T * C);
    auto mean_cpu = std::vector<float>(B * T);
    auto rstd_cpu = std::vector<float>(B * T);
    auto inp = make_random_float_vec(B * T * C);
    auto weight = make_random_float_vec(C);
    auto bias = make_random_float_vec(C);

    // CPU ground truth
    layernorm_forward_cpu1(out_cpu, mean_cpu, rstd_cpu, inp, weight, bias, B, T, C);

    // Host inputs for Device
    std::vector<uint32_t> inp_tilized = fp32_to_raw(tilize(inp, shape));
    std::vector<uint32_t> weight_rowmajor = fp32_to_raw(weight);
    std::vector<uint32_t> bias_rowmajor = fp32_to_raw(bias);

    // make dram buffers
    // outputs
    auto out_buffer = MakeBuffer(device, inp_volume * sizeof(float), tile_size_b, false); // tilized
    auto mean_buffer = MakeBuffer(device, B * T * sizeof(float), 32 * sizeof(float), false); // RM, page_size = 32
    auto rstd_buffer = MakeBuffer(device, B * T * sizeof(float), 32 * sizeof(float), false); // RM, page_size = 32
    // inputs
    auto inp_buffer = MakeBuffer(device, inp_volume * sizeof(float), tile_size_b, false); //  tilized
    // page size of mean_buffer implies how we can parallelize. If page is 32, then one core can write 32 rows at a time
    auto weight_buffer = MakeBuffer(device, C * sizeof(float), 32 * sizeof(float), false); // RM, page_size = 32
    auto bias_buffer = MakeBuffer(device, C * sizeof(float), 32 * sizeof(float), false); // RM, page_size = 32

    // write buffer to dram
    EnqueueWriteBuffer(cq, inp_buffer, inp_tilized, false);
    EnqueueWriteBuffer(cq, weight_buffer, weight_rowmajor, false);
    EnqueueWriteBuffer(cq, bias_buffer, bias_rowmajor, false);

    // make program
    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0}; // TODO: Parallelize
    auto reader = CreateKernel(
        program,
        "/localdev/cglagovich/llm.c/dev/metal/kernels/layernorm_forward/reader_interleaved.cpp",
        core,
        DataMovementConfig {.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
    );
    auto writer = CreateKernel(
        program,
        "/localdev/cglagovich/llm.c/dev/metal/kernels/layernorm_forward/writer_interleaved.cpp",
        core,
        DataMovementConfig {.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default}
    );
    auto compute = CreateKernel(
        program,
        "/localdev/cglagovich/llm.c/dev/metal/kernels/layernorm_forward/compute.cpp",
        core,
        ComputeConfig{
            .fp32_dest_acc_en = true,
            .math_approx_mode = false,
            .compile_args = {},
            .defines = {}
        }
    );

    // Make CBs
    // TODO: Double buffering for inputs and outputs
    // input_cb
    auto inp_cb_num_tiles = C / 32; // buffer 32 in T and full C
    MakeCircularBuffer(program, core, tt::CB::c_in0, inp_cb_num_tiles * tile_size_b, tile_size_b, tt::DataFormat::Float32);
    // weight_cb
    MakeCircularBuffer(program, core, tt::CB::c_in1, weight_buffer->size(), weight_buffer->page_size(), tt::DataFormat::Float32);
    // bias_cb
    MakeCircularBuffer(program, core, tt::CB::c_in2, bias_buffer->size(), bias_buffer->page_size(), tt::DataFormat::Float32);
    // out_cb
    MakeCircularBuffer(program, core, tt::CB::c_out0, inp_cb_num_tiles * tile_size_b, tile_size_b, tt::DataFormat::Float32);
    // mean_cb
    uint32_t stats_cb_num_pages = 1; // buffer one page of mean_buffer, size 32 
    MakeCircularBuffer(program, core, tt::CB::c_out1, stats_cb_num_pages * mean_buffer->page_size(), mean_buffer->page_size(), tt::DataFormat::Float32);
    // rstd_cb
    MakeCircularBuffer(program, core, tt::CB::c_out2, stats_cb_num_pages * rstd_buffer->page_size(), rstd_buffer->page_size(), tt::DataFormat::Float32);

    SetRuntimeArgs(program, reader, core, {
        B,
        T,
        C,
        inp_buffer->address(),
        weight_buffer->address(),
        bias_buffer->address(),

    });
    SetRuntimeArgs(program, writer, core, {
        B,
        T,
        C,
        out_buffer->address(),
        mean_buffer->address(),
        rstd_buffer->address(),
    });

    SetRuntimeArgs(program, compute, core, {
        B,
        T,
        C,
    });

    // launch program
    EnqueueProgram(cq, program, true);
    std::cout << "Kernel execution finished" << std::endl;

    // Read the output buffer.
    std::vector<uint32_t> out_tt;
    std::vector<uint32_t> mean_tt;
    std::vector<uint32_t> rstd_tt;
    EnqueueReadBuffer(cq, out_buffer, out_tt, true);
    EnqueueReadBuffer(cq, mean_buffer, mean_tt, true);
    EnqueueReadBuffer(cq, rstd_buffer, rstd_tt, true);


    std::vector<float> out_untilized = raw_to_fp32(untilize(out_tt, shape));

    // check equality
    for (int i = 0; i < inp_volume; i++) {
        if (out_cpu[i] != out_untilized[i]) {
            std::cerr << "Mismatch at index " << i << ": " << out_cpu[i] << " != " << out_untilized[i] << std::endl;
        }
        TT_FATAL(out_cpu[i] == out_untilized[i]);
    }
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        CloseDevice(device);
        return -1;
    }

    CloseDevice(device);
    return 0;
}