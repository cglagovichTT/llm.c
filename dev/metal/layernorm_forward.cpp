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

void layernorm_setup_program(Program& program, Device* device, uint32_t B, uint32_t T, uint32_t C, 
    std::shared_ptr<Buffer>& inp_buffer, std::shared_ptr<Buffer>& weight_buffer, std::shared_ptr<Buffer>& bias_buffer,
    std::shared_ptr<Buffer>& out_buffer, std::shared_ptr<Buffer>& mean_buffer, std::shared_ptr<Buffer>& rstd_buffer) {

    uint32_t tile_size = 32 * 32;
    const int sizeof_bfloat = 2;
    uint32_t tile_size_b = sizeof_bfloat * tile_size;
    const int stick_size = 32 * sizeof_bfloat;
    const uint32_t log_stick_size = std::log2(stick_size);

    const float one_scalar = 1.0;
    bfloat16 bf_one_scalar = bfloat16(one_scalar);
    uint32_t packed_one_scalar = pack_two_bfloat16_into_uint32({bf_one_scalar, bf_one_scalar});
    union { float f; uint32_t u; } one_packed; one_packed.f = one_scalar;
    const float mean_recip_scalar = 1.0 / C;
    union { float f; uint32_t u; } mean_recip_packed; mean_recip_packed.f = mean_recip_scalar;
    constexpr float epsilon = 1e-5f;
    union { float f; uint32_t u; } epsilon_packed; epsilon_packed.f = epsilon;

    constexpr CoreCoord core = {0, 0}; // TODO: Parallelize
    const CoreCoord device_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores = device_grid_size.x * device_grid_size.y;
    const CoreCoord end_core = {device_grid_size.x - 1, device_grid_size.y - 1};
    std::cout << "range x, y: " << end_core.x << ", " << end_core.y << std::endl;
    const CoreRange core_range = CoreRange(core, end_core);
    std::vector<uint32_t> reader_ct_args = {
        packed_one_scalar,
        mean_recip_packed.u,
        epsilon_packed.u,
        // log of stick_size
        log_stick_size // should be 6 for bf16
    };
    std::vector<uint32_t> writer_ct_args = {
        log_stick_size
    };
    std::cout << "log of stick_size: " << log_stick_size << std::endl;
    auto reader = CreateKernel(
        program,
        "kernels/layernorm_forward/reader_interleaved.cpp",
        core_range,
        DataMovementConfig {.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = reader_ct_args}
    );
    auto writer = CreateKernel(
        program,
        "kernels/layernorm_forward/writer_interleaved.cpp",
        core_range,
        DataMovementConfig {.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = writer_ct_args}
    );
    auto compute = CreateKernel(
        program,
        "kernels/layernorm_forward/compute.cpp",
        core_range,
        ComputeConfig{
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = {},
            .defines = {}
        }
    );

    // Make CBs
    // TODO: Double buffering for inputs and outputs
    // input_cb
    auto inp_cb_num_tiles = C / 32; // buffer 32 in T and full C
    MakeCircularBuffer(program, core_range, tt::CB::c_in0, inp_cb_num_tiles * tile_size_b, tile_size_b, tt::DataFormat::Float16_b);
    // weight_cb
    const uint32_t weight_cb_num_tiles = C / 32; // Allocate space for C/32 tiles for reader
    MakeCircularBuffer(program, core_range, tt::CB::c_in1, weight_cb_num_tiles * tile_size_b, tile_size_b, tt::DataFormat::Float16_b);
    // bias_cb
    MakeCircularBuffer(program, core_range, tt::CB::c_in2, weight_cb_num_tiles * tile_size_b, tile_size_b, tt::DataFormat::Float16_b);
    // one scalar
    MakeCircularBuffer(program, core_range, tt::CB::c_in3, tile_size_b, tile_size_b, tt::DataFormat::Float16_b);
    // mean_recip scalar
    MakeCircularBuffer(program, core_range, tt::CB::c_in4, tile_size_b, tile_size_b, tt::DataFormat::Float16_b);
    // epsilon scalar
    MakeCircularBuffer(program, core_range, tt::CB::c_in5, tile_size_b, tile_size_b, tt::DataFormat::Float16_b);
    // out_cb
    MakeCircularBuffer(program, core_range, tt::CB::c_out0, inp_cb_num_tiles * tile_size_b, tile_size_b, tt::DataFormat::Float16_b);
    // Stats CBs will contain 1 tile at a time, where the top row is 32 values for the currently processed input row
    // mean_cb
    MakeCircularBuffer(program, core_range, tt::CB::c_out1, tile_size_b, tile_size_b, tt::DataFormat::Float16_b);
    // rstd_cb
    MakeCircularBuffer(program, core_range, tt::CB::c_out2, tile_size_b, tile_size_b, tt::DataFormat::Float16_b);

    // intermediate compute buffers
    // x - mean
    MakeCircularBuffer(program, core_range, tt::CB::c_intermed0, inp_cb_num_tiles * tile_size_b, tile_size_b, tt::DataFormat::Float16_b);
    // x - mean squared
    MakeCircularBuffer(program, core_range, tt::CB::c_intermed1, inp_cb_num_tiles * tile_size_b, tile_size_b, tt::DataFormat::Float16_b);
    // mean
    MakeCircularBuffer(program, core_range, tt::CB::c_intermed2, tile_size_b, tile_size_b, tt::DataFormat::Float16_b);
    // rstd
    MakeCircularBuffer(program, core_range, tt::CB::c_intermed3, tile_size_b, tile_size_b, tt::DataFormat::Float16_b);
    // (x - mean) * rstd
    MakeCircularBuffer(program, core_range, tt::CB::c_intermed4, inp_cb_num_tiles * tile_size_b, tile_size_b, tt::DataFormat::Float16_b);
    // (x - mean) * rstd * weight
    MakeCircularBuffer(program, core_range, tt::CB::c_intermed5, inp_cb_num_tiles * tile_size_b, tile_size_b, tt::DataFormat::Float16_b);

    uint32_t seq_tiles = T / 32;
    uint32_t batch_parallel_factor = std::min(B, num_cores);
    uint32_t seq_parallel_factor = std::min(num_cores / batch_parallel_factor, seq_tiles);
    uint32_t batch_per_core = std::ceil((float)B / batch_parallel_factor);
    uint32_t seq_tiles_per_core = std::ceil((float)seq_tiles / seq_parallel_factor);

    for (uint32_t core_idx = 0; core_idx < num_cores; ++core_idx) {
        CoreCoord core = {core_idx % device_grid_size.x, core_idx / device_grid_size.x};

        uint32_t start_b = (core_idx / seq_parallel_factor) * batch_per_core;
        uint32_t end_b = start_b + batch_per_core;
        uint32_t start_t = (core_idx % seq_parallel_factor) * seq_tiles_per_core;
        uint32_t end_t = start_t + seq_tiles_per_core;
                
        start_b = std::min(start_b, B);
        end_b = std::min(end_b, B);
        start_t = std::min(start_t, seq_tiles);
        end_t = std::min(end_t, seq_tiles);
        
        std::cout << "core: " << core_idx << " start_b: " << start_b << " start_t: " << start_t << " end_b: " << end_b << " end_t: " << end_t << std::endl;

        SetRuntimeArgs(program, reader, core, {
            B, T, C,
            inp_buffer->address(),
            weight_buffer->address(),
            bias_buffer->address(),
            start_b, start_t, end_b, end_t
        });

        SetRuntimeArgs(program, writer, core, {
            B, T, C,
            out_buffer->address(),
            mean_buffer->address(),
            rstd_buffer->address(),
            start_b, start_t, end_b, end_t
        });

        SetRuntimeArgs(program, compute, core, {
            B, T, C,
            start_b, start_t, end_b, end_t
        });
    }
}

int main(int argc, char **argv) {

    srand(0);

    uint32_t B = 8;
    uint32_t T = 1024;
    uint32_t C = 768;

    // create host memory of random numbers
    auto out_cpu = std::vector<float>(B * T * C);
    auto mean_cpu = std::vector<float>(B * T);
    auto rstd_cpu = std::vector<float>(B * T);
    auto inp = make_random_float_vec(B * T * C);
    auto weight = make_random_float_vec(C);
    auto bias = make_random_float_vec(C);
    
    // CPU ground truth
    layernorm_forward_cpu1(out_cpu, mean_cpu, rstd_cpu, inp, weight, bias, B, T, C);    

    auto shape = std::vector<uint32_t>{B, T, C};
    auto inp_volume = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<uint32_t>());

    int device_id = 0;
    Device *device = CreateDevice(device_id);
    CommandQueue& cq = device->command_queue();

    try {

    uint32_t tile_size = 32 * 32;
    const int sizeof_bfloat = 2;
    uint32_t tile_size_b = sizeof_bfloat * tile_size;

    // Host inputs for Device
    std::vector<uint32_t> inp_tilized = fp32_to_bfloat16_packed(tilize(inp, shape));
    std::vector<uint32_t> weight_rowmajor = fp32_to_bfloat16_packed(weight);
    std::vector<uint32_t> bias_rowmajor = fp32_to_bfloat16_packed(bias);

    // make dram buffers
    // outputs
    const int stick_size = 32 * sizeof_bfloat;
    auto out_buffer = MakeBuffer(device, inp_volume * sizeof_bfloat, tile_size_b, false); // tilized
    auto mean_buffer = MakeBuffer(device, B * T * sizeof_bfloat, stick_size, false); // RM, page_size = 32
    auto rstd_buffer = MakeBuffer(device, B * T * sizeof_bfloat, stick_size, false); // RM, page_size = 32
    // inputs
    auto inp_buffer = MakeBuffer(device, inp_volume * sizeof_bfloat, tile_size_b, false); //  tilized
    // page size of mean_buffer implies how we can parallelize. If page is 32, then one core can write 32 rows at a time
    auto weight_buffer = MakeBuffer(device, C * sizeof_bfloat, stick_size, false); // RM, page_size = 32
    auto bias_buffer = MakeBuffer(device, C * sizeof_bfloat, stick_size, false); // RM, page_size = 32

    // write buffer to dram
    EnqueueWriteBuffer(cq, inp_buffer, inp_tilized, false);
    EnqueueWriteBuffer(cq, weight_buffer, weight_rowmajor, false);
    EnqueueWriteBuffer(cq, bias_buffer, bias_rowmajor, false);

    Program layernorm_program = CreateProgram();
    layernorm_setup_program(layernorm_program, device, B, T, C, inp_buffer, weight_buffer, bias_buffer, out_buffer, mean_buffer, rstd_buffer);

    // launch program
    EnqueueProgram(cq, layernorm_program, true);
    std::cout << "Kernel execution finished" << std::endl;

    // Read the output buffer.
    std::vector<uint32_t> out_tt;
    std::vector<uint32_t> mean_tt;
    std::vector<uint32_t> rstd_tt;
    EnqueueReadBuffer(cq, out_buffer, out_tt, true);
    EnqueueReadBuffer(cq, mean_buffer, mean_tt, true);
    EnqueueReadBuffer(cq, rstd_buffer, rstd_tt, true);


    std::vector<float> tt_out_untilized = untilize(bfloat16_packed_to_fp32(out_tt), shape);
    std::vector<float> tt_mean = bfloat16_packed_to_fp32(mean_tt);
    std::vector<float> tt_rstd = bfloat16_packed_to_fp32(rstd_tt);

    validate_result_vec(out_cpu, tt_out_untilized, "out", B * T * C, 1e-1f);
    validate_result_vec(mean_cpu, tt_mean, "mean", B * T, 1e-1f);
    validate_result_vec(rstd_cpu, tt_rstd, "rstd", B * T, 1e-1f);

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        CloseDevice(device);
        return -1;
    }

    CloseDevice(device);
    return 0;
}