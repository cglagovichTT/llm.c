#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <vector>

using namespace tt::tt_metal;

template<class T>
T ceil_div(T dividend, T divisor) {
    return (dividend + divisor-1) / divisor;
}


// ----------------------------------------------------------------------------
// random utils

float* make_random_float_01(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        arr[i] = ((float)rand() / RAND_MAX); // range 0..1
    }
    return arr;
}

float* make_random_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        arr[i] = ((float)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    return arr;
}

std::vector<float> make_random_float_vec(size_t N) {
    auto arr = std::vector<float>(N);
    for (size_t i = 0; i < N; i++) {
        arr[i] = ((float)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    return arr;
}

std::vector<float> arange_float_vec(size_t N) {
    auto arr = std::vector<float>(N);
    for (size_t i = 0; i < N; i++) {
        arr[i] = i;
    }
    return arr;
}


int* make_random_int(size_t N, int V) {
    int* arr = (int*)malloc(N * sizeof(int));
    for (size_t i = 0; i < N; i++) {
        arr[i] = rand() % V; // range 0..V-1
    }
    return arr;
}

// float* make_zeros_float(size_t N) {
//     float* arr = (float*)malloc(N * sizeof(float));
//     memset(arr, 0, N * sizeof(float)); // all zero
//     return arr;
// }

float* make_ones_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        arr[i] = 1.0f;
    }
    return arr;
}

std::vector<uint32_t> fp32_to_raw(const std::vector<float>& vec) {
    std::vector<uint32_t> raw;
    raw.reserve(vec.size());
    for (auto& v : vec) {
        raw.push_back(*reinterpret_cast<const uint32_t*>(&v));
    }
    return raw;
}

std::vector<float> raw_to_fp32(const std::vector<uint32_t>& raw) {
    std::vector<float> vec;
    vec.reserve(raw.size());
    for (auto& r : raw) {
        vec.push_back(*reinterpret_cast<const float*>(&r));
    }
    return vec;
}

template<class D, class T>
void validate_result_vec(const D* device_result, const T* cpu_reference, const char* name, std::size_t num_elements, T tolerance=1e-4) {
    int nfaults = 0;
    for (int i = 0; i < num_elements; i++) {
        // print the first few comparisons
        if (i < 5) {
            printf("%f %f\n", cpu_reference[i], (T)device_result[i]);
        }
        // ensure correctness for all elements. We can set an "ignore" mask by writing NaN
        if (fabs(cpu_reference[i] - (T)device_result[i]) > tolerance && std::isfinite(cpu_reference[i])) {
            printf("Mismatch of %s at %d: CPU_ref: %f vs GPU: %f\n", name, i, cpu_reference[i], (T)device_result[i]);
            nfaults ++;
            if (nfaults >= 10) {
                // exit(EXIT_FAILURE);
                throw std::runtime_error("Too many mismatches");
            }
        }
    }
}

template <typename T>
std::vector<T> tilize(std::vector<T> data, std::vector<uint32_t> shape) {
    // shape is n-dimensional vector where last two dims are rows and cols
    TT_FATAL(shape.size() >= 2);
    // TT_FATAL volume of shape equals size of data
    size_t volume = 1;
    for (auto i = 0; i < shape.size(); i++) {
        volume *= shape[i];
    }
    TT_FATAL(volume == data.size());
    auto upper_dim_volume = volume / (shape[shape.size() - 2] * shape[shape.size() - 1]);

    for (auto i = shape.size() - 2; i < shape.size(); i++) {
        TT_FATAL(shape[i] % 32 == 0);
    }
    size_t rows = shape[shape.size() - 2];
    size_t cols = shape[shape.size() - 1];
    TT_FATAL(rows % 32 == 0);
    TT_FATAL(cols % 32 == 0);

    int tile_volume = 32 * 32;
    int num_tiles = volume / tile_volume;
    TT_FATAL(volume == num_tiles * tile_volume);

    int num_tiles_r = rows / 32;
    int num_tiles_c = cols / 32;
    const int face_size = 16*16;
    std::vector<T> result;
    result.reserve(volume);

    for (size_t upper_dim = 0, untilized_face_offset = 0; upper_dim < upper_dim_volume; ++upper_dim, untilized_face_offset += rows * cols){

        // For this tile_idx, find out the index into the shape
        for (auto r_tile = 0; r_tile < num_tiles_r; ++r_tile) {
            for (auto c_tile = 0; c_tile < num_tiles_c; ++c_tile) {
                auto tile_vector = std::vector<T>(32 * 32, 0);
                for(auto j = 0; j < 32; j++) { // tile rows
                    for(auto i = 0; i < 32; i++) { // tile cols
                        bool is_lower_face = j >= 16;
                        bool is_right_face = i >= 16;
                        int r_idx = r_tile * 32 + j; // index into 
                        int c_idx = c_tile * 32 + i;
                        int untilized_index = untilized_face_offset + r_idx * cols + c_idx;

                        int in_face_idx_r = j % 16;
                        int in_face_idx_c = i % 16;
                        int in_face_idx = in_face_idx_r * 16 + in_face_idx_c;
                        int out_face_idx = is_lower_face * face_size * 2 + is_right_face * face_size + in_face_idx;
                        TT_FATAL(tile_vector.at(out_face_idx) == 0);
                        // std::cout << "r_idx: " << r_idx << " c_idx: " << c_idx << " untilized_index: " << untilized_index << " out_face_idx: " << out_face_idx << std::endl;
                        tile_vector[out_face_idx] = data.at(untilized_index);
                    }
                }
                result.insert(result.end(), tile_vector.begin(), tile_vector.end());
            }
        }
    }
    return result;
}

template <typename T>
std::vector<T> untilize(std::vector<T> data, std::vector<uint32_t> shape) {
    TT_FATAL(shape.size() >= 2);
    size_t volume = shape[0];
    for (size_t i = 1; i < shape.size(); i++) {
        volume *= shape[i];
    }
    TT_FATAL(volume == data.size());
    size_t upper_dim_volume = volume / (shape[shape.size() - 2] * shape[shape.size() - 1]);

    size_t rows = shape[shape.size() - 2];
    size_t cols = shape[shape.size() - 1];
    TT_FATAL(rows % 32 == 0);
    TT_FATAL(cols % 32 == 0);

    size_t num_tiles_r = rows / 32;
    size_t num_tiles_c = cols / 32;
    std::vector<T> result(volume);

    for (size_t upper_dim = 0, tilized_face_offset = 0; upper_dim < upper_dim_volume; ++upper_dim) {
        for (size_t r_tile = 0; r_tile < num_tiles_r; ++r_tile) {
            for (size_t c_tile = 0; c_tile < num_tiles_c; ++c_tile) {
                for (size_t j = 0; j < 32; j++) {
                    for (size_t i = 0; i < 32; i++) {
                        bool is_lower_face = j >= 16;
                        bool is_right_face = i >= 16;
                        size_t r_idx = r_tile * 32 + j;
                        size_t c_idx = c_tile * 32 + i;
                        size_t untilized_index = upper_dim * rows * cols + r_idx * cols + c_idx;

                        size_t in_face_idx_r = j % 16;
                        size_t in_face_idx_c = i % 16;
                        size_t in_face_idx = in_face_idx_r * 16 + in_face_idx_c;
                        size_t tilized_face_idx = is_lower_face * 16 * 16 * 2 + is_right_face * 16 * 16 + in_face_idx;
                        TT_FATAL(untilized_index < result.size());
                        TT_FATAL(tilized_face_offset + tilized_face_idx < data.size());
                        result[untilized_index] = data[tilized_face_offset + tilized_face_idx];
                    }
                }
                tilized_face_offset += 32 * 32;
            }
        }
    }
    return result;
}

std::shared_ptr<Buffer> MakeBuffer(Device *device, uint32_t size, uint32_t page_size, bool sram, bool log=true)
{
    TT_FATAL(size % page_size == 0);
    if (log) {
        std::cout << "Creating buffer with size " << size << " and page size " << page_size << std::endl;
    }
    InterleavedBufferConfig config{
        .device= device,
        .size = size,
        .page_size = page_size,
        .buffer_type = (sram ? BufferType::L1 : BufferType::DRAM)
    };
    return CreateBuffer(config);
}

std::shared_ptr<Buffer> MakeBufferFP32(Device *device, uint32_t n_tiles, bool sram)
{
    constexpr uint32_t tile_size = 4 * (32 * 32);
    // For simplicity, all DRAM buffers have page size = tile size.
    const uint32_t page_tiles = sram ? n_tiles : 1;
    return MakeBuffer(device, tile_size * n_tiles, page_tiles * tile_size, sram);
}

CBHandle MakeCircularBuffer(Program& program, const CoreCoord& core, tt::CB cb, uint32_t size, uint32_t page_size, tt::DataFormat format, bool log=true)
{
    TT_FATAL(size % page_size == 0);
    if (log) {
        std::cout << "Creating CB " << cb << " with size " << size << " and page size " << page_size << std::endl;
    }
    CircularBufferConfig cb_src0_config = CircularBufferConfig(
        size,
        {{
            cb,
            format
    }})
    .set_page_size(cb, page_size);
    return CreateCircularBuffer(program, core, cb_src0_config);
}

CBHandle MakeCircularBufferFP32(Program& program, const CoreCoord& core, tt::CB cb, uint32_t n_tiles)
{
    constexpr uint32_t tile_size = 4 * (32 * 32);
    return MakeCircularBuffer(program, core, cb, n_tiles * tile_size, tile_size, tt::DataFormat::Float32);
}
