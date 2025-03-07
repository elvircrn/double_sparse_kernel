#include "doublesparse_cuda_kernel.cu"

#include <cfloat>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

struct SparsifiedLinear {
  u32 m{};
  u32 n{};
  u32 k{};
  u32 non_zero_rows;
  u32 *d_a_row_offsets;
  u32 *d_a_col_vals;
  u32 *d_b_row_offsets;
  u32 *d_b_col_vals;

  void free() {
    cudaFree(d_a_row_offsets);
    cudaFree(d_a_col_vals);
    cudaFree(d_b_row_offsets);
    cudaFree(d_b_col_vals);
  }
};

template <class T> T *device_from_size(int s) {
  T *d_buff;
  cudaMalloc(reinterpret_cast<void **>(&d_buff), sizeof(T) * s);
  return d_buff;
}

template <class T> T *device_from_file(const std::string &file_path) {
  // Open the binary file
  std::ifstream file(file_path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file: " + file_path);
  }

  // Get the file size
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  // Allocate a buffer and read the data
  std::vector<T> buffer(size / sizeof(T));
  if (!file.read(reinterpret_cast<char *>(buffer.data()), size)) {
    throw std::runtime_error("Error reading file: " + file_path);
  }

  T *d_buff;
  cudaMalloc(reinterpret_cast<void **>(&d_buff), sizeof(T) * buffer.size());

  cudaDeviceSynchronize();

  cudaMemcpy(d_buff, buffer.data(), sizeof(T) * buffer.size(),
             cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  return d_buff;
}

using XType = uint16_t;

struct Result {
  float min;
  float mean;
};

Result mul_with_time(const SparsifiedLinear &d_s, XType *d_x, XType *d_y,
                     float *measurements, int k, Features features) {
  u32 *d_workspace;
  cudaMalloc(reinterpret_cast<void **>(&d_workspace), k * 165536 * sizeof(u32));
  cudaDeviceSynchronize();

  doublesparse_matmul(d_s.m, d_s.n, d_s.k, d_s.d_a_row_offsets,
                      d_s.d_a_col_vals, d_s.d_b_row_offsets, d_s.d_b_col_vals,
                      d_s.non_zero_rows, k, d_x, d_y, d_workspace, nullptr,
                      measurements, features._);
  cudaFree(d_workspace);
  return Result{.min = *measurements, .mean = *measurements};
}

SparsifiedLinear from_path(const std::string &base_path) {
  std::ifstream meta_stream(base_path + "meta.txt");
  u32 m, n, k, non_zero_rows;
  meta_stream >> m >> n >> k >> non_zero_rows;

  return SparsifiedLinear{
      .m = m,
      .n = n,
      .k = k,
      .non_zero_rows = non_zero_rows,
      .d_a_row_offsets = device_from_file<u32>(base_path + "a_row_offsets.bin"),
      .d_a_col_vals = device_from_file<u32>(base_path + "a_col_vals.bin"),
      .d_b_row_offsets = device_from_file<u32>(base_path + "b_row_offsets.bin"),
      .d_b_col_vals = device_from_file<u32>(base_path + "b_col_vals.bin")};
}

int main() {
  std::string tag = "baseline_fp8_csr";
  Features features{._ = 0u};
  features.flags.is_async = false;
  features.flags.is_fp8 = true;

  std::ofstream results("results.txt", std::ios_base::app);
  static constexpr int XY_SIZE = 11008 * 4;
  static constexpr int NUM_REPS = 512;
  int num_layers = 20;
  auto d_x = device_from_size<uint16_t>(XY_SIZE);
  auto d_y = device_from_size<uint16_t>(XY_SIZE);
  const std::vector<std::string> &_layer_names{
      "mlp.down_proj",    "mlp.gate_proj",    "mlp.up_proj",
      "self_attn.k_proj", "self_attn.o_proj", "self_attn.q_proj",
      "self_attn.v_proj"};

  const std::vector<std::string> &layer_names{
    "self_attn.k_proj"};

  auto measurements = new float[NUM_REPS];

  float mean_runtime = 0.f;
  int tests{};


  for (int i = 0; i < num_layers; i++) {
    for (const auto &layer_name : layer_names) {
      std::string quant_linear_path =
          "/mnt/6e3c126c-c6bb-43eb-9d82-1e59b2111688/ecrncevi/double_sparse_data/compressed_csr/bin/experiment0.70/" +
          std::to_string(i) + "/" + layer_name + "/";

      SparsifiedLinear sparsified_linear = from_path(quant_linear_path);
      auto result =
          mul_with_time(sparsified_linear, d_x, d_y, measurements, 1, features);

      mean_runtime += result.min;

      std::cout << std::left << std::setw(3) << i << " " << std::left
                << std::setw(20) << layer_name << "     " << std::setw(5)
                << std::left << std::setprecision(3) << result.min << std::endl;
      tests++;

      sparsified_linear.free();
    }
  }

  results << std::left << std::setw(16) << tag << " " << (mean_runtime / tests)
          << std::endl;

  delete[] measurements;

  return 0;
}
