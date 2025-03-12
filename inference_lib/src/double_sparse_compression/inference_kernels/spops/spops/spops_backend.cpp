#include <torch/extension.h>
#include "./lib/sputnik_spops.cpp"
#include <vector>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sputnik_spmm_fp32", &sputnik_spmm_fp32, "Sputnik SpMM FP32 (CUDA)");
  m.def("sputnik_spmm_fp16", &sputnik_spmm_fp16, "Sputnik SpMM FP16 (CUDA)");
}