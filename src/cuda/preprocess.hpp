#pragma once

#include <cstddef>

#include "core.hpp"
#include "cuda/gpu_buffers.hpp"

namespace camera_chessboard_detector {
namespace cuda {

// GPU image preprocessing: uint8 BGR (or gray) input -> normalized float
// [0, 1] device image. Mirrors the host pipeline used by the CPU front end
// (BGR2GRAY + 9x9 Gaussian sigma=1.5 + min-max normalize). Device buffers are
// reused across frames (resize is a no-op at constant resolution).
class CudaPreprocessor {
public:
  CudaPreprocessor();
  ~CudaPreprocessor();
  CudaPreprocessor(const CudaPreprocessor &) = delete;
  CudaPreprocessor &operator=(const CudaPreprocessor &) = delete;

  // `h_src` points to width*height*channels uint8 (channels == 1 or 3, BGR).
  // Returns a device float image normalized to [0, 1], owned by this object.
  const GpuImagePtr &run(const unsigned char *h_src, int width, int height,
                         int channels);

private:
  void ensureKernel();

  GpuImagePtr gray_;
  GpuImagePtr tmp_;
  GpuImagePtr blur_;
  GpuImagePtr norm_;
  unsigned char *d_src_{nullptr};
  std::size_t src_capacity_{0};
  float *d_kernel_{nullptr};
  bool kernel_ready_{false};
};

}  // namespace cuda
}  // namespace camera_chessboard_detector
