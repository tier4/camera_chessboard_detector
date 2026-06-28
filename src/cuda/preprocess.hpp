// This file is part of camera_chessboard_detector.
// Copyright (C) 2026 TIER IV, Inc.
//
// camera_chessboard_detector is free software: you can redistribute it and/or
// modify it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or (at your
// option) any later version.
//
// camera_chessboard_detector is distributed in the hope that it will be
// useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
// Public License for more details.
//
// You should have received a copy of the GNU General Public License along with
// this program. If not, see <https://www.gnu.org/licenses/>.

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
