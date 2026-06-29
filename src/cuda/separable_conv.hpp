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

// Opt-in CUDA path: rank-k separable Geiger convolution.
//
// Used only when ChessboardDetectorConfig::acceleration ==
// CUDA_SEPARABLE. Each 2D sub-kernel is decomposed on the host by SVD
// into rank-1 outer products K = sum_k sigma_k * u_k * v_k^T, which
// are then applied on the GPU as paired 1D horizontal (v_k) and
// vertical (u_k) passes. At full rank (rank == kernel side length) the
// result equals the dense dense-convolution path up to FP precision;
// the probe (the test suite) is the standing
// proof of that. Lower ranks trade accuracy for speed; the synthetic-
// board accuracy test in regression_test.cpp is the gating signal.

#include "core.hpp"
#include "cuda/gpu_buffers.hpp"

#include <cuda_runtime.h>

#include <memory>
#include <vector>

namespace camera_chessboard_detector
{
namespace cuda
{

class CudaSeparableConvolver
{
public:
  CudaSeparableConvolver();
  ~CudaSeparableConvolver();

  // Copy/move are not deleted so this class can be a member of
  // CudaLikelihoodEstimator, which std::vector::resize must default-
  // construct. Matches the existing pattern in CudaConvolver. Do not
  // actually copy instances: the destructor frees the cudaStream and
  // device buffers.

  // Configure the separable representation from a dense kernel.
  // requested_rank <= 0 selects full rank (== 2*radius + 1).
  // Idempotent: safe to call repeatedly with the same kernel.
  void set_kernel(const camera_chessboard_detector::CpuKernelF32 &kernel, int requested_rank);

  // Apply (sum over k of u_k * v_k^T) cross-correlated against the
  // input image; writes into output. Output is resized and zeroed.
  void convolve(const GpuImagePtr &image, GpuImagePtr &output);

  int radius() const { return radius_; }
  int rank() const { return rank_; }

private:
  void release_();

  int radius_{0};
  int rank_{0};
  // Per-component 1D weight buffers (length S = 2*radius+1 each).
  // Stored as plain row-major float vectors on the device -- no width
  // doubling, no padding, no alignment tricks. Speed lives elsewhere.
  std::vector<float *> d_u_;
  std::vector<float *> d_v_;
  GpuImagePtr tmp_;  // intermediate horizontal-pass scratch
  cudaStream_t stream_;
};

}  // namespace cuda
}  // namespace camera_chessboard_detector
