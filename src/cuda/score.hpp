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

#include "core.hpp"
#include "cuda/gpu_buffers.hpp"

#include <cstddef>

namespace camera_chessboard_detector
{
namespace cuda
{

// On-GPU per-corner scoring for the Geiger detector.
//
// GPU replacement for the CPU `per-corner scorer` loop and the precursor
// `weights->download(cpu_weights)`. The class scores every corner in
// `CornerArray` against the device-resident grayscale image and the
// device-resident weights map, sweeping radius R ∈ {4, 8, 12} and
// retaining the per-corner maximum — the same contract the host
// implementation provided to the downstream prune pipeline.
//
// For each (corner, radius) the kernel computes:
//   1. A geometric `filter` mask (-1 / +1) from the corner's two edge
//      directions.
//   2. weights_roi mean / std and filter mean / std, used to
//      normalise both before an inner product. The resulting
//      `score_gradient` is `dot(filter_norm, weights_norm) /
//      (patch_pixels - 1)`, clipped at 0.
//   3. Four unnormalised quadrant kernels (a, b, c, d) built inline
//      per pixel from the corner's edge angles plus a Gaussian
//      radial envelope, the same math as `buildQuadrantKernels` in the host
//      code. Each kernel's sum is reduced separately so the per-
//      pixel value can be divided to give a unit-sum kernel.
//   4. Inner products a1 = sum(kernels.a * img_roi), etc.
//   5. `score_intensity = max(0, min/max chain over (a1, a2, b1, b2,
//      mu))` reproducing the host `per-corner scorer` exactly.
//   6. `score = score_gradient * score_intensity`; `d_score[i] =
//      max(d_score[i], score)` per launch (one launch per R).
//
// Corners that the kernel must skip (zero (x, y), patch out of
// bounds, edge degenerate) get a no-op — the existing `d_score[i]`
// (typically the value left by an earlier radius, or 0 on the first
// pass) survives.
//
// The class owns device buffers for (x, y, e1_cos, e1_sin, e2_cos,
// e2_sin, score) lazily resized on demand. It is not thread-safe;
// callers must serialise.
class CudaScoreCorners
{
public:
  // Initial device-array capacity; grown on demand. The 4K production
  // sample yields a few hundred corners, so 4 k is generous.
  static constexpr std::size_t kDefaultCapacity = 4096;

  CudaScoreCorners();
  ~CudaScoreCorners();

  CudaScoreCorners(const CudaScoreCorners &) = delete;
  CudaScoreCorners &operator=(const CudaScoreCorners &) = delete;
  CudaScoreCorners(CudaScoreCorners &&) = delete;
  CudaScoreCorners &operator=(CudaScoreCorners &&) = delete;

  // Score every corner in `corners` against (img, weights). The
  // function reads `corners.{x, y, edge1_cos, edge1_sin, edge2_cos,
  // edge2_sin}` and writes `corners.score`. Initial score is 0; the
  // per-radius max is taken on the device. Other fields are left
  // unchanged.
  void run(const GpuImagePtr &img, const GpuImagePtr &weights, CornerArray &corners);

private:
  void ensureCapacity(std::size_t min_capacity);

  float *d_x_{nullptr};
  float *d_y_{nullptr};
  float *d_e1c_{nullptr};
  float *d_e1s_{nullptr};
  float *d_e2c_{nullptr};
  float *d_e2s_{nullptr};
  float *d_score_{nullptr};
  std::size_t capacity_{0};
};

}  // namespace cuda
}  // namespace camera_chessboard_detector
