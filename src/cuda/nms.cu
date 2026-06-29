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

#include "cuda/gpu_buffers.hpp"
#include "cuda/nms.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <vector>

namespace camera_chessboard_detector
{
namespace cuda
{

namespace
{

// Lightweight CUDA error check macro mirroring NVCHK used elsewhere
// in the package. Prints and continues — favours diagnostics over
// aborting on first failure, matching the surrounding detector code.
inline void checkCuda(cudaError_t status, const char *what)
{
  if (status != cudaSuccess)
  {
    std::fprintf(stderr, "[cuda_nms_gpu] %s: %s\n", what, cudaGetErrorString(status));
  }
}

// Fused NMS kernel:
//   - Margin / threshold reject early.
//   - Iterate over (2R+1)*(2R+1) window. Reject if any pixel has a
//     strictly greater value, or has the same value at a row-major
//     earlier position. Surviving pixels are the lexicographically
//     first plateau representative — exactly one per plateau, by
//     construction.
//   - Atomic-append (x, y, value) to a compact device array.
//
// The kernel is launched over the full image (no padding); border
// reads inside the window are guarded by the margin reject above.
// `radius >= 1` and `margin + radius < min(rows, cols)` are
// preconditions enforced by the host wrapper.
__global__ void nms_compact_kernel(
  const float *__restrict__ src, int rows, int cols, int radius, int margin, float threshold,
  int *__restrict__ d_out_x, int *__restrict__ d_out_y, float *__restrict__ d_out_v,
  int *__restrict__ d_count, int capacity
)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= cols || y >= rows)
  {
    return;
  }
  const int low = margin + radius;
  if (x < low || x >= cols - low || y < low || y >= rows - low)
  {
    return;
  }

  const float val = src[y * cols + x];
  if (val <= threshold)
  {
    return;
  }

  for (int dy = -radius; dy <= radius; ++dy)
  {
    const int yy = y + dy;
    for (int dx = -radius; dx <= radius; ++dx)
    {
      const int xx = x + dx;
      if (xx == x && yy == y)
      {
        continue;
      }
      const float other = src[yy * cols + xx];
      if (other > val)
      {
        return;
      }
      if (other == val && (yy < y || (yy == y && xx < x)))
      {
        // Plateau tie: a lexicographically earlier pixel already
        // represents this plateau.
        return;
      }
    }
  }

  const int pos = atomicAdd(d_count, 1);
  if (pos < capacity)
  {
    d_out_x[pos] = x;
    d_out_y[pos] = y;
    d_out_v[pos] = val;
  }
}

}  // namespace

CudaNonMaxSuppression::CudaNonMaxSuppression()
{
  checkCuda(cudaMalloc(&d_count_, sizeof(int)), "alloc d_count");
  ensureCapacity(kDefaultCapacity);
}

CudaNonMaxSuppression::~CudaNonMaxSuppression()
{
  if (d_out_x_) cudaFree(d_out_x_);
  if (d_out_y_) cudaFree(d_out_y_);
  if (d_out_v_) cudaFree(d_out_v_);
  if (d_count_) cudaFree(d_count_);
}

void CudaNonMaxSuppression::ensureCapacity(std::size_t min_capacity)
{
  if (min_capacity <= capacity_)
  {
    return;
  }
  if (d_out_x_) cudaFree(d_out_x_);
  if (d_out_y_) cudaFree(d_out_y_);
  if (d_out_v_) cudaFree(d_out_v_);
  d_out_x_ = nullptr;
  d_out_y_ = nullptr;
  d_out_v_ = nullptr;
  checkCuda(cudaMalloc(&d_out_x_, min_capacity * sizeof(int)), "alloc d_out_x");
  checkCuda(cudaMalloc(&d_out_y_, min_capacity * sizeof(int)), "alloc d_out_y");
  checkCuda(cudaMalloc(&d_out_v_, min_capacity * sizeof(float)), "alloc d_out_v");
  capacity_ = min_capacity;
}

std::size_t CudaNonMaxSuppression::run(
  const GpuImagePtr &likelihood_map, int radius, int margin, float threshold, CornerArray &corners
)
{
  corners.x.clear();
  corners.y.clear();
  corners.v.clear();
  corners.edge1_cos.clear();
  corners.edge1_sin.clear();
  corners.edge2_cos.clear();
  corners.edge2_sin.clear();
  corners.score.clear();

  if (!likelihood_map || radius < 1 || margin < 0)
  {
    return 0;
  }
  const int rows = likelihood_map->height();
  const int cols = likelihood_map->width();
  const int low = margin + radius;
  if (rows <= 2 * low || cols <= 2 * low)
  {
    return 0;
  }

  checkCuda(cudaMemset(d_count_, 0, sizeof(int)), "memset d_count");

  const dim3 block(32, 8);
  const dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
  nms_compact_kernel<<<grid, block>>>(
    likelihood_map->data(), rows, cols, radius, margin, threshold, d_out_x_, d_out_y_, d_out_v_,
    d_count_, static_cast<int>(capacity_)
  );
  checkCuda(cudaGetLastError(), "launch nms_compact_kernel");

  int h_count = 0;
  checkCuda(cudaMemcpy(&h_count, d_count_, sizeof(int), cudaMemcpyDeviceToHost), "copy d_count");

  const std::size_t emitted = static_cast<std::size_t>(h_count);
  const std::size_t kept = std::min(emitted, capacity_);
  if (kept == 0)
  {
    return emitted;
  }

  std::vector<int> h_x(kept);
  std::vector<int> h_y(kept);
  std::vector<float> h_v(kept);
  checkCuda(
    cudaMemcpy(h_x.data(), d_out_x_, kept * sizeof(int), cudaMemcpyDeviceToHost), "copy d_out_x"
  );
  checkCuda(
    cudaMemcpy(h_y.data(), d_out_y_, kept * sizeof(int), cudaMemcpyDeviceToHost), "copy d_out_y"
  );
  checkCuda(
    cudaMemcpy(h_v.data(), d_out_v_, kept * sizeof(float), cudaMemcpyDeviceToHost), "copy d_out_v"
  );

  corners.x.reserve(kept);
  corners.y.reserve(kept);
  corners.v.reserve(kept);
  corners.edge1_cos.reserve(kept);
  corners.edge1_sin.reserve(kept);
  corners.edge2_cos.reserve(kept);
  corners.edge2_sin.reserve(kept);
  for (std::size_t i = 0; i < kept; ++i)
  {
    corners.x.push_back(static_cast<float>(h_x[i]));
    corners.y.push_back(static_cast<float>(h_y[i]));
    corners.v.push_back(h_v[i]);
    // Placeholder edge fields — same contract as the host
    // non-max suppression, consumed by the refiner.
    corners.edge1_cos.push_back(2.0f);
    corners.edge1_sin.push_back(0.0f);
    corners.edge2_cos.push_back(2.0f);
    corners.edge2_sin.push_back(0.0f);
  }
  return emitted;
}

}  // namespace cuda
}  // namespace camera_chessboard_detector
