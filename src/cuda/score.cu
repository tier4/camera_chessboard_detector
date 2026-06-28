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

#include "cuda/score.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

namespace camera_chessboard_detector {
namespace cuda {

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr int kBlockThreads = 256;

inline void checkCuda(cudaError_t status, const char * what) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "[cuda_score_gpu] %s: %s\n", what,
                 cudaGetErrorString(status));
  }
}

// Per-thread cooperative sum reduce over `kBlockThreads` threads via
// shared memory. Result is valid only in thread 0; other threads
// return undefined.
__device__ float blockReduceSum(float val, float * smem) {
  const int tid = threadIdx.x;
  smem[tid] = val;
  __syncthreads();
  for (int s = kBlockThreads / 2; s > 0; s >>= 1) {
    if (tid < s) {
      smem[tid] += smem[tid + s];
    }
    __syncthreads();
  }
  return smem[0];
}

// Compute the `filter` value for one patch pixel (relative offset
// `dx`, `dy` from the corner). Mirrors the host loop:
//   if (norm1 <= 1.5 || norm2 <= 1.5) filter = +1; else -1
// The centre pixel (dx == 0 && dy == 0) is forced to -1, matching
// the host's `if (dx == 0 && dy == 0) continue;` (which leaves the
// default -1 value the matrix was zero-initialised to).
__device__ float computeFilter(int dx_int, int dy_int, float e1c, float e1s,
                               float e2c, float e2s) {
  if (dx_int == 0 && dy_int == 0) {
    return -1.0f;
  }
  const float dx = static_cast<float>(dx_int);
  const float dy = static_cast<float>(dy_int);
  const float p1x = dx * e1c * e1c + dy * e1c * e1s;
  const float p1y = dx * e1c * e1s + dy * e1s * e1s;
  const float p2x = dx * e2c * e2c + dy * e2c * e2s;
  const float p2y = dx * e2c * e2s + dy * e2s * e2s;
  const float n1 = sqrtf((p1x - dx) * (p1x - dx) + (p1y - dy) * (p1y - dy));
  const float n2 = sqrtf((p2x - dx) * (p2x - dx) + (p2y - dy) * (p2y - dy));
  return (n1 <= 1.5f || n2 <= 1.5f) ? 1.0f : -1.0f;
}

// Unnormalised quadrant kernel value at relative offset (dx, dy).
// Returns four values via out-params, exactly matching the host
// `buildQuadrantKernels` math (Gaussian radial envelope, gated by a |side| >
// 0.1 quadrant test). The host code's `sigma = kernelSize / 2`
// (integer division) is reproduced.
__device__ void computeQuadrantKernels(int dx_int, int dy_int, int radius,
                                       float sin1, float cos1, float sin2,
                                       float cos2, float & ka, float & kb,
                                       float & kc, float & kd) {
  ka = 0.0f;
  kb = 0.0f;
  kc = 0.0f;
  kd = 0.0f;
  const float dx = static_cast<float>(dx_int);
  const float dy = static_cast<float>(dy_int);
  const float dis = sqrtf(dx * dx + dy * dy);
  const float sigma = static_cast<float>(radius / 2);
  const float pdf = expf(-0.5f * dis * dis / (sigma * sigma)) /
                    (sqrtf(2.0f * kPi) * sigma);
  const float side1 = dx * (-sin1) + dy * cos1;
  const float side2 = dx * (-sin2) + dy * cos2;
  if (side1 <= -0.1f && side2 <= -0.1f) ka = pdf;
  if (side1 >= 0.1f && side2 >= 0.1f) kb = pdf;
  if (side1 <= -0.1f && side2 >= 0.1f) kc = pdf;
  if (side1 >= 0.1f && side2 <= -0.1f) kd = pdf;
}

__global__ void scoreCorners_k(
    const float * __restrict__ img, int img_w, int img_h,
    const float * __restrict__ weights, const float * __restrict__ d_x,
    const float * __restrict__ d_y, const float * __restrict__ d_e1c,
    const float * __restrict__ d_e1s, const float * __restrict__ d_e2c,
    const float * __restrict__ d_e2s, float * __restrict__ d_score,
    int num_corners, int radius) {
  __shared__ float s_reduce[kBlockThreads];
  __shared__ float s_sums[8];

  const int corner = blockIdx.x;
  if (corner >= num_corners) {
    return;
  }

  const float x_f = d_x[corner];
  const float y_f = d_y[corner];
  const int u = static_cast<int>(x_f + 0.5f);
  const int v = static_cast<int>(y_f + 0.5f);

  // Skip zeroed corners (the prune pipeline marks invalid ones with
  // (0, 0)). The host `per-corner scorer` does the same.
  if (u == 0 && v == 0) {
    return;
  }
  if (u - radius < 0 || u + radius >= img_w || v - radius < 0 ||
      v + radius >= img_h) {
    return;
  }

  const float e1c = d_e1c[corner];
  const float e1s = d_e1s[corner];
  const float e2c = d_e2c[corner];
  const float e2s = d_e2s[corner];

  // Host `per-corner scorer` rejects the same degenerate configuration.
  if (e1c == e2c && e1s == e2s && e1c == e1s && e1c < 0.0f) {
    return;
  }

  const float ang1 = atan2f(e1s, e1c);
  const float ang2 = atan2f(e2s, e2c);
  const float sin1 = sinf(ang1);
  const float cos1 = cosf(ang1);
  const float sin2 = sinf(ang2);
  const float cos2 = cosf(ang2);

  const int W = 2 * radius + 1;
  const int total = W * W;
  const float inv_total = 1.0f / static_cast<float>(total);

  // -------- Pass 1: per-pixel sums needed for normalisation ----------
  float sum_w = 0.0f;
  float sum_w2 = 0.0f;
  float sum_f = 0.0f;
  float sum_f2 = 0.0f;
  float sum_ka = 0.0f;
  float sum_kb = 0.0f;
  float sum_kc = 0.0f;
  float sum_kd = 0.0f;
  for (int p = threadIdx.x; p < total; p += kBlockThreads) {
    const int dx_int = (p % W) - radius;
    const int dy_int = (p / W) - radius;
    const int xx = u + dx_int;
    const int yy = v + dy_int;
    const float wv = weights[yy * img_w + xx];
    const float fv = computeFilter(dx_int, dy_int, e1c, e1s, e2c, e2s);

    float ka, kb, kc, kd;
    computeQuadrantKernels(dx_int, dy_int, radius, sin1, cos1, sin2, cos2, ka,
                           kb, kc, kd);
    sum_w += wv;
    sum_w2 += wv * wv;
    sum_f += fv;
    sum_f2 += fv * fv;
    sum_ka += ka;
    sum_kb += kb;
    sum_kc += kc;
    sum_kd += kd;
  }

  // Reduce each sum across the block. blockReduceSum returns the
  // total in thread 0; we publish via s_sums[].
  const float r_w = blockReduceSum(sum_w, s_reduce);
  if (threadIdx.x == 0) s_sums[0] = r_w;
  __syncthreads();
  const float r_w2 = blockReduceSum(sum_w2, s_reduce);
  if (threadIdx.x == 0) s_sums[1] = r_w2;
  __syncthreads();
  const float r_f = blockReduceSum(sum_f, s_reduce);
  if (threadIdx.x == 0) s_sums[2] = r_f;
  __syncthreads();
  const float r_f2 = blockReduceSum(sum_f2, s_reduce);
  if (threadIdx.x == 0) s_sums[3] = r_f2;
  __syncthreads();
  const float r_ka = blockReduceSum(sum_ka, s_reduce);
  if (threadIdx.x == 0) s_sums[4] = r_ka;
  __syncthreads();
  const float r_kb = blockReduceSum(sum_kb, s_reduce);
  if (threadIdx.x == 0) s_sums[5] = r_kb;
  __syncthreads();
  const float r_kc = blockReduceSum(sum_kc, s_reduce);
  if (threadIdx.x == 0) s_sums[6] = r_kc;
  __syncthreads();
  const float r_kd = blockReduceSum(sum_kd, s_reduce);
  if (threadIdx.x == 0) s_sums[7] = r_kd;
  __syncthreads();

  const float mean_w = s_sums[0] * inv_total;
  const float var_w = fmaxf(0.0f, s_sums[1] * inv_total - mean_w * mean_w);
  const float std_w = sqrtf(var_w);
  const float mean_f = s_sums[2] * inv_total;
  const float var_f = fmaxf(0.0f, s_sums[3] * inv_total - mean_f * mean_f);
  const float std_f = sqrtf(var_f);
  const float sum_ka_total = s_sums[4];
  const float sum_kb_total = s_sums[5];
  const float sum_kc_total = s_sums[6];
  const float sum_kd_total = s_sums[7];

  // Guard against zero std (uniform patch): skip gracefully.
  if (std_w == 0.0f || std_f == 0.0f) {
    return;
  }

  // -------- Pass 2: normalised inner products ----------
  float dot_fw = 0.0f;
  float dot_ka_img = 0.0f;
  float dot_kb_img = 0.0f;
  float dot_kc_img = 0.0f;
  float dot_kd_img = 0.0f;
  for (int p = threadIdx.x; p < total; p += kBlockThreads) {
    const int dx_int = (p % W) - radius;
    const int dy_int = (p / W) - radius;
    const int xx = u + dx_int;
    const int yy = v + dy_int;
    const float wv = weights[yy * img_w + xx];
    const float iv = img[yy * img_w + xx];
    const float fv = computeFilter(dx_int, dy_int, e1c, e1s, e2c, e2s);
    const float wn = (wv - mean_w) / std_w;
    const float fn = (fv - mean_f) / std_f;
    dot_fw += wn * fn;

    float ka, kb, kc, kd;
    computeQuadrantKernels(dx_int, dy_int, radius, sin1, cos1, sin2, cos2, ka,
                           kb, kc, kd);
    dot_ka_img += ka * iv;
    dot_kb_img += kb * iv;
    dot_kc_img += kc * iv;
    dot_kd_img += kd * iv;
  }

  const float r_dot_fw = blockReduceSum(dot_fw, s_reduce);
  if (threadIdx.x == 0) s_sums[0] = r_dot_fw;
  __syncthreads();
  const float r_a = blockReduceSum(dot_ka_img, s_reduce);
  if (threadIdx.x == 0) s_sums[1] = r_a;
  __syncthreads();
  const float r_b = blockReduceSum(dot_kb_img, s_reduce);
  if (threadIdx.x == 0) s_sums[2] = r_b;
  __syncthreads();
  const float r_c = blockReduceSum(dot_kc_img, s_reduce);
  if (threadIdx.x == 0) s_sums[3] = r_c;
  __syncthreads();
  const float r_d = blockReduceSum(dot_kd_img, s_reduce);
  if (threadIdx.x == 0) s_sums[4] = r_d;
  __syncthreads();

  if (threadIdx.x == 0) {
    const float gradient_raw = s_sums[0] / static_cast<float>(total - 1);
    const float score_gradient = gradient_raw >= 0.0f ? gradient_raw : 0.0f;

    const float a1 = s_sums[1] / sum_ka_total;
    const float a2 = s_sums[2] / sum_kb_total;
    const float b1 = s_sums[3] / sum_kc_total;
    const float b2 = s_sums[4] / sum_kd_total;
    const float mu = (a1 + a2 + b1 + b2) * 0.25f;

    // Replicate the host min/max chain literally so floating-point
    // semantics line up corner-for-corner with the reference impl.
    float score_a = (a1 - mu) >= (a2 - mu) ? (a2 - mu) : (a1 - mu);
    float score_b = (mu - b1) >= (mu - b2) ? (mu - b2) : (mu - b1);
    float score_1 = score_a >= score_b ? score_b : score_a;

    score_b = (b1 - mu) >= (b2 - mu) ? (b2 - mu) : (b1 - mu);
    score_a = (mu - a1) >= (mu - a2) ? (mu - a2) : (mu - a1);
    const float score_2 = score_a >= score_b ? score_b : score_a;

    float score_intensity = score_1 >= score_2 ? score_1 : score_2;
    score_intensity = score_intensity > 0.0f ? score_intensity : 0.0f;

    const float score = score_gradient * score_intensity;
    if (score > d_score[corner]) {
      d_score[corner] = score;
    }
  }
}

}  // namespace

CudaScoreCorners::CudaScoreCorners() {
  ensureCapacity(kDefaultCapacity);
}

CudaScoreCorners::~CudaScoreCorners() {
  if (d_x_) cudaFree(d_x_);
  if (d_y_) cudaFree(d_y_);
  if (d_e1c_) cudaFree(d_e1c_);
  if (d_e1s_) cudaFree(d_e1s_);
  if (d_e2c_) cudaFree(d_e2c_);
  if (d_e2s_) cudaFree(d_e2s_);
  if (d_score_) cudaFree(d_score_);
}

void CudaScoreCorners::ensureCapacity(std::size_t min_capacity) {
  if (min_capacity <= capacity_) {
    return;
  }
  if (d_x_) cudaFree(d_x_);
  if (d_y_) cudaFree(d_y_);
  if (d_e1c_) cudaFree(d_e1c_);
  if (d_e1s_) cudaFree(d_e1s_);
  if (d_e2c_) cudaFree(d_e2c_);
  if (d_e2s_) cudaFree(d_e2s_);
  if (d_score_) cudaFree(d_score_);
  d_x_ = nullptr;
  d_y_ = nullptr;
  d_e1c_ = nullptr;
  d_e1s_ = nullptr;
  d_e2c_ = nullptr;
  d_e2s_ = nullptr;
  d_score_ = nullptr;
  const std::size_t bytes = min_capacity * sizeof(float);
  checkCuda(cudaMalloc(&d_x_, bytes), "alloc d_x");
  checkCuda(cudaMalloc(&d_y_, bytes), "alloc d_y");
  checkCuda(cudaMalloc(&d_e1c_, bytes), "alloc d_e1c");
  checkCuda(cudaMalloc(&d_e1s_, bytes), "alloc d_e1s");
  checkCuda(cudaMalloc(&d_e2c_, bytes), "alloc d_e2c");
  checkCuda(cudaMalloc(&d_e2s_, bytes), "alloc d_e2s");
  checkCuda(cudaMalloc(&d_score_, bytes), "alloc d_score");
  capacity_ = min_capacity;
}

void CudaScoreCorners::run(const GpuImagePtr & img, const GpuImagePtr & weights,
                           CornerArray & corners) {
  const std::size_t n = corners.x.size();
  if (n == 0 || !img || !weights) {
    return;
  }
  if (corners.score.size() != n) {
    corners.score.assign(n, 0.0f);
  }
  ensureCapacity(n);

  // Upload corner state. Score is initialised on device with the
  // current host values (typically all zeros on first prune call;
  // the kernel does a max against this on each radius pass).
  const std::size_t bytes = n * sizeof(float);
  checkCuda(cudaMemcpy(d_x_, corners.x.data(), bytes, cudaMemcpyHostToDevice),
            "copy d_x");
  checkCuda(cudaMemcpy(d_y_, corners.y.data(), bytes, cudaMemcpyHostToDevice),
            "copy d_y");
  checkCuda(cudaMemcpy(d_e1c_, corners.edge1_cos.data(), bytes,
                       cudaMemcpyHostToDevice),
            "copy d_e1c");
  checkCuda(cudaMemcpy(d_e1s_, corners.edge1_sin.data(), bytes,
                       cudaMemcpyHostToDevice),
            "copy d_e1s");
  checkCuda(cudaMemcpy(d_e2c_, corners.edge2_cos.data(), bytes,
                       cudaMemcpyHostToDevice),
            "copy d_e2c");
  checkCuda(cudaMemcpy(d_e2s_, corners.edge2_sin.data(), bytes,
                       cudaMemcpyHostToDevice),
            "copy d_e2s");
  checkCuda(cudaMemcpy(d_score_, corners.score.data(), bytes,
                       cudaMemcpyHostToDevice),
            "copy d_score");

  const int img_w = img->width();
  const int img_h = img->height();
  const dim3 grid(static_cast<unsigned int>(n));
  const dim3 block(kBlockThreads);

  // Radius sweep — same R ∈ {4, 8, 12} the host code used. Each
  // launch updates d_score with a per-corner max.
  const int radii[3] = {4, 8, 12};
  for (int r : radii) {
    scoreCorners_k<<<grid, block>>>(
        img->data(), img_w, img_h, weights->data(), d_x_, d_y_, d_e1c_, d_e1s_,
        d_e2c_, d_e2s_, d_score_, static_cast<int>(n), r);
    checkCuda(cudaGetLastError(), "launch scoreCorners_k");
  }

  // Download the per-corner score back to the host.
  checkCuda(cudaMemcpy(corners.score.data(), d_score_, bytes,
                       cudaMemcpyDeviceToHost),
            "copy back d_score");
}

}  // namespace cuda
}  // namespace camera_chessboard_detector
