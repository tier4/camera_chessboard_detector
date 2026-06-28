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

#include "cuda/preprocess.hpp"

#include <cmath>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

namespace camera_chessboard_detector {
namespace cuda {
namespace {

constexpr int kGaussKsize = 9;
constexpr int kGaussRadius = 4;
constexpr double kGaussSigma = 1.5;

// BORDER_REFLECT_101 (gfedcb|abcdefgh|gfedcba), matching cv::GaussianBlur's
// default border handling.
__device__ inline int reflect101(int i, int n) {
  if (n == 1) return 0;
  while (i < 0 || i >= n) {
    if (i < 0) i = -i;
    if (i >= n) i = 2 * (n - 1) - i;
  }
  return i;
}

// BGR (or gray) uint8 -> float gray, using OpenCV's fixed-point BGR2GRAY
// coefficients so the value matches cv::cvtColor exactly before the blur.
__global__ void bgr2gray_f_k(float *gray, const unsigned char *src, int w,
                             int h, int channels) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  const int idx = y * w + x;
  if (channels == 1) {
    gray[idx] = static_cast<float>(src[idx]);
    return;
  }
  const unsigned char *p = src + static_cast<std::size_t>(idx) * 3;
  const int B = p[0], G = p[1], R = p[2];
  gray[idx] = static_cast<float>((R * 4899 + G * 9617 + B * 1868 + 8192) >> 14);
}

__global__ void gauss_h_k(float *dst, const float *src, const float *k, int w,
                          int h) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  float acc = 0.0f;
#pragma unroll
  for (int t = 0; t < kGaussKsize; ++t) {
    acc += k[t] * src[y * w + reflect101(x + t - kGaussRadius, w)];
  }
  dst[y * w + x] = acc;
}

__global__ void gauss_v_k(float *dst, const float *src, const float *k, int w,
                          int h) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  float acc = 0.0f;
#pragma unroll
  for (int t = 0; t < kGaussKsize; ++t) {
    acc += k[t] * src[reflect101(y + t - kGaussRadius, h) * w + x];
  }
  dst[y * w + x] = acc;
}

__global__ void normalize_k(float *dst, const float *src, float mn,
                            float inv_range, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = (src[i] - mn) * inv_range;
}

}  // namespace

CudaPreprocessor::CudaPreprocessor() {}

CudaPreprocessor::~CudaPreprocessor() {
  if (d_src_ != nullptr) cudaFree(d_src_);
  if (d_kernel_ != nullptr) cudaFree(d_kernel_);
}

void CudaPreprocessor::ensureKernel() {
  if (kernel_ready_) return;
  double w[kGaussKsize];
  double sum = 0.0;
  for (int i = 0; i < kGaussKsize; ++i) {
    const double x = i - kGaussRadius;
    w[i] = std::exp(-0.5 * x * x / (kGaussSigma * kGaussSigma));
    sum += w[i];
  }
  float wf[kGaussKsize];
  for (int i = 0; i < kGaussKsize; ++i) {
    wf[i] = static_cast<float>(w[i] / sum);
  }
  NVCHK(cudaMalloc(&d_kernel_, kGaussKsize * sizeof(float)));
  NVCHK(cudaMemcpy(d_kernel_, wf, kGaussKsize * sizeof(float),
                   cudaMemcpyHostToDevice));
  kernel_ready_ = true;
}

const GpuImagePtr &CudaPreprocessor::run(const unsigned char *h_src, int width,
                                         int height, int channels) {
  ensureKernel();

  const std::size_t pixels = static_cast<std::size_t>(width) * height;
  const std::size_t src_bytes = pixels * channels;
  if (src_bytes > src_capacity_) {
    if (d_src_ != nullptr) cudaFree(d_src_);
    NVCHK(cudaMalloc(&d_src_, src_bytes));
    src_capacity_ = src_bytes;
  }
  NVCHK(cudaMemcpy(d_src_, h_src, src_bytes, cudaMemcpyHostToDevice));

  if (!gray_) gray_ = std::make_shared<GpuImageF32>();
  if (!tmp_) tmp_ = std::make_shared<GpuImageF32>();
  if (!blur_) blur_ = std::make_shared<GpuImageF32>();
  if (!norm_) norm_ = std::make_shared<GpuImageF32>();
  gray_->resize(width, height);
  tmp_->resize(width, height);
  blur_->resize(width, height);
  norm_->resize(width, height);

  const dim3 block(32, 8);
  const dim3 grid((width + block.x - 1) / block.x,
                  (height + block.y - 1) / block.y);
  bgr2gray_f_k<<<grid, block>>>(gray_->data(), d_src_, width, height, channels);
  gauss_h_k<<<grid, block>>>(tmp_->data(), gray_->data(), d_kernel_, width,
                             height);
  gauss_v_k<<<grid, block>>>(blur_->data(), tmp_->data(), d_kernel_, width,
                             height);

  // min / max of the blurred image for the [0, 1] normalisation.
  thrust::device_ptr<float> bp(blur_->data());
  auto mm = thrust::minmax_element(thrust::device, bp, bp + pixels);
  const float mn = *mm.first;
  const float mx = *mm.second;
  const float inv_range = (mx > mn) ? (1.0f / (mx - mn)) : 0.0f;

  const int total = static_cast<int>(pixels);
  const int tb = 256;
  normalize_k<<<(total + tb - 1) / tb, tb>>>(norm_->data(), blur_->data(), mn,
                                             inv_range, total);
  return norm_;
}

}  // namespace cuda
}  // namespace camera_chessboard_detector
