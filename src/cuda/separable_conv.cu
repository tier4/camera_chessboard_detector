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

// Opt-in CUDA separable Geiger convolution. See header for the
// design rationale and gating conditions.

#include "cuda/gpu_buffers.hpp"
#include "cuda/separable_conv.hpp"

#include <Eigen/Dense>
#include <Eigen/SVD>

#include <cassert>
#include <cmath>
#include <vector>

namespace camera_chessboard_detector
{
namespace cuda
{

namespace
{

// Horizontal 1D cross-correlation:
//   out[j, i] = sum_{s=-R..R} v[s + R] * in[j, i + s]
// Valid-output region is the inset [R, W-R) x [0, H); pixels outside
// that region are written as zero (caller relies on this to keep the
// vertical pass's input clean).
__global__ void sep_conv_h_kernel(
  float *__restrict__ out, const float *__restrict__ in, const float *__restrict__ v, int W, int H,
  int R
)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= W || j >= H) return;
  if (i < R || i >= W - R)
  {
    out[j * W + i] = 0.0f;
    return;
  }
  const float *row = in + j * W;
  float acc = 0.0f;
  for (int s = -R; s <= R; ++s)
  {
    acc = fmaf(v[s + R], row[i + s], acc);
  }
  out[j * W + i] = acc;
}

// Vertical 1D cross-correlation, accumulating into `out`:
//   out[j, i] += sum_{r=-R..R} u[r + R] * in[j + r, i]
// Valid-output region is [0, W) x [R, H-R); pixels outside that region
// are left unchanged so earlier accumulations (k > 0) survive.
__global__ void sep_conv_v_accum_kernel(
  float *__restrict__ out, const float *__restrict__ in, const float *__restrict__ u, int W, int H,
  int R
)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= W || j < R || j >= H - R) return;
  float acc = 0.0f;
  for (int r = -R; r <= R; ++r)
  {
    acc = fmaf(u[r + R], in[(j + r) * W + i], acc);
  }
  out[j * W + i] += acc;
}

// SVD decompose K = U Σ V^T on the host and return (sqrt(σ_k) * u_k,
// sqrt(σ_k) * v_k) per component for the requested rank. Components
// whose singular value is essentially zero are skipped.
void decomposeKernel(
  const camera_chessboard_detector::CpuKernelF32 &kernel, int requested_rank,
  std::vector<std::vector<float>> &u_out, std::vector<std::vector<float>> &v_out
)
{
  const int S = kernel.width();
  assert(S == kernel.height());

  Eigen::MatrixXf K(S, S);
  for (int r = 0; r < S; ++r)
  {
    for (int c = 0; c < S; ++c)
    {
      // K(r, c) corresponds to kernel row r, col c -- the same logical
      // layout used by the references and by the dense
      // dense-convolution kernel.
      K(r, c) = kernel.at(c, r);
    }
  }

  Eigen::JacobiSVD<Eigen::MatrixXf> svd(K, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const auto &U = svd.matrixU();
  const auto &V = svd.matrixV();
  const auto &sv = svd.singularValues();

  const int max_rank = static_cast<int>(sv.size());
  const int rank = (requested_rank <= 0) ? max_rank : std::min(requested_rank, max_rank);

  u_out.clear();
  v_out.clear();
  u_out.reserve(rank);
  v_out.reserve(rank);

  // A singular value below sv(0) * eps is numerical noise; dropping it
  // keeps the accumulator clean and matches what "full rank" means in
  // practice for a numerically rank-deficient stored kernel.
  const float drop_threshold = (sv.size() > 0) ? sv(0) * 1e-7f : 0.0f;

  for (int k = 0; k < rank; ++k)
  {
    const float s = sv(k);
    if (s <= drop_threshold) break;
    const float root_s = std::sqrt(s);
    std::vector<float> u(S), v(S);
    for (int t = 0; t < S; ++t)
    {
      u[t] = U(t, k) * root_s;  // vertical (row direction)
      v[t] = V(t, k) * root_s;  // horizontal (col direction)
    }
    u_out.push_back(std::move(u));
    v_out.push_back(std::move(v));
  }
}

}  // namespace

CudaSeparableConvolver::CudaSeparableConvolver()
{
  NVCHK(cudaStreamCreate(&stream_));
  tmp_ = std::make_shared<GpuImageF32>();
}

CudaSeparableConvolver::~CudaSeparableConvolver()
{
  release();
  cudaStreamDestroy(stream_);  // non-throwing dtor; ignore error code
}

void CudaSeparableConvolver::release()
{
  for (auto *p : d_u_)
    if (p) cudaFree(p);
  for (auto *p : d_v_)
    if (p) cudaFree(p);
  d_u_.clear();
  d_v_.clear();
}

void CudaSeparableConvolver::setKernel(
  const camera_chessboard_detector::CpuKernelF32 &kernel, int requested_rank
)
{
  const int S = kernel.width();
  assert(S == kernel.height());
  assert(S >= 3 && (S % 2) == 1);

  std::vector<std::vector<float>> u_components, v_components;
  decomposeKernel(kernel, requested_rank, u_components, v_components);

  release();
  radius_ = S / 2;
  rank_ = static_cast<int>(u_components.size());
  d_u_.resize(rank_, nullptr);
  d_v_.resize(rank_, nullptr);

  for (int k = 0; k < rank_; ++k)
  {
    NVCHK(cudaMalloc(&d_u_[k], S * sizeof(float)));
    NVCHK(cudaMalloc(&d_v_[k], S * sizeof(float)));
    NVCHK(cudaMemcpy(d_u_[k], u_components[k].data(), S * sizeof(float), cudaMemcpyHostToDevice));
    NVCHK(cudaMemcpy(d_v_[k], v_components[k].data(), S * sizeof(float), cudaMemcpyHostToDevice));
  }
}

void CudaSeparableConvolver::convolve(const GpuImagePtr &image, GpuImagePtr &output)
{
  const int W = image->width();
  const int H = image->height();
  output->resize(W, H);
  output->fill(0);
  tmp_->resize(W, H);
  // tmp_ is fully written by sep_conv_h_kernel inside the valid region
  // and explicitly zeroed outside that region, so a per-iteration
  // fill is unnecessary.

  if (rank_ == 0) return;  // pathological all-zero kernel
  const int R = radius_;
  assert(R > 0);

  const dim3 block(32, 8);
  const dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);

  for (int k = 0; k < rank_; ++k)
  {
    sep_conv_h_kernel<<<grid, block, 0, stream_>>>(tmp_->data(), image->data(), d_v_[k], W, H, R);
    sep_conv_v_accum_kernel<<<grid, block, 0, stream_>>>(
      output->data(), tmp_->data(), d_u_[k], W, H, R
    );
  }
  NVCHK(cudaStreamSynchronize(stream_));
}

}  // namespace cuda
}  // namespace camera_chessboard_detector
