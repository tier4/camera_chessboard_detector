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

#include <cassert>
#include <cstddef>
#include <cstdio>

// Abort with a CUDA error message if `err` is not cudaSuccess.
#define NVCHK(err)                                                \
  if (err != cudaSuccess)                                         \
  {                                                               \
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
    assert(false);                                                \
  }

namespace camera_chessboard_detector
{

namespace cuda
{

template <typename T>
class GpuImage
{
public:
  GpuImage();
  GpuImage(int width, int height);
  // GpuImage(int width, int height, T *data);
  // GpuImage(GpuImage<T> &&other);
  ~GpuImage();

  bool resize(int width, int height);
  void fill(T value);
  // void set(int x, int y, T value);

  std::size_t size() const;
  bool valid() const;
  int width() const;
  int height() const;
  T *data();
  // T at(int x, int y) const;

  // static GpuImage<T> from_cpu(const CpuImage<T> &other);
  void download(const CpuImage<T> &other);
  void download(T *data, std::size_t size);
  void upload(const CpuImage<T> &other);
  void upload(const T *data, std::size_t size);
  static GpuImage<T> from_cpu(const CpuImage<T> &other);

private:
  bool initialized_;
  int width_;
  int height_;
  T *data_;
};

template <typename T>
class GpuKernel : public GpuImage<T>
{
public:
  GpuKernel(int radius);
  GpuKernel(const CpuKernel<T> &other);
  static GpuKernel<T> from_cpu(const CpuKernel<T> &other);
};

typedef GpuImage<float> GpuImageF32;
typedef std::shared_ptr<GpuImageF32> GpuImagePtr;
typedef GpuKernel<float> GpuKernelF32;
typedef std::shared_ptr<GpuKernelF32> GpuKernelPtr;

}  // namespace cuda

}  // namespace camera_chessboard_detector
