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

// Unit tests for the GpuImage device-buffer owner (src/cuda/gpu_buffers.hpp).
//
// GpuImage owns a raw cudaMalloc allocation, so a copy would share the
// pointer and free it twice; and every member must be initialised by the
// default constructor, because that is how the pipeline creates all of its
// buffers (std::make_shared<GpuImageF32>() followed by resize()).

#include "cuda/gpu_buffers.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstring>
#include <new>
#include <type_traits>
#include <vector>

namespace
{

using camera_chessboard_detector::cuda::GpuImageF32;
using camera_chessboard_detector::cuda::GpuKernelF32;

bool cudaDeviceAvailable()
{
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

// Default-constructs a GpuImageF32 in storage pre-filled with `pattern`, so a
// member the constructor forgets to initialise keeps that pattern instead of
// whatever the allocator happened to leave behind.
template <typename Body>
void withDefaultConstructedIn(unsigned char pattern, Body &&body)
{
  alignas(GpuImageF32) unsigned char storage[sizeof(GpuImageF32)];
  std::memset(storage, pattern, sizeof(storage));
  auto *image = new (storage) GpuImageF32();
  body(*image);
  image->~GpuImageF32();
}

}  // namespace

TEST(GpuImage, IsNotCopyable)
{
  EXPECT_FALSE(std::is_copy_constructible<GpuImageF32>::value);
  EXPECT_FALSE(std::is_copy_assignable<GpuImageF32>::value);
  EXPECT_FALSE(std::is_copy_constructible<GpuKernelF32>::value);
  EXPECT_FALSE(std::is_copy_assignable<GpuKernelF32>::value);
}

TEST(GpuImage, DefaultConstructedIsEmptyAndInvalid)
{
  for (const unsigned char pattern : {0x00, 0x01})
  {
    SCOPED_TRACE(static_cast<int>(pattern));
    withDefaultConstructedIn(pattern, [](GpuImageF32 &image) {
      EXPECT_FALSE(image.valid());
      EXPECT_EQ(image.data(), nullptr);
      EXPECT_EQ(image.width(), 0);
      EXPECT_EQ(image.height(), 0);
      EXPECT_EQ(image.size(), 0u);
    });
  }
}

TEST(GpuImage, ResizeFromDefaultAllocatesAndBecomesValid)
{
  if (!cudaDeviceAvailable()) GTEST_SKIP() << "no CUDA device";
  withDefaultConstructedIn(0x00, [](GpuImageF32 &image) {
    ASSERT_TRUE(image.resize(16, 8));
    EXPECT_TRUE(image.valid());
    EXPECT_NE(image.data(), nullptr);
    EXPECT_EQ(image.width(), 16);
    EXPECT_EQ(image.height(), 8);
  });
}

TEST(GpuImage, SizedConstructorIsValid)
{
  if (!cudaDeviceAvailable()) GTEST_SKIP() << "no CUDA device";
  GpuImageF32 image(16, 8);
  EXPECT_TRUE(image.valid());
  EXPECT_NE(image.data(), nullptr);
  EXPECT_EQ(image.size(), 16u * 8u);
}

TEST(GpuImage, RoundTripsDataAfterShrinkingResize)
{
  if (!cudaDeviceAvailable()) GTEST_SKIP() << "no CUDA device";
  GpuImageF32 image(16, 8);
  ASSERT_TRUE(image.resize(8, 4));

  std::vector<float> uploaded(8 * 4);
  for (std::size_t i = 0; i < uploaded.size(); ++i) uploaded[i] = static_cast<float>(i) * 0.5f;
  image.upload(uploaded.data(), uploaded.size());

  std::vector<float> downloaded(uploaded.size(), -1.0f);
  image.download(downloaded.data(), downloaded.size());
  EXPECT_EQ(downloaded, uploaded);
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}
