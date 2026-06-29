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

#include "core.hpp"

#include <opencv2/imgproc/imgproc.hpp>

#include <cstring>

namespace camera_chessboard_detector
{

namespace utils
{

}  // namespace utils

#ifdef OPENCV_ENABLED
template <typename T>
CpuImage<T>::CpuImage(const cv::Mat &mat)
{
  assert((mat.type() % 8) == cv::DataType<T>::type);
  width_ = mat.cols;
  height_ = mat.rows;
  int channels = mat.channels();
  if (channels == 1)
  {
    // For some reason, static_cast doesn't work here.
    data_.resize(width_ * height_);
    std::memcpy(data_.data(), mat.data, width_ * height_ * sizeof(T));
  }
  else if (channels == 3)
  {
    cv::Mat gray;
    cv::cvtColor(mat, gray, cv::COLOR_BGR2GRAY);
    data_.resize(width_ * height_);
    std::memcpy(data_.data(), mat.data, width_ * height_ * sizeof(T));
  }
}

template <typename T>
cv::Mat CpuImage<T>::toMatF32() const
{
  cv::Mat mat(height_, width_, CV_32FC1);
  std::memcpy(mat.data, data_.data(), width_ * height_ * sizeof(T));
  return mat;
}

template <typename T>
cv::Mat CpuImage<T>::toMatU8() const
{
  cv::Mat mat(height_, width_, CV_8UC1);
  std::memcpy(mat.data, data_.data(), width_ * height_ * sizeof(T));
  return mat;
}

template <typename T>
cv::Mat CpuImage<T>::toMat() const
{
  if constexpr (std::is_same<T, float>::value)
  {
    return toMatF32();
  }
  else if constexpr (std::is_same<T, uint8_t>::value)
  {
    return toMatU8();
  }
  else
  {
    assert(false);
  }
}
#endif

// template <typename T>
// CpuImage(int width, int height, std::vector<T> && data) {}

template <typename T>
CpuImage<T>::CpuImage(int width, int height)
{
  width_ = width;
  height_ = height;
  data_.resize(width * height);
  fill(0);
}

template <typename T>
bool CpuImage<T>::resize(int width, int height)
{
  if (width == width_ && height == height_) return false;

  width_ = width;
  height_ = height;
  data_.resize(width * height);
  return true;
}

template <typename T>
void CpuImage<T>::fill(T value)
{
  std::fill(data_.begin(), data_.end(), value);
}

template <typename T>
void CpuImage<T>::set(int x, int y, T value)
{
  data_[y * width_ + x] = value;
}

template <typename T>
CpuKernel<T>::CpuKernel(int radius) : CpuImage<T>(2 * radius + 1, 2 * radius + 1)
{
}

static float gaussianPdf(float x, float mu, float sigma)
{
  return std::exp(-(x - mu) * (x - mu) / (2 * sigma * sigma));
}

template <typename T>
CpuKernel<T> CpuKernel<T>::geiger(int radius, float angle1, float angle2, int quadrant)
{
  CpuKernel<T> kernel(radius);
  float sum = 0;
  for (int y = -radius; y <= radius; ++y)
  {
    for (int x = -radius; x <= radius; ++x)
    {
      float to_axis1 = -x * std::sin(angle1) + y * std::cos(angle1);
      float to_axis2 = -x * std::sin(angle2) + y * std::cos(angle2);

      float value = 0;
      float pdf = gaussianPdf(std::sqrt(x * x + y * y), 0, static_cast<int>(radius / 2));
      if (to_axis1 <= -0.1 && to_axis2 <= -0.1 && quadrant == 0)
      {
        value = pdf;
        sum += pdf;
      }
      else if (to_axis1 >= 0.1 && to_axis2 >= 0.1 && quadrant == 2)
      {
        value = pdf;
        sum += pdf;
      }
      else if (to_axis1 <= -0.1 && to_axis2 >= 0.1 && quadrant == 1)
      {
        value = pdf;
        sum += pdf;
      }
      else if (to_axis1 >= 0.1 && to_axis2 <= -0.1 && quadrant == 3)
      {
        value = pdf;
        sum += pdf;
      }

      kernel.set(radius - x, radius - y, value);
    }
  }
  for (int y = -radius; y <= radius; ++y)
  {
    for (int x = -radius; x <= radius; ++x)
    {
      kernel.set(x + radius, y + radius, kernel.at(x + radius, y + radius) / sum);
    }
  }
  return kernel;
}

template class CpuImage<float>;
template class CpuKernel<float>;

}  // namespace camera_chessboard_detector
