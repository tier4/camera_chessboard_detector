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

#include <opencv2/opencv.hpp>

namespace camera_chessboard_detector
{
namespace cpu
{

enum ConvolutionType {
  // Full convolution, including the border region.
  CONVOLUTION_FULL,
  // Same size as the input image.
  CONVOLUTION_SAME,
  // Only the region unaffected by the border.
  CONVOLUTION_VALID
};

// 2D correlation matching MATLAB's conv2() for the requested output mode.
cv::Mat conv2(const cv::Mat &img, const cv::Mat &kernel, ConvolutionType type);

}  // namespace cpu
}  // namespace camera_chessboard_detector
