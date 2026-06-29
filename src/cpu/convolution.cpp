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

#include "cpu/convolution.hpp"

namespace camera_chessboard_detector
{
namespace cpu
{

using namespace cv;

Mat conv2(const Mat &img, const Mat &ikernel, ConvolutionType type)
{
  Mat dest;
  Mat kernel;
  flip(ikernel, kernel, -1);
  Mat source = img;
  if (CONVOLUTION_FULL == type)
  {
    source = Mat();
    const int additionalRows = kernel.rows - 1;
    const int additionalCols = kernel.cols - 1;
    copyMakeBorder(
      img, source, (additionalRows + 1) / 2, additionalRows / 2, (additionalCols + 1) / 2,
      additionalCols / 2, BORDER_CONSTANT, Scalar(0)
    );
  }
  Point anchor(kernel.cols - kernel.cols / 2 - 1, kernel.rows - kernel.rows / 2 - 1);
  filter2D(source, dest, img.depth(), kernel, anchor, 0, BORDER_CONSTANT);

  if (CONVOLUTION_VALID == type)
  {
    dest = dest.colRange((kernel.cols - 1) / 2, dest.cols - kernel.cols / 2)
             .rowRange((kernel.rows - 1) / 2, dest.rows - kernel.rows / 2);
  }
  return dest;
}

}  // namespace cpu
}  // namespace camera_chessboard_detector
