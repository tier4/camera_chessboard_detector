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

#include <vector>

namespace camera_chessboard_detector
{
namespace cpu
{

// Raw corner candidates produced by the CPU front end before structure
// recovery: sub-pixel positions `p`, the two unit edge directions
// `v1` / `v2`, and the per-corner score.
struct CornerCandidates
{
  std::vector<cv::Point2f> p;
  std::vector<cv::Vec2f> v1;
  std::vector<cv::Vec2f> v2;
  std::vector<float> score;
};

}  // namespace cpu
}  // namespace camera_chessboard_detector
