#pragma once

#include <vector>

#include <opencv2/opencv.hpp>

namespace camera_chessboard_detector {
namespace cpu {

// Raw corner candidates produced by the CPU front end before structure
// recovery: sub-pixel positions `p`, the two unit edge directions
// `v1` / `v2`, and the per-corner score.
struct CornerCandidates {
  std::vector<cv::Point2f> p;
  std::vector<cv::Vec2f> v1;
  std::vector<cv::Vec2f> v2;
  std::vector<float> score;
};

}  // namespace cpu
}  // namespace camera_chessboard_detector
