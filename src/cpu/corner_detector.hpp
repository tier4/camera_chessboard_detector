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

#include "cpu/corner_candidates.hpp"
#include "score/correlation_kernel_cache.hpp"

#include <opencv2/opencv.hpp>

#include <utility>
#include <vector>

namespace camera_chessboard_detector
{
namespace cpu
{

// Scalar type and matching OpenCV depth used throughout the CPU front end.
// Switch both lines together to build the detector in double precision.
using RealT = float;
constexpr int kMatDepth = CV_32F;

// CPU corner front end: Geiger likelihood map, non-maximum suppression,
// sub-pixel refinement and per-corner scoring.
class CpuCornerDetector
{
public:
  enum RefinementOption : bool { NO_REFINE = false, DO_REFINE = true };

  CpuCornerDetector();
  ~CpuCornerDetector();
  explicit CpuCornerDetector(cv::Mat img);

  void detect(
    cv::Mat &src, CornerCandidates &mcorners, RealT score_threshold,
    RefinementOption refinement_option = DO_REFINE
  );

private:
  void buildLikelihoodMap(cv::Mat &src, cv::Mat &dst);

  RealT gaussian1d(RealT dist, RealT mu, RealT sigma);

  void elementwiseMin(cv::Mat src1, cv::Mat src2, cv::Mat &dst);
  void elementwiseMax(cv::Mat src1, cv::Mat src2, cv::Mat &dst);

  void computeGradientOrientation(
    cv::Mat img, cv::Mat &img_du, cv::Mat &img_dv, cv::Mat &img_angle, cv::Mat &img_weight
  );

  void estimateEdgeOrientations(cv::Mat img_angle, cv::Mat img_weight, int index);

  void findHistogramModes(
    std::vector<RealT> hist, std::vector<RealT> &hist_smoothed,
    std::vector<std::pair<RealT, int>> &modes, RealT sigma
  );

  void scoreAllCorners(
    cv::Mat img, cv::Mat img_angle, cv::Mat img_weight, std::vector<cv::Point2f> &corners,
    std::vector<int> radius, std::vector<float> &score
  );

  void correlationScore(
    cv::Mat img, cv::Mat img_weight, std::vector<cv::Point2f> corners_edge, float &score
  );

  void refineCornersSubpixel(
    std::vector<cv::Point2f> &corners, cv::Mat img_du, cv::Mat img_dv, cv::Mat img_angle,
    cv::Mat img_weight, float radius
  );

  void buildQuadrantKernels(
    float angle1, float angle2, int kernel_size, cv::Mat &kernel_a, cv::Mat &kernel_b,
    cv::Mat &kernel_c, cv::Mat &kernel_d
  );

  void nonMaxSuppress(
    cv::Mat &input_corners, std::vector<cv::Point2f> &output_corners, int patch_size,
    RealT threshold, int margin
  );

  float vecNorm(cv::Point2f o);

  std::vector<cv::Point2f> template_angles_;
  std::vector<int> radius_;
  std::vector<cv::Point2f> corner_points_;
  std::vector<std::vector<RealT>> edge_dirs1_;
  std::vector<std::vector<RealT>> edge_dirs2_;
  CorrelationKernelCache score_kernel_cache_;
};

}  // namespace cpu
}  // namespace camera_chessboard_detector
