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
using real_t = float;
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
    cv::Mat &src, CornerCandidates &mcorners, real_t scoreThreshold,
    RefinementOption refinementOption = DO_REFINE
  );

private:
  void buildLikelihoodMap(cv::Mat &src, cv::Mat &dst);

  real_t gaussian1d(real_t dist, real_t mu, real_t sigma);

  void elementwiseMin(cv::Mat src1, cv::Mat src2, cv::Mat &dst);
  void elementwiseMax(cv::Mat src1, cv::Mat src2, cv::Mat &dst);

  void computeGradientOrientation(
    cv::Mat img, cv::Mat &imgDu, cv::Mat &imgDv, cv::Mat &imgAngle, cv::Mat &imgWeight
  );

  void estimateEdgeOrientations(cv::Mat imgAngle, cv::Mat imgWeight, int index);

  void findHistogramModes(
    std::vector<real_t> hist, std::vector<real_t> &hist_smoothed,
    std::vector<std::pair<real_t, int>> &modes, real_t sigma
  );

  void scoreAllCorners(
    cv::Mat img, cv::Mat imgAngle, cv::Mat imgWeight, std::vector<cv::Point2f> &corners,
    std::vector<int> radius, std::vector<float> &score
  );

  void correlationScore(
    cv::Mat img, cv::Mat imgWeight, std::vector<cv::Point2f> cornersEdge, float &score
  );

  void refineCornersSubpixel(
    std::vector<cv::Point2f> &corners, cv::Mat imgDu, cv::Mat imgDv, cv::Mat imgAngle,
    cv::Mat imgWeight, float radius
  );

  void buildQuadrantKernels(
    float angle1, float angle2, int kernelSize, cv::Mat &kernelA, cv::Mat &kernelB,
    cv::Mat &kernelC, cv::Mat &kernelD
  );

  void nonMaxSuppress(
    cv::Mat &inputCorners, std::vector<cv::Point2f> &outputCorners, int patchSize, real_t threshold,
    int margin
  );

  float vecNorm(cv::Point2f o);

  std::vector<cv::Point2f> template_angles_;
  std::vector<int> radius;
  std::vector<cv::Point2f> corner_points_;
  std::vector<std::vector<real_t>> edge_dirs1_;
  std::vector<std::vector<real_t>> edge_dirs2_;
  CorrelationKernelCache score_kernel_cache_;
};

}  // namespace cpu
}  // namespace camera_chessboard_detector
