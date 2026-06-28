#pragma once

#include <opencv2/opencv.hpp>

namespace camera_chessboard_detector {
namespace cpu {

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
