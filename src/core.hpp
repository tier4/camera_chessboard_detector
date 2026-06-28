#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include <opencv2/opencv.hpp>

namespace camera_chessboard_detector {

constexpr float PI = 3.14159265358979323846f;

// Structure-of-arrays corner set passed between detection stages. Each
// corner carries its sub-pixel position and the two unit (cos, sin) edge
// directions of the board axes meeting at it; `v` is a reserved scratch
// slot kept for layout stability.
struct CornerArray {
  std::vector<float> x;
  std::vector<float> y;
  std::vector<float> v;
  std::vector<float> edge1_sin;
  std::vector<float> edge1_cos;
  std::vector<float> edge2_sin;
  std::vector<float> edge2_cos;
  std::vector<float> score;
};

// Non-maximum-suppression and score thresholds for the GPU pipeline.
struct DetectThresholds {
  int nms_margin;
  int nms_radius;
  float nms_threshold;
  float score_threshold;
};

// Host-side image buffer. Wrapped in a shared pointer at the call sites to
// avoid copies. Used to stage the Geiger correlation kernels before they
// are uploaded to the GPU.
template <typename T>
class CpuImage {
public:
  CpuImage(const cv::Mat &mat);
  CpuImage(int width, int height);
  CpuImage() : initialized_(false), width_(0), height_(0) {}

  bool resize(int width, int height);
  void fill(T value);
  void set(int x, int y, T value);

  std::size_t size() const { return data_.size(); }
  bool valid() const { return initialized_; }
  int width() const { return width_; }
  int height() const { return height_; }
  T at(int x, int y) const { return data_[y * width_ + x]; }
  const T *data() const { return data_.data(); }

  cv::Mat to_mat_f32() const;
  cv::Mat to_mat_u8() const;
  cv::Mat to_mat() const;

protected:
  bool initialized_;
  int width_;
  int height_;
  std::vector<T> data_;
};

// Square (2*radius+1) correlation kernel. `geiger` builds one of the four
// quadrant templates for the given pair of edge angles.
template <typename T>
class CpuKernel : public CpuImage<T> {
public:
  CpuKernel(int radius);

  static CpuKernel<T> geiger(int radius, float angle1, float angle2,
                             int quadrant);
};

typedef CpuImage<float> CpuImageF32;
typedef CpuKernel<float> CpuKernelF32;

// Board structure recovery. Grows a 3x3 seed neighbourhood into full
// chessboards over a CornerArray, minimising a Geiger/ROCHADE-style
// energy. CPU-only C++, shared by both front ends.
namespace cpu {

class BoardBuilder {
public:
  cv::Mat seedBoard(CornerArray &corners, int idx);

  void setVerboseLogging(bool enabled) { verbose_logging_ = enabled; }

  void assembleBoards(CornerArray &corners,
                              std::vector<cv::Mat> &chessboards,
                              float lambda = 0.5);

  int nearestNeighborAlong(int idx, cv::Vec2f v, cv::Mat chessboard,
                          CornerArray &corners, int &neighbor_idx,
                          float &min_dist);

  float boardEnergy(cv::Mat chessboard, CornerArray &corners);

  void predictNextCorners(std::vector<cv::Vec2f> &p1, std::vector<cv::Vec2f> &p2,
                      std::vector<cv::Vec2f> &p3, std::vector<cv::Vec2f> &pred);

  cv::Mat growBoard(cv::Mat chessboard, CornerArray &corners,
                         int border_type);

  void matchClosestCandidates(std::vector<cv::Vec2f> &cand,
                            std::vector<cv::Vec2f> &pred,
                            std::vector<int> &idx);

  cv::Mat chessboard;
  float lambda_;
  bool verbose_logging_{false};
};

}  // namespace cpu

}  // namespace camera_chessboard_detector
