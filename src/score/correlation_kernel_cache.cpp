#include "score/correlation_kernel_cache.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/core.hpp>

namespace camera_chessboard_detector {
namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kTwoPi = 2.0f * kPi;
constexpr float kBinStep =
    kTwoPi / static_cast<float>(CorrelationKernelCache::kNumAngleBins);

// Gaussian (mu=0) probability density used to weight the correlation
// kernels:
//
//   s = exp(-0.5 * dist^2 / sigma^2) / (sqrt(2*pi) * sigma)
//
// `sigma` is `radius / 2` (integer division: sigma = 2 for R=4, 4 for
// R=8, 6 for R=12).
float gaussianPdf(float dist, float sigma) {
  const float s = std::exp(-0.5f * dist * dist / (sigma * sigma));
  return s / (std::sqrt(2.0f * kPi) * sigma);
}

}  // namespace

int CorrelationKernelCache::angleToBin(float angle) noexcept {
  // Wrap into [-pi, +pi).
  float wrapped = angle - kTwoPi * std::floor((angle + kPi) / kTwoPi);
  // Shift to [0, 2*pi) then divide into N bins.
  const float shifted = wrapped + kPi;
  int bin = static_cast<int>(std::floor(shifted / kBinStep));
  // The wrap above can in principle land exactly on +pi due to float
  // rounding; clamp instead of risking an out-of-range bin.
  if (bin < 0) bin = 0;
  if (bin >= kNumAngleBins) bin = kNumAngleBins - 1;
  return bin;
}

float CorrelationKernelCache::binToAngle(int bin) noexcept {
  return -kPi + kBinStep * (static_cast<float>(bin) + 0.5f);
}

const cv::Mat & CorrelationKernelCache::disTable(int radius) {
  auto it = dis_tables_.find(radius);
  if (it != dis_tables_.end()) {
    return it->second;
  }
  const int size = 2 * radius + 1;
  cv::Mat dis(size, size, CV_32F);
  for (int v = 0; v < size; ++v) {
    float * row = dis.ptr<float>(v);
    const float dv = static_cast<float>(v - radius);
    for (int u = 0; u < size; ++u) {
      const float du = static_cast<float>(u - radius);
      row[u] = std::sqrt(du * du + dv * dv);
    }
  }
  return dis_tables_.emplace(radius, std::move(dis)).first->second;
}

CorrelationKernelSet CorrelationKernelCache::build(int angle1_bin, int angle2_bin,
                                       int radius) {
  const cv::Mat & dis = disTable(radius);

  const float angle1 = binToAngle(angle1_bin);
  const float angle2 = binToAngle(angle2_bin);
  const float sin1 = std::sin(angle1);
  const float cos1 = std::cos(angle1);
  const float sin2 = std::sin(angle2);
  const float cos2 = std::cos(angle2);
  const float sigma = static_cast<float>(radius / 2);

  const int size = 2 * radius + 1;
  CorrelationKernelSet set;
  set.a = cv::Mat::zeros(size, size, CV_32F);
  set.b = cv::Mat::zeros(size, size, CV_32F);
  set.c = cv::Mat::zeros(size, size, CV_32F);
  set.d = cv::Mat::zeros(size, size, CV_32F);

  for (int v = 0; v < size; ++v) {
    const float * dis_row = dis.ptr<float>(v);
    float * a_row = set.a.ptr<float>(v);
    float * b_row = set.b.ptr<float>(v);
    float * c_row = set.c.ptr<float>(v);
    float * d_row = set.d.ptr<float>(v);
    const float vec1 = static_cast<float>(v - radius);
    for (int u = 0; u < size; ++u) {
      const float vec0 = static_cast<float>(u - radius);
      const float side1 = vec0 * (-sin1) + vec1 * cos1;
      const float side2 = vec0 * (-sin2) + vec1 * cos2;
      const float w = gaussianPdf(dis_row[u], sigma);
      if (side1 <= -0.1f && side2 <= -0.1f) a_row[u] = w;
      if (side1 >= 0.1f && side2 >= 0.1f) b_row[u] = w;
      if (side1 <= -0.1f && side2 >= 0.1f) c_row[u] = w;
      if (side1 >= 0.1f && side2 <= -0.1f) d_row[u] = w;
    }
  }

  // Normalise each quadrant so the four kernels each sum to 1.
  set.a /= cv::sum(set.a)[0];
  set.b /= cv::sum(set.b)[0];
  set.c /= cv::sum(set.c)[0];
  set.d /= cv::sum(set.d)[0];
  return set;
}

const CorrelationKernelSet & CorrelationKernelCache::get(float angle1, float angle2,
                                             int radius) {
  Key key;
  key.angle1_bin = static_cast<std::int16_t>(angleToBin(angle1));
  key.angle2_bin = static_cast<std::int16_t>(angleToBin(angle2));
  key.radius = static_cast<std::int16_t>(radius);

  auto it = entries_.find(key);
  if (it != entries_.end()) {
    return it->second;
  }
  CorrelationKernelSet set = build(key.angle1_bin, key.angle2_bin, radius);
  return entries_.emplace(key, std::move(set)).first->second;
}

}  // namespace camera_chessboard_detector
