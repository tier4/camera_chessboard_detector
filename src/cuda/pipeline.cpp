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

#include "cuda/pipeline.hpp"
#include "cuda/kernels.hpp"

#include <chrono>
#include <cmath>
#include <iostream>

#include <opencv2/core/core.hpp>

using camera_chessboard_detector::cuda::GpuImageF32;
using camera_chessboard_detector::cuda::GpuKernelF32;

#ifdef TIME_LOGGER
#define TimeNow() std::chrono::high_resolution_clock::now()
#define TimeDiff(start, end) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
#endif

namespace camera_chessboard_detector {

void CudaLikelihoodEstimator::configureKernels(int radius, float angle1,
                                                  float angle2) {
  for (int i = 0; i < 4; ++i) {
    const auto cpu_kernel = CpuKernelF32::geiger(radius, angle1, angle2, i);
    set_kernel(i, std::make_shared<GpuKernelF32>(cpu_kernel));
    if (use_separable_) {
      sep_convolvers_[i].set_kernel(cpu_kernel, separable_rank_);
    }
    convolved_images_[i] = std::make_shared<GpuImageF32>();
    diffs_[i] = std::make_shared<GpuImageF32>();
    diffs_[i + 4] = std::make_shared<GpuImageF32>();
  }
  a_ = std::make_shared<GpuImageF32>();
  b_ = std::make_shared<GpuImageF32>();
  mean_ = std::make_shared<GpuImageF32>();
}

GpuImagePtr CudaLikelihoodEstimator::buildMap(const GpuImagePtr &image) {
  // Reuse the per-estimator map across frames; resize() only reallocs when
  // the resolution changes. fill(0) reproduces the fresh-buffer state.
  if (!likelihood_map_) likelihood_map_ = std::make_shared<GpuImageF32>();
  likelihood_map_->resize(image->width(), image->height());
  likelihood_map_->fill(0);

  for (int i = 0; i < 4; ++i) {
    if (use_separable_) {
      sep_convolvers_[i].convolve(image, convolved_images_[i]);
    } else {
      convolvers_[i].convolve(image, kernels_[i], convolved_images_[i]);
    }
  }

  cuda::likelihood(likelihood_map_,
                   convolved_images_[0], convolved_images_[2],
                   convolved_images_[1], convolved_images_[3]);

  return likelihood_map_;
}

CudaDetector::CudaDetector(std::vector<int> &radius, std::vector<float> &angle1,
                           std::vector<float> &angle2,
                           bool verbose_logging,
                           bool use_separable,
                           int separable_rank) {
  impl_ = new CudaDetectorImpl(radius, angle1, angle2, verbose_logging,
                               use_separable, separable_rank);
}

CudaDetector::~CudaDetector() { delete impl_; }

CornerArray CudaDetector::detect(const float *data, int width, int height,
                                  DetectThresholds s, bool convert_to_gray) {
#ifdef TIME_LOGGER
  auto start = TimeNow();
 #endif
  if (!input_) input_ = std::make_shared<GpuImageF32>();
  input_->resize(width, height);
  if (convert_to_gray) {
    assert(false);
  } else {
    input_->upload(data, width * height);
  }
#ifdef TIME_LOGGER
  auto end = TimeNow();
  std::cout << "Upload time: " << TimeDiff(start, end)*1e-3 << " ms" << std::endl;
#endif
  return impl_->runDetect(input_, s.nms_margin, s.nms_radius,
                           s.nms_threshold, s.score_threshold);
}

CornerArray CudaDetector::detect(const GpuImagePtr &image, DetectThresholds s) {
  return impl_->runDetect(image, s.nms_margin, s.nms_radius,
                           s.nms_threshold, s.score_threshold);
}

CornerArray CudaDetector::detect(const unsigned char *data, int width,
                                 int height, int channels, DetectThresholds s) {
  return impl_->detectHost(data, width, height, channels, s);
}

CudaDetectorImpl::CudaDetectorImpl(std::vector<int> &radius,
                                   std::vector<float> &angle1,
                                   std::vector<float> &angle2,
                                   bool verbose_logging,
                                   bool use_separable,
                                   int separable_rank)
: verbose_logging_(verbose_logging) {
  assert(radius.size() == angle1.size());
  assert(radius.size() == angle2.size());
  estimators_.resize(radius.size());
  for (int i = 0; i < radius.size(); ++i) {
    // use_separable must be set BEFORE configureKernels so the
    // estimator knows whether to populate sep_convolvers_ too.
    estimators_[i].set_use_separable(use_separable, separable_rank);
    estimators_[i].configureKernels(radius[i], angle1[i], angle2[i]);
  }
}

CudaDetectorImpl::~CudaDetectorImpl() {}

CornerArray CudaDetectorImpl::detectHost(const unsigned char *data, int width,
                                         int height, int channels,
                                         DetectThresholds s) {
#ifdef TIME_LOGGER
  auto pstart = TimeNow();
#endif
  const GpuImagePtr &norm = preproc_.run(data, width, height, channels);
#ifdef TIME_LOGGER
  cudaDeviceSynchronize();
  std::cout << "Preprocess (GPU, incl upload) time: "
            << TimeDiff(pstart, TimeNow()) * 1e-3 << " ms" << std::endl;
#endif
  return runDetect(norm, s.nms_margin, s.nms_radius, s.nms_threshold,
                   s.score_threshold);
}

CornerArray CudaDetectorImpl::runDetect(const GpuImagePtr &devimg,
                                          int nms_margin, int nms_radius,
                                          float nms_threshold,
                                          float score_threshold) {
#ifdef TIME_LOGGER
  auto start = TimeNow();
#endif
  if (!map_) map_ = std::make_shared<GpuImageF32>();
  map_->resize(devimg->width(), devimg->height());
  map_->fill(0);
  for (int i = 0; i < estimators_.size(); ++i) {
    GpuImagePtr likelihood_map = estimators_[i].buildMap(devimg);
    cuda::max(likelihood_map, map_, map_);
  }
#ifdef TIME_LOGGER
  auto end = TimeNow();
  std::cout << "Map generation time: " << TimeDiff(start, end) * 1e-3 << " ms"
            << std::endl;
#endif

#ifdef TIME_LOGGER
  start = TimeNow();
#endif
  CornerArray corners;
  nms_.run(map_, nms_radius, nms_margin, nms_threshold, corners);
#ifdef TIME_LOGGER
  end = TimeNow();
  std::cout << "NMS time: " << TimeDiff(start, end) * 1e-3 << " ms" << std::endl;
#endif
  if (verbose_logging_) {
    std::cout << "NMS Detected " << corners.x.size() << " corners" << std::endl;
  }
#ifdef TIME_LOGGER
  start = TimeNow();
#endif
  // Compute the input gradients once and share them with both the refiner
  // and the weight map below (avoids a redundant Sobel pass per frame).
  if (!gx_) gx_ = std::make_shared<GpuImageF32>();
  if (!gy_) gy_ = std::make_shared<GpuImageF32>();
  cuda::gradients(gx_, gy_, devimg);
  refiner_.refine(devimg, gx_, gy_, corners, 10, cuda::CudaRefiner::REFINE_ALL);
#ifdef TIME_LOGGER
  end = TimeNow();
  std::cout << "Refinement time: " << TimeDiff(start, end) * 1e-3 << " ms"
            << std::endl;
#endif

#ifdef TIME_LOGGER
  start = TimeNow();
#endif
  if (!weights_) weights_ = std::make_shared<GpuImageF32>();
  weights_->resize(devimg->width(), devimg->height());
  weights_->fill(0);
  cuda::weight(weights_, gx_, gy_);
  pruneCorners(devimg, weights_, corners, score_threshold);
#ifdef TIME_LOGGER
  end = TimeNow();
  std::cout << "Pruning time: " << TimeDiff(start, end) * 1e-3 << " ms"
            << std::endl;
#endif
  return corners;
}

void CudaDetectorImpl::pruneCorners(const GpuImagePtr &devimg,
                                     const GpuImagePtr &weights,
                                     CornerArray &corners, float threshold) {
  for (int i = 0; i < corners.x.size(); ++i) {
    if (corners.edge1_cos[i] == 0 && corners.edge1_sin[i] == 0 &&
        corners.edge2_cos[i] == 0 && corners.edge2_sin[i] == 0) {
      corners.x[i] = 0;
      corners.y[i] = 0;
    }
  }

  score_gpu_.run(devimg, weights, corners);

  // Drop corners that did not clear the score threshold; they get
  // compacted out at the end.
  for (int i = 0; i < corners.x.size(); ++i) {
    if (corners.score[i] < threshold) {
      corners.x[i] = 0;
      corners.y[i] = 0;
    }
  }

  // 1 px dedup, retaining the higher-scoring edge fields.
  for (int i = 0; i < corners.x.size(); ++i) {
    if (corners.x[i] == 0 && corners.y[i] == 0) {
      continue;
    }
    for (int j = i + 1; j < corners.x.size(); ++j) {
      if (fabsf(corners.x[i] - corners.x[j]) < 1.0 &&
          fabsf(corners.y[i] - corners.y[j]) < 1.0) {
        corners.x[j] = 0;
        corners.y[j] = 0;
        if (corners.score[j] > corners.score[i]) {
          corners.score[i] = corners.score[j];
          corners.edge1_cos[i] = corners.edge1_cos[j];
          corners.edge1_sin[i] = corners.edge1_sin[j];
          corners.edge2_cos[i] = corners.edge2_cos[j];
          corners.edge2_sin[i] = corners.edge2_sin[j];
        }
      }
    }
  }

  // Remove zeroed Corners
  for (int i = 0; i < corners.x.size(); ++i) {
    if (corners.x[i] == 0 && corners.y[i] == 0) {
      corners.x.erase(corners.x.begin() + i);
      corners.y.erase(corners.y.begin() + i);
      corners.edge1_cos.erase(corners.edge1_cos.begin() + i);
      corners.edge1_sin.erase(corners.edge1_sin.begin() + i);
      corners.edge2_cos.erase(corners.edge2_cos.begin() + i);
      corners.edge2_sin.erase(corners.edge2_sin.begin() + i);
      corners.score.erase(corners.score.begin() + i);
      --i;
    }
  }

  std::vector<cv::Vec2f> corners_n1(corners.x.size());
  for (int i = 0; i < corners_n1.size(); i++) {
    if (corners.edge1_cos[i] + corners.edge1_sin[i] < 0.0) {
      corners.edge1_cos[i] = -corners.edge1_cos[i];
      corners.edge1_sin[i] = -corners.edge1_sin[i];
    }
    corners_n1[i] = cv::Vec2f(corners.edge1_cos[i], corners.edge1_sin[i]);
    float flipflag = corners_n1[i][0] * corners.edge2_cos[i] +
                     corners_n1[0][1] * corners.edge2_sin[i];
    if (flipflag > 0)
      flipflag = -1;
    else
      flipflag = 1;
    corners.edge2_sin[i] = flipflag * corners.edge2_sin[i];
    corners.edge2_cos[i] = flipflag * corners.edge2_cos[i];
  }

  if (verbose_logging_) {
    std::cout << "Pruned corners: " << corners.x.size() << std::endl;
  }
}

} // namespace camera_chessboard_detector
