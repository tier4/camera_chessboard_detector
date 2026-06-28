#pragma once

#include "core.hpp"
#include "cuda/kernels.hpp"
#include "cuda/nms.hpp"
#include "cuda/score.hpp"
#include "cuda/separable_conv.hpp"
#include "cuda/gpu_buffers.hpp"
#include "cuda/preprocess.hpp"

#include <unordered_map>

namespace camera_chessboard_detector {

using cuda::GpuImagePtr;
using cuda::GpuKernelPtr;

class CudaDetectorImpl;

/**
 * @brief Generates a likelihood map for a given image
 * with four Geiger kernels.
 * Implemented using CUDA in such a way that it's possible to
 * easily run multiple instances in parallel with cudaStreams
 * TODO: implement streams
 */
class CudaLikelihoodEstimator {
  // enum {
  //   NUM_KERNELS = 4,
  //   NUM_RADIUS = 3,
  //   NUM_ANGLES = 2,
  //   TOTAL_KERNELS = NUM_KERNELS * NUM_RADIUS * NUM_ANGLES
  // };
  enum {
    NUM_KERNELS = 4,
  };

public:
  CudaLikelihoodEstimator() {}
  ~CudaLikelihoodEstimator() {}

  inline void set_kernel(int index, GpuKernelPtr &&kernel) {
    kernels_[index] = std::move(kernel);
  }

  // Opt-in: configure the estimator to dispatch map-gen via
  // the SVD-separable path with the requested rank (<= 0 = full rank
  // = kernel side length). Call before configureKernels(). When
  // left false, behaviour is bit-stable with the historical dense
  // dense-convolution path.
  inline void set_use_separable(bool enable, int rank) {
    use_separable_ = enable;
    separable_rank_ = rank;
  }

  // void configureKernels();
  void configureKernels(int radius, float angle1, float angle2);

  /**
   * @brief Generates a likelihood map for a given image.
   * @param image The image to generate a likelihood map for.
   * @return The likelihood map.
   */
  GpuImagePtr buildMap(const GpuImagePtr &image);

  // void set_normalized_image(const GpuImagePtr &image) {
  //   normalized_image_ = image;
  // }

  void cache(const std::string &name, const GpuImagePtr &image) {
    output_maps_[std::hash<std::string>{}(name)] = image;
  }

  GpuImagePtr get_cached(const std::string &name) {
    return output_maps_[std::hash<std::string>{}(name)];
  }

private:
  // void convolve(const ImageF &image, const KernelF &kernel, ImageF &output);

  GpuKernelPtr kernels_[NUM_KERNELS];
  // GpuImagePtr normalized_image_;
  // GpuImagePtr output_map_;
  cuda::CudaConvolver convolvers_[NUM_KERNELS];
  // Opt-in separable convolvers. Constructed eagerly so
  // buildMap can dispatch without checking pointer state, but
  // they hold no device memory until configureKernels runs in
  // separable mode.
  cuda::CudaSeparableConvolver sep_convolvers_[NUM_KERNELS];
  bool use_separable_{false};
  int separable_rank_{-1};
  GpuImagePtr convolved_images_[NUM_KERNELS];
  GpuImagePtr mean_;
  GpuImagePtr diffs_[NUM_KERNELS * 2];
  GpuImagePtr a_;
  GpuImagePtr b_;
  // Reused across frames; resize() is a no-op at constant resolution.
  GpuImagePtr likelihood_map_;

  std::unordered_map<int, GpuImagePtr> output_maps_;
};

class CudaDetector {
public:
  CudaDetector(std::vector<int> &radius, std::vector<float> &angle1,
               std::vector<float> &angle2,
               bool verbose_logging = false,
               bool use_separable = false,
               int separable_rank = -1);
  ~CudaDetector();

  // #ifdef OPENCV_ENABLED
  //   void detect(const cv::Mat &image);
  // #endif
  CornerArray detect(const float *data, int width, int height,
                      DetectThresholds s, bool convert_to_gray = true);
  CornerArray detect(const GpuImagePtr &image, DetectThresholds s);
  // Uint8 BGR/gray host image -> GPU preprocess (BGR2GRAY + blur + normalize)
  // -> detect. `channels` is 1 or 3.
  CornerArray detect(const unsigned char *data, int width, int height,
                     int channels, DetectThresholds s);

private:
  CudaDetectorImpl *impl_;
  // Reused device input buffer (resized per frame, no realloc at constant size).
  GpuImagePtr input_;
};

class CudaDetectorImpl {
public:
  CudaDetectorImpl(std::vector<int> &radius, std::vector<float> &angle1,
                   std::vector<float> &angle2,
                   bool verbose_logging = false,
                   bool use_separable = false,
                   int separable_rank = -1);
  ~CudaDetectorImpl();

  CornerArray runDetect(const GpuImagePtr &devimg,
                          int nms_margin, int nms_radius, float nms_threshold,
                          float score_threshold);
  CornerArray detectHost(const unsigned char *data, int width, int height,
                         int channels, DetectThresholds s);

private:
  void pruneCorners(const GpuImagePtr &devimg, const GpuImagePtr &weights,
                     CornerArray &corners, float threshold);

  std::vector<CudaLikelihoodEstimator> estimators_;
  cuda::CudaPreprocessor preproc_;
  cuda::CudaRefiner refiner_;
  cuda::CudaNonMaxSuppression nms_;
  cuda::CudaScoreCorners score_gpu_;
  bool verbose_logging_{false};
  // Reused across frames (resize() is a no-op at constant resolution).
  GpuImagePtr map_;
  GpuImagePtr weights_;
  // Input gradients, computed once per frame and shared by the refiner and
  // the weight map (previously each recomputed Sobel independently).
  GpuImagePtr gx_;
  GpuImagePtr gy_;
};

} // namespace camera_chessboard_detector
