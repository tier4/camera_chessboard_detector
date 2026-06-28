#ifndef CCD__CHESSBOARD_DETECTOR_HPP
#define CCD__CHESSBOARD_DETECTOR_HPP

#include <cstdint>
#include <memory>
#include <vector>

#include <opencv2/core.hpp>

#include "camera_chessboard_detector/types.hpp"

// Public detector API. The growth-based (Geiger-style) corner detection and
// board structure-recovery algorithm lives in src/; this header declares only
// the namespace, configuration, and result types.
namespace camera_chessboard_detector
{

enum class ChessboardAccelerationMode
{
  CPU,
  CUDA,
  // Opt-in separable Geiger map-gen path: each non-separable 2D sub-kernel
  // is decomposed by SVD on the host into `separable_rank` rank-1 outer
  // products and applied as paired 1D horizontal + vertical convolutions.
  // `separable_rank <= 0` means "use full rank", which is mathematically
  // equivalent to the dense path up to floating-point precision.
  CUDA_SEPARABLE,
};

struct ChessboardDetectorConfig
{
  float score_threshold{0.01f};
  int cuda_nms_margin{5};
  int cuda_nms_radius{3};
  float cuda_nms_threshold{0.025f};
  float structure_lambda{0.6f};
  bool refine{true};
  bool verbose_logs{false};
  ChessboardAccelerationMode acceleration{ChessboardAccelerationMode::CUDA};
  // Used only when `acceleration == CUDA_SEPARABLE`. <= 0 selects full
  // rank (== kernel side length), which is exact w.r.t. the dense
  // path up to FP precision. Smaller values approximate the kernel
  // and run proportionally faster but may lose detection accuracy;
  // the gating signal is the synthetic-board accuracy test in
  // regression_test.cpp.
  int separable_rank{-1};
  std::uint32_t target_id{0};
};

struct ChessboardDetection
{
  Header header{};
  Chessboard2dList boards{};
  bool has_boards{false};

  void clear();
};

class ChessboardDetector
{
public:
  ChessboardDetector(const ChessboardModel &model,
                     const ChessboardDetectorConfig &config = ChessboardDetectorConfig());

  void setBoardModel(const ChessboardModel &model);
  void setConfig(const ChessboardDetectorConfig &config);

  bool detectChessboards(const cv::Mat &image, ChessboardDetection &detection_out);
  bool detectChessboard(const cv::Mat &image, Chessboard2d &board_out);

  void drawChessboardCorners(cv::Mat &image,
                             const Chessboard2d &board,
                             bool draw_index = false) const;

private:
  bool detectChessboardsCpu(const cv::Mat &image,
                            Chessboard2dList &boards_out) const;
  bool detectChessboardsCuda(const cv::Mat &image,
                             Chessboard2dList &boards_out) const;

  ChessboardModel board_model_{};
  ChessboardDetectorConfig config_{};

  // Cached GPU corner detector (a camera_chessboard_detector::CudaDetector), kept as an
  // opaque pointer so this public header stays CUDA-free. It is built lazily on
  // the first CUDA detect and reused across frames -- rebuilding it per call
  // cost kernel setup plus a device allocation every frame. The estimators are
  // image-size independent (their scratch buffers resize on demand), so one
  // instance serves every frame and every ROI. setConfig() resets it so new
  // kernel settings take effect. Reuse is safe only for serial calls on a single
  // detector; each detection stage owns its own instance, so the fast and
  // precise stages never share one. Unused on CPU-only builds. Mutable because
  // the CUDA detect helper is const.
  mutable std::shared_ptr<void> cuda_pipeline_{};
};

}  // namespace camera_chessboard_detector

#endif  // CCD__CHESSBOARD_DETECTOR_HPP
