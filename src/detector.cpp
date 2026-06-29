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

#include "camera_chessboard_detector/detector.hpp"

#include "core.hpp"  // CornerArray + board structure recovery
#include "cpu/corner_detector.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#ifdef CUDA_ENABLED
#include "cuda/pipeline.hpp"

#include <cuda_runtime.h>
#endif

namespace camera_chessboard_detector
{
namespace
{

// Near-duplicate corner merge distance.
constexpr float kDedupPx = 1.0f;  // merge corners closer than this (px)

// Convert the CPU front end's AoS corner candidates (p / v1 / v2 / score)
// to the SoA CornerArray consumed by structure recovery. v1 / v2 are the
// two (cos, sin) edge directions.
CornerArray toCornerArray(const cpu::CornerCandidates &src)
{
  camera_chessboard_detector::CornerArray s;
  const std::size_t n = src.p.size();
  s.x.resize(n);
  s.y.resize(n);
  s.v.assign(n, 0.0f);
  s.edge1_cos.resize(n);
  s.edge1_sin.resize(n);
  s.edge2_cos.resize(n);
  s.edge2_sin.resize(n);
  s.score.resize(n);
  for (std::size_t i = 0; i < n; ++i)
  {
    s.x[i] = src.p[i].x;
    s.y[i] = src.p[i].y;
    s.edge1_cos[i] = src.v1[i][0];
    s.edge1_sin[i] = src.v1[i][1];
    s.edge2_cos[i] = src.v2[i][0];
    s.edge2_sin[i] = src.v2[i][1];
    s.score[i] = src.score.empty() ? 0.0f : src.score[i];
  }
  return s;
}

// The CPU front end does not de-duplicate, so it can emit several
// near-duplicate corners at a single location, which structure recovery
// cannot assemble. Mirror the GPU path: drop near-duplicates within
// `min_sep_px`, keeping the higher-scoring corner of each pair. This is a
// no-op on well-separated real-board corners.
CornerArray dedupNearbyCorners(const CornerArray &s, float min_sep_px)
{
  const std::size_t m = s.x.size();
  std::vector<char> drop(m, 0);
  for (std::size_t i = 0; i < m; ++i)
  {
    if (drop[i])
    {
      continue;
    }
    for (std::size_t j = i + 1; j < m; ++j)
    {
      if (drop[j])
      {
        continue;
      }
      if (std::abs(s.x[i] - s.x[j]) < min_sep_px && std::abs(s.y[i] - s.y[j]) < min_sep_px)
      {
        if (s.score[j] > s.score[i])
        {
          drop[i] = 1;
          break;
        }
        drop[j] = 1;
      }
    }
  }
  camera_chessboard_detector::CornerArray d;
  for (std::size_t i = 0; i < m; ++i)
  {
    if (drop[i])
    {
      continue;
    }
    d.x.push_back(s.x[i]);
    d.y.push_back(s.y[i]);
    d.v.push_back(0.0f);
    d.edge1_cos.push_back(s.edge1_cos[i]);
    d.edge1_sin.push_back(s.edge1_sin[i]);
    d.edge2_cos.push_back(s.edge2_cos[i]);
    d.edge2_sin.push_back(s.edge2_sin[i]);
    d.score.push_back(s.score[i]);
  }
  return d;
}

// Verbose-only diagnostic: prints the corner count, the near-duplicate
// count, and a few sample corners with their edge directions.
void dumpCornerDiag(const char *tag, const CornerArray &s)
{
  const std::size_t n = s.x.size();
  std::size_t dup_pts = 0;
  for (std::size_t i = 0; i < n; ++i)
  {
    for (std::size_t j = i + 1; j < n; ++j)
    {
      if (std::abs(s.x[i] - s.x[j]) < 1.0f && std::abs(s.y[i] - s.y[j]) < 1.0f)
      {
        ++dup_pts;
        break;
      }
    }
  }
  std::cerr << "[ccd] " << tag << " N=" << n << " near-dup-pts(<1px)=" << dup_pts << std::endl;
  for (std::size_t i = 0; i < n && i < 6; ++i)
  {
    std::cerr << "  c" << i << " p=(" << s.x[i] << "," << s.y[i] << ")" << " e1=(" << s.edge1_cos[i]
              << "," << s.edge1_sin[i] << ")|n|=" << std::hypot(s.edge1_cos[i], s.edge1_sin[i])
              << " e2=(" << s.edge2_cos[i] << "," << s.edge2_sin[i]
              << ")|n|=" << std::hypot(s.edge2_cos[i], s.edge2_sin[i]) << std::endl;
  }
}

bool modelHasSize(const ChessboardModel &model) { return model.rows > 0 && model.cols > 0; }

bool sizeMatchesModel(const cv::Mat &checkerboard, const ChessboardModel &model)
{
  if (!modelHasSize(model))
  {
    return true;
  }
  const int rows = checkerboard.rows;
  const int cols = checkerboard.cols;
  if (rows <= 0 || cols <= 0)
  {
    return false;
  }
  const int model_rows = static_cast<int>(model.rows);
  const int model_cols = static_cast<int>(model.cols);
  return (rows == model_rows && cols == model_cols) || (rows == model_cols && cols == model_rows);
}

bool reorderDetectedCorners(std::vector<cv::Point2f> &corners, const ChessboardModel &model)
{
  if (!modelHasSize(model))
  {
    return false;
  }

  const std::size_t rows = static_cast<std::size_t>(model.rows);
  const std::size_t cols = static_cast<std::size_t>(model.cols);
  if (rows == 0 || cols == 0 || rows * cols != corners.size())
  {
    return false;
  }

  std::vector<cv::Point2f> row_sorted = corners;
  for (std::size_t row = 0; row < rows; ++row)
  {
    const std::size_t base = row * cols;
    const auto begin_it = row_sorted.begin() + static_cast<std::ptrdiff_t>(base);
    const auto end_it = begin_it + static_cast<std::ptrdiff_t>(cols);
    std::sort(begin_it, end_it, [](const cv::Point2f &lhs, const cv::Point2f &rhs) {
      return lhs.x < rhs.x;
    });
  }

  std::vector<std::size_t> row_indices(rows);
  for (std::size_t row = 0; row < rows; ++row)
  {
    row_indices[row] = row;
  }
  std::sort(
    row_indices.begin(), row_indices.end(),
    [&row_sorted, cols](std::size_t lhs, std::size_t rhs) {
      return row_sorted[lhs * cols].y < row_sorted[rhs * cols].y;
    }
  );

  std::vector<cv::Point2f> reordered;
  reordered.reserve(corners.size());
  for (const std::size_t row_index : row_indices)
  {
    const std::size_t base = row_index * cols;
    const auto begin_it = row_sorted.begin() + static_cast<std::ptrdiff_t>(base);
    const auto end_it = begin_it + static_cast<std::ptrdiff_t>(cols);
    reordered.insert(reordered.end(), begin_it, end_it);
  }

  corners.swap(reordered);
  return true;
}

template <typename CornerAccessor>
bool buildOrderedCorners(
  const cv::Mat &checkerboard, std::size_t corner_count, const CornerAccessor &corner_at,
  std::vector<cv::Point2f> &ordered
)
{
  if (checkerboard.empty())
  {
    return false;
  }
  cv::Mat board = checkerboard.t();
  const int rows = board.rows;
  const int cols = board.cols;
  if (rows <= 0 || cols <= 0)
  {
    return false;
  }
  ordered.clear();
  ordered.reserve(static_cast<std::size_t>(rows * cols));

  auto append_corner = [&](int index) -> bool {
    if (index < 0 || static_cast<std::size_t>(index) >= corner_count)
    {
      return false;
    }
    ordered.push_back(corner_at(index));
    return true;
  };

  if (cols < rows)
  {
    for (int col = 0; col < cols; ++col)
    {
      for (int row = 0; row < rows; ++row)
      {
        if (!append_corner(board.at<int>(row, col)))
        {
          return false;
        }
      }
    }
  }
  else
  {
    for (int row = 0; row < rows; ++row)
    {
      for (int col = 0; col < cols; ++col)
      {
        if (!append_corner(board.at<int>(row, col)))
        {
          return false;
        }
      }
    }
  }

  return ordered.size() == static_cast<std::size_t>(rows * cols);
}

#ifdef CUDA_ENABLED
bool isCudaAvailable()
{
  int device_count = 0;
  const cudaError_t status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess)
  {
    return false;
  }
  return device_count > 0;
}
#endif

}  // namespace

void ChessboardDetection::clear()
{
  header = Header{};
  boards.clear();
  has_boards = false;
}

ChessboardDetector::ChessboardDetector(
  const ChessboardModel &model, const ChessboardDetectorConfig &config
)
: board_model_(model), config_(config)
{
}

void ChessboardDetector::setBoardModel(const ChessboardModel &model) { board_model_ = model; }

void ChessboardDetector::setConfig(const ChessboardDetectorConfig &config)
{
  config_ = config;
  // Drop the cached GPU detector so the next detect rebuilds it with the new
  // kernel settings (separable rank, verbosity). shared_ptr<void>::reset is
  // CUDA-free, so this also compiles on CPU-only builds.
  cuda_pipeline_.reset();
}

bool ChessboardDetector::detectChessboards(const cv::Mat &image, ChessboardDetection &detection_out)
{
  detection_out.clear();

  if (image.empty())
  {
    return false;
  }

  Chessboard2dList boards;

  bool has_boards = false;

  if (config_.acceleration == ChessboardAccelerationMode::CUDA ||
      config_.acceleration == ChessboardAccelerationMode::CUDA_SEPARABLE)
  {
#ifdef CUDA_ENABLED
    if (isCudaAvailable())
    {
      has_boards = detectChessboardsCuda(image, boards);
    }
    else
    {
      std::cerr << "[camera_chessboard_detector] CUDA acceleration requested but no CUDA device "
                   "was found; returning no detection."
                << std::endl;
    }
#else
    std::cerr << "[camera_chessboard_detector] CUDA acceleration requested but this build has no "
                 "CUDA support; returning no detection."
              << std::endl;
#endif
  }
  else
  {
    has_boards = detectChessboardsCpu(image, boards);
  }

  if (!has_boards)
  {
    return false;
  }

  detection_out.boards.reserve(boards.size());
  for (auto &board : boards)
  {
    board.header = detection_out.header;
    board.target_id = config_.target_id;
    board.model = board_model_;
    detection_out.boards.push_back(std::move(board));
  }

  detection_out.has_boards = !detection_out.boards.empty();
  return detection_out.has_boards;
}

bool ChessboardDetector::detectChessboard(const cv::Mat &image, Chessboard2d &board_out)
{
  ChessboardDetection detection;
  if (!detectChessboards(image, detection))
  {
    return false;
  }
  if (detection.boards.empty())
  {
    return false;
  }
  board_out = std::move(detection.boards.front());
  return true;
}

void ChessboardDetector::drawChessboardCorners(
  cv::Mat &image, const Chessboard2d &board, bool draw_index
) const
{
  std::vector<cv::Point2f> corners;
  toCvPoints(board, corners);
  if (corners.empty())
  {
    return;
  }

  const bool has_model = modelHasSize(board.model);
  const bool size_matches =
    has_model && corners.size() == static_cast<std::size_t>(board.model.rows * board.model.cols);

  if (size_matches)
  {
    cv::drawChessboardCorners(
      image, cv::Size(static_cast<int>(board.model.cols), static_cast<int>(board.model.rows)),
      corners, true
    );
  }
  else
  {
    for (const auto &corner : corners)
    {
      cv::circle(image, corner, 3, cv::Scalar(0, 255, 0), -1);
    }
  }

  if (draw_index)
  {
    for (std::size_t i = 0; i < corners.size(); ++i)
    {
      const std::string label = std::to_string(i);
      cv::putText(
        image, label, corners[i], cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 0, 0), 1,
        cv::LINE_AA
      );
    }
  }
}

bool ChessboardDetector::detectChessboardsCpu(const cv::Mat &image, Chessboard2dList &boards_out)
  const
{
  boards_out.clear();
  cv::Mat image_view = image;

#ifdef TIME_LOGGER
  auto _pt = std::chrono::high_resolution_clock::now();
  auto _lap = [&_pt](const char *w) {
    const auto _n = std::chrono::high_resolution_clock::now();
    std::cout << "[prof][cpu] " << w << ": "
              << std::chrono::duration_cast<std::chrono::microseconds>(_n - _pt).count() * 1e-3
              << " ms" << std::endl;
    _pt = _n;
  };
#endif

  cpu::CpuCornerDetector corner_detector(image_view);
  const auto refine_option = config_.refine ? cpu::CpuCornerDetector::RefinementOption::DO_REFINE
                                            : cpu::CpuCornerDetector::RefinementOption::NO_REFINE;

  cpu::CornerCandidates corners;
  corner_detector.detect(image_view, corners, config_.score_threshold, refine_option);
#ifdef TIME_LOGGER
  _lap("corner detect (conv+nms+refine+score, incl host preprocess)");
#endif
  if (config_.verbose_logs)
  {
    std::cerr << "[ccd][cpu] corner detect -> " << corners.p.size() << " corners" << std::endl;
  }
  if (corners.p.empty())
  {
    return false;
  }

  // De-duplicate near-coincident CPU corners before structure recovery
  // (the CPU front end has no built-in dedup); no-op on real boards.
  auto soa = dedupNearbyCorners(toCornerArray(corners), kDedupPx);
  if (config_.verbose_logs)
  {
    dumpCornerDiag("cpu ", soa);
  }
  cpu::BoardBuilder chessboard;
  chessboard.setVerboseLogging(config_.verbose_logs);
  std::vector<cv::Mat> checkerboards;
  chessboard.assembleBoards(soa, checkerboards, config_.structure_lambda);
#ifdef TIME_LOGGER
  _lap("structure recovery (CPU)");
#endif
  if (config_.verbose_logs)
  {
    std::cerr << "[ccd][cpu] structure recovery -> " << checkerboards.size() << " board(s)"
              << std::endl;
  }
  if (checkerboards.empty())
  {
    return false;
  }

  for (const auto &checkerboard : checkerboards)
  {
    if (!sizeMatchesModel(checkerboard, board_model_))
    {
      continue;
    }

    std::vector<cv::Point2f> ordered;
    const bool ok = buildOrderedCorners(
      checkerboard, soa.x.size(),
      [&soa](int index) {
        return cv::Point2f(
          soa.x[static_cast<std::size_t>(index)], soa.y[static_cast<std::size_t>(index)]
        );
      },
      ordered
    );
    if (!ok)
    {
      continue;
    }
    reorderDetectedCorners(ordered, board_model_);

    Chessboard2d board_msg;
    fromCvPoints(ordered, board_msg);
    boards_out.push_back(std::move(board_msg));
  }

  return !boards_out.empty();
}

bool ChessboardDetector::detectChessboardsCuda(const cv::Mat &image, Chessboard2dList &boards_out)
  const
{
  boards_out.clear();

#ifdef CUDA_ENABLED
#ifdef TIME_LOGGER
  auto _pt = std::chrono::high_resolution_clock::now();
  auto _lap = [&_pt](const char *w) {
    const auto _n = std::chrono::high_resolution_clock::now();
    std::cout << "[prof][cuda] " << w << ": "
              << std::chrono::duration_cast<std::chrono::microseconds>(_n - _pt).count() * 1e-3
              << " ms" << std::endl;
    _pt = _n;
  };
#endif
  // Preprocessing (BGR2GRAY + Gaussian blur + min-max normalize) runs on the
  // GPU; upload the raw 8-bit image directly.
  const cv::Mat image_u8 = image.isContinuous() ? image : image.clone();
#ifdef TIME_LOGGER
  _lap("preprocess host (none; runs on GPU)");
#endif

  // six Geiger templates: radii {4, 8, 12}, each at two angle orientations
  std::vector<int> radius = {4, 4, 8, 8, 12, 12};
  std::vector<float> angle1 = {0.0f, static_cast<float>(CV_PI) / 4.0f,
                               0.0f, static_cast<float>(CV_PI) / 4.0f,
                               0.0f, static_cast<float>(CV_PI) / 4.0f};
  std::vector<float> angle2 = {static_cast<float>(CV_PI) / 2.0f, -static_cast<float>(CV_PI) / 4.0f,
                               static_cast<float>(CV_PI) / 2.0f, -static_cast<float>(CV_PI) / 4.0f,
                               static_cast<float>(CV_PI) / 2.0f, -static_cast<float>(CV_PI) / 4.0f};

  const bool use_separable = config_.acceleration == ChessboardAccelerationMode::CUDA_SEPARABLE;
  // Build the GPU corner detector (six likelihood estimators, their kernels and
  // device scratch buffers) once and reuse it across frames. Constructing it per
  // call rebuilt every kernel and forced a device allocation each frame; since
  // the estimators are image-size independent (their buffers resize on demand),
  // a single instance serves every frame and every ROI. It is cached behind an
  // opaque pointer (see the header note); setConfig() resets it when the kernel
  // settings change.
  auto *corner_detector =
    static_cast<camera_chessboard_detector::CudaDetector *>(cuda_pipeline_.get());
  if (corner_detector == nullptr)
  {
    auto built = std::make_shared<camera_chessboard_detector::CudaDetector>(
      radius, angle1, angle2, config_.verbose_logs, use_separable, config_.separable_rank
    );
    corner_detector = built.get();
    cuda_pipeline_ = std::move(built);
  }
  camera_chessboard_detector::DetectThresholds settings;
  settings.nms_margin = config_.cuda_nms_margin;
  settings.nms_radius = config_.cuda_nms_radius;
  settings.nms_threshold = config_.cuda_nms_threshold;
  settings.score_threshold = config_.score_threshold;

  auto corners = corner_detector->detect(
    image_u8.data, image_u8.cols, image_u8.rows, image_u8.channels(), settings
  );
#ifdef TIME_LOGGER
  _lap("corner detect (GPU pipeline total; see Map/NMS/Refine/Prune above)");
#endif
  if (config_.verbose_logs)
  {
    std::cerr << "[ccd][cuda] corner detect -> " << corners.x.size() << " corners" << std::endl;
    dumpCornerDiag("cuda", corners);
  }

  if (corners.x.empty())
  {
    return false;
  }

  camera_chessboard_detector::cpu::BoardBuilder chessboard;
  chessboard.setVerboseLogging(config_.verbose_logs);
  std::vector<cv::Mat> checkerboards;
  chessboard.assembleBoards(corners, checkerboards, config_.structure_lambda);
#ifdef TIME_LOGGER
  _lap("structure recovery (CPU)");
#endif
  if (config_.verbose_logs)
  {
    std::cerr << "[ccd][cuda] structure recovery -> " << checkerboards.size() << " board(s)"
              << std::endl;
  }
  if (checkerboards.empty())
  {
    return false;
  }

  for (const auto &checkerboard : checkerboards)
  {
    if (!sizeMatchesModel(checkerboard, board_model_))
    {
      continue;
    }

    std::vector<cv::Point2f> ordered;
    const bool ok = buildOrderedCorners(
      checkerboard, corners.x.size(),
      [&corners](int index) {
        return cv::Point2f(
          corners.x[static_cast<std::size_t>(index)], corners.y[static_cast<std::size_t>(index)]
        );
      },
      ordered
    );
    if (!ok)
    {
      continue;
    }
    reorderDetectedCorners(ordered, board_model_);

    Chessboard2d board_msg;
    fromCvPoints(ordered, board_msg);
    boards_out.push_back(std::move(board_msg));
  }

  return !boards_out.empty();
#else
  (void)image;
  return false;
#endif
}

}  // namespace camera_chessboard_detector
