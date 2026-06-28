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

#ifndef CCD__TYPES_HPP
#define CCD__TYPES_HPP

#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

// Self-contained result types for this package. Keeping them local avoids any
// message-definition or external schema dependency. Mapping these into a
// downstream consumer's own board representation is intentionally left to that
// consumer, which can convert from `ChessboardDetection` / `Chessboard2dList`
// to whatever shape it prefers.
namespace camera_chessboard_detector
{

// Lightweight image header. The detector only default-constructs and copies
// this; it never inspects the fields, so a minimal timestamp + frame-id shape
// is sufficient.
struct Header
{
  std::uint64_t stamp_ns{0};
  std::string frame_id{};
};

struct Point2f
{
  float x{0.0f};
  float y{0.0f};
};

struct ChessboardModel
{
  float square_size{};        // [m]
  std::uint32_t rows{};       // inner-corner rows (0 == size-agnostic)
  std::uint32_t cols{};       // inner-corner cols (0 == size-agnostic)
};

struct Chessboard2d
{
  Header header{};
  std::uint32_t target_id{};
  ChessboardModel model{};
  std::vector<Point2f> corners2d{};
};

using Chessboard2dList = std::vector<Chessboard2d>;

// Small helpers to convert between the package's Point2f list and OpenCV's
// cv::Point2f, used at the detector's input/output boundary.
inline void toCvPoints(const Chessboard2d &board, std::vector<cv::Point2f> &out)
{
  out.clear();
  out.reserve(board.corners2d.size());
  for (const auto &p : board.corners2d)
  {
    out.emplace_back(p.x, p.y);
  }
}

inline void fromCvPoints(const std::vector<cv::Point2f> &in, Chessboard2d &board)
{
  board.corners2d.clear();
  board.corners2d.reserve(in.size());
  for (const auto &p : in)
  {
    board.corners2d.push_back(Point2f{p.x, p.y});
  }
}

}  // namespace camera_chessboard_detector

#endif  // CCD__TYPES_HPP
