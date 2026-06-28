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

#include <cstddef>
#include <cstdint>
#include <unordered_map>

#include <opencv2/core.hpp>

namespace camera_chessboard_detector {

// Four normalised quadrant kernels for one (angle1, angle2, radius) triple
// — the per-corner output that `buildQuadrantKernels` builds on the fly inside
// `per-corner scorer`. Each matrix is (2R+1)x(2R+1) float32 and sums to 1.
struct CorrelationKernelSet {
  cv::Mat a;
  cv::Mat b;
  cv::Mat c;
  cv::Mat d;
};

// Cache for `buildQuadrantKernels`'s output. The four matrices depend only on
// (angle1, angle2, R); inside `per-corner scorer` they are otherwise rebuilt
// for every corner, which is O((2R+1)^2) per call. The cache quantises
// the two angles into `kNumAngleBins` bins, pre-computes the radial
// distance table per radius (it does not depend on the angles), and
// memoises the four kernels per (angle1_bin, angle2_bin, R) key. The
// per-corner cost drops to a single bin lookup plus, on miss, the same
// O((2R+1)^2) build as before — the miss cost is paid once per unique
// quantised triple over the lifetime of the cache.
//
// Behaviour:
//   * Lazy fill, never evict. Per-frame work is bounded; the cache is
//     intended to live for the lifetime of a detector instance.
//   * Worst-case size: kNumAngleBins^2 * #radii entries. With 32 bins
//     and the three production radii (4, 8, 12) that is 3072 entries.
//   * Not thread-safe — callers must serialise access from multiple
//     threads.
//
// The returned reference is stable while the cache is alive.
class CorrelationKernelCache {
public:
  // Number of quantisation bins covering [-pi, +pi). Step = 2*pi/N.
  // 32 bins give a step of ~11.25 degrees, which keeps the cache small
  // while staying well within the dead-zone of the |side| < 0.1
  // quadrant test inside buildQuadrantKernels for any practical edge angle.
  static constexpr int kNumAngleBins = 32;

  CorrelationKernelCache() = default;

  // Returns the four kernels for the (angle1, angle2, R) input.
  // The angles are quantised to the nearest bin centre before the
  // lookup; consecutive calls with the same bin reuse the entry.
  const CorrelationKernelSet & get(float angle1, float angle2, int radius);

  // Diagnostics / tests.
  std::size_t size() const noexcept { return entries_.size(); }

  // Convert an angle in [-inf, +inf] to its bin index in [0, N).
  static int angleToBin(float angle) noexcept;

  // The representative angle for a bin index (its centre).
  static float binToAngle(int bin) noexcept;

private:
  struct Key {
    std::int16_t angle1_bin;
    std::int16_t angle2_bin;
    std::int16_t radius;
    bool operator==(const Key & other) const noexcept {
      return angle1_bin == other.angle1_bin &&
             angle2_bin == other.angle2_bin &&
             radius == other.radius;
    }
  };

  struct KeyHash {
    std::size_t operator()(const Key & k) const noexcept {
      // Three small ints into one 64-bit word — exact, collision-free.
      std::uint64_t w = static_cast<std::uint16_t>(k.angle1_bin);
      w = (w << 16) | static_cast<std::uint16_t>(k.angle2_bin);
      w = (w << 16) | static_cast<std::uint16_t>(k.radius);
      return static_cast<std::size_t>(w);
    }
  };

  // Pre-baked distance table for the given radius. Each entry is
  // sqrt((u-R)^2 + (v-R)^2). Built lazily on first request per R.
  const cv::Mat & disTable(int radius);

  // Construct the four kernels for the (bin1, bin2, R) triple. Uses the
  // bin centres as the angle values so the output is what
  // buildQuadrantKernels(binToAngle(bin1), binToAngle(bin2), R) would produce.
  CorrelationKernelSet build(int angle1_bin, int angle2_bin, int radius);

  std::unordered_map<int, cv::Mat> dis_tables_;
  std::unordered_map<Key, CorrelationKernelSet, KeyHash> entries_;
};

}  // namespace camera_chessboard_detector
