#pragma once

#include "core.hpp"
#include "cuda/gpu_buffers.hpp"

namespace camera_chessboard_detector {
namespace cuda {

// On-GPU non-maximum suppression for the Geiger likelihood map.
//
// GPU replacement for the CPU `non-max suppression` plus the prerequisite
// download of the full W*H float map. The class runs a fused kernel
// that, for every pixel:
//   1. Skips if the pixel is inside the `margin + radius` border.
//   2. Skips if `src(x, y) <= threshold`.
//   3. Compares `src(x, y)` against every pixel in the (2R+1)*(2R+1)
//      window around it. Emits only if `src(x, y)` is the maximum,
//      with plateau ties resolved by row-major lexicographic order
//      so each plateau yields exactly one candidate.
// Surviving (x, y, value) triples are atomically appended to a device
// array; only the compact list is copied back to the host.
//
// The output `CornerArray` is populated with (x, y) plus the
// placeholder edge fields (`edge1_cos=2.0`, `edge1_sin=0.0`,
// `edge2_cos=2.0`, `edge2_sin=0.0`) that the downstream refiner
// expects — the same contract the host implementation provides.
class CudaNonMaxSuppression {
public:
  // Default capacity for the compact device array. The likelihood map
  // at NMS R ~= 4 typically yields a few hundred candidates on the
  // production 4K sample; 50k is generous and uses < 1 MB of device
  // memory.
  static constexpr std::size_t kDefaultCapacity = 50000;

  CudaNonMaxSuppression();
  ~CudaNonMaxSuppression();

  CudaNonMaxSuppression(const CudaNonMaxSuppression &) = delete;
  CudaNonMaxSuppression & operator=(const CudaNonMaxSuppression &) = delete;
  CudaNonMaxSuppression(CudaNonMaxSuppression &&) = delete;
  CudaNonMaxSuppression & operator=(CudaNonMaxSuppression &&) = delete;

  // Run NMS on `likelihood_map`. `corners` is cleared and refilled.
  // Returns the number of candidates emitted on the device; if this
  // equals the current capacity, the device buffer overflowed and some
  // candidates were lost (the host output is still consistent — only
  // the first `capacity` are returned).
  std::size_t run(const GpuImagePtr & likelihood_map, int radius,
                  int margin, float threshold, CornerArray & corners);

  std::size_t capacity() const noexcept { return capacity_; }

private:
  // Grow the device buffers to at least `min_capacity` slots.
  void ensureCapacity(std::size_t min_capacity);

  int * d_out_x_{nullptr};
  int * d_out_y_{nullptr};
  float * d_out_v_{nullptr};
  int * d_count_{nullptr};
  std::size_t capacity_{0};
};

}  // namespace cuda
}  // namespace camera_chessboard_detector
