# camera_chessboard_detector

A growth-based ("Geiger" / ROCHADE-style) chessboard corner detector. It
recovers partially visible and strongly distorted boards (e.g. fisheye)
where OpenCV's `findChessboardCorners` fails outright. Detection only — no
pose estimation.

## Features

- **CPU pipeline** — always available, depends only on OpenCV.
- **CUDA pipeline** — built automatically when a CUDA compiler is present.
  Both front ends feed one shared structure-recovery back end, so they
  produce the same board topology.
- **Size-agnostic** — detects any inner-grid size (set `rows = cols = 0`),
  or validates against a known board size.
- **Sub-pixel accurate** — verified on a synthetic board with known
  ground-truth corners and frozen against per-pipeline golden outputs.

## Dependencies

- **OpenCV** (`core`, `imgproc`, `calib3d`, `imgcodecs`) — required.
- **Eigen3** — required.
- **CUDA toolkit** — optional. With it, the GPU pipeline is built (default
  architecture is the build host's GPU; override with
  `-DCMAKE_CUDA_ARCHITECTURES=<arch>`). Without it, the package builds
  CPU-only with no source change.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Per-stage timing is opt-in: add `-DCCD_PROFILE=ON`.

## Usage

```cpp
#include <opencv2/imgcodecs.hpp>
#include <camera_chessboard_detector/detector.hpp>

int main()
{
  using namespace camera_chessboard_detector;

  ChessboardModel model;                // rows = cols = 0 -> detect any board
  ChessboardDetectorConfig cfg;
  cfg.acceleration = ChessboardAccelerationMode::CUDA;  // or ::CPU

  ChessboardDetector detector(model, cfg);

  const cv::Mat image = cv::imread("board.jpg", cv::IMREAD_COLOR);
  ChessboardDetection out;
  if (detector.detectChessboards(image, out) && !out.boards.empty()) {
    for (const auto &p : out.boards.front().corners2d) {
      std::printf("%.4f %.4f\n", p.x, p.y);
    }
  }
}
```

Link against the exported target:

```cmake
find_package(camera_chessboard_detector REQUIRED)
target_link_libraries(your_target camera_chessboard_detector::camera_chessboard_detector)
```

## Test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The suite freezes CPU and CUDA corner output against golden files in
`test/golden/`, checks CPU/CUDA agreement, and validates sub-pixel accuracy
on a synthetic board with known ground-truth corners. CUDA-dependent tests
skip cleanly on a CPU-only build. To regenerate goldens after an intended
behavioural change, run the test binary with `CCD_RECORD_GOLDEN=1`.

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).

