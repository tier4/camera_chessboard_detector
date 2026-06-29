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

### ROS 2 / colcon

The library is pure CMake and never links against ROS. It is also shipped with
a `package.xml`, so it can be dropped into a colcon workspace and built as an
ament-packaged CMake library:

```bash
colcon build --packages-select camera_chessboard_detector
```

When `ament_cmake` is present (i.e. a ROS 2 environment is sourced) the build
registers the package with the ament index and exports its dependencies; when
it is absent, the same target is installed as a standalone CMake package. No
flag is needed — the build detects which mode applies.

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

## Contributing

Code style and identifier naming are enforced by the `.clang-format` and
`.clang-tidy` files at the repository root. Please run both before opening a PR.

### Formatting — `.clang-format`

Google-based style with Allman braces, a 100-column limit, and pointer /
reference bound to the variable name (`const cv::Mat &img`, `int *p`). It is
written to be portable across clang-format 14 through 22+ (the bracket style
uses the legacy `AlignAfterOpenBracket: BlockIndent` spelling, which newer
clang-format still accepts for backward compatibility).

Format every source file in place:

```bash
find src include test \( -name '*.cpp' -o -name '*.hpp' -o -name '*.cu' -o -name '*.cuh' \) \
  -print0 | xargs -0 clang-format -i
```

Check formatting without editing (CI-friendly — fails on any diff):

```bash
find src include test \( -name '*.cpp' -o -name '*.hpp' -o -name '*.cu' -o -name '*.cuh' \) \
  -print0 | xargs -0 clang-format --dry-run --Werror
```

### Naming / lint — `.clang-tidy`

`.clang-tidy` enables `readability-identifier-naming` with these conventions:

| Kind | Style | Example |
| --- | --- | --- |
| types / enums / type aliases | `CamelCase` | `CornerArray`, `RealT` |
| enum constants | `CamelCase` | `CudaSeparable` |
| methods / functions | `camelBack` | `detectChessboards`, `toMat` |
| variables / parameters / **local** constants | `lower_case` | `kernel_size`, `rows_left` |
| namespaces | `lower_case` | `camera_chessboard_detector` |
| static / global constants, constexpr | `k` + `CamelCase` | `kPi`, `kMatDepth` |
| private / protected / static members | `lower_case_` (trailing `_`) | `radius_`, `lambda_` |

Note the trailing `_` is for **non-public** members only; keep public data
members suffix-free (or make internal state `private`).

clang-tidy needs a compile database, so configure with export enabled first:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

Report violations across the C++ translation units and their headers:

```bash
run-clang-tidy -p build -header-filter='camera_chessboard_detector/(src|include)/' '\.cpp$'
```

Apply fixes automatically (add `-fix`):

```bash
run-clang-tidy -p build -fix -header-filter='camera_chessboard_detector/(src|include)/' '\.cpp$'
```

On Debian/Ubuntu the binaries are version-suffixed; pass them explicitly, e.g.
`run-clang-tidy-18 -clang-tidy-binary=clang-tidy-18 -clang-apply-replacements-binary=clang-apply-replacements-18 ...`.

#### CUDA (`.cu`) caveat — read before running `-fix`

`.cu` files are compiled by `nvcc`, so clang-tidy cannot analyse them; the lint
above deliberately restricts the translation units to `'\.cpp$'`. Identifiers
that are local to a `.cu` file are therefore **not** enforced and may keep any
style.

The catch is *shared symbols* — names declared in a header and defined or used
inside a `.cu` (CUDA class methods, enums, kernel-wrapper functions). A `-fix`
run renames the header and `.cpp` side but **not** the `.cu` side, so after any
automated rename you must:

1. Build CPU **and** CUDA (`cmake --build build`) and update the `.cu`
   references to the renamed symbols by hand until it compiles.
2. Run `ctest`, and ideally diff detector output before/after. Automated renames
   can silently *collide* — e.g. stripping the trailing `_` from a public member
   that shares a name with a parameter turns `member_ = value` into the
   self-assignment `member = value`, which compiles and may still pass coarse
   tests while changing results. Such state belongs in `private` (with the `_`).

## Credits

This detector implements, and is a derivative work of, the growth-based
checkerboard detection algorithm of Andreas Geiger et al. ("libcbdetect", KIT).
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the full upstream
attribution and license notice. If you use this software in academic work,
please cite:

> A. Geiger, F. Moosmann, Ö. Car, B. Schuster.
> "Automatic Camera and Range Sensor Calibration using a Single Shot."
> IEEE International Conference on Robotics and Automation (ICRA), 2012.

Upstream reference implementations:
[cvlibs libcbdetect](https://www.cvlibs.net/software/libcbdetect/) (GPL) and
the C++ port [ftdlyc/libcbdetect](https://github.com/ftdlyc/libcbdetect) (GPL-3.0).

## License

Copyright (C) 2026 TIER IV, Inc.

GPL-3.0-or-later. See [LICENSE](LICENSE).

This project is a derivative of the GPL-licensed `libcbdetect` (GPL-3.0-or-later)
and is therefore distributed under the same `GPL-3.0-or-later` terms, which are
compatible with and preserve the upstream license. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

