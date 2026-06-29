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

// Wall-clock benchmark across resolutions and acceleration modes.
//
// Usage: camera_chessboard_detector_benchmark [image] [warm_iters] [preset_filter]
//   image         - source image (defaults to the bundled 4K sample)
//   warm_iters    - timed iterations after one warm-up (default 5)
//   preset_filter - run only presets whose name contains this string
//                   (e.g. "4K"); default runs all presets
//
// Build with -DCCD_PROFILE=ON to also emit the per-stage timing the library
// prints internally (map-gen / NMS / refine / score / structure recovery).
#include <camera_chessboard_detector/detector.hpp>
#include <opencv2/opencv.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#ifndef CCD_SAMPLE_IMAGE
#define CCD_SAMPLE_IMAGE ""
#endif

namespace ccd = camera_chessboard_detector;

namespace
{

struct Preset
{
  const char *name;
  int w;
  int h;
};

double runMode(const cv::Mat &img, ccd::ChessboardAccelerationMode mode, int warm, int &corners_out)
{
  ccd::ChessboardModel model;  // rows = cols = 0 -> size-agnostic
  ccd::ChessboardDetectorConfig cfg;
  cfg.acceleration = mode;
  cfg.separable_rank = 1;  // used only by CUDA_SEPARABLE
  ccd::ChessboardDetector detector(model, cfg);
  ccd::ChessboardDetection out;

  detector.detectChessboards(img, out);  // warm-up (CUDA ctx/JIT, pipeline build)

  double total_ms = 0.0;
  corners_out = 0;
  for (int i = 0; i < warm; ++i)
  {
    const auto t0 = std::chrono::high_resolution_clock::now();
    const bool ok = detector.detectChessboards(img, out);
    const auto t1 = std::chrono::high_resolution_clock::now();
    total_ms += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() * 1e-3;
    if (ok && !out.boards.empty())
    {
      corners_out = static_cast<int>(out.boards.front().corners2d.size());
    }
  }
  return total_ms / warm;
}

}  // namespace

int main(int argc, char **argv)
{
  const std::string path = (argc > 1) ? argv[1] : CCD_SAMPLE_IMAGE;
  const int warm = (argc > 2) ? std::max(1, std::atoi(argv[2])) : 5;
  const std::string filter = (argc > 3) ? argv[3] : "";

  const cv::Mat src = cv::imread(path, cv::IMREAD_COLOR);
  if (src.empty())
  {
    std::fprintf(stderr, "cannot load image: %s\n", path.c_str());
    return 1;
  }

  const std::vector<Preset> presets = {
    {"VGA", 640, 360}, {"SD", 720, 405}, {"HD", 1280, 720}, {"FHD", 1920, 1080}, {"4K", 3840, 2160},
  };

  std::printf("source %dx%d  warm-iterations=%d\n", src.cols, src.rows, warm);
  std::printf(
    "%-5s %-11s %11s %11s %13s   corners(cpu/cuda/sep)\n", "preset", "resolution", "CPU ms",
    "CUDA ms", "CUDA-sep1 ms"
  );

  for (const auto &p : presets)
  {
    if (!filter.empty() && std::string(p.name).find(filter) == std::string::npos)
    {
      continue;
    }
    cv::Mat img;
    cv::resize(src, img, cv::Size(p.w, p.h));

    int c_cpu = 0, c_cuda = 0, c_sep = 0;
    const double cpu = runMode(img, ccd::ChessboardAccelerationMode::CPU, warm, c_cpu);
    const double cuda = runMode(img, ccd::ChessboardAccelerationMode::CUDA, warm, c_cuda);
    const double sep = runMode(img, ccd::ChessboardAccelerationMode::CUDA_SEPARABLE, warm, c_sep);

    char res[32];
    std::snprintf(res, sizeof(res), "%dx%d", p.w, p.h);
    std::printf(
      "%-5s %-11s %11.1f %11.1f %13.1f   %d/%d/%d\n", p.name, res, cpu, cuda, sep, c_cpu, c_cuda,
      c_sep
    );
    std::fflush(stdout);
  }
  return 0;
}
