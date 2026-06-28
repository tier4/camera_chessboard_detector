// Golden / regression safety net.
//
// Freezes the corner output of the CPU and CUDA pipelines so refactors
// cannot silently change behaviour. CPU and CUDA differ at the ~0.01 px
// level (different convolution paths), so each pipeline has its own golden.
// A change that legitimately alters output is applied by re-recording the
// goldens (run with CCD_RECORD_GOLDEN=1).
//
// Inputs:
//   * synthetic checkerboard — deterministic, with known ground-truth inner
//     corners (validates accuracy, not just regression).
//   * the real sample image  — regression freeze of real-world behaviour.
//
// Comparison is set-based (order-independent nearest-neighbour): the
// meaningful invariant is the detected corner *set*, not its order.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

#include "camera_chessboard_detector/detector.hpp"

#ifndef CCD_SAMPLE_IMAGE
#define CCD_SAMPLE_IMAGE ""
#endif
#ifndef CCD_GOLDEN_DIR
#define CCD_GOLDEN_DIR ""
#endif

namespace
{

namespace fb = camera_chessboard_detector;

struct Pt
{
  float x;
  float y;
};

// --- synthetic checkerboard -------------------------------------------------
// `sq_cols` x `sq_rows` squares -> (sq_cols-1) x (sq_rows-1) inner corners.
struct Synthetic
{
  cv::Mat image;
  std::vector<Pt> gt_inner_corners;
};

Synthetic makeSyntheticBoard(int sq_cols, int sq_rows, int square_px, int margin_px)
{
  const int w = margin_px * 2 + sq_cols * square_px;
  const int h = margin_px * 2 + sq_rows * square_px;
  cv::Mat img(h, w, CV_8UC1, cv::Scalar(255));  // white quiet zone

  for (int r = 0; r < sq_rows; ++r)
  {
    for (int c = 0; c < sq_cols; ++c)
    {
      if (((r + c) & 1) == 0)
      {
        continue;  // leave white
      }
      const cv::Rect cell(margin_px + c * square_px,
                          margin_px + r * square_px,
                          square_px, square_px);
      cv::rectangle(img, cell, cv::Scalar(0), cv::FILLED);
    }
  }
  // mild anti-alias; the detector also blurs internally.
  cv::GaussianBlur(img, img, cv::Size(3, 3), 0.8);

  cv::Mat bgr;
  cv::cvtColor(img, bgr, cv::COLOR_GRAY2BGR);

  Synthetic s;
  s.image = bgr;
  for (int j = 1; j < sq_rows; ++j)
  {
    for (int i = 1; i < sq_cols; ++i)
    {
      s.gt_inner_corners.push_back(
        Pt{static_cast<float>(margin_px + i * square_px),
           static_cast<float>(margin_px + j * square_px)});
    }
  }
  return s;
}

// --- detection helper -------------------------------------------------------
bool detect(const cv::Mat &image, fb::ChessboardAccelerationMode mode,
            std::vector<Pt> &corners_out,
            int separable_rank = -1)
{
  fb::ChessboardModel model;  // rows=cols=0 -> size-agnostic
  fb::ChessboardDetectorConfig cfg;
  cfg.acceleration = mode;
  cfg.separable_rank = separable_rank;  // ignored unless mode == CUDA_SEPARABLE
  fb::ChessboardDetector det(model, cfg);
  fb::ChessboardDetection out;
  const bool ok = det.detectChessboards(image, out);
  corners_out.clear();
  if (!ok || out.boards.empty())
  {
    return false;
  }
  for (const auto &p : out.boards.front().corners2d)
  {
    corners_out.push_back(Pt{p.x, p.y});
  }
  return !corners_out.empty();
}

// --- golden IO --------------------------------------------------------------
std::string goldenPath(const std::string &name)
{
  return std::string(CCD_GOLDEN_DIR) + "/" + name + ".csv";
}

void sortPts(std::vector<Pt> &v)
{
  std::sort(v.begin(), v.end(), [](const Pt &a, const Pt &b) {
    if (std::abs(a.y - b.y) > 1.0f)
    {
      return a.y < b.y;
    }
    return a.x < b.x;
  });
}

void writeGolden(const std::string &name, std::vector<Pt> pts)
{
  sortPts(pts);
  std::ofstream ofs(goldenPath(name));
  ofs << "# camera_chessboard_detector golden: " << name
      << " (sorted y,x; " << pts.size() << " pts)\n";
  ofs.setf(std::ios::fixed);
  ofs.precision(4);
  for (const auto &p : pts)
  {
    ofs << p.x << "," << p.y << "\n";
  }
}

bool readGolden(const std::string &name, std::vector<Pt> &pts)
{
  std::ifstream ifs(goldenPath(name));
  if (!ifs.is_open())
  {
    return false;
  }
  pts.clear();
  std::string line;
  while (std::getline(ifs, line))
  {
    if (line.empty() || line[0] == '#')
    {
      continue;
    }
    const auto comma = line.find(',');
    if (comma == std::string::npos)
    {
      continue;
    }
    pts.push_back(Pt{std::stof(line.substr(0, comma)),
                     std::stof(line.substr(comma + 1))});
  }
  return true;
}

// Order-independent set match: equal sizes + a greedy 1-1 nearest assignment
// within `tol` px. Returns the worst matched distance (or -1 on failure).
double setMatchMaxDist(const std::vector<Pt> &a, const std::vector<Pt> &b,
                       double tol)
{
  if (a.size() != b.size())
  {
    return -1.0;
  }
  std::vector<bool> used(b.size(), false);
  double worst = 0.0;
  for (const auto &pa : a)
  {
    int best = -1;
    double best_d = 1e18;
    for (std::size_t k = 0; k < b.size(); ++k)
    {
      if (used[k])
      {
        continue;
      }
      const double d = std::hypot(pa.x - b[k].x, pa.y - b[k].y);
      if (d < best_d)
      {
        best_d = d;
        best = static_cast<int>(k);
      }
    }
    if (best < 0 || best_d > tol)
    {
      return -1.0;
    }
    used[best] = true;
    worst = std::max(worst, best_d);
  }
  return worst;
}

bool recordMode()
{
  const char *e = std::getenv("CCD_RECORD_GOLDEN");
  return e != nullptr && std::string(e) == "1";
}

constexpr double kRegressionTolPx = 0.05;  // same pipeline, deterministic
constexpr double kParityTolPx = 0.50;      // CPU vs CUDA conv differences
constexpr double kAccuracyTolPx = 1.50;    // detected-vs-ground-truth

// Shared fixtures (built once).
const Synthetic &synthetic()
{
  static Synthetic s = makeSyntheticBoard(10, 8, 70, 100);  // 9x7 = 63 inner
  return s;
}

cv::Mat sampleImage()
{
  return cv::imread(CCD_SAMPLE_IMAGE, cv::IMREAD_COLOR);
}

void goldenCheckOrRecord(const std::string &name, const std::vector<Pt> &pts)
{
  if (recordMode())
  {
    writeGolden(name, pts);
    GTEST_SKIP() << "recorded golden " << name << " (" << pts.size() << " pts)";
  }
  std::vector<Pt> golden;
  ASSERT_TRUE(readGolden(name, golden))
    << "missing golden " << name << " — run with CCD_RECORD_GOLDEN=1";
  const double w = setMatchMaxDist(pts, golden, kRegressionTolPx);
  EXPECT_GE(w, 0.0) << name << ": set mismatch vs golden (size "
                    << pts.size() << " vs " << golden.size() << ")";
}

}  // namespace

// --- regression anchors: real-world behaviour ------------------------------
// The real sample is detected by both pipelines (48 corners each); these two
// goldens are the primary regression net.
TEST(Regression, SampleCpuMatchesGolden)
{
  const cv::Mat img = sampleImage();
  ASSERT_FALSE(img.empty()) << "sample image not found: " << CCD_SAMPLE_IMAGE;
  std::vector<Pt> c;
  ASSERT_TRUE(detect(img, fb::ChessboardAccelerationMode::CPU, c));
  goldenCheckOrRecord("sample_cpu", c);
}

TEST(Regression, SampleCudaMatchesGolden)
{
  const cv::Mat img = sampleImage();
  ASSERT_FALSE(img.empty()) << "sample image not found: " << CCD_SAMPLE_IMAGE;
  std::vector<Pt> c;
  if (!detect(img, fb::ChessboardAccelerationMode::CUDA, c))
  {
    GTEST_SKIP() << "CUDA pipeline unavailable";
  }
  goldenCheckOrRecord("sample_cuda", c);
}

// CUDA detects the synthetic board fully (63 = 9x7 inner corners).
TEST(Regression, SyntheticCudaMatchesGolden)
{
  std::vector<Pt> c;
  if (!detect(synthetic().image, fb::ChessboardAccelerationMode::CUDA, c))
  {
    GTEST_SKIP() << "CUDA pipeline unavailable";
  }
  goldenCheckOrRecord("synthetic_cuda", c);
}

// --- CPU vs CUDA parity on the input both pipelines handle -----------------
TEST(Parity, SampleCpuVsCuda)
{
  const cv::Mat img = sampleImage();
  ASSERT_FALSE(img.empty());
  std::vector<Pt> cpu;
  std::vector<Pt> cuda;
  ASSERT_TRUE(detect(img, fb::ChessboardAccelerationMode::CPU, cpu));
  if (!detect(img, fb::ChessboardAccelerationMode::CUDA, cuda))
  {
    GTEST_SKIP() << "CUDA pipeline unavailable";
  }
  const double w = setMatchMaxDist(cpu, cuda, kParityTolPx);
  EXPECT_GE(w, 0.0) << "CPU/CUDA disagree (sizes " << cpu.size()
                    << " vs " << cuda.size() << ")";
}

// --- accuracy vs known ground truth (CUDA handles the synthetic board) -----
TEST(Accuracy, SyntheticCudaVsGroundTruth)
{
  std::vector<Pt> c;
  if (!detect(synthetic().image, fb::ChessboardAccelerationMode::CUDA, c))
  {
    GTEST_SKIP() << "CUDA pipeline unavailable";
  }
  std::vector<Pt> gt = synthetic().gt_inner_corners;
  // 1-1 match detected<->GT within sub-pixel tolerance (recall + precision).
  const double w = setMatchMaxDist(c, gt, kAccuracyTolPx);
  EXPECT_GE(w, 0.0) << "CUDA synthetic detection (" << c.size()
                    << ") does not match the " << gt.size()
                    << " ground-truth inner corners within "
                    << kAccuracyTolPx << " px";
}

// --- CPU synthetic detection ----------------------------------------------
// The CPU front end must de-duplicate near-coincident corners (as the CUDA
// path does) for structure recovery to assemble the synthetic board; this
// verifies that the CPU path detects it like CUDA.
TEST(Regression, SyntheticCpuMatchesGolden)
{
  std::vector<Pt> c;
  ASSERT_TRUE(detect(synthetic().image, fb::ChessboardAccelerationMode::CPU, c))
    << "CPU should detect the synthetic board";
  goldenCheckOrRecord("synthetic_cpu", c);
}

TEST(Accuracy, SyntheticCpuVsGroundTruth)
{
  std::vector<Pt> c;
  ASSERT_TRUE(detect(synthetic().image, fb::ChessboardAccelerationMode::CPU, c));
  std::vector<Pt> gt = synthetic().gt_inner_corners;
  const double w = setMatchMaxDist(c, gt, kAccuracyTolPx);
  EXPECT_GE(w, 0.0) << "CPU synthetic detection (" << c.size()
                    << ") does not match the " << gt.size()
                    << " ground-truth inner corners within "
                    << kAccuracyTolPx << " px";
}

TEST(Parity, SyntheticCpuVsCuda)
{
  std::vector<Pt> cpu;
  std::vector<Pt> cuda;
  ASSERT_TRUE(detect(synthetic().image, fb::ChessboardAccelerationMode::CPU, cpu));
  if (!detect(synthetic().image, fb::ChessboardAccelerationMode::CUDA, cuda))
  {
    GTEST_SKIP() << "CUDA pipeline unavailable";
  }
  const double w = setMatchMaxDist(cpu, cuda, kParityTolPx);
  EXPECT_GE(w, 0.0) << "CPU/CUDA disagree on synthetic (sizes " << cpu.size()
                    << " vs " << cuda.size() << ")";
}

// --- opt-in SVD-separable path: gating signals for adoption ----------------
//
// The real-image golden alone is an insufficient correctness signal: a
// defective separable implementation can still recover the same 48 fisheye
// corners by accident. The accuracy test on the synthetic regular
// checkerboard is the required tripwire. These three tests together
// (synthetic accuracy at full rank, dense<->separable parity at full rank,
// real-sample regression at full rank) are the gate that must stay green
// before any non-full-rank approximation ships.

constexpr double kSeparableParityTolPx = 1.0;  // FP accumulation in 2 passes

TEST(SeparableFullRank, SyntheticAccuracy)
{
  std::vector<Pt> c;
  if (!detect(synthetic().image,
              fb::ChessboardAccelerationMode::CUDA_SEPARABLE, c, -1))
  {
    GTEST_SKIP() << "CUDA pipeline unavailable";
  }
  const std::vector<Pt> &gt = synthetic().gt_inner_corners;
  const double w = setMatchMaxDist(c, gt, kAccuracyTolPx);
  EXPECT_GE(w, 0.0) << "CUDA separable (full rank) synthetic detection ("
                    << c.size() << ") does not match the " << gt.size()
                    << " ground-truth inner corners within "
                    << kAccuracyTolPx
                    << " px. Full rank must equal the dense path; a "
                       "miss here indicates a separable-impl defect.";
}

TEST(SeparableFullRank, SampleMatchesCudaDense)
{
  const cv::Mat img = sampleImage();
  ASSERT_FALSE(img.empty());
  std::vector<Pt> dense;
  std::vector<Pt> separable;
  if (!detect(img, fb::ChessboardAccelerationMode::CUDA, dense))
  {
    GTEST_SKIP() << "CUDA dense pipeline unavailable";
  }
  if (!detect(img, fb::ChessboardAccelerationMode::CUDA_SEPARABLE,
              separable, -1))
  {
    GTEST_SKIP() << "CUDA separable pipeline unavailable";
  }
  const double w = setMatchMaxDist(dense, separable, kSeparableParityTolPx);
  EXPECT_GE(w, 0.0) << "Full-rank separable diverged from CUDA dense on "
                       "the real sample (sizes "
                    << dense.size() << " vs " << separable.size() << ").";
}

TEST(SeparableFullRank, SyntheticMatchesCudaDense)
{
  std::vector<Pt> dense;
  std::vector<Pt> separable;
  if (!detect(synthetic().image, fb::ChessboardAccelerationMode::CUDA, dense))
  {
    GTEST_SKIP() << "CUDA dense pipeline unavailable";
  }
  if (!detect(synthetic().image,
              fb::ChessboardAccelerationMode::CUDA_SEPARABLE,
              separable, -1))
  {
    GTEST_SKIP() << "CUDA separable pipeline unavailable";
  }
  const double w = setMatchMaxDist(dense, separable, kSeparableParityTolPx);
  EXPECT_GE(w, 0.0) << "Full-rank separable diverged from CUDA dense on "
                       "the synthetic board (sizes "
                    << dense.size() << " vs " << separable.size() << ").";
}

// Emit the exact synthetic board as a PNG (record mode only) for manual
// inspection. Not a regression assertion.
TEST(Fixture, EmitSyntheticImage)
{
  if (!recordMode())
  {
    GTEST_SKIP() << "synthetic PNG emitted only with CCD_RECORD_GOLDEN=1";
  }
  const std::string path = std::string(CCD_GOLDEN_DIR) + "/synthetic_board.png";
  ASSERT_TRUE(cv::imwrite(path, synthetic().image)) << "imwrite failed: " << path;
  GTEST_SKIP() << "emitted " << path;
}
