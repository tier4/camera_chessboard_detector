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

// Derived from the libcbdetect checkerboard corner detector of A. Geiger et al.
// (KIT), "Automatic Camera and Range Sensor Calibration using a Single Shot",
// ICRA 2012. Distributed under GPL-3.0-or-later; see THIRD_PARTY_NOTICES.md.

#include "cpu/corner_detector.hpp"
#include "cpu/convolution.hpp"

#include <algorithm>
#include <cmath>

namespace camera_chessboard_detector {
namespace cpu {

using namespace cv;
using namespace std;

namespace {
// Tuning constants for the growth-based corner detector. The values are
// inherent to the Geiger / cbdetect formulation; they are named here for
// readability and typed to match the original literals exactly (double for
// the thresholds, int for counts) so the arithmetic is unchanged.
constexpr int    kCornerRadii[3]   = {4, 8, 12};  // the three template radii
constexpr double kQuadrantDeadZone = 0.1;     // |side| dead-zone for quadrant assignment
constexpr int    kOrientationBins  = 32;      // edge-orientation histogram bins over [0, pi)
constexpr double kMinEdgeAngleRad  = 0.3;     // reject corners whose two edges are nearly parallel
constexpr double kMinGradNorm      = 0.1;     // ignore pixels with near-zero gradient
constexpr double kInlierCos        = 0.25;    // |cos| threshold for an edge inlier
constexpr int    kEdgeDistPx       = 3;       // max distance (px) of a pixel from an edge line
constexpr int    kMaxCornerShiftPx = 4;       // reject sub-pixel updates that jump further than this
constexpr double kSvdRankEps       = 0.0001;  // singular value treated as non-zero above this
constexpr double kFlatHistEps      = 0.0001;  // histogram treated as flat below this spread
constexpr double kGradientBandPx   = 1.5;     // half-width (px) of the gradient template band
constexpr int    kNmsPatchSize     = 3;       // non-max-suppression window half-size
constexpr double kNmsThreshold     = 0.025;   // non-max-suppression likelihood floor
constexpr int    kNmsMargin        = 5;       // non-max-suppression image border to skip
constexpr int    kRefineRadiusPx   = 10;      // ROI radius for sub-pixel refinement
constexpr int    kBlurKernel       = 9;       // preprocess Gaussian kernel size
constexpr double kBlurSigma        = 1.5;     // preprocess Gaussian sigma
}  // namespace

CpuCornerDetector::CpuCornerDetector() {}

CpuCornerDetector::~CpuCornerDetector() {}

CpuCornerDetector::CpuCornerDetector(cv::Mat img) {
  // one scale per radius, each with two Geiger angle templates
  for (int r : kCornerRadii) {
    radius.push_back(r);
    template_angles_.push_back(Point2f((real_t)0, (real_t)CV_PI / 2));
    template_angles_.push_back(Point2f((real_t)CV_PI / 4, (real_t)-CV_PI / 4));
  }
}

// Gaussian probability density (mu, sigma).
real_t CpuCornerDetector::gaussian1d(real_t dist, real_t mu, real_t sigma) {
  real_t s = exp(-0.5 * (dist - mu) * (dist - mu) / (sigma * sigma));
  s = s / (std::sqrt(2 * CV_PI) * sigma);
  return s;
}

void CpuCornerDetector::buildQuadrantKernels(float angle1, float angle2, int kernelSize,
                                   Mat &kernelA, Mat &kernelB, Mat &kernelC,
                                   Mat &kernelD) {
  int width = (int)kernelSize * 2 + 1;
  int height = (int)kernelSize * 2 + 1;
  kernelA = cv::Mat::zeros(height, width, kMatDepth);
  kernelB = cv::Mat::zeros(height, width, kMatDepth);
  kernelC = cv::Mat::zeros(height, width, kMatDepth);
  kernelD = cv::Mat::zeros(height, width, kMatDepth);

  for (int u = 0; u < width; ++u) {
    for (int v = 0; v < height; ++v) {
      real_t vec[] = {u - kernelSize, v - kernelSize};
      real_t dis = std::sqrt(vec[0] * vec[0] + vec[1] * vec[1]);
      real_t side1 = vec[0] * (-sin(angle1)) + vec[1] * cos(angle1);
      real_t side2 = vec[0] * (-sin(angle2)) +
                    vec[1] * cos(angle2); // X=X0*cos+Y0*sin;Y=Y0*cos-X0*sin
      if (side1 <= -kQuadrantDeadZone && side2 <= -kQuadrantDeadZone) {
        kernelA.ptr<real_t>(v)[u] = gaussian1d(dis, 0, kernelSize / 2);
      }
      if (side1 >= kQuadrantDeadZone && side2 >= kQuadrantDeadZone) {
        kernelB.ptr<real_t>(v)[u] = gaussian1d(dis, 0, kernelSize / 2);
      }
      if (side1 <= -kQuadrantDeadZone && side2 >= kQuadrantDeadZone) {
        kernelC.ptr<real_t>(v)[u] = gaussian1d(dis, 0, kernelSize / 2);
      }
      if (side1 >= kQuadrantDeadZone && side2 <= -kQuadrantDeadZone) {
        kernelD.ptr<real_t>(v)[u] = gaussian1d(dis, 0, kernelSize / 2);
      }
    }
  }

  kernelA = kernelA / cv::sum(kernelA)[0];
  kernelB = kernelB / cv::sum(kernelB)[0];
  kernelC = kernelC / cv::sum(kernelC)[0];
  kernelD = kernelD / cv::sum(kernelD)[0];
}

void CpuCornerDetector::elementwiseMin(Mat src1, Mat src2, Mat &dst) {
  int rowsLeft = src1.rows;
  int colsLeft = src1.cols;
  int rowsRight = src2.rows;
  int colsRight = src2.cols;
  if (rowsLeft != rowsRight || colsLeft != colsRight)
    return;

  int channels = src1.channels();

  int nr = rowsLeft;
  int nc = colsLeft;
  if (src1.isContinuous()) {
    nc = nc * nr;
    nr = 1;
  }
  for (int i = 0; i < nr; i++) {
    const real_t *dataLeft = src1.ptr<real_t>(i);
    const real_t *dataRight = src2.ptr<real_t>(i);
    real_t *dataResult = dst.ptr<real_t>(i);
    for (int j = 0; j < nc * channels; ++j) {
      dataResult[j] = (dataLeft[j] < dataRight[j]) ? dataLeft[j] : dataRight[j];
    }
  }
}

void CpuCornerDetector::elementwiseMax(Mat src1, Mat src2, Mat &dst) {
  int rowsLeft = src1.rows;
  int colsLeft = src1.cols;
  int rowsRight = src2.rows;
  int colsRight = src2.cols;
  if (rowsLeft != rowsRight || colsLeft != colsRight)
    return;

  int channels = src1.channels();

  int nr = rowsLeft;
  int nc = colsLeft;
  if (src1.isContinuous()) {
    nc = nc * nr;
    nr = 1;
  }
  for (int i = 0; i < nr; i++) {
    const real_t *dataLeft = src1.ptr<real_t>(i);
    const real_t *dataRight = src2.ptr<real_t>(i);
    real_t *dataResult = dst.ptr<real_t>(i);
    for (int j = 0; j < nc * channels; ++j) {
      dataResult[j] =
        (dataLeft[j] >= dataRight[j]) ? dataLeft[j] : dataRight[j];
    }
  }
}

void CpuCornerDetector::computeGradientOrientation(Mat img, Mat &imgDu, Mat &imgDv,
    Mat &imgAngle, Mat &imgWeight) {
  Mat sobelKernel(3, 3, kMatDepth);
  Mat sobelKernelTrs(3, 3, kMatDepth);

  sobelKernel.col(0).setTo(cv::Scalar(-1.0));
  sobelKernel.col(1).setTo(cv::Scalar(0.0));
  sobelKernel.col(2).setTo(cv::Scalar(1.0));

  sobelKernelTrs = sobelKernel.t();

  imgDu = conv2(img, sobelKernel, CONVOLUTION_SAME);
  imgDv = conv2(img, sobelKernelTrs, CONVOLUTION_SAME);

  if (imgDu.size() != imgDv.size())
    return;

  cartToPolar(imgDu, imgDv, imgWeight, imgAngle, false);
  for (int i = 0; i < imgDu.rows; i++) {
    for (int j = 0; j < imgDu.cols; j++) {
      real_t *dataAngle = imgAngle.ptr<real_t>(i);
      if (dataAngle[j] < 0)
        dataAngle[j] = dataAngle[j] + CV_PI;
      else if (dataAngle[j] > CV_PI)
        dataAngle[j] = dataAngle[j] - CV_PI;
    }
  }
}

void CpuCornerDetector::nonMaxSuppress(Mat &inputCorners,
    vector<Point2f> &outputCorners,
    int patchSize, real_t threshold,
    int margin) {
  if (inputCorners.empty()) {
    return;
  }
  for (int i = margin + patchSize;
       i <= inputCorners.cols - (margin + patchSize + 1);
       i = i + patchSize + 1) {
    for (int j = margin + patchSize;
         j <= inputCorners.rows - (margin + patchSize + 1);
         j = j + patchSize + 1) {
      real_t maxVal = inputCorners.ptr<real_t>(j)[i];
      int maxX = i;
      int maxY = j;
      for (int m = i; m <= i + patchSize; m++) {
        for (int n = j; n <= j + patchSize; n++) {
          real_t temp = inputCorners.ptr<real_t>(n)[m];
          if (temp > maxVal) {
            maxVal = temp;
            maxX = m;
            maxY = n;
          }
        }
      }
      if (maxVal < threshold)
        continue;
      int flag = 0;
      for (int m = maxX - patchSize;
           m <= min(maxX + patchSize, inputCorners.cols - margin - 1);
           m++) {
        for (int n = maxY - patchSize;
             n <= min(maxY + patchSize, inputCorners.rows - margin - 1); n++) {
          if (inputCorners.ptr<real_t>(n)[m] > maxVal &&
              (m < i || m > i + patchSize || n < j || n > j + patchSize)) {
            flag = 1;
            break;
          }
        }
        if (flag)
          break;
      }
      if (flag)
        continue;
      outputCorners.push_back(Point(maxX, maxY));
      std::vector<real_t> e1(2, 0.0);
      std::vector<real_t> e2(2, 0.0);
      edge_dirs1_.push_back(e1);
      edge_dirs2_.push_back(e2);
    }
  }
}

int cmp(const pair<real_t, int> &a, const pair<real_t, int> &b) {
  return a.first > b.first;
}

// locate the peaks of the smoothed histogram
void CpuCornerDetector::findHistogramModes(vector<real_t> hist,
    vector<real_t> &hist_smoothed,
    vector<pair<real_t, int>> &modes,
    real_t sigma) {
  // approximate mean-shift by histogram smoothing, then hill-climb
  // smooth the histogram
  bool allZeros = true;
  for (int i = 0; i < hist.size(); i++) {
    real_t sum = 0;
    for (int j = -(int)round(2 * sigma); j <= (int)round(2 * sigma); j++) {
      int idx = 0;
      idx = (i + j) % hist.size();
      sum = sum + hist[idx] * gaussian1d(j, 0, sigma);
    }
    hist_smoothed[i] = sum;
    if (abs(hist_smoothed[i] - hist_smoothed[0]) > kFlatHistEps)
      allZeros = false; // check if at least one entry is non - zero
    // (a flat histogram would otherwise loop forever)
  }
  if (allZeros)
    return;

  // hill-climb each bin to its local peak
  for (int i = 0; i < hist.size(); i++) {
    int j = i;
    while (true) {
      float h0 = hist_smoothed[j];
      int j1 = (j + 1) % hist.size();
      int j2 = (j - 1) % hist.size();
      float h1 = hist_smoothed[j1];
      float h2 = hist_smoothed[j2];
      if (h1 >= h0 && h1 >= h2)
        j = j1;
      else if (h2 > h0 && h2 > h1)
        j = j2;
      else
        break;
    }
    bool ys = true;
    if (modes.size() == 0) {
      ys = true;
    } else {
      for (int k = 0; k < modes.size(); k++) {
        if (modes[k].second == j) {
          ys = false;
          break;
        }
      }
    }
    if (ys == true) {
      modes.push_back(std::make_pair(hist_smoothed[j], j));
    }
  }
  std::sort(modes.begin(), modes.end(), cmp);
}

// estimate the two dominant edge orientations at a corner
void CpuCornerDetector::estimateEdgeOrientations(Mat imgAngle, Mat imgWeight, int index) {
  // orientation histogram resolution
  int num_bins = kOrientationBins;

  // flatten the ROI into angle / weight samples
  if (imgAngle.size() != imgWeight.size())
    return;

  vector<real_t> vec_angle, vec_weight;
  for (int i = 0; i < imgAngle.cols; i++) {
    for (int j = 0; j < imgAngle.rows; j++) {
      // rotate normals by 90deg to get edge directions
      float angle = imgAngle.ptr<real_t>(j)[i] + CV_PI / 2;
      angle = angle > CV_PI ? (angle - CV_PI) : angle;
      vec_angle.push_back(angle);

      vec_weight.push_back(imgWeight.ptr<real_t>(j)[i]);
    }
  }

  // weighted orientation histogram
  real_t bin_width = (CV_PI / num_bins);
  vector<real_t> angleHist(num_bins, 0);
  for (int i = 0; i < vec_angle.size(); i++) {
    int bin = max(min((int)floor(vec_angle[i] / bin_width), num_bins - 1), 0);
    angleHist[bin] = angleHist[bin] + vec_weight[i];
  }

  // peaks of the smoothed histogram
  vector<real_t> hist_smoothed(angleHist);
  vector<std::pair<real_t, int>> modes;
  findHistogramModes(angleHist, hist_smoothed, modes, 1);

  // need at least two distinct orientations
  if (modes.size() <= 1)
    return;

  // the two strongest peaks, ordered by angle
  float fo[2];
  fo[0] = modes[0].second * bin_width;
  fo[1] = modes[1].second * bin_width;
  real_t deltaAngle = 0;
  if (fo[0] > fo[1]) {
    real_t t = fo[0];
    fo[0] = fo[1];
    fo[1] = t;
  }

  deltaAngle = MIN(fo[1] - fo[0], fo[0] - fo[1] + (real_t)CV_PI);
  // reject near-parallel edge pairs
  if (deltaAngle <= kMinEdgeAngleRad)
    return;

  // store the two edge unit vectors
  edge_dirs1_[index][0] = cos(fo[0]);
  edge_dirs1_[index][1] = sin(fo[0]);
  edge_dirs2_[index][0] = cos(fo[1]);
  edge_dirs2_[index][1] = sin(fo[1]);
}

float CpuCornerDetector::vecNorm(cv::Point2f o) {
  return sqrt(o.x * o.x + o.y * o.y);
}

void CpuCornerDetector::refineCornersSubpixel(vector<Point2f> &corners, Mat imgDu,
                                    Mat imgDv, Mat imgAngle, Mat imgWeight,
                                    float radius) {
  // image dimensions
  int width = imgDu.cols;
  int height = imgDu.rows;

  // for all corners do
  for (int i = 0; i < corners.size(); i++) {
    // integer corner location
    int cu = corners[i].x;
    int cv = corners[i].y;
    // estimate edge orientations
    int startX, startY, ROIwidth, ROIheight;
    startX = MAX(cu - radius, (real_t)0);
    startY = MAX(cv - radius, (real_t)0);
    ROIwidth = MIN(cu + radius + 1, (real_t)width - 1) - startX;
    ROIheight = MIN(cv + radius + 1, (real_t)height - 1) - startY;

    Mat roiAngle, roiWeight;
    roiAngle = imgAngle(Rect(startX, startY, ROIwidth, ROIheight));
    roiWeight = imgWeight(Rect(startX, startY, ROIwidth, ROIheight));
    estimateEdgeOrientations(roiAngle, roiWeight, i);

    // skip corners with degenerate edges
    if (edge_dirs1_[i][0] == 0 && edge_dirs1_[i][1] == 0 ||
        edge_dirs2_[i][0] == 0 && edge_dirs2_[i][1] == 0)
      continue;

    cv::Mat A1 = cv::Mat::zeros(cv::Size(2, 2), kMatDepth);
    cv::Mat A2 = cv::Mat::zeros(cv::Size(2, 2), kMatDepth);


    for (int u = startX; u < startX + ROIwidth; u++)
      for (int v = startY; v < startY + ROIheight; v++) {
        // gradient direction at this pixel
        cv::Point2f o(imgDu.at<real_t>(v, u), imgDv.at<real_t>(v, u));
        float no = vecNorm(o);


        if (no < kMinGradNorm)
          continue;
        o = o / no;
        // accumulate inliers for edge 1
        real_t t0 = abs(o.x * edge_dirs1_[i][0] + o.y * edge_dirs1_[i][1]);
        if (t0 < kInlierCos) { // inlier ?
          Mat outer(1, 2, kMatDepth);
          outer.col(0).setTo(imgDu.at<real_t>(v, u));
          outer.col(1).setTo(imgDv.at<real_t>(v, u));
          Mat outer_u = imgDu.at<real_t>(v, u) * outer;
          Mat outer_v = imgDv.at<real_t>(v, u) * outer;
          for (int j = 0; j < A1.cols; j++) {
            A1.at<real_t>(0, j) = A1.at<real_t>(0, j) + outer_u.at<real_t>(0, j);
            A1.at<real_t>(1, j) = A1.at<real_t>(1, j) + outer_v.at<real_t>(0, j);
          }

        }
        // accumulate inliers for edge 2
        real_t t1 = abs(o.x * edge_dirs2_[i][0] + o.y * edge_dirs2_[i][1]);
        if (t1 < kInlierCos) { // inlier ?
          Mat outer(1, 2, kMatDepth);
          outer.col(0).setTo(imgDu.at<real_t>(v, u));
          outer.col(1).setTo(imgDv.at<real_t>(v, u));
          Mat outer_u = imgDu.at<real_t>(v, u) * outer;
          Mat outer_v = imgDv.at<real_t>(v, u) * outer;
          for (int j = 0; j < A2.cols; j++) {
            A2.at<real_t>(0, j) = A2.at<real_t>(0, j) + outer_u.at<real_t>(0, j);
            A2.at<real_t>(1, j) = A2.at<real_t>(1, j) + outer_v.at<real_t>(0, j);
          }

        }
      } // end for
    // refined orientation = smallest eigenvector of each scatter matrix
    cv::Mat v1, evec1;
    cv::Mat v2, evec2;
    cv::eigen(A1, v1, evec1);
    cv::eigen(A2, v2, evec2);
    edge_dirs1_[i][0] = -evec1.at<real_t>(1, 0);
    edge_dirs1_[i][1] = -evec1.at<real_t>(1, 1);
    edge_dirs2_[i][0] = -evec2.at<real_t>(1, 0);
    edge_dirs2_[i][1] = -evec2.at<real_t>(1, 1);



    cv::Mat G = cv::Mat::zeros(cv::Size(2, 2), kMatDepth);
    cv::Mat b = cv::Mat::zeros(cv::Size(1, 2), kMatDepth);
    for (int u = startX; u < startX + ROIwidth; u++)
      for (int v = startY; v < startY + ROIheight; v++) {
        // gradient direction at this pixel
        cv::Point2f o(imgDu.at<real_t>(v, u), imgDv.at<real_t>(v, u));
        float no = vecNorm(o);
        if (no < kMinGradNorm)
          continue;
        o = o / no;
        // least-squares sub-pixel position from the edge constraints
        if (u != cu || v != cv) { // % do not consider center pixel
          // distance of this pixel from each edge line
          cv::Point2f w(u - cu, v - cv);
          float proj1 = w.x * edge_dirs1_[i][0] + w.y * edge_dirs1_[i][1];
          float proj2 = w.x * edge_dirs2_[i][0] + w.y * edge_dirs2_[i][1];

          cv::Point2f wv1(proj1 * edge_dirs1_[i][0], proj1 * edge_dirs1_[i][1]);
          cv::Point2f wv2(proj2 * edge_dirs2_[i][0], proj2 * edge_dirs2_[i][1]);
          cv::Point2f vd1(w.x - wv1.x, w.y - wv1.y);
          cv::Point2f vd2(w.x - wv2.x, w.y - wv2.y);
          real_t d1 = vecNorm(vd1), d2 = vecNorm(vd2);
          // pixel lies on one of the two edges
          if ((d1 < kEdgeDistPx) && abs(o.x * edge_dirs1_[i][0] +
                              o.y * edge_dirs1_[i][1]) < kInlierCos ||
              (d2 < kEdgeDistPx) && abs(o.x * edge_dirs2_[i][0] +
                              o.y * edge_dirs2_[i][1]) < kInlierCos) {
            real_t du = imgDu.at<real_t>(v, u), dv = imgDv.at<real_t>(v, u);
            cv::Mat uvt = (Mat_<real_t>(2, 1) << u, v);
            cv::Mat H =
              (Mat_<real_t>(2, 2) << du * du, du * dv, dv * du, dv * dv);
            G = G + H;
            cv::Mat t = H * (uvt);
            b = b + t;
          }
        }
      } // endfor
    // solve for the sub-pixel corner when well-conditioned



    Mat s, u, v;
    SVD::compute(G, s, u, v);
    int rank = 0;
    for (int k = 0; k < s.rows; k++) {
      if (s.at<real_t>(k, 0) > kSvdRankEps ||
          s.at<real_t>(k, 0) < -kSvdRankEps) { // treat as non-zero
        rank++;
      }
    }
    if (rank == 2) {
      cv::Mat mp = G.inv() * b;
      cv::Point2f new_pos(mp.at<real_t>(0, 0), mp.at<real_t>(1, 0));
      // discard if the update jumps too far
      if (vecNorm(cv::Point2f(new_pos.x - cu, new_pos.y - cv)) >=
          kMaxCornerShiftPx) {
        edge_dirs1_[i][0] = 0;
        edge_dirs1_[i][1] = 0;
        edge_dirs2_[i][0] = 0;
        edge_dirs2_[i][1] = 0;
      } else {
        corners[i].x = mp.at<real_t>(0, 0);
        corners[i].y = mp.at<real_t>(1, 0);
      }
    } else { // ill-conditioned: mark the corner invalid
      edge_dirs1_[i][0] = 0;
      edge_dirs1_[i][1] = 0;
      edge_dirs2_[i][0] = 0;
      edge_dirs2_[i][1] = 0;
    }
  }

}

// correlation score for one corner patch
void CpuCornerDetector::correlationScore(Mat img, Mat imgWeight,
    vector<Point2f> cornersEdge, float &score) {
  // center
  int c[] = {imgWeight.cols / 2, imgWeight.cols / 2};

  // gradient template mask (~3 px band)
  Mat img_filter = Mat::ones(imgWeight.size(), imgWeight.type());
  img_filter = img_filter * -1;
  for (int i = 0; i < imgWeight.cols; i++) {
    for (int j = 0; j < imgWeight.rows; j++) {
      Point2f p1 = Point2f(i - c[0], j - c[1]);
      Point2f p2 = Point2f(p1.x * cornersEdge[0].x * cornersEdge[0].x +
                           p1.y * cornersEdge[0].x * cornersEdge[0].y,
                           p1.x * cornersEdge[0].x * cornersEdge[0].y +
                           p1.y * cornersEdge[0].y * cornersEdge[0].y);
      Point2f p3 = Point2f(p1.x * cornersEdge[1].x * cornersEdge[1].x +
                           p1.y * cornersEdge[1].x * cornersEdge[1].y,
                           p1.x * cornersEdge[1].x * cornersEdge[1].y +
                           p1.y * cornersEdge[1].y * cornersEdge[1].y);
      float norm1 =
        sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
      float norm2 =
        sqrt((p1.x - p3.x) * (p1.x - p3.x) + (p1.y - p3.y) * (p1.y - p3.y));
      if (norm1 <= kGradientBandPx || norm2 <= kGradientBandPx) {
        img_filter.ptr<real_t>(j)[i] = 1;
      }
    }
  }

  // normalize
  Mat mean, std, mean1, std1;
  meanStdDev(imgWeight, mean, std);
  meanStdDev(img_filter, mean1, std1);
  for (int i = 0; i < imgWeight.cols; i++) {
    for (int j = 0; j < imgWeight.rows; j++) {
      imgWeight.ptr<real_t>(j)[i] =
        (real_t)(imgWeight.ptr<real_t>(j)[i] - mean.ptr<double>(0)[0]) /
        (real_t)std.ptr<double>(0)[0];
      img_filter.ptr<real_t>(j)[i] =
        (real_t)(img_filter.ptr<real_t>(j)[i] - mean1.ptr<double>(0)[0]) /
        (real_t)std1.ptr<double>(0)[0];
    }
  }

  // convert into vectors
  vector<float> vec_filter, vec_weight;
  for (int i = 0; i < imgWeight.cols; i++) {
    for (int j = 0; j < imgWeight.rows; j++) {
      vec_filter.push_back(img_filter.ptr<real_t>(j)[i]);
      vec_weight.push_back(imgWeight.ptr<real_t>(j)[i]);
    }
  }

  // compute gradient score
  float sum = 0;
  for (int i = 0; i < vec_weight.size(); i++) {
    sum += vec_weight[i] * vec_filter[i];
  }
  sum = (real_t)sum / (real_t)(vec_weight.size() - 1);
  real_t gradient_score = sum >= 0 ? sum : 0;

  // intensity quadrant kernels
  const camera_chessboard_detector::CorrelationKernelSet & kernels = score_kernel_cache_.get(
      atan2(cornersEdge[0].y, cornersEdge[0].x),
      atan2(cornersEdge[1].y, cornersEdge[1].x), c[0]);

  // the four quadrant responses
  const float a1 = kernels.a.dot(img);
  const float a2 = kernels.b.dot(img);
  const float b1 = kernels.c.dot(img);
  const float b2 = kernels.d.dot(img);

  float mu = (a1 + a2 + b1 + b2) / 4;

  float score_a = (a1 - mu) >= (a2 - mu) ? (a2 - mu) : (a1 - mu);
  float score_b = (mu - b1) >= (mu - b2) ? (mu - b2) : (mu - b1);
  float score_1 = score_a >= score_b ? score_b : score_a;

  score_b = (b1 - mu) >= (b2 - mu) ? (b2 - mu) : (b1 - mu);
  score_a = (mu - a1) >= (mu - a2) ? (mu - a2) : (mu - a1);
  float score_2 = score_a >= score_b ? score_b : score_a;

  float intensity_score = score_1 >= score_2 ? score_1 : score_2;
  intensity_score = intensity_score > 0.0 ? intensity_score : 0.0;

  score = gradient_score * intensity_score;
}

// score every corner across the radius sweep
void CpuCornerDetector::scoreAllCorners(Mat img, Mat imgAngle, Mat imgWeight,
                                   vector<Point2f> &corners, vector<int> radius,
                                   vector<float> &score) {
  // for all corners do
  for (int i = 0; i < corners.size(); i++) {
    // corner location
    int u = corners[i].x + 0.5;
    int v = corners[i].y + 0.5;

    // correlation score at each radius
    vector<float> scores;
    for (int j = 0; j < radius.size(); j++) {
      scores.push_back(0);
      int r = radius[j];
      if (u > r && u <= (img.cols - r - 1) && v > r &&
          v <= (img.rows - r - 1)) {
        int startX, startY, ROIwidth, ROIheight;
        startX = u - r;
        startY = v - r;
        ROIwidth = 2 * r + 1;
        ROIheight = 2 * r + 1;

        Mat sub_img = img(Rect(startX, startY, ROIwidth, ROIheight)).clone();
        Mat sub_imgWeight =
          imgWeight(Rect(startX, startY, ROIwidth, ROIheight)).clone();
        vector<Point2f> cornersEdge;
        cornersEdge.push_back(
          Point2f((float)edge_dirs1_[i][0], (float)edge_dirs1_[i][1]));
        cornersEdge.push_back(
          Point2f((float)edge_dirs2_[i][0], (float)edge_dirs2_[i][1]));
        correlationScore(sub_img, sub_imgWeight, cornersEdge, scores[j]);
      }
    }
    // keep the best radius
    score.push_back(*max_element(begin(scores), end(scores)));
  }
}

void CpuCornerDetector::buildLikelihoodMap(cv::Mat &src, cv::Mat &dst) {
  Mat respA(src.size(), kMatDepth);
  Mat respB(src.size(), kMatDepth);
  Mat respC(src.size(), kMatDepth);
  Mat respD(src.size(), kMatDepth);

  Mat lowA(src.size(), kMatDepth);
  Mat lowB(src.size(), kMatDepth);
  Mat cand1(src.size(), kMatDepth);
  Mat cand2(src.size(), kMatDepth);
  Mat mean_resp(src.size(), kMatDepth);

  for (int i = 0; i < 6; i++) {
    Mat kA, kB, kC, kD;
    buildQuadrantKernels(template_angles_[i].x, template_angles_[i].y, radius[i / 2],
                 kA, kB, kC,
                 kD);

    // correlate the image with the four quadrant kernels
    respA =
      conv2(src, kA, CONVOLUTION_SAME);
    respB =
      conv2(src, kB, CONVOLUTION_SAME);
    respC =
      conv2(src, kC, CONVOLUTION_SAME);
    respD =
      conv2(src, kD, CONVOLUTION_SAME);

    // compute mean
    mean_resp = (respA + respB + respC + respD) /
                    4.0;

    // case 1: a = white, b = black
    elementwiseMin(respA - mean_resp, respB - mean_resp,
          lowA);
    elementwiseMin(mean_resp - respC, mean_resp - respD,
          lowB);
    elementwiseMin(lowA, lowB, cand1);
    // case 2: b = white, a = black
    elementwiseMin(mean_resp - respA, mean_resp - respB,
          lowA);
    elementwiseMin(respC - mean_resp, respD - mean_resp,
          lowB);
    elementwiseMin(lowA, lowB, cand2);

    // keep the per-pixel maximum
    elementwiseMax(dst, cand1, dst);
    elementwiseMax(dst, cand2, dst);
  }
}

void CpuCornerDetector::detect(Mat &src, CornerCandidates &mcorners,
                                    real_t scoreThreshold,
                                    RefinementOption refinementOption) {

  Mat gray, normalized;
  gray = Mat(src.size(), CV_8U);

  // grayscale
  if (src.channels() == 3) {
    cvtColor(src, gray, COLOR_BGR2GRAY);
  } else {
    gray = src.clone();
  }

  cv::GaussianBlur(gray, gray, cv::Size(kBlurKernel, kBlurKernel), kBlurSigma);

  // normalise to [0, 1]
  normalize(gray, normalized, 0, 1, cv::NORM_MINMAX, kMatDepth);

  Mat likelihood = Mat::zeros(normalized.size(), kMatDepth);

  buildLikelihoodMap(normalized, likelihood);

  // candidate corners via non-maximum suppression
  nonMaxSuppress(likelihood, corner_points_, kNmsPatchSize, kNmsThreshold,
                        kNmsMargin);


  // post processing

  Mat grad_x(gray.size(), kMatDepth);
  Mat grad_y(gray.size(), kMatDepth);

  Mat angle_map = cv::Mat::zeros(gray.size(), kMatDepth);
  Mat weight_map = cv::Mat::zeros(gray.size(), kMatDepth);

  computeGradientOrientation(normalized, grad_x, grad_y, angle_map, weight_map);


  if (refinementOption == RefinementOption::DO_REFINE) {
    // subpixel refinement
    refineCornersSubpixel(corner_points_, grad_x, grad_y, angle_map, weight_map, kRefineRadiusPx);

    if (corner_points_.size() > 0) {
      for (int i = 0; i < corner_points_.size(); i++) {
        if (edge_dirs1_[i][0] == 0 && edge_dirs1_[i][0] == 0) {
          corner_points_[i].x = 0;
          corner_points_[i].y = 0;
        }
      }
    }

  }
  // drop corners whose edges were rejected

  // score corners
  vector<float> score;
  scoreAllCorners(normalized, angle_map, weight_map, corner_points_, radius, score);



  // keep only corners above the score threshold
  int nlen = corner_points_.size();
  if (nlen > 0) {
    for (int i = 0; i < nlen; i++) {
      if (score[i] > scoreThreshold) {
        mcorners.p.push_back(corner_points_[i]);
        mcorners.v1.push_back(
          cv::Vec2f(edge_dirs1_[i][0], edge_dirs1_[i][1]));
        mcorners.v2.push_back(
          cv::Vec2f(edge_dirs2_[i][0], edge_dirs2_[i][1]));
        mcorners.score.push_back(score[i]);
      }
    }
  }

  std::vector<cv::Vec2f> corners_n1(mcorners.p.size());
  for (int i = 0; i < corners_n1.size(); i++) {
    if (mcorners.v1[i][0] + mcorners.v1[i][1] < 0.0) {
      mcorners.v1[i] = -mcorners.v1[i];
    }
    corners_n1[i] = mcorners.v1[i];
    float flipflag = corners_n1[i][0] * mcorners.v2[i][0] +
                     corners_n1[0][1] * mcorners.v2[i][1];
    if (flipflag > 0)
      flipflag = -1;
    else
      flipflag = 1;
    mcorners.v2[i] = flipflag * mcorners.v2[i];
  }
}

}  // namespace cpu
}  // namespace camera_chessboard_detector
