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

namespace camera_chessboard_detector
{
namespace cpu
{

using namespace cv;
using namespace std;

namespace
{
// Tuning constants for the growth-based corner detector. The values are
// inherent to the Geiger / cbdetect formulation; they are named here for
// readability and typed to match the original literals exactly (double for
// the thresholds, int for counts) so the arithmetic is unchanged.
constexpr int kCornerRadii[3] = {4, 8, 12};  // the three template radii
constexpr double kQuadrantDeadZone = 0.1;    // |side| dead-zone for quadrant assignment
constexpr int kOrientationBins = 32;         // edge-orientation histogram bins over [0, pi)
constexpr double kMinEdgeAngleRad = 0.3;     // reject corners whose two edges are nearly parallel
constexpr double kMinGradNorm = 0.1;         // ignore pixels with near-zero gradient
constexpr double kInlierCos = 0.25;          // |cos| threshold for an edge inlier
constexpr int kEdgeDistPx = 3;               // max distance (px) of a pixel from an edge line
constexpr int kMaxCornerShiftPx = 4;         // reject sub-pixel updates that jump further than this
constexpr double kSvdRankEps = 0.0001;       // singular value treated as non-zero above this
constexpr double kFlatHistEps = 0.0001;      // histogram treated as flat below this spread
constexpr double kGradientBandPx = 1.5;      // half-width (px) of the gradient template band
constexpr int kNmsPatchSize = 3;             // non-max-suppression window half-size
constexpr double kNmsThreshold = 0.025;      // non-max-suppression likelihood floor
constexpr int kNmsMargin = 5;                // non-max-suppression image border to skip
constexpr int kRefineRadiusPx = 10;          // ROI radius for sub-pixel refinement
constexpr int kBlurKernel = 9;               // preprocess Gaussian kernel size
constexpr double kBlurSigma = 1.5;           // preprocess Gaussian sigma
}  // namespace

CpuCornerDetector::CpuCornerDetector() {}

CpuCornerDetector::~CpuCornerDetector() {}

CpuCornerDetector::CpuCornerDetector(cv::Mat img)
{
  // one scale per radius, each with two Geiger angle templates
  for (int r : kCornerRadii)
  {
    radius_.push_back(r);
    template_angles_.push_back(Point2f((RealT)0, (RealT)CV_PI / 2));
    template_angles_.push_back(Point2f((RealT)CV_PI / 4, (RealT)-CV_PI / 4));
  }
}

// Gaussian probability density (mu, sigma).
RealT CpuCornerDetector::gaussian1d(RealT dist, RealT mu, RealT sigma)
{
  RealT s = exp(-0.5 * (dist - mu) * (dist - mu) / (sigma * sigma));
  s = s / (std::sqrt(2 * CV_PI) * sigma);
  return s;
}

void CpuCornerDetector::buildQuadrantKernels(
  float angle1, float angle2, int kernel_size, Mat &kernel_a, Mat &kernel_b, Mat &kernel_c,
  Mat &kernel_d
)
{
  int width = (int)kernel_size * 2 + 1;
  int height = (int)kernel_size * 2 + 1;
  kernel_a = cv::Mat::zeros(height, width, kMatDepth);
  kernel_b = cv::Mat::zeros(height, width, kMatDepth);
  kernel_c = cv::Mat::zeros(height, width, kMatDepth);
  kernel_d = cv::Mat::zeros(height, width, kMatDepth);

  for (int u = 0; u < width; ++u)
  {
    for (int v = 0; v < height; ++v)
    {
      RealT vec[] = {static_cast<RealT>(u - kernel_size), static_cast<RealT>(v - kernel_size)};
      RealT dis = std::sqrt(vec[0] * vec[0] + vec[1] * vec[1]);
      RealT side1 = vec[0] * (-sin(angle1)) + vec[1] * cos(angle1);
      RealT side2 =
        vec[0] * (-sin(angle2)) + vec[1] * cos(angle2);  // X=X0*cos+Y0*sin;Y=Y0*cos-X0*sin
      if (side1 <= -kQuadrantDeadZone && side2 <= -kQuadrantDeadZone)
      {
        kernel_a.ptr<RealT>(v)[u] = gaussian1d(dis, 0, kernel_size / 2);
      }
      if (side1 >= kQuadrantDeadZone && side2 >= kQuadrantDeadZone)
      {
        kernel_b.ptr<RealT>(v)[u] = gaussian1d(dis, 0, kernel_size / 2);
      }
      if (side1 <= -kQuadrantDeadZone && side2 >= kQuadrantDeadZone)
      {
        kernel_c.ptr<RealT>(v)[u] = gaussian1d(dis, 0, kernel_size / 2);
      }
      if (side1 >= kQuadrantDeadZone && side2 <= -kQuadrantDeadZone)
      {
        kernel_d.ptr<RealT>(v)[u] = gaussian1d(dis, 0, kernel_size / 2);
      }
    }
  }

  kernel_a = kernel_a / cv::sum(kernel_a)[0];
  kernel_b = kernel_b / cv::sum(kernel_b)[0];
  kernel_c = kernel_c / cv::sum(kernel_c)[0];
  kernel_d = kernel_d / cv::sum(kernel_d)[0];
}

void CpuCornerDetector::elementwiseMin(Mat src1, Mat src2, Mat &dst)
{
  int rows_left = src1.rows;
  int cols_left = src1.cols;
  int rows_right = src2.rows;
  int cols_right = src2.cols;
  if (rows_left != rows_right || cols_left != cols_right) return;

  int channels = src1.channels();

  int nr = rows_left;
  int nc = cols_left;
  if (src1.isContinuous())
  {
    nc = nc * nr;
    nr = 1;
  }
  for (int i = 0; i < nr; i++)
  {
    const RealT *data_left = src1.ptr<RealT>(i);
    const RealT *data_right = src2.ptr<RealT>(i);
    RealT *data_result = dst.ptr<RealT>(i);
    for (int j = 0; j < nc * channels; ++j)
    {
      data_result[j] = (data_left[j] < data_right[j]) ? data_left[j] : data_right[j];
    }
  }
}

void CpuCornerDetector::elementwiseMax(Mat src1, Mat src2, Mat &dst)
{
  int rows_left = src1.rows;
  int cols_left = src1.cols;
  int rows_right = src2.rows;
  int cols_right = src2.cols;
  if (rows_left != rows_right || cols_left != cols_right) return;

  int channels = src1.channels();

  int nr = rows_left;
  int nc = cols_left;
  if (src1.isContinuous())
  {
    nc = nc * nr;
    nr = 1;
  }
  for (int i = 0; i < nr; i++)
  {
    const RealT *data_left = src1.ptr<RealT>(i);
    const RealT *data_right = src2.ptr<RealT>(i);
    RealT *data_result = dst.ptr<RealT>(i);
    for (int j = 0; j < nc * channels; ++j)
    {
      data_result[j] = (data_left[j] >= data_right[j]) ? data_left[j] : data_right[j];
    }
  }
}

void CpuCornerDetector::computeGradientOrientation(
  Mat img, Mat &img_du, Mat &img_dv, Mat &img_angle, Mat &img_weight
)
{
  Mat sobel_kernel(3, 3, kMatDepth);
  Mat sobel_kernel_trs(3, 3, kMatDepth);

  sobel_kernel.col(0).setTo(cv::Scalar(-1.0));
  sobel_kernel.col(1).setTo(cv::Scalar(0.0));
  sobel_kernel.col(2).setTo(cv::Scalar(1.0));

  sobel_kernel_trs = sobel_kernel.t();

  img_du = conv2(img, sobel_kernel, ConvolutionSame);
  img_dv = conv2(img, sobel_kernel_trs, ConvolutionSame);

  if (img_du.size() != img_dv.size()) return;

  cartToPolar(img_du, img_dv, img_weight, img_angle, false);
  for (int i = 0; i < img_du.rows; i++)
  {
    for (int j = 0; j < img_du.cols; j++)
    {
      RealT *data_angle = img_angle.ptr<RealT>(i);
      if (data_angle[j] < 0)
        data_angle[j] = data_angle[j] + CV_PI;
      else if (data_angle[j] > CV_PI)
        data_angle[j] = data_angle[j] - CV_PI;
    }
  }
}

void CpuCornerDetector::nonMaxSuppress(
  Mat &input_corners, vector<Point2f> &output_corners, int patch_size, RealT threshold, int margin
)
{
  if (input_corners.empty())
  {
    return;
  }
  for (int i = margin + patch_size; i <= input_corners.cols - (margin + patch_size + 1);
       i = i + patch_size + 1)
  {
    for (int j = margin + patch_size; j <= input_corners.rows - (margin + patch_size + 1);
         j = j + patch_size + 1)
    {
      RealT max_val = input_corners.ptr<RealT>(j)[i];
      int max_x = i;
      int max_y = j;
      for (int m = i; m <= i + patch_size; m++)
      {
        for (int n = j; n <= j + patch_size; n++)
        {
          RealT temp = input_corners.ptr<RealT>(n)[m];
          if (temp > max_val)
          {
            max_val = temp;
            max_x = m;
            max_y = n;
          }
        }
      }
      if (max_val < threshold) continue;
      int flag = 0;
      for (int m = max_x - patch_size;
           m <= min(max_x + patch_size, input_corners.cols - margin - 1); m++)
      {
        for (int n = max_y - patch_size;
             n <= min(max_y + patch_size, input_corners.rows - margin - 1); n++)
        {
          if (input_corners.ptr<RealT>(n)[m] > max_val &&
              (m < i || m > i + patch_size || n < j || n > j + patch_size))
          {
            flag = 1;
            break;
          }
        }
        if (flag) break;
      }
      if (flag) continue;
      output_corners.push_back(Point(max_x, max_y));
      std::vector<RealT> e1(2, 0.0);
      std::vector<RealT> e2(2, 0.0);
      edge_dirs1_.push_back(e1);
      edge_dirs2_.push_back(e2);
    }
  }
}

int cmp(const pair<RealT, int> &a, const pair<RealT, int> &b) { return a.first > b.first; }

// locate the peaks of the smoothed histogram
void CpuCornerDetector::findHistogramModes(
  vector<RealT> hist, vector<RealT> &hist_smoothed, vector<pair<RealT, int>> &modes, RealT sigma
)
{
  // approximate mean-shift by histogram smoothing, then hill-climb
  // smooth the histogram
  bool all_zeros = true;
  for (int i = 0; i < hist.size(); i++)
  {
    RealT sum = 0;
    for (int j = -(int)round(2 * sigma); j <= (int)round(2 * sigma); j++)
    {
      int idx = 0;
      idx = (i + j) % hist.size();
      sum = sum + hist[idx] * gaussian1d(j, 0, sigma);
    }
    hist_smoothed[i] = sum;
    if (abs(hist_smoothed[i] - hist_smoothed[0]) > kFlatHistEps)
      all_zeros = false;  // check if at least one entry is non - zero
    // (a flat histogram would otherwise loop forever)
  }
  if (all_zeros) return;

  // hill-climb each bin to its local peak
  for (int i = 0; i < hist.size(); i++)
  {
    int j = i;
    while (true)
    {
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
    if (modes.size() == 0)
    {
      ys = true;
    }
    else
    {
      for (int k = 0; k < modes.size(); k++)
      {
        if (modes[k].second == j)
        {
          ys = false;
          break;
        }
      }
    }
    if (ys == true)
    {
      modes.push_back(std::make_pair(hist_smoothed[j], j));
    }
  }
  std::sort(modes.begin(), modes.end(), cmp);
}

// estimate the two dominant edge orientations at a corner
void CpuCornerDetector::estimateEdgeOrientations(Mat img_angle, Mat img_weight, int index)
{
  // orientation histogram resolution
  int num_bins = kOrientationBins;

  // flatten the ROI into angle / weight samples
  if (img_angle.size() != img_weight.size()) return;

  vector<RealT> vec_angle, vec_weight;
  for (int i = 0; i < img_angle.cols; i++)
  {
    for (int j = 0; j < img_angle.rows; j++)
    {
      // rotate normals by 90deg to get edge directions
      float angle = img_angle.ptr<RealT>(j)[i] + CV_PI / 2;
      angle = angle > CV_PI ? (angle - CV_PI) : angle;
      vec_angle.push_back(angle);

      vec_weight.push_back(img_weight.ptr<RealT>(j)[i]);
    }
  }

  // weighted orientation histogram
  RealT bin_width = (CV_PI / num_bins);
  vector<RealT> angle_hist(num_bins, 0);
  for (int i = 0; i < vec_angle.size(); i++)
  {
    int bin = max(min((int)floor(vec_angle[i] / bin_width), num_bins - 1), 0);
    angle_hist[bin] = angle_hist[bin] + vec_weight[i];
  }

  // peaks of the smoothed histogram
  vector<RealT> hist_smoothed(angle_hist);
  vector<std::pair<RealT, int>> modes;
  findHistogramModes(angle_hist, hist_smoothed, modes, 1);

  // need at least two distinct orientations
  if (modes.size() <= 1) return;

  // the two strongest peaks, ordered by angle
  float fo[2];
  fo[0] = modes[0].second * bin_width;
  fo[1] = modes[1].second * bin_width;
  RealT delta_angle = 0;
  if (fo[0] > fo[1])
  {
    RealT t = fo[0];
    fo[0] = fo[1];
    fo[1] = t;
  }

  delta_angle = MIN(fo[1] - fo[0], fo[0] - fo[1] + (RealT)CV_PI);
  // reject near-parallel edge pairs
  if (delta_angle <= kMinEdgeAngleRad) return;

  // store the two edge unit vectors
  edge_dirs1_[index][0] = cos(fo[0]);
  edge_dirs1_[index][1] = sin(fo[0]);
  edge_dirs2_[index][0] = cos(fo[1]);
  edge_dirs2_[index][1] = sin(fo[1]);
}

float CpuCornerDetector::vecNorm(cv::Point2f o) { return sqrt(o.x * o.x + o.y * o.y); }

void CpuCornerDetector::refineCornersSubpixel(
  vector<Point2f> &corners, Mat img_du, Mat img_dv, Mat img_angle, Mat img_weight, float radius
)
{
  // image dimensions
  int width = img_du.cols;
  int height = img_du.rows;

  // for all corners do
  for (int i = 0; i < corners.size(); i++)
  {
    // integer corner location
    int cu = corners[i].x;
    int cv = corners[i].y;
    // estimate edge orientations
    int start_x, start_y, ro_iwidth, ro_iheight;
    start_x = MAX(cu - radius, (RealT)0);
    start_y = MAX(cv - radius, (RealT)0);
    ro_iwidth = MIN(cu + radius + 1, (RealT)width - 1) - start_x;
    ro_iheight = MIN(cv + radius + 1, (RealT)height - 1) - start_y;

    Mat roi_angle, roi_weight;
    roi_angle = img_angle(Rect(start_x, start_y, ro_iwidth, ro_iheight));
    roi_weight = img_weight(Rect(start_x, start_y, ro_iwidth, ro_iheight));
    estimateEdgeOrientations(roi_angle, roi_weight, i);

    // skip corners with degenerate edges
    if (edge_dirs1_[i][0] == 0 && edge_dirs1_[i][1] == 0 ||
        edge_dirs2_[i][0] == 0 && edge_dirs2_[i][1] == 0)
      continue;

    cv::Mat a1 = cv::Mat::zeros(cv::Size(2, 2), kMatDepth);
    cv::Mat a2 = cv::Mat::zeros(cv::Size(2, 2), kMatDepth);

    for (int u = start_x; u < start_x + ro_iwidth; u++)
      for (int v = start_y; v < start_y + ro_iheight; v++)
      {
        // gradient direction at this pixel
        cv::Point2f o(img_du.at<RealT>(v, u), img_dv.at<RealT>(v, u));
        float no = vecNorm(o);

        if (no < kMinGradNorm) continue;
        o = o / no;
        // accumulate inliers for edge 1
        RealT t0 = abs(o.x * edge_dirs1_[i][0] + o.y * edge_dirs1_[i][1]);
        if (t0 < kInlierCos)
        {  // inlier ?
          Mat outer(1, 2, kMatDepth);
          outer.col(0).setTo(img_du.at<RealT>(v, u));
          outer.col(1).setTo(img_dv.at<RealT>(v, u));
          Mat outer_u = img_du.at<RealT>(v, u) * outer;
          Mat outer_v = img_dv.at<RealT>(v, u) * outer;
          for (int j = 0; j < a1.cols; j++)
          {
            a1.at<RealT>(0, j) = a1.at<RealT>(0, j) + outer_u.at<RealT>(0, j);
            a1.at<RealT>(1, j) = a1.at<RealT>(1, j) + outer_v.at<RealT>(0, j);
          }
        }
        // accumulate inliers for edge 2
        RealT t1 = abs(o.x * edge_dirs2_[i][0] + o.y * edge_dirs2_[i][1]);
        if (t1 < kInlierCos)
        {  // inlier ?
          Mat outer(1, 2, kMatDepth);
          outer.col(0).setTo(img_du.at<RealT>(v, u));
          outer.col(1).setTo(img_dv.at<RealT>(v, u));
          Mat outer_u = img_du.at<RealT>(v, u) * outer;
          Mat outer_v = img_dv.at<RealT>(v, u) * outer;
          for (int j = 0; j < a2.cols; j++)
          {
            a2.at<RealT>(0, j) = a2.at<RealT>(0, j) + outer_u.at<RealT>(0, j);
            a2.at<RealT>(1, j) = a2.at<RealT>(1, j) + outer_v.at<RealT>(0, j);
          }
        }
      }  // end for
    // refined orientation = smallest eigenvector of each scatter matrix
    cv::Mat v1, evec1;
    cv::Mat v2, evec2;
    cv::eigen(a1, v1, evec1);
    cv::eigen(a2, v2, evec2);
    edge_dirs1_[i][0] = -evec1.at<RealT>(1, 0);
    edge_dirs1_[i][1] = -evec1.at<RealT>(1, 1);
    edge_dirs2_[i][0] = -evec2.at<RealT>(1, 0);
    edge_dirs2_[i][1] = -evec2.at<RealT>(1, 1);

    cv::Mat g = cv::Mat::zeros(cv::Size(2, 2), kMatDepth);
    cv::Mat b = cv::Mat::zeros(cv::Size(1, 2), kMatDepth);
    for (int u = start_x; u < start_x + ro_iwidth; u++)
      for (int v = start_y; v < start_y + ro_iheight; v++)
      {
        // gradient direction at this pixel
        cv::Point2f o(img_du.at<RealT>(v, u), img_dv.at<RealT>(v, u));
        float no = vecNorm(o);
        if (no < kMinGradNorm) continue;
        o = o / no;
        // least-squares sub-pixel position from the edge constraints
        if (u != cu || v != cv)
        {  // % do not consider center pixel
          // distance of this pixel from each edge line
          cv::Point2f w(u - cu, v - cv);
          float proj1 = w.x * edge_dirs1_[i][0] + w.y * edge_dirs1_[i][1];
          float proj2 = w.x * edge_dirs2_[i][0] + w.y * edge_dirs2_[i][1];

          cv::Point2f wv1(proj1 * edge_dirs1_[i][0], proj1 * edge_dirs1_[i][1]);
          cv::Point2f wv2(proj2 * edge_dirs2_[i][0], proj2 * edge_dirs2_[i][1]);
          cv::Point2f vd1(w.x - wv1.x, w.y - wv1.y);
          cv::Point2f vd2(w.x - wv2.x, w.y - wv2.y);
          RealT d1 = vecNorm(vd1), d2 = vecNorm(vd2);
          // pixel lies on one of the two edges
          if ((d1 < kEdgeDistPx) &&
                abs(o.x * edge_dirs1_[i][0] + o.y * edge_dirs1_[i][1]) < kInlierCos ||
              (d2 < kEdgeDistPx) &&
                abs(o.x * edge_dirs2_[i][0] + o.y * edge_dirs2_[i][1]) < kInlierCos)
          {
            RealT du = img_du.at<RealT>(v, u), dv = img_dv.at<RealT>(v, u);
            cv::Mat uvt = (Mat_<RealT>(2, 1) << u, v);
            cv::Mat h = (Mat_<RealT>(2, 2) << du * du, du * dv, dv * du, dv * dv);
            g = g + h;
            cv::Mat t = h * (uvt);
            b = b + t;
          }
        }
      }  // endfor
    // solve for the sub-pixel corner when well-conditioned

    Mat s, u, v;
    SVD::compute(g, s, u, v);
    int rank = 0;
    for (int k = 0; k < s.rows; k++)
    {
      if (s.at<RealT>(k, 0) > kSvdRankEps || s.at<RealT>(k, 0) < -kSvdRankEps)
      {  // treat as non-zero
        rank++;
      }
    }
    if (rank == 2)
    {
      cv::Mat mp = g.inv() * b;
      cv::Point2f new_pos(mp.at<RealT>(0, 0), mp.at<RealT>(1, 0));
      // discard if the update jumps too far
      if (vecNorm(cv::Point2f(new_pos.x - cu, new_pos.y - cv)) >= kMaxCornerShiftPx)
      {
        edge_dirs1_[i][0] = 0;
        edge_dirs1_[i][1] = 0;
        edge_dirs2_[i][0] = 0;
        edge_dirs2_[i][1] = 0;
      }
      else
      {
        corners[i].x = mp.at<RealT>(0, 0);
        corners[i].y = mp.at<RealT>(1, 0);
      }
    }
    else
    {  // ill-conditioned: mark the corner invalid
      edge_dirs1_[i][0] = 0;
      edge_dirs1_[i][1] = 0;
      edge_dirs2_[i][0] = 0;
      edge_dirs2_[i][1] = 0;
    }
  }
}

// correlation score for one corner patch
void CpuCornerDetector::correlationScore(
  Mat img, Mat img_weight, vector<Point2f> corners_edge, float &score
)
{
  // center
  int c[] = {img_weight.cols / 2, img_weight.cols / 2};

  // gradient template mask (~3 px band)
  Mat img_filter = Mat::ones(img_weight.size(), img_weight.type());
  img_filter = img_filter * -1;
  for (int i = 0; i < img_weight.cols; i++)
  {
    for (int j = 0; j < img_weight.rows; j++)
    {
      Point2f p1 = Point2f(i - c[0], j - c[1]);
      Point2f p2 = Point2f(
        p1.x * corners_edge[0].x * corners_edge[0].x + p1.y * corners_edge[0].x * corners_edge[0].y,
        p1.x * corners_edge[0].x * corners_edge[0].y + p1.y * corners_edge[0].y * corners_edge[0].y
      );
      Point2f p3 = Point2f(
        p1.x * corners_edge[1].x * corners_edge[1].x + p1.y * corners_edge[1].x * corners_edge[1].y,
        p1.x * corners_edge[1].x * corners_edge[1].y + p1.y * corners_edge[1].y * corners_edge[1].y
      );
      float norm1 = sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
      float norm2 = sqrt((p1.x - p3.x) * (p1.x - p3.x) + (p1.y - p3.y) * (p1.y - p3.y));
      if (norm1 <= kGradientBandPx || norm2 <= kGradientBandPx)
      {
        img_filter.ptr<RealT>(j)[i] = 1;
      }
    }
  }

  // normalize
  Mat mean, std, mean1, std1;
  meanStdDev(img_weight, mean, std);
  meanStdDev(img_filter, mean1, std1);
  for (int i = 0; i < img_weight.cols; i++)
  {
    for (int j = 0; j < img_weight.rows; j++)
    {
      img_weight.ptr<RealT>(j)[i] = (RealT)(img_weight.ptr<RealT>(j)[i] - mean.ptr<double>(0)[0]) /
                                    (RealT)std.ptr<double>(0)[0];
      img_filter.ptr<RealT>(j)[i] = (RealT)(img_filter.ptr<RealT>(j)[i] - mean1.ptr<double>(0)[0]) /
                                    (RealT)std1.ptr<double>(0)[0];
    }
  }

  // convert into vectors
  vector<float> vec_filter, vec_weight;
  for (int i = 0; i < img_weight.cols; i++)
  {
    for (int j = 0; j < img_weight.rows; j++)
    {
      vec_filter.push_back(img_filter.ptr<RealT>(j)[i]);
      vec_weight.push_back(img_weight.ptr<RealT>(j)[i]);
    }
  }

  // compute gradient score
  float sum = 0;
  for (int i = 0; i < vec_weight.size(); i++)
  {
    sum += vec_weight[i] * vec_filter[i];
  }
  sum = (RealT)sum / (RealT)(vec_weight.size() - 1);
  RealT gradient_score = sum >= 0 ? sum : 0;

  // intensity quadrant kernels
  const camera_chessboard_detector::CorrelationKernelSet &kernels = score_kernel_cache_.get(
    atan2(corners_edge[0].y, corners_edge[0].x), atan2(corners_edge[1].y, corners_edge[1].x), c[0]
  );

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
void CpuCornerDetector::scoreAllCorners(
  Mat img, Mat img_angle, Mat img_weight, vector<Point2f> &corners, vector<int> radius,
  vector<float> &score
)
{
  // for all corners do
  for (int i = 0; i < corners.size(); i++)
  {
    // corner location
    int u = corners[i].x + 0.5;
    int v = corners[i].y + 0.5;

    // correlation score at each radius
    vector<float> scores;
    for (int j = 0; j < radius.size(); j++)
    {
      scores.push_back(0);
      int r = radius[j];
      if (u > r && u <= (img.cols - r - 1) && v > r && v <= (img.rows - r - 1))
      {
        int start_x, start_y, ro_iwidth, ro_iheight;
        start_x = u - r;
        start_y = v - r;
        ro_iwidth = 2 * r + 1;
        ro_iheight = 2 * r + 1;

        Mat sub_img = img(Rect(start_x, start_y, ro_iwidth, ro_iheight)).clone();
        Mat sub_img_weight = img_weight(Rect(start_x, start_y, ro_iwidth, ro_iheight)).clone();
        vector<Point2f> corners_edge;
        corners_edge.push_back(Point2f((float)edge_dirs1_[i][0], (float)edge_dirs1_[i][1]));
        corners_edge.push_back(Point2f((float)edge_dirs2_[i][0], (float)edge_dirs2_[i][1]));
        correlationScore(sub_img, sub_img_weight, corners_edge, scores[j]);
      }
    }
    // keep the best radius
    score.push_back(*max_element(begin(scores), end(scores)));
  }
}

void CpuCornerDetector::buildLikelihoodMap(cv::Mat &src, cv::Mat &dst)
{
  Mat resp_a(src.size(), kMatDepth);
  Mat resp_b(src.size(), kMatDepth);
  Mat resp_c(src.size(), kMatDepth);
  Mat resp_d(src.size(), kMatDepth);

  Mat low_a(src.size(), kMatDepth);
  Mat low_b(src.size(), kMatDepth);
  Mat cand1(src.size(), kMatDepth);
  Mat cand2(src.size(), kMatDepth);
  Mat mean_resp(src.size(), kMatDepth);

  for (int i = 0; i < 6; i++)
  {
    Mat k_a, k_b, k_c, k_d;
    buildQuadrantKernels(
      template_angles_[i].x, template_angles_[i].y, radius_[i / 2], k_a, k_b, k_c, k_d
    );

    // correlate the image with the four quadrant kernels
    resp_a = conv2(src, k_a, ConvolutionSame);
    resp_b = conv2(src, k_b, ConvolutionSame);
    resp_c = conv2(src, k_c, ConvolutionSame);
    resp_d = conv2(src, k_d, ConvolutionSame);

    // compute mean
    mean_resp = (resp_a + resp_b + resp_c + resp_d) / 4.0;

    // case 1: a = white, b = black
    elementwiseMin(resp_a - mean_resp, resp_b - mean_resp, low_a);
    elementwiseMin(mean_resp - resp_c, mean_resp - resp_d, low_b);
    elementwiseMin(low_a, low_b, cand1);
    // case 2: b = white, a = black
    elementwiseMin(mean_resp - resp_a, mean_resp - resp_b, low_a);
    elementwiseMin(resp_c - mean_resp, resp_d - mean_resp, low_b);
    elementwiseMin(low_a, low_b, cand2);

    // keep the per-pixel maximum
    elementwiseMax(dst, cand1, dst);
    elementwiseMax(dst, cand2, dst);
  }
}

void CpuCornerDetector::detect(
  Mat &src, CornerCandidates &mcorners, RealT score_threshold, RefinementOption refinement_option
)
{
  Mat gray, normalized;
  gray = Mat(src.size(), CV_8U);

  // grayscale
  if (src.channels() == 3)
  {
    cvtColor(src, gray, COLOR_BGR2GRAY);
  }
  else
  {
    gray = src.clone();
  }

  cv::GaussianBlur(gray, gray, cv::Size(kBlurKernel, kBlurKernel), kBlurSigma);

  // normalise to [0, 1]
  normalize(gray, normalized, 0, 1, cv::NORM_MINMAX, kMatDepth);

  Mat likelihood = Mat::zeros(normalized.size(), kMatDepth);

  buildLikelihoodMap(normalized, likelihood);

  // candidate corners via non-maximum suppression
  nonMaxSuppress(likelihood, corner_points_, kNmsPatchSize, kNmsThreshold, kNmsMargin);

  // post processing

  Mat grad_x(gray.size(), kMatDepth);
  Mat grad_y(gray.size(), kMatDepth);

  Mat angle_map = cv::Mat::zeros(gray.size(), kMatDepth);
  Mat weight_map = cv::Mat::zeros(gray.size(), kMatDepth);

  computeGradientOrientation(normalized, grad_x, grad_y, angle_map, weight_map);

  if (refinement_option == RefinementOption::DoRefine)
  {
    // subpixel refinement
    refineCornersSubpixel(corner_points_, grad_x, grad_y, angle_map, weight_map, kRefineRadiusPx);

    if (corner_points_.size() > 0)
    {
      for (int i = 0; i < corner_points_.size(); i++)
      {
        if (edge_dirs1_[i][0] == 0 && edge_dirs1_[i][0] == 0)
        {
          corner_points_[i].x = 0;
          corner_points_[i].y = 0;
        }
      }
    }
  }
  // drop corners whose edges were rejected

  // score corners
  vector<float> score;
  scoreAllCorners(normalized, angle_map, weight_map, corner_points_, radius_, score);

  // keep only corners above the score threshold
  int nlen = corner_points_.size();
  if (nlen > 0)
  {
    for (int i = 0; i < nlen; i++)
    {
      if (score[i] > score_threshold)
      {
        mcorners.p.push_back(corner_points_[i]);
        mcorners.v1.push_back(cv::Vec2f(edge_dirs1_[i][0], edge_dirs1_[i][1]));
        mcorners.v2.push_back(cv::Vec2f(edge_dirs2_[i][0], edge_dirs2_[i][1]));
        mcorners.score.push_back(score[i]);
      }
    }
  }

  std::vector<cv::Vec2f> corners_n1(mcorners.p.size());
  for (int i = 0; i < corners_n1.size(); i++)
  {
    if (mcorners.v1[i][0] + mcorners.v1[i][1] < 0.0)
    {
      mcorners.v1[i] = -mcorners.v1[i];
    }
    corners_n1[i] = mcorners.v1[i];
    float flipflag = corners_n1[i][0] * mcorners.v2[i][0] + corners_n1[0][1] * mcorners.v2[i][1];
    if (flipflag > 0)
      flipflag = -1;
    else
      flipflag = 1;
    mcorners.v2[i] = flipflag * mcorners.v2[i];
  }
}

}  // namespace cpu
}  // namespace camera_chessboard_detector
