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

// Board structure recovery for the libcbdetect-derived detector of
// A. Geiger et al. (KIT), "Automatic Camera and Range Sensor Calibration using
// a Single Shot", ICRA 2012. Distributed under GPL-3.0-or-later; see
// THIRD_PARTY_NOTICES.md.

#include "core.hpp"

#include <opencv2/core/types.hpp>

#include <algorithm>
#include <numeric>

namespace camera_chessboard_detector {


namespace cpu {

namespace {
// Structure-recovery tuning constants (inherent to the cbdetect growth
// formulation; typed to match the original literals so the math is unchanged).
constexpr int    kMinSeedCorners = 9;     // need a full 3x3 seed neighbourhood
constexpr float  kPerpWeight     = 5.0f;  // perpendicular-offset penalty when choosing a neighbour
constexpr double kMaxSpacingCV   = 0.3;   // reject a seed if neighbour spacing varies more than this
constexpr double kPredictStep    = 0.75;  // fraction of the extrapolated step used to place a prediction
constexpr float  kAcceptEnergy   = -8.0f; // accept a grown board once its energy drops below this
}  // namespace

inline float vdist(cv::Vec2f a, cv::Vec2f b) {
  return std::sqrt((a[0] - b[0]) * (a[0] - b[0]) +
                   (a[1] - b[1]) * (a[1] - b[1]));
}

inline float vmean(std::vector<float> &resultSet) {
  double sum = std::accumulate(std::begin(resultSet), std::end(resultSet), 0.0);
  double mean = sum / resultSet.size();
  return mean;
}

inline float vstddev(std::vector<float> &resultSet, float &mean) {
  double accum = 0.0;
  mean = vmean(resultSet);
  std::for_each(std::begin(resultSet), std::end(resultSet),
                [&](const double d) { accum += (d - mean) * (d - mean); });
  double stdev = sqrt(accum / (resultSet.size() - 1));
  return stdev;
}

inline float coeffVariation(std::vector<float> &resultSet) {
  float stdvalue, meanvalue;

  stdvalue = vstddev(resultSet, meanvalue);

  return stdvalue / meanvalue;
}

int BoardBuilder::nearestNeighborAlong(int idx, cv::Vec2f v, cv::Mat chessboard,
                                    CornerArray &corners, int &neighbor_idx,
                                    float &min_dist) {

  // indices of corners not yet placed on the board
  std::vector<int> available(corners.x.size());
  for (int i = 0; i < available.size(); i++) {
    available[i] = i;
  }
  for (int i = 0; i < chessboard.rows; i++)
    for (int j = 0; j < chessboard.cols; j++) {
      int xy = chessboard.at<int>(i, j);
      if (xy >= 0) {
        available[xy] = -1;
      }
    }

  int nsize = available.size();

  for (int i = 0; i < nsize;) {
    if (available[i] < 0) {
      std::vector<int>::iterator iter = available.begin() + i;
      available.erase(iter);
      i = 0;
      nsize = available.size();
      continue;
    }
    i++;
  }

  std::vector<float> perp_dist;
  std::vector<float> along_dist;

  cv::Vec2f origin = cv::Vec2f(corners.x[idx], corners.y[idx]);
  // signed distance along v and perpendicular offset for each free corner
  for (int i = 0; i < available.size(); i++) {
    int ind = available[i];
    cv::Vec2f delta_vec = cv::Vec2f(corners.x[ind], corners.y[ind]) - origin;
    float along = delta_vec[0] * v[0] + delta_vec[1] * v[1];

    cv::Vec2f de = delta_vec - along * v;
    perp_dist.push_back(vdist(de, cv::Vec2f(0, 0)));
    // along-axis distance
    along_dist.push_back(along);
  }

  // closest corner ahead along v (perpendicular offset penalised)
  int min_idx = 0;
  min_dist = std::numeric_limits<float>::max();

  for (int i = 0; i < along_dist.size(); i++) {
    if (along_dist[i] > 0) {
      float m = along_dist[i] + kPerpWeight * perp_dist[i];
      if (m < min_dist) {
        min_dist = m;
        min_idx = i;
      }
    }
  }
  neighbor_idx = available[min_idx];

  return 1;
}

cv::Mat BoardBuilder::seedBoard(CornerArray &corners, int idx) {
  // need at least a 3x3 neighbourhood
  if (corners.x.size() < kMinSeedCorners) {
    std::cerr << "not enough corners!\n";
    chessboard.release(); // return empty!
    return chessboard;
  }
  // start from a 3x3 hypothesis centred on idx
  chessboard = -1 * cv::Mat::ones(3, 3, CV_32S);

  // the two edge directions of the central corner
  cv::Vec2f v1 = cv::Vec2f(corners.edge1_cos[idx], corners.edge1_sin[idx]);
  cv::Vec2f v2 = cv::Vec2f(corners.edge2_cos[idx], corners.edge2_sin[idx]);
  chessboard.at<int>(1, 1) = idx;
  std::vector<float> dist1(2), dist2(6);

  // place the four edge neighbours
  nearestNeighborAlong(idx, +1 * v1, chessboard, corners,
                      chessboard.at<int>(1, 2), dist1[0]);
  nearestNeighborAlong(idx, -1 * v1, chessboard, corners,
                      chessboard.at<int>(1, 0), dist1[1]);
  nearestNeighborAlong(idx, +1 * v2, chessboard, corners,
                      chessboard.at<int>(2, 1), dist2[0]);
  nearestNeighborAlong(idx, -1 * v2, chessboard, corners,
                      chessboard.at<int>(0, 1), dist2[1]);

  // place the four diagonal neighbours

  nearestNeighborAlong(chessboard.at<int>(1, 0), -1 * v2, chessboard, corners,
                      chessboard.at<int>(0, 0), dist2[2]);
  nearestNeighborAlong(chessboard.at<int>(1, 0), +1 * v2, chessboard, corners,
                      chessboard.at<int>(2, 0), dist2[3]);
  nearestNeighborAlong(chessboard.at<int>(1, 2), -1 * v2, chessboard, corners,
                      chessboard.at<int>(0, 2), dist2[4]);
  nearestNeighborAlong(chessboard.at<int>(1, 2), +1 * v2, chessboard, corners,
                      chessboard.at<int>(2, 2), dist2[5]);

  // keep the seed only if the spacing is roughly uniform

  bool reject = false;
  reject = reject || (dist1[0] < 0) || (dist1[1] < 0);
  reject = reject || (dist2[0] < 0) || (dist2[1] < 0) || (dist2[2] < 0) ||
           (dist2[3] < 0) || (dist2[4] < 0) || (dist2[5] < 0);

  reject = reject || (coeffVariation(dist1) > kMaxSpacingCV) ||
           (coeffVariation(dist2) > kMaxSpacingCV);

  chessboard = chessboard.t();

  if (reject == true) {
    chessboard.release();
    return chessboard;
  }
  return chessboard;
}

float BoardBuilder::boardEnergy(cv::Mat chessboard, CornerArray &corners) {
  float lambda = lambda_;
  // term 1: reward for board size
  float energy_corners = -1 * chessboard.size().area();
  // term 2: penalty for non-collinear triples
  float energy_structure = 0;
  // scan each row triple
  for (int i = 0; i < chessboard.rows; i++)
    for (int j = 0; j < chessboard.cols - 2; j++) {
      std::vector<cv::Vec2f> x;
      float triple_energy = 0;
      for (int k = j; k <= j + 2; k++) {
        int n = chessboard.at<int>(i, k);
        x.push_back(cv::Vec2f(corners.x[n], corners.y[n]));
      }
      triple_energy = vdist(x[0] + x[2] - 2 * x[1], cv::Vec2f(0, 0));
      float span = vdist(x[0] - x[2], cv::Vec2f(0, 0));
      triple_energy = triple_energy / span;
      if (energy_structure < triple_energy)
        energy_structure = triple_energy;
    }

  // scan each column triple
  for (int i = 0; i < chessboard.cols; i++)
    for (int j = 0; j < chessboard.rows - 2; j++) {
      std::vector<cv::Vec2f> x;
      float triple_energy = 0;
      for (int k = j; k <= j + 2; k++) {
        int n = chessboard.at<int>(k, i);
        x.push_back(cv::Vec2f(corners.x[n], corners.y[n]));
      }
      triple_energy = vdist(x[0] + x[2] - 2 * x[1], cv::Vec2f(0, 0));
      float span = vdist(x[0] - x[2], cv::Vec2f(0, 0));
      triple_energy = triple_energy / span;
      if (energy_structure < triple_energy)
        energy_structure = triple_energy;
    }

  // combined energy
  float E = energy_corners + lambda * chessboard.size().area() * energy_structure;

  return E;
}

// extrapolate the next corner from the last three along a line
void BoardBuilder::predictNextCorners(std::vector<cv::Vec2f> &p1,
                                std::vector<cv::Vec2f> &p2,
                                std::vector<cv::Vec2f> &p3,
                                std::vector<cv::Vec2f> &predicted) {
  cv::Vec2f v1, v2;
  float a1, a2, a3;
  float s1, s2, s3;
  predicted.resize(p1.size());
  for (int i = 0; i < p1.size(); i++) {
    // step vectors
    v1 = p2[i] - p1[i];
    v2 = p3[i] - p2[i];
    // extrapolate the turn angle
    a1 = atan2(v1[1], v1[0]);
    a2 = atan2(v2[1], v2[0]);
    a3 = 2.0 * a2 - a1;

    // extrapolate the step length
    s1 = vdist(v1, cv::Vec2f(0, 0));
    s2 = vdist(v2, cv::Vec2f(0, 0));
    s3 = 2 * s2 - s1;
    predicted[i] = p3[i] + kPredictStep * s3 * cv::Vec2f(cos(a3), sin(a3));
  }
}

void BoardBuilder::matchClosestCandidates(std::vector<cv::Vec2f> &candidates,
                                      std::vector<cv::Vec2f> &predicted,
                                      std::vector<int> &idx) {
  // fail if there are fewer candidates than predictions
  if (candidates.size() < predicted.size()) {
    idx.resize(1);
    idx[0] = -1;
    return;
  }
  idx.resize(predicted.size());

  // candidate-to-prediction distance matrix
  cv::Mat D = cv::Mat::zeros(candidates.size(), predicted.size(), CV_32FC1);
  float mind = FLT_MAX;
  for (int i = 0; i < D.cols; i++)
  {
    cv::Vec2f delta;
    for (int j = 0; j < D.rows; j++) {
      delta = candidates[j] - predicted[i];
      float s = vdist(delta, cv::Vec2f(0, 0));
      D.at<float>(j, i) = s;
      if (s < mind) {
        mind = s;
      }
    }
  }


  // greedily match the nearest candidate to each prediction
  for (int k = 0; k < predicted.size(); k++) {
    bool isbreak = false;
    for (int i = 0; i < D.rows; i++) {
      for (int j = 0; j < D.cols; j++) {
        if (fabs(D.at<float>(i, j) - mind) < 10e-10) {
          idx[j] = i;
          for (int m = 0; m < D.cols; m++) {
            D.at<float>(i, m) = FLT_MAX;
          }
          for (int m = 0; m < D.rows; m++) {
            D.at<float>(m, j) = FLT_MAX;
          }
          isbreak = true;
          break;
        }
      }
      if (isbreak == true)
        break;
    }
    mind = FLT_MAX;
    for (int i = 0; i < D.rows; i++) {
      for (int j = 0; j < D.cols; j++) {
        if (D.at<float>(i, j) < mind) {
          mind = D.at<float>(i, j);
        }
      }
    }
  }
}

cv::Mat BoardBuilder::growBoard(cv::Mat chessboard, CornerArray &corners,
                                   int border_type) {
  if (chessboard.empty() == true) {
    return chessboard;
  }
  std::vector<int> available(corners.x.size());
  for (int i = 0; i < available.size(); i++) {
    available[i] = i;
  }
  for (int i = 0; i < chessboard.rows; i++)
    for (int j = 0; j < chessboard.cols; j++) {
      int xy = chessboard.at<int>(i, j);
      if (xy >= 0) {
        available[xy] = -1;
      }
    }

  int nsize = available.size();

  for (int i = 0; i < nsize;) {
    if (available[i] < 0) {
      std::vector<int>::iterator iter = available.begin() + i;
      available.erase(iter);
      i = 0;
      nsize = available.size();
      continue;
    }
    i++;
  }
  // free corners are the growth candidates
  std::vector<cv::Vec2f> candidates;
  for (int i = 0; i < available.size(); i++) {
    candidates.push_back(cv::Vec2f(corners.x[available[i]], corners.y[available[i]]));
  }
  // grow on one of the four borders
  cv::Mat grown;

  switch (border_type) {
  case 0: {
    std::vector<cv::Vec2f> p1, p2, p3, predicted;
    for (int row = 0; row < chessboard.rows; row++)
      for (int col = 0; col < chessboard.cols; col++) {
        if (col == chessboard.cols - 3) {
          p1.push_back(cv::Point2f(corners.x[chessboard.at<int>(row, col)],
                                 corners.y[chessboard.at<int>(row, col)]));
        }
        if (col == chessboard.cols - 2) {
          p2.push_back(cv::Point2f(corners.x[chessboard.at<int>(row, col)],
                                 corners.y[chessboard.at<int>(row, col)]));
        }
        if (col == chessboard.cols - 1) {
          p3.push_back(cv::Point2f(corners.x[chessboard.at<int>(row, col)],
                                 corners.y[chessboard.at<int>(row, col)]));
        }
      }
    std::vector<int> idx;
    predictNextCorners(p1, p2, p3, predicted);
    matchClosestCandidates(candidates, predicted, idx);
    if (idx[0] < 0) {
      return chessboard;
    }

    cv::copyMakeBorder(chessboard, grown, 0, 0, 0, 1, 0, 0);

    for (int i = 0; i < grown.rows; i++) {
      grown.at<int>(i, grown.cols - 1) = available[idx[i]];
    }
    chessboard = grown.clone();

    break;
  }
  case 1: {
    std::vector<cv::Vec2f> p1, p2, p3, predicted;
    for (int row = 0; row < chessboard.rows; row++)
      for (int col = 0; col < chessboard.cols; col++) {
        if (row == chessboard.rows - 3) {
          p1.push_back(cv::Point2f(corners.x[chessboard.at<int>(row, col)],
                                 corners.y[chessboard.at<int>(row, col)]));
        }
        if (row == chessboard.rows - 2) {
          p2.push_back(cv::Point2f(corners.x[chessboard.at<int>(row, col)],
                                 corners.y[chessboard.at<int>(row, col)]));
        }
        if (row == chessboard.rows - 1) {
          p3.push_back(cv::Point2f(corners.x[chessboard.at<int>(row, col)],
                                 corners.y[chessboard.at<int>(row, col)]));
        }
      }
    std::vector<int> idx;
    predictNextCorners(p1, p2, p3, predicted);
    matchClosestCandidates(candidates, predicted, idx);
    if (idx[0] < 0) {
      return chessboard;
    }

    cv::copyMakeBorder(chessboard, grown, 0, 1, 0, 0, 0, 0);
    for (int i = 0; i < grown.cols; i++) {
      grown.at<int>(grown.rows - 1, i) = available[idx[i]];
    }
    chessboard = grown.clone();

    break;
  }
  case 2: {
    std::vector<cv::Vec2f> p1, p2, p3, predicted;
    for (int row = 0; row < chessboard.rows; row++)
      for (int col = 0; col < chessboard.cols; col++) {
        if (col == 2) {
          p1.push_back(cv::Point2f(corners.x[chessboard.at<int>(row, col)],
                                 corners.y[chessboard.at<int>(row, col)]));
        }
        if (col == 1) {
          p2.push_back(cv::Point2f(corners.x[chessboard.at<int>(row, col)],
                                 corners.y[chessboard.at<int>(row, col)]));
        }
        if (col == 0) {
          p3.push_back(cv::Point2f(corners.x[chessboard.at<int>(row, col)],
                                 corners.y[chessboard.at<int>(row, col)]));
        }
      }
    std::vector<int> idx;
    predictNextCorners(p1, p2, p3, predicted);
    matchClosestCandidates(candidates, predicted, idx);
    if (idx[0] < 0) {
      return chessboard;
    }

    cv::copyMakeBorder(chessboard, grown, 0, 0, 1, 0, 0, 0);
    for (int i = 0; i < grown.rows; i++) {
      grown.at<int>(i, 0) = available[idx[i]];
    }
    chessboard = grown.clone();

    break;
  }
  case 3: {
    std::vector<cv::Vec2f> p1, p2, p3, predicted;
    for (int row = 0; row < chessboard.rows; row++)
      for (int col = 0; col < chessboard.cols; col++) {
        if (row == 2) {
          p1.push_back(cv::Point2f(corners.x[chessboard.at<int>(row, col)],
                                 corners.y[chessboard.at<int>(row, col)]));
        }
        if (row == 1) {
          p2.push_back(cv::Point2f(corners.x[chessboard.at<int>(row, col)],
                                 corners.y[chessboard.at<int>(row, col)]));
        }
        if (row == 0) {
          p3.push_back(cv::Point2f(corners.x[chessboard.at<int>(row, col)],
                                 corners.y[chessboard.at<int>(row, col)]));
        }
      }
    std::vector<int> idx;
    predictNextCorners(p1, p2, p3, predicted);
    matchClosestCandidates(candidates, predicted, idx);
    if (idx[0] < 0) {
      return chessboard;
    }
    cv::copyMakeBorder(chessboard, grown, 1, 0, 0, 0, 0, 0);
    for (int i = 0; i < grown.cols; i++) {
      grown.at<int>(0, i) = available[idx[i]];
    }
    chessboard = grown.clone();
    break;
  }
  default:
    break;
  }
  return chessboard;
}

void BoardBuilder::assembleBoards(CornerArray &corners,
                                        std::vector<cv::Mat> &chessboards,
                                        float lambda) {
  if (verbose_logging_) {
    std::cerr << "Structure recovery:\n";
  }
  lambda_ = lambda;
  for (int i = 0; i < corners.x.size(); i++) {
    cv::Mat seed = seedBoard(corners, i);
    if (seed.empty() == true) {
      continue;
    }
    float E = boardEnergy(seed, corners);
    if (E > 0) {
      continue;
    }



    int s = 0;
    // grow until the energy stops improving
    while (true) {
      s++;
      // current energy
      float energy = boardEnergy(chessboard, corners);


      std::vector<cv::Mat> proposal(4);
      std::vector<float> proposal_energy(4);
      // energy of each of the four growth proposals
      for (int j = 0; j < 4; j++) {
        proposal[j] = growBoard(chessboard, corners, j);
        proposal_energy[j] = boardEnergy(proposal[j], corners);
      }
      // lowest-energy proposal
      float min_value = proposal_energy[0];
      int min_idx = 0;
      for (int i0 = 1; i0 < proposal_energy.size(); i0++) {
        if (min_value > proposal_energy[i0]) {
          min_value = proposal_energy[i0];
          min_idx = i0;
        }
      }
      // accept it only if it lowers the energy
      cv::Mat best_proposal;
      if (proposal_energy[min_idx] < energy) {
        best_proposal = proposal[min_idx];
        chessboard = best_proposal.clone();
      } else {
        break;
      }
    }

    if (boardEnergy(chessboard, corners) < kAcceptEnergy) {
      // does this board share any corner with an accepted one?
      cv::Mat overlap =
          cv::Mat::zeros(cv::Size(2, chessboards.size()), CV_32FC1);
      for (int j = 0; j < chessboards.size(); j++) {
        bool isbreak = false;
        for (int k = 0; k < chessboards[j].size().area(); k++) {
          int existing_cell = chessboards[j].at<int>(k / chessboards[j].cols,
                                            k % chessboards[j].cols);
          for (int l = 0; l < chessboard.size().area(); l++) {
            int new_cell =
                chessboard.at<int>(l / chessboard.cols, l % chessboard.cols);
            if (existing_cell == new_cell) {
              overlap.at<float>(j, 0) = 1.0;
              float s = boardEnergy(chessboards[j], corners);
              overlap.at<float>(j, 1) = s;
              isbreak = true;
              break;
            }
          }
        }

      }

      // add it, replacing any lower-energy overlapping boards
      bool isoverlap = false;
      for (int i0 = 0; i0 < overlap.rows; i0++) {
        if (overlap.empty() == false) {
          if (fabs(overlap.at<float>(i0, 0)) > 0.000001) // ==1
          {
            isoverlap = true;
            break;
          }
        }
      }
      if (isoverlap == false) {
        chessboards.push_back(chessboard);
      } else {
        bool keep_new = true;
        std::vector<bool> erase_flags(overlap.rows);
        for (int m = 0; m < erase_flags.size(); m++) {
          erase_flags[m] = false;
        }
        float new_energy = boardEnergy(chessboard, corners);
        for (int i1 = 0; i1 < overlap.rows; i1++) {
          if (fabs(overlap.at<float>(i1, 0)) > 0.0001) {

            int a = int(overlap.at<float>(i1, 1) * 1000);
            int b = int(new_energy * 1000);

            bool isb2 = a > b;

            if (isb2) {
              erase_flags[i1] = true;
            } else {
              keep_new = false;
            }
          }

        }

        if (keep_new == true) {
          for (int i1 = 0; i1 < chessboards.size();) {
            std::vector<cv::Mat>::iterator it = chessboards.begin() + i1;
            std::vector<bool>::iterator it1 = erase_flags.begin() + i1;
            if (*it1 == true) {
              chessboards.erase(it);
              erase_flags.erase(it1);
              i1 = 0;
            }
            i1++;
          }
          chessboards.push_back(chessboard);
        }

      }

    }
  }
}


};

} // namespace camera_chessboard_detector
