#include "cuda/kernels.hpp"
#include "cuda/gpu_buffers.hpp"

#include <cassert>
#include <math.h>

////////////////////////////////////////////////////////////////////////////////
#include <opencv2/core.hpp>
// #include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
// #include <xmmintrin.h>
////////////////////////////////////////////////////////////////////////////////

template <typename T>
__global__ void padBorder_k(T *dst, const T *src, int rows, int cols, int pad) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < cols && j < rows) {
    int index = j * cols + i;
    int dst_index = (j + pad) * (cols + 2 * pad) + i + pad;
    dst[dst_index] = src[index];
  }
}

__global__ void conv2dShared(float *output, const float *input, const float *kernel,
                                  int H, int W, int S) {
#ifdef CONV_SHARED
  extern __shared__  float smem[];
  float *data = smem;
#endif
  int R = S >> 1;
  int R2 = R << 1;
  int S2 = S << 1;
  int i = 2 * threadIdx.x + __mul24(blockIdx.x, blockDim.x) * 2 + R;
  // int i = threadIdx.x + __mul24(blockIdx.x, blockDim.x) + R;
  int j = threadIdx.y + __mul24(blockIdx.y, blockDim.y) + R;
  int gindex = __mul24(j, W) + i;
#if CONV_SHARED
#if 0
  int block_width = 2 * blockDim.x + R2;
  int block_height = blockDim.y + R2;
  // int block_loc = threadIdx.x + __mul24(threadIdx.y, block_width) / 2;
  int block_loc = (2 * threadIdx.x + __mul24(threadIdx.y, block_width)) / 2;
  int RW = __mul24(R, W);
#else
	if (threadIdx.x <= R && threadIdx.y <= R2) {
		data[threadIdx.x + __mul24(threadIdx.y, S2)] = kernel[threadIdx.x + __mul24(threadIdx.y, S2)];
		data[threadIdx.x + __mul24(threadIdx.y, S2) + R] = kernel[threadIdx.x + __mul24(threadIdx.y, S2) + R];
		data[threadIdx.x + __mul24(threadIdx.y, S2) + S] = kernel[threadIdx.x + __mul24(threadIdx.y, S2) + S];
		data[threadIdx.x + __mul24(threadIdx.y, S2) + S + R] = kernel[threadIdx.x + __mul24(threadIdx.y, S2) + R + S];
	}
	__syncthreads();
#endif
#endif

  if (i >= R && i < W - R && j >= R && j < H - R) {
    // Image-tile staging in shared memory was scaffolded here by the
    // original authors (shared_size in convolve() still reserves the tile
    // halo) but never wired up; the inner loop below reads `input` from
    // global memory directly. See the design notes for the
    // optimisation attempts that explored re-enabling this and why they
    // did not improve performance on this Jetson.

    int x, y;
    x = 2 * threadIdx.x;
    y = threadIdx.y;
    float2 sum = make_float2(0, 0);
    // float sum = 0;
#if 1
    for (int r = -R; r <= R; ++r) {
        // for (int s = -R; s <= R; ++s) {
        //     sum += input[gindex + r * W + s]
        //         * kernel[__mul24(r + R, S2) + s + R];
        // }
      for (int s = -R; s <= R - 1 ; s += 2) {
				#ifdef CONV_SHARED
          // float2 data1 = reinterpret_cast<float2 *>(data)[(__mul24((y + r + R), block_width) + x + s + R) / 2];
          // float2 kernel1 = reinterpret_cast<const float2 *>(kernel)[(__mul24(r + R, S2) + s + R) / 2];
          float2 data1 = reinterpret_cast<const float2 *>(input)[(gindex + r * W + s) / 2];
          float2 kernel1 = reinterpret_cast<const float2 *>(data)[(__mul24(r + R, S2) + s + R) / 2];
				#else
          float2 data1 = reinterpret_cast<const float2 *>(input)[(gindex + r * W + s) / 2];
          float2 kernel1 = reinterpret_cast<const float2 *>(kernel)[(__mul24(r + R, S2) + s + R) / 2];
				#endif
          // sum += data1.x * kernel1.x;
          // sum += data1.y * kernel1.y;
          // sum = fma(data1.x, kernel1.x, sum);
          // sum = fma(data1.y, kernel1.y, sum);
          sum.x = fma(data1.x, kernel1.x, sum.x);
          sum.x = fma(data1.y, kernel1.y, sum.x);
				#ifdef CONV_SHARED
          // float2 data2 = reinterpret_cast<float2 *>(data)[(__mul24((y + r + R), block_width) + x + s + R) / 2 + 1];
          // float2 kernel2 = reinterpret_cast<const float2 *>(kernel)[(__mul24(r + R, S2) + s + R) / 2 + 1 + R];
          float2 data2 = reinterpret_cast<const float2 *>(input)[(gindex + r * W + s) / 2 + 1];
          float2 kernel2 = reinterpret_cast<const float2 *>(data)[(__mul24(r + R, S2) + s + R) / 2 + 1 + R];
				#else
          float2 data2 = reinterpret_cast<const float2 *>(input)[(gindex + r * W + s) / 2 + 1];
          float2 kernel2 = reinterpret_cast<const float2 *>(kernel)[(__mul24(r + R, S2) + s + R) / 2 + 1 + R];
				#endif
          sum.y = fma(data2.x, kernel2.x, sum.y);
          sum.y = fma(data2.y, kernel2.y, sum.y);
      }
			#ifdef CONV_SHARED
      // sum.x += data[__mul24((y + r + R), block_width) + x + R2]
      //     * kernel[__mul24(r + R, S2) + R2];
      // sum.y += data[__mul24((y + r + R), block_width) + x + 1]
      //     * kernel[__mul24(r + R, S2)];
      sum.x += input[gindex + r * W + R]
          * data[__mul24(r + R, S2) + R2];
      sum.y += input[gindex + r * W - R + 1]
          * data[__mul24(r + R, S2)];
			#else
      sum.x += input[gindex + r * W + R]
          * kernel[__mul24(r + R, S2) + R2];
      sum.y += input[gindex + r * W - R + 1]
          * kernel[__mul24(r + R, S2)];
			#endif
    }
#else
    for (int r = -R; r <= R; ++r) {
      for (int s = -R; s <= R - 1 ; s += 2) {
          // float2 data2 = reinterpret_cast<float2 *>(data)[(__mul24((y + r + R), block_width) + x + s + R) / 2];
          float2 data2 = reinterpret_cast<const float2 *>(input)[(gindex + r * W + s) / 2];
          float2 kernel2 = reinterpret_cast<const float2 *>(kernel)[(__mul24(r + R, S2) + s + R) / 2];
          sum.x = fma(data2.x, kernel2.x, sum.x);
          sum.x = fma(data2.y, kernel2.y, sum.x);
      }
      // sum.x += data[__mul24((y + r + R), block_width) + x + R2]
      //     * kernel[__mul24(r + R, S) + R2];
      sum.x += input[gindex + r * W + R]
          * kernel[__mul24(r + R, S) + R2];
    }

    x = 2 * threadIdx.x + 1;
    for (int r = -R; r <= R; ++r) {
      for (int s = -R + 1; s <= R - 1 ; s += 2) {
          // float2 data2 = reinterpret_cast<float2 *>(data)[(__mul24((y + r + R), block_width) + x + s + R) / 2];
          float2 data2 = reinterpret_cast<const float2 *>(input)[(gindex + r * W + s) / 2];
          float2 kernel2 = reinterpret_cast<const float2 *>(kernel)[(S + __mul24(r + R, S2) + s + R) / 2];
          sum.y = fma(data2.x, kernel2.x, sum.y);
          sum.y = fma(data2.y, kernel2.y, sum.y);
      }
      // sum.y += data[__mul24((y + r + R), block_width) + x]
      //     * kernel[__mul24(r + R, S)];
      sum.y += input[gindex + r * W - R + 1]
          * kernel[__mul24(r + R, S)];
    }
#endif
    reinterpret_cast<float2*>(output)[gindex >> 1] = sum;
    // output[gindex] = sum;
  }
}

template <typename T>
__global__ void nonMaxSuppress_k(T *dst, const T *src, int rows, int cols, int R,
                          int stride, int pad) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < cols && j < rows) {
    int index = j * cols + i;
    int input_index = (j * stride + pad) * (cols + 2 * pad) + i * stride + pad;
    T max = 0;
    for (int r = 0; r < R; ++r) {
      for (int s = 0; s < R; ++s) {
        max = src[input_index + r * (cols + 2 * pad) + s] > max
                  ? src[input_index + r * (cols + 2 * pad) + s]
                  : max;
      }
    }
    dst[index] = max;
  }
}

template <typename T>
__global__ void min_k(T *dst, const T *src1, const T *src2, int rows,
                      int cols) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < rows && j < cols) {
    int index = i * cols + j;
    dst[index] = min(src1[index], src2[index]);
  }
}

template <typename T>
__global__ void min4_k(T *dst, const T *src1, const T *src2, const T *src3,
                       const T *src4, int rows, int cols) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < rows && j < cols) {
    int index = i * cols + j;
    dst[index] =
        min(min(src1[index], src2[index]), min(src3[index], src4[index]));
  }
}

template <typename T>
__global__ void max_k(T *dst, const T *src1, const T *src2, int rows,
                      int cols) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < rows && j < cols) {
    int index = i * cols + j;
    dst[index] = max(src1[index], src2[index]);
  }
}

template <typename T>
__global__ void sub_k(T *dst, const T *src1, const T *src2, int rows,
                      int cols) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < rows && j < cols) {
    int index = i * cols + j;
    dst[index] = src1[index] - src2[index];
  }
}

template <typename T>
__global__ void neg_k(T *dst, const T *src, int rows, int cols) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < rows && j < cols) {
    int index = i * cols + j;
    dst[index] = -src[index];
  }
}

template <typename T>
__global__ void mean4_k(T *dst, const T *src1, const T *src2, const T *src3,
                        const T *src4, int rows, int cols) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < rows && j < cols) {
    int index = i * cols + j;
    dst[index] = (src1[index] + src2[index] + src3[index] + src4[index]) / 4;
  }
}

template <typename T>
__global__ void combineLikelihood_k(T *dst,
                           const T *A, const T *B,
                           const T *C, const T *D,
                           int rows, int cols) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < rows && j < cols) {
    int index = __mul24(i, cols) + j;
#if 0
    T a = A[index];
    float mean = a;
    T b = B[index];
    __fadd_rz(mean, b);
    T c = C[index];
    __fadd_rz(mean, c);
    T d = D[index];
    __fadd_rz(mean, d);
    __fmul_rz(mean, 0.25f);
#elif 0
    float mean = 0;
    T a = A[index];
    mean += a;
    T b = B[index];
    mean += b;
    T c = C[index];
    mean += c;
    T d = D[index];
    mean += d;
    mean *= 0.25f;
    float a2 = a - mean;
    float b2 = b - mean;
    float c2 = mean - c;
    float d2 = mean - d;

    float min1 = min(min(a2, b2), min(c2, d2));
    float min2 = -max(max(a2, b2), max(c2, d2));
    float max1 = max(min1, min2);
#else
    float mean = 0;
    T a = A[index];
    mean += a;
    T b = B[index];
    mean += b;
    float a2 = min(a, b);
    float b2 = max(a, b);
    T c = C[index];
    mean += c;
    T d = D[index];
    mean += d;
    mean *= 0.25f;
    float c2 = min(c, d);
    float d2 = max(c, d);

    float min1 = min(a2 - mean, mean - d2);
    float min2 = -max(b2 - mean, mean - c2);
    float max1 = max(min1, min2);
#endif
    // float a2 = a - mean;
    // float b2 = b - mean;
    // float c2 = mean - c;
    // float d2 = mean - d;

    // float min1 = min(min(a2, b2), min(c2, d2));
    // float min2 = -max(max(a2, b2), max(c2, d2));
    // float max1 = max(min1, min2);
    dst[index] = (T)max(max1, 0.0f);
    // dst[index] = (T)(a2 * a2 + b2 * b2 + c2 * c2 + d2 * d2);
  }
}


template <typename T>
__global__ void bgr_to_gray_interleaved_k(T *dst, const T *src, int width,
                                          int height) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < height && j < width) {
    int index = i * width + j;
		float3 bgr = reinterpret_cast<const float3 *>(src)[index];
		dst[index] = 0.299 * bgr.z + 0.587 * bgr.y + 0.114 * bgr.x;
		// dst[index] = (bgr.x + bgr.y + bgr.z) / 3;
  }
}

template <typename T>
__global__ void bgr_to_gray_planar_k(T *dst, const T *src, int width,
                                     int height) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < height && j < width) {
    int b_index = i * width + j;
    int g_index = b_index + width * height;
    int r_index = g_index + width * height;
    dst[b_index] =
        0.299 * src[r_index] + 0.587 * src[g_index] + 0.114 * src[b_index];
  }
}

template <typename T>
__global__ void gradientWeight_k(T *dst, const T *gx, const T *gy, int rows, int cols) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < rows && j < cols) {
    int index = i * cols + j;
    dst[index] = sqrtf(gx[index] * gx[index] + gy[index] * gy[index]);
  }
}

template <typename T>
__global__ void gaussian9x9_k(T *dst, const T *src, float sigma,
														int rows, int cols) {
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < rows && j < cols) {
		int index = i * cols + j;
		T sum = 0;
		for (int r = -4; r <= 4; ++r) {
			for (int s = -4; s <= 4; ++s) {
				int x_index = j + s;
				int y_index = i + r;
				if (x_index >= 0 && x_index < cols && y_index >= 0 && y_index < rows) {
					T value = src[y_index * cols + x_index];
					T weight = expf(-(r * r + s * s) / (2 * sigma * sigma));
					sum += value * weight;
				}
			}
		}
		dst[index] = sum;
	}
}

// Finds the single minimum value in the array
template <typename T>
__global__ void minimum_k(T *dst, const T *src, int size) {
	__shared__ T sdata[256];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	sdata[tid] = src[i];
	__syncthreads();
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			sdata[tid] = min(sdata[tid], sdata[tid + s]);
		}
		__syncthreads();
	}
	if (tid == 0) {
		dst[blockIdx.x] = sdata[0];
	}
}

template <typename T>
__global__ void maximum_k(T *dst, const T *src, int size) {
	__shared__ T sdata[256];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	sdata[tid] = src[i];
	__syncthreads();
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			sdata[tid] = max(sdata[tid], sdata[tid + s]);
		}
		__syncthreads();
	}
	if (tid == 0) {
		dst[blockIdx.x] = sdata[0];
	}
}

template <typename T>
__global__ void normalize_k(float *dst, const T *src, T min, T max, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		dst[i] = (src[i] - min) / (max - min);
	}
}

template <typename T> struct Point {
  T x;
  T y;
  __device__ Point(T x, T y) : x(x), y(y) {}
};

template <typename T> __device__ Point<T> smallestEigenvector(T *A) {
  T a = A[0];
  T b = A[1];
  T c = A[2];
  T d = A[3];

  T D2 = (a - d) * (a - d) + 4 * b * c;
  if (D2 <= 0)
    return Point<T>(-1, 0);

  T D = sqrtf(D2);
  T l1 = (a + d - D) / 2;

  // if (fabsf(a - l1) > 1e-6) {
  //   float v2 = -b / (a - l1);
  //   float inorm = rsqrtf(1 + v2 * v2);
  //   return Point<T>(v2 * inorm, inorm);
  // } else {
  //   float v1 = -c / (d - l1);
  //   float inorm = rsqrtf(1 + v1 * v1);
  //   return Point<T>(inorm, v1 * inorm);
  // }
  if (fabsf(a - l1) < 1e-6) {
      return Point<T>(-1, 0);
  }

  float v = b / (a - l1);
  float inorm = rsqrtf(1 + v * v);
  return Point<T>(v * inorm, -inorm);
}

template <typename T>
__device__ bool smoothHistogram(T *bins, int bin_size, float sigma) {
  T *smoothed = new T[bin_size];
  for (int i = 0; i < bin_size; ++i) {
    smoothed[i] = 0;
    for (int j = -(int)round(2 * sigma); j <= (int)round(2 * sigma); ++j) {
      int index = i + j;
      if (index < 0)
        index += bin_size;
      else if (index >= bin_size)
        index -= bin_size;
      smoothed[i] += bins[index] * exp(-(j * j) / (2 * sigma * sigma));
    }
  }

  // TODO: Check this
  bool nonzero = false;
  for (int i = 0; i < bin_size; ++i) {
    bins[i] = smoothed[i];
    if (fabsf(bins[i] - bins[0]) > 0.0001)
      nonzero = true;
  }

  delete[] smoothed;
  return nonzero;
}

template <typename T>
__global__ void cornerEdges_k(T *corners_edge1_sin, T *corners_edge1_cos,
                                   T *corners_edge2_sin, T *corners_edge2_cos,
                                   int num_corners, const T *gx, const T *gy,
                                   const T *corners_x, const T *corners_y,
                                   int radius, int width, int height,
                                   bool refine) {
#define BIN_SIZE 32

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_corners) {
    T x = corners_x[i];
    T y = corners_y[i];
    T sin1 = 0;
    T cos1 = 0;
    T sin2 = 0;
    T cos2 = 0;

    T bins[BIN_SIZE] = {0};
    float pin = M_PI / BIN_SIZE;
    for (int r = -radius; r <= radius; ++r) {
      for (int s = -radius; s <= radius; ++s) {
        int x_index = x + s;
        int y_index = y + r;
        if (x_index >= 0 && x_index < width && y_index >= 0 &&
            y_index < height) {
          T ga = atan2(gy[y_index * width + x_index],
                       gx[y_index * width + x_index]);
          T gw = gx[y_index * width + x_index] * gx[y_index * width + x_index] +
                 gy[y_index * width + x_index] * gy[y_index * width + x_index];

#if 1
          if (ga > M_PI) ga -= M_PI;
          else if (ga < 0) ga += M_PI;
          ga += M_PI / 2;
          if (ga > M_PI) ga -= M_PI;
          int bin = max(min((int)(ga / pin), BIN_SIZE - 1), 0);
#else
          // int bin = (int)((ga + M_PI) * 16 / M_PI); // TODO: Check this
          int bin = max(min((int)((ga + M_PI) * 16 / M_PI), 31), 0);
#endif
          bins[bin] += sqrt(gw); // sqrt?
        }
      }
    }

    bool nonzero = smoothHistogram(bins, BIN_SIZE, 1);
    if (nonzero) {
      // float angle1 = 0;
      // float angle2 = 0;
      // FIXME: Check this
      // for (int i = 1; i < 32; ++i) {
      //   if (bins[i] > bins[max_bin])
      //     max_bin = i;
      // }
#if 0
      int max_bin1 = 0;
      int max_bin2 = 0;
      for (int i = 1; i < 32; ++i) {
        if (bins[i] > bins[max_bin1])
          max_bin1 = i;
      }
      for (int i = 1; i < 32; ++i) {
        if (i == max_bin1) continue;
        if (bins[i] > bins[max_bin2])
          max_bin2 = i;
      }
#else
      int max_bin1 = -1;
      int max_bin2 = -1;
      for (int i = 0; i < BIN_SIZE; ++i) {
        int j = i;
        while (true) {
          T h0 = bins[j];
          int j1 = (j + 1) % BIN_SIZE;
          int j2 = (j + BIN_SIZE - 1) % BIN_SIZE;
          T h1 = bins[j1];
          T h2 = bins[j2];
          if (h1 >= h0 && h1 >= h2) {
            j = j1;
          } else if (h2 > h0 && h2 > h1) {
            j = j2;
          } else
            break;
        }
#if 0
        if (max_bin1 != j) {
          max_bin2 = max_bin1;
          max_bin1 = j;
        }
#else
        // printf("===>");
        if (j == max_bin1 || j == max_bin2) {
            // printf("0\n");
            continue;
        }

        if (max_bin1 < 0) {
            max_bin1 = j;
            // printf("1\n");
            // printf("max_bin1 = %d (%f)", j, bins[j]);
            continue;
        }
        if (max_bin2 < 0) {
            max_bin2 = max_bin1;
            max_bin1 = j;
            if (bins[max_bin1] < bins[max_bin2]) {
                max_bin1 = max_bin2;
                max_bin2 = j;
            }
            // printf("2\n");
            // printf("max_bin1 = %d (%f)", max_bin1, bins[max_bin1]);
            // printf("max_bin2 = %d (%f)", max_bin2, bins[max_bin2]);
            continue;
        }
        if (bins[j] > bins[max_bin1]) {
            max_bin2 = max_bin1;
            max_bin1 = j;
            // printf("3\n");
            // printf("max_bin1 = %d (%f)", max_bin1, bins[max_bin1]);
            // printf("max_bin2 = %d (%f)", max_bin2, bins[max_bin2]);
        } else if (bins[j] > bins[max_bin2]) {
            max_bin2 = j;
            // printf("4\n");
            // printf("max_bin2 = %d (%f)", max_bin2, bins[max_bin2]);
        }
#endif
      }
#endif

      if (max_bin1 > max_bin2) {
        int tmp = max_bin1;
        max_bin1 = max_bin2;
        max_bin2 = tmp;
      }

      if (max_bin2 - max_bin1 > 1) {
        // sin1 += sin(max_bin1 * M_PI / 16 + M_PI / 2);
        // cos1 += cos(max_bin1 * M_PI / 16 + M_PI / 2);
        // sin2 += sin(max_bin2 * M_PI / 16 + M_PI / 2);
        // cos2 += cos(max_bin2 * M_PI / 16 + M_PI / 2);

        sin1 = sin(max_bin1 * pin);
        cos1 = cos(max_bin1 * pin);
        sin2 = sin(max_bin2 * pin);
        cos2 = cos(max_bin2 * pin);

        // printf("x = %f, y = %f, max_bin1 = %d, max_bin2 = %d, bins[max_bin1] = %f, bins[max_bin2] = %f, cos1 = %f, cos2 = %f\n",
        //        x, y, max_bin1, max_bin2, bins[max_bin1], bins[max_bin2], cos1, cos2);
      }
    }
    // corners_edge1_sin[i] = sin2;
    // corners_edge1_cos[i] = cos2;
    // corners_edge2_sin[i] = sin1;
    // corners_edge2_cos[i] = cos1;
    corners_edge1_sin[i] = sin1;
    corners_edge1_cos[i] = cos1;
    corners_edge2_sin[i] = sin2;
    corners_edge2_cos[i] = cos2;

    if (!refine)
      return;

    T A1[4] = {0};
    T A2[4] = {0};
    for (int r = -radius; r <= radius; ++r) {
      for (int s = -radius; s <= radius; ++s) {
        int x_index = x + s;
        int y_index = y + r;
        if (x_index >= 0 && x_index < width && y_index >= 0 &&
            y_index < height) {
          T du = gx[y_index * width + x_index];
          T dv = gy[y_index * width + x_index];
          T gw = sqrt(du * du + dv * dv);
          T ox = du / gw;
          T oy = dv / gw;
          float t0 = fabsf(ox * cos1 + oy * sin1);
          if (t0 < 0.25) {
            A1[0] += du * du;
            A1[1] += du * dv;
            A1[2] += du * dv;
            A1[3] += dv * dv;
          }

          float t1 = fabsf(ox * cos2 + oy * sin2);
          if (t1 < 0.25) {
            A2[0] += du * du;
            A2[1] += du * dv;
            A2[2] += du * dv;
            A2[3] += dv * dv;
          }
        }
      }
    }

    Point<T> e1 = smallestEigenvector<T>(A1);
    Point<T> e2 = smallestEigenvector<T>(A2);

    corners_edge1_cos[i] = e1.x;
    corners_edge1_sin[i] = e1.y;
    corners_edge2_cos[i] = e2.x;
    corners_edge2_sin[i] = e2.y;

    // float tan2a = 2 * A1[1] / (A1[0] - A1[3]);
    // float tan2b = 2 * A2[1] / (A2[0] - A2[3]);
    // float a = atan2f(tan2a, 1);
    // float b = atan2f(tan2b, 1);
    // corners_edge1_sin[i] = sin(a);
    // corners_edge1_cos[i] = cos(a);
    // corners_edge2_sin[i] = sin(b);
    // corners_edge2_cos[i] = cos(b);

    // float sin2a = tan2a / sqrtf(1 + tan2a * tan2a);
    // float cos2a = 1 / sqrtf(1 + tan2a * tan2a);
    // float sin2b = tan2b / sqrtf(1 + tan2b * tan2b);
    // float cos2b = 1 / sqrtf(1 + tan2b * tan2b);
    //
    // corners_edge1_sin[i] = sin2a;
    // corners_edge1_cos[i] = cos2a;
    // corners_edge2_sin[i] = sin2b;
    // corners_edge2_cos[i] = cos2b;
  }
}

#if 0
template <typename T>
__global__ void cornerEdges_k(T *corners_edge1_sin, T *corners_edge1_cos,
                                   T *corners_edge2_sin, T *corners_edge2_cos,
                                   int num_corners, const T *gx, const T *gy,
                                   const T *corners_x, const T *corners_y,
                                   int radius, int width, int height,
                                   bool refine) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_corners) {
    T x = corners_x[i];
    T y = corners_y[i];
    T sin1 = 0;
    T cos1 = 0;
    T sin2 = 0;
    T cos2 = 0;

    T bins[32] = {0};
    T pin = M_PI / 32;
    for (int r = -radius; r <= radius; ++r) {
      for (int s = -radius; s <= radius; ++s) {
        int x_index = x + s;
        int y_index = y + r;
        if (x_index >= 0 && x_index < width && y_index >= 0 &&
            y_index < height) {
          T ga = atan2(gy[y_index * width + x_index],
                       gx[y_index * width + x_index]);
          T gw = gx[y_index * width + x_index] * gx[y_index * width + x_index] +
                 gy[y_index * width + x_index] * gy[y_index * width + x_index];

#if 1
          if (ga > M_PI) ga -= M_PI;
          else if (ga < 0) ga += M_PI;
          int bin = max(min((int)(ga / pin), 31), 0);
#else
          // int bin = (int)((ga + M_PI) * 16 / M_PI); // TODO: Check this
          int bin = max(min((int)((ga + M_PI) * 16 / M_PI), 31), 0);
#endif
          bins[bin] += sqrt(gw); // sqrt?
        }
      }
    }

    bool nonzero = smoothHistogram(bins, 32, 1);
    if (nonzero) {
      int max_bin1 = -1;
      int max_bin2 = -1;
      float angle1 = 0;
      float angle2 = 0;
      // FIXME: Check this
      // for (int i = 1; i < 32; ++i) {
      //   if (bins[i] > bins[max_bin])
      //     max_bin = i;
      // }
      for (int i = 0; i < 32; ++i) {
        int j = i;
        while (true) {
          T h0 = bins[j];
          int j1 = (j + 1) % 32;
          int j2 = (j + 31) % 32;
          T h1 = bins[j1];
          T h2 = bins[j2];
          if (h1 >= h0 && h1 >= h2) {
            j = j1;
          } else if (h2 > h0 && h2 > h1) {
            j = j2;
          } else
            break;
        }
#if 1
        if (max_bin1 != j) {
          max_bin2 = max_bin1;
          max_bin1 = j;
        }
#else
        if (max_bin1 < 0) max_bin1 = j;
        if (max_bin2 < 0) {
            if (max_bin1 != j) {
                max_bin2 = max_bin1;
                max_bin1 = j;
            }
            if (bins[max_bin1] < bins[max_bin2]) {
                max_bin1 = max_bin2;
                max_bin2 = j;
            }
        }
        if (bins[j] > bins[max_bin1]) {
            max_bin2 = max_bin1;
            max_bin1 = j;
        } else if (bins[j] > bins[max_bin2]) {
            max_bin2 = j;
        }
#endif
      }

      if (abs(max_bin2 - max_bin1) > 1) {
        // sin1 += sin(max_bin1 * M_PI / 16 - M_PI / 2);
        // cos1 += cos(max_bin1 * M_PI / 16 - M_PI / 2);
        // sin2 += sin(max_bin2 * M_PI / 16 - M_PI / 2);
        // cos2 += cos(max_bin2 * M_PI / 16 - M_PI / 2);
        sin1 += sin(max_bin1 * pin);
        cos1 += cos(max_bin1 * pin);
        sin2 += sin(max_bin2 * pin);
        cos2 += cos(max_bin2 * pin);
      }
    }
#if 1
    corners_edge1_sin[i] = sin2;
    corners_edge1_cos[i] = cos2;
    corners_edge2_sin[i] = sin1;
    corners_edge2_cos[i] = cos1;
#else
    corners_edge1_sin[i] = sin1;
    corners_edge1_cos[i] = cos1;
    corners_edge2_sin[i] = sin2;
    corners_edge2_cos[i] = cos2;
#endif

    if (!refine)
      return;

    T A1[4] = {0};
    T A2[4] = {0};
    for (int r = -radius; r <= radius; ++r) {
      for (int s = -radius; s <= radius; ++s) {
        int x_index = x + s;
        int y_index = y + r;
        if (x_index >= 0 && x_index < width && y_index >= 0 &&
            y_index < height) {
          T du = gx[y_index * width + x_index];
          T dv = gy[y_index * width + x_index];
          T gw = sqrt(du * du + dv * dv);
          T ox = du / gw;
          T oy = dv / gw;
          float t0 = fabsf(ox * sin1 + oy * cos1);
          if (t0 < 0.25) {
            A1[0] += du * du;
            A1[1] += du * dv;
            A1[2] += du * dv;
            A1[3] += dv * dv;
          }

          float t1 = fabsf(ox * sin2 + oy * cos2);
          if (t1 < 0.25) {
            A2[0] += du * du;
            A2[1] += du * dv;
            A2[2] += du * dv;
            A2[3] += dv * dv;
          }
        }
      }
    }

    Point<T> e1 = smallestEigenvector<T>(A1);
    Point<T> e2 = smallestEigenvector<T>(A2);

    corners_edge1_cos[i] = e1.x;
    corners_edge1_sin[i] = e1.y;
    corners_edge2_cos[i] = e2.x;
    corners_edge2_sin[i] = e2.y;

    // float tan2a = 2 * A1[1] / (A1[0] - A1[3]);
    // float tan2b = 2 * A2[1] / (A2[0] - A2[3]);
    // float a = atan2f(tan2a, 1);
    // float b = atan2f(tan2b, 1);
    // corners_edge1_sin[i] = sin(a);
    // corners_edge1_cos[i] = cos(a);
    // corners_edge2_sin[i] = sin(b);
    // corners_edge2_cos[i] = cos(b);

    // float sin2a = tan2a / sqrtf(1 + tan2a * tan2a);
    // float cos2a = 1 / sqrtf(1 + tan2a * tan2a);
    // float sin2b = tan2b / sqrtf(1 + tan2b * tan2b);
    // float cos2b = 1 / sqrtf(1 + tan2b * tan2b);
    //
    // corners_edge1_sin[i] = sin2a;
    // corners_edge1_cos[i] = cos2a;
    // corners_edge2_sin[i] = sin2b;
    // corners_edge2_cos[i] = cos2b;
  }
}
#endif

template <typename T>
__global__ void
refine_corners_k(T *corners_x, T *corners_y, int num_corners,
                 const T *corners_edge1_sin, const T *corners_edge1_cos,
                 const T *corners_edge2_sin, const T *corners_edge2_cos,
                 const T *gx, const T *gy, int radius, int width, int height) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_corners) {
    T x = corners_x[i];
    T y = corners_y[i];
    T sin1 = corners_edge1_sin[i];
    T cos1 = corners_edge1_cos[i];
    T sin2 = corners_edge2_sin[i];
    T cos2 = corners_edge2_cos[i];

    T G[4] = {0};
    T b[2] = {0};
    for (int r = -radius; r <= radius; ++r) {
      for (int s = -radius; s <= radius; ++s) {
        int x_index = x + s;
        int y_index = y + r;
        if (s == 0 && r == 0)
            continue;

        if (x_index >= 0 && x_index < width && y_index >= 0 &&
            y_index < height ) {
          float wvv1 = s * cos1 + r * sin1;
          float wvv2 = s * cos2 + r * sin2;

          Point<T> wv1 = Point<T>(wvv1 * cos1, wvv1 * sin1);
          Point<T> wv2 = Point<T>(wvv2 * cos2, wvv2 * sin2);
          Point<T> vd1 = Point<T>(s - wv1.x, r - wv1.y);
          Point<T> vd2 = Point<T>(s - wv2.x, r - wv2.y);
          float d1 = sqrtf(vd1.x * vd1.x + vd1.y * vd1.y);
          float d2 = sqrtf(vd2.x * vd2.x + vd2.y * vd2.y);

          T du = gx[y_index * width + x_index];
          T dv = gy[y_index * width + x_index];
          T gw = sqrt(du * du + dv * dv);
          T ox = du / gw;
          T oy = dv / gw;

          if (((d1 < 3) && (fabs(ox * cos1 + oy * sin1) < 0.25)) ||
              ((d2 < 3) && (fabs(ox * cos2 + oy * sin2) < 0.25))) {
            G[0] += du * du;
            G[1] += du * dv;
            G[2] += du * dv;
            G[3] += dv * dv;
            b[0] += x_index * du * du + y_index * du * dv;
            b[1] += x_index * du * dv + y_index * dv * dv;
          }
        }
      }
    }

    T det = G[0] * G[3] - G[1] * G[2];
    // int rank = 2;
    // if (G[0] < 1e-6 && G[3] < 1e-6 && G[1] < 1e-6 && G[2] < 1e-6) {
    //   rank = 0;
    // } else if (det < 1e-6) {
    //   rank = 1;
    // }

    if (fabsf(det) > 1e-6) {
      T new_x = (G[3] * b[0] - G[1] * b[1]) / det;
      T new_y = (G[0] * b[1] - G[2] * b[0]) / det;

      if ((new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) < 16) {
#ifndef NDEBUG
        printf("refined: %f %f -> %f %f\n", x, y, new_x, new_y);
#endif
        corners_x[i] = new_x;
        corners_y[i] = new_y;
      } else {
#ifndef NDEBUG
        printf("not refined: %f %f -> %f %f\n", x, y, new_x, new_y);
#endif
        corners_x[i] = 0;
        corners_y[i] = 0;
      }
    } else {
#ifndef NDEBUG
      printf("not refined: %f %f\n", x, y);
#endif
      corners_x[i] = 0;
      corners_y[i] = 0;
    }
  }
}

__device__ float gaussianPdf(float x, float sigma) {
	return (1 / (sigma * sqrt(2 * M_PI))) * exp(-0.5 * x * x / (sigma * sigma));
}

#if 0
/*
 * Load a kernel size (< 32x32) block of threads for each corner
 * each thread will compute the template, A, B, C, D filter values
 * corresponding to its location in the source image.
 * Afterwards, reduce the sum of the arrays (dot product) to obtain
 * the final score.
 */
__global__ void intensityScore_k(float *score, const float *img,
														 const float *gx, const float *gy,
														 const float *x, const float *y,
														 const float *sin1, const float *cos1,
														 const float *sin2, const float *cos2,
														 int radius, int num_corners,
														 int width, int height) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int corner = blockIdx.z;

	__shared__ float s_img[32][32];
	__shared__ float s_w[32][32];
	__shared__ float s_mw[32][32];
	__shared__ float s_sw[32][32];
	__shared__ float s_template[32][32];
	__shared__ float s_mtemplate[32][32];
	__shared__ float s_stemplate[32][32];
	__shared__ float s_A[32][32];
	__shared__ float s_B[32][32];
	__shared__ float s_C[32][32];
	__shared__ float s_D[32][32];
	__shared__ float s_mA[32][32];
	__shared__ float s_mB[32][32];
	__shared__ float s_mC[32][32];
	__shared__ float s_mD[32][32];
	__shared__ float s_sin1[1];
	__shared__ float s_cos1[1];
	__shared__ float s_sin2[1];
	__shared__ float s_cos2[1];

	if (threadIdx.x < 32 && threadIdx.y < 32) {
		int x_index = x[corner] + threadIdx.x - 16;
		int y_index = y[corner] + threadIdx.y - 16;
		if (x_index >= 0 && x_index < width && y_index >= 0 && y_index < height) {
			s_img[threadIdx.y][threadIdx.x] = img[y_index * width + x_index];
			s_w[threadIdx.y][threadIdx.x] = sqrt(
				gx[y_index * width + x_index] * gx[y_index * width + x_index] +
				gy[y_index * width + x_index] * gy[y_index * width + x_index]);
		} else {
			s_img[threadIdx.y][threadIdx.x] = 0;
			s_w[threadIdx.y][threadIdx.x] = 0;
		}

		if (threadIdx.x == 0 && threadIdx.y == 0) {
			s_sin1[0] = sin1[corner];
			s_cos1[0] = cos1[corner];
			s_sin2[0] = sin2[corner];
			s_cos2[0] = cos2[corner];
		}

		s_w[threadIdx.y][threadIdx.x] = 0;
		s_mw[threadIdx.y][threadIdx.x] = 0;
		s_sw[threadIdx.y][threadIdx.x] = 0;
		s_template[threadIdx.y][threadIdx.x] = 0;
		s_mtemplate[threadIdx.y][threadIdx.x] = 0;
		s_stemplate[threadIdx.y][threadIdx.x] = 0;
		s_A[threadIdx.y][threadIdx.x] = 0;
		s_B[threadIdx.y][threadIdx.x] = 0;
		s_C[threadIdx.y][threadIdx.x] = 0;
		s_D[threadIdx.y][threadIdx.x] = 0;
		s_mA[threadIdx.y][threadIdx.x] = 0;
		s_mB[threadIdx.y][threadIdx.x] = 0;
		s_mC[threadIdx.y][threadIdx.x] = 0;
		s_mD[threadIdx.y][threadIdx.x] = 0;
	}
	__syncthreads();

	if (i < 32 && j < 32) {
		float y1 = s_sin1[0];
		float x1 = s_cos1[0];
		float y2 = s_sin2[0];
		float x2 = s_cos2[0];
		float t = 0;
		float A = 0;
		float B = 0;
		float C = 0;
		float D = 0;
		float r = threadIdx.x - 16;
		float s = threadIdx.y - 16;

		float t1x = r * x1 * x1 + s * y1 * x1;
		float t1y = r * y1 * x1 + s * y1 * y1;
		float t2x = r * x2 * x2 + s * y2 * x2;
		float t2y = r * y2 * x2 + s * y2 * y2;
		float norm1 = sqrtf(t1x * t1x + t1y * t1y);
		float norm2 = sqrtf(t2x * t2x + t2y * t2y);

		if (norm1 < 1.5 && norm2 < 1.5) {
			s_template[threadIdx.y][threadIdx.x] = 1;
		}

		float s1 = s * x1 - r * y1;
		float s2 = s * x2 - r * y2;
		float dist = sqrtf(r * r + s * s);
#if 1
		if (s1 <= -0.1 && s2 <= -0.1) {
			A = gaussianPdf(dist, 1);
			s_A[threadIdx.y][threadIdx.x] = A;
		}
		if (s1 >= 0.1 && s2 >= 0.1) {
			B = gaussianPdf(dist, 1);
			s_B[threadIdx.y][threadIdx.x] = B;
		}
		if (s1 <= -0.1 && s2 >= 0.1) {
			C = gaussianPdf(dist, 1);
			s_C[threadIdx.y][threadIdx.x] = C;
		}
		if (s1 >= 0.1 && s2 <= -0.1) {
			D = gaussianPdf(dist, 1);
			s_D[threadIdx.y][threadIdx.x] = D;
		}
#else
		if (s1 <= -0.1 && s2 <= -0.1) {
			s_loc[threadIdx.y][threadIdx.x] = 0;
		}
		if (s1 >= 0.1 && s2 >= 0.1) {
			s_loc[threadIdx.y][threadIdx.x] = 1;
		}
		if (s1 <= -0.1 && s2 >= 0.1) {
			s_loc[threadIdx.y][threadIdx.x] = 2;
		}
		if (s1 >= 0.1 && s2 <= -0.1) {
			s_loc[threadIdx.y][threadIdx.x] = 3;
		}
		s_K[threadIdx.y][threadIdx.x] = gaussianPdf(dist, 1);
#endif
	}
	__syncthreads();

	// Normalization

	unsigned mask = __ballot_sync(0xFFFFFFFF, threadIdx.x < 32 && threadIdx.y < 32);
	unsigned a_mask = __ballot_sync(0xFFFFFFFF, s_loc[threadIdx.y][threadIdx.x] == 0);
	unsigned b_mask = __ballot_sync(0xFFFFFFFF, s_loc[threadIdx.y][threadIdx.x] == 1);
	unsigned c_mask = __ballot_sync(0xFFFFFFFF, s_loc[threadIdx.y][threadIdx.x] == 2);
	unsigned d_mask = __ballot_sync(0xFFFFFFFF, s_loc[threadIdx.y][threadIdx.x] == 3);
	for (int offset = 16; offset > 0; offset /= 2) {
		float t = __shfl_down_sync(mask, s_template[threadIdx.y][threadIdx.x], offset);
		float w = __shfl_down_sync(mask, s_w[threadIdx.y][threadIdx.x], offset);
		float A = __shfl_down_sync(a_mask, s_K[threadIdx.y][threadIdx.x], offset);
		float B = __shfl_down_sync(b_mask, s_K[threadIdx.y][threadIdx.x], offset);
		float C = __shfl_down_sync(c_mask, s_K[threadIdx.y][threadIdx.x], offset);
		float D = __shfl_down_sync(d_mask, s_K[threadIdx.y][threadIdx.x], offset);
		// float K = __shfl_down_sync(mask, s_K[threadIdx.y][threadIdx.x], offset);
		s_template[threadIdx.y][threadIdx.x] += t;
		// s_mA[i][j] += A;
		// s_mB[i][j] += B;
		// s_mC[i][j] += C;
		// s_mD[i][j] += D;
		s_mw[threadIdx.y][threadIdx.x] += w;
		if (s_loc[threadIdx.y][threadIdx.x] == 0) {
			s_mA[threadIdx.y][threadIdx.x] += A;
		}
		if (s_loc[threadIdx.y][threadIdx.x] == 1) {
			s_mB[threadIdx.y][threadIdx.x] += B;
		}
		if (s_loc[threadIdx.y][threadIdx.x] == 2) {
			s_mC[threadIdx.y][threadIdx.x] += C;
		}
		if (s_loc[threadIdx.y][threadIdx.x] == 3) {
			s_mD[threadIdx.y][threadIdx.x] += D;
		}
	}
	__syncthreads();
	float mean_template = s_template[31][31] / 1024;
	float sum_A = s_mA[31][31];
	float sum_B = s_mB[31][31];
	float sum_C = s_mC[31][31];
	float sum_D = s_mD[31][31];
	float mean_w = s_mw[31][31] / 1024;
	if (threadIdx.x < 32 && threadIdx.y < 32) {
		s_stemplate[threadIdx.y][threadIdx.x] =
			(s_template[threadIdx.y][threadIdx.x] - mean_template)
			* (s_template[i][j] - mean_template);
		s_sw[threadIdx.y][threadIdx.x] =
			(s_w[threadIdx.y][threadIdx.x] - mean_w)
			* (s_w[threadIdx.y][threadIdx.x] - mean_w);
	}
	__syncthreads();
	for (int offset = 16; offset > 0; offset /= 2) {
		float st = __shfl_down_sync(mask, s_stemplate[threadIdx.y][threadIdx.x], offset);
		float sw = __shfl_down_sync(mask, s_sw[threadIdx.y][threadIdx.x], offset);
		s_stemplate[threadIdx.y][threadIdx.x] += st;
		s_sw[threadIdx.y][threadIdx.x] += sw;
	}
	__syncthreads();
	float std_template = sqrtf(s_stemplate[31][31]) / 1024;
	float std_w = sqrtf(s_sw[31][31]) / 1024;

	if (threadIdx.x < 32 && threadIdx.y < 32) {
		s_template[threadIdx.y][threadIdx.x] =
			(s_template[threadIdx.y][threadIdx.x] - mean_template) / std_template;
		s_w[threadIdx.y][threadIdx.x] =
			(s_w[threadIdx.y][threadIdx.x] - mean_w) / std_w;
		// s_A[threadIdx.y][threadIdx.x] = s_mA[threadIdx.y][threadIdx.x] / sum_A;
		// s_B[threadIdx.y][threadIdx.x] = s_mB[threadIdx.y][threadIdx.x] / sum_B;
		// s_C[threadIdx.y][threadIdx.x] = s_mC[threadIdx.y][threadIdx.x] / sum_C;
		// s_D[threadIdx.y][threadIdx.x] = s_mD[threadIdx.y][threadIdx.x] / sum_D;
		if (s_loc[threadIdx.y][threadIdx.x] == 0) {
			s_K[threadIdx.y][threadIdx.x] = s_mA[threadIdx.y][threadIdx.x] / sum_A;
		} else if (s_loc[threadIdx.y][threadIdx.x] == 1) {
			s_K[threadIdx.y][threadIdx.x] = s_mB[threadIdx.y][threadIdx.x] / sum_B;
		} else if (s_loc[threadIdx.y][threadIdx.x] == 2) {
			s_K[threadIdx.y][threadIdx.x] = s_mC[threadIdx.y][threadIdx.x] / sum_C;
		} else if (s_loc[threadIdx.y][threadIdx.x] == 3) {
			s_K[threadIdx.y][threadIdx.x] = s_mD[threadIdx.y][threadIdx.x] / sum_D;
		}
	}
	__syncthreads();

	// Compute the dot products
	if (threadIdx.x < 32 && threadIdx.y < 32) {
		s_template[threadIdx.y][threadIdx.x] =
			s_template[threadIdx.y][threadIdx.x] * s_w[threadIdx.y][threadIdx.x];
		// s_A[threadIdx.y][threadIdx.x] =
		// 	s_A[threadIdx.y][threadIdx.x] * s_img[threadIdx.y][threadIdx.x];
		// s_B[threadIdx.y][threadIdx.x] =
		// 	s_B[threadIdx.y][threadIdx.x] * s_img[threadIdx.y][threadIdx.x];
		// s_C[threadIdx.y][threadIdx.x] =
		// 	s_C[threadIdx.y][threadIdx.x] * s_img[threadIdx.y][threadIdx.x];
		// s_D[threadIdx.y][threadIdx.x] =
		// 	s_D[threadIdx.y][threadIdx.x] * s_img[threadIdx.y][threadIdx.x];
		s_K[threadIdx.y][threadIdx.x] =
			s_K[threadIdx.y][threadIdx.x] * s_img[threadIdx.y][threadIdx.x];
	}
	__syncthreads();
	for (int offset = 16; offset > 0; offset /= 2) {
		float t = __shfl_down_sync(mask, s_template[threadIdx.y][threadIdx.x], offset);
		// float A = __shfl_down_sync(mask, s_A[threadIdx.y][threadIdx.x], offset);
		// float B = __shfl_down_sync(mask, s_B[threadIdx.y][threadIdx.x], offset);
		// float C = __shfl_down_sync(mask, s_C[threadIdx.y][threadIdx.x], offset);
		// float D = __shfl_down_sync(mask, s_D[threadIdx.y][threadIdx.x], offset);
		float A = __shfl_down_sync(a_mask, s_K[threadIdx.y][threadIdx.x], offset);
		float B = __shfl_down_sync(b_mask, s_K[threadIdx.y][threadIdx.x], offset);
		float C = __shfl_down_sync(c_mask, s_K[threadIdx.y][threadIdx.x], offset);
		float D = __shfl_down_sync(d_mask, s_K[threadIdx.y][threadIdx.x], offset);
		s_template[threadIdx.y][threadIdx.x] += t;
		// s_A[threadIdx.y][threadIdx.x] += A;
		// s_B[threadIdx.y][threadIdx.x] += B;
		// s_C[threadIdx.y][threadIdx.x] += C;
		// s_D[threadIdx.y][threadIdx.x] += D;
		if (s_loc[threadIdx.y][threadIdx.x] == 0) {
			s_K[threadIdx.y][threadIdx.x] += A;
		} else if (s_loc[threadIdx.y][threadIdx.x] == 1) {
			s_K[threadIdx.y][threadIdx.x] += B;
		} else if (s_loc[threadIdx.y][threadIdx.x] == 2) {
			s_K[threadIdx.y][threadIdx.x] += C;
		} else if (s_loc[threadIdx.y][threadIdx.x] == 3) {
			s_K[threadIdx.y][threadIdx.x] += D;
		}
	}
	__syncthreads();
	if (threadIdx.x == 0 && threadIdx.y == 0) {
		float score_gradient = s_template[0][0];
		float a = s_A[0][0];
		float b = s_B[0][0];
		float c = s_C[0][0];
		float d = s_D[0][0];
		float mu = (a + b + c + d) / 4;
		float score_a = (a - mu) >= (b - mu) ? (b - mu) : (a - mu);
		float score_b = (mu - c) >= (mu - d) ? (mu - d) : (mu - c);
		float score_1 = (score_a >= score_b) ? score_b : score_a;
		score_a = (mu - a) >= (mu - b) ? (mu - b) : (mu - a);
		score_b = (c - mu) >= (d - mu) ? (d - mu) : (c - mu);
		float score_2 = (score_a >= score_b) ? score_b : score_a;
		float score_intensity = (score_1 >= score_2) ? score_1 : score_2;
		score_intensity = score_intensity > 0 ? score_intensity : 0;
		score[corner] = score_intensity * score_gradient;
	}
}
#endif

__global__ void gradientScore_k(float *score,
															 const float *gx, const float *gy,
															 const float *x, const float *y,
															 const float *sin1, const float *cos1,
															 const float *sin2, const float *cos2,
															 int radius, int num_corners,
															 int width, int height) {
#if 0
	int corner = blockIdx.z;
	int roi_size = 2 * radius + 1;
	float iroisize2 = 1 / ((2 * radius + 1) * (2 * radius + 1));

	constexpr int kBlockSize = 32 * 32;
	constexpr int kNumWarps = kBlockSize / 32;

	__shared__ float s_w[32][32];
	__shared__ float s_template[32][32];
	__shared__ float s_mtemplate[kNumWarps];
	__shared__ float s_mw[kNumWarps];
	__shared__ float s_sin1[1];
	__shared__ float s_cos1[1];
	__shared__ float s_sin2[1];
	__shared__ float s_cos2[1];

	if (threadIdx.x < roi_size && threadIdx.y < roi_size) {
		int x_index = x[corner] + threadIdx.x - radius;
		int y_index = y[corner] + threadIdx.y - radius;
		if (x_index >= 0 && x_index < width && y_index >= 0 && y_index < height) {
			s_w[threadIdx.y][threadIdx.x] = sqrt(
				gx[y_index * width + x_index] * gx[y_index * width + x_index] +
				gy[y_index * width + x_index] * gy[y_index * width + x_index]);
		} else {
			s_w[threadIdx.y][threadIdx.x] = 0;
		}

		if (threadIdx.x == 0 && threadIdx.y == 0) {
			s_sin1[0] = sin1[corner];
			s_cos1[0] = cos1[corner];
			s_sin2[0] = sin2[corner];
			s_cos2[0] = cos2[corner];
		}

		s_template[threadIdx.y][threadIdx.x] = 0;
	}
	__syncthreads();

	if (threadIdx.x < roi_size && threadIdx.y < roi_size) {
		float y1 = s_sin1[0];
		float x1 = s_cos1[0];
		float y2 = s_sin2[0];
		float x2 = s_cos2[0];
		float r = threadIdx.x - 16;
		float s = threadIdx.y - 16;

		float t1x = r * x1 * x1 + s * y1 * x1;
		float t1y = r * y1 * x1 + s * y1 * y1;
		float t2x = r * x2 * x2 + s * y2 * x2;
		float t2y = r * y2 * x2 + s * y2 * y2;
		float norm1 = sqrtf(t1x * t1x + t1y * t1y);
		float norm2 = sqrtf(t2x * t2x + t2y * t2y);

		if (norm1 < 1.5 && norm2 < 1.5) {
			s_template[threadIdx.y][threadIdx.x] = 1;
		}
	}
	__syncthreads();

	// Normalization
	unsigned mask = __ballot_sync(0xFFFFFFFF,
																threadIdx.x < roi_size &&
																threadIdx.y < roi_size);
	float mean_t = s_template[threadIdx.y][threadIdx.x];
	float mean_w = s_w[threadIdx.y][threadIdx.x];
	for (int offset = 16; offset > 0; offset /= 2) {
		mean_t += __shfl_down_sync(mask, mean_t, offset);
		mean_w += __shfl_down_sync(mask, mean_w, offset);
	}
	__syncthreads();
	s_mtemplate[threadIdx.y / 32] = mean_t;
	s_mw[threadIdx.y / 32] = mean_w;
	for (int offset = 16; offset > 0; offset /= 2) {
		s_mtemplate[threadIdx.y / 32] += __shfl_down_sync(mask, s_mtemplate[threadIdx.y / 32], offset);
		s_mw[threadIdx.y / 32] += __shfl_down_sync(mask, s_mw[threadIdx.y / 32], offset);
	}
	__syncthreads();
	if (threadIdx.x < roi_size && threadIdx.y < roi_size) {
		s_template[threadIdx.y][threadIdx.x] = s_template[threadIdx.y][threadIdx.x] - s_mtemplate[0];
		s_w[threadIdx.y][threadIdx.x] = s_w[threadIdx.y][threadIdx.x] - s_mw[0];
	}
	mean_t = s_mtemplate[0] * iroisize2;
	mean_w = s_mw[0] * iroisize2;
	float std_t = s_template[threadIdx.y][threadIdx.x] - mean_t;
	float std_w = s_w[threadIdx.y][threadIdx.x] - mean_w;
	std_t = std_t * std_t;
	std_w = std_w * std_w;
	for (int offset = 16; offset > 0; offset /= 2) {
		float t = __shfl_down_sync(mask, std_t, offset);
		float w = __shfl_down_sync(mask, std_w, offset);
		std_t += t * t;
		std_w += w * w;
	}
	__syncthreads();
	s_mtemplate[threadIdx.y / 32] = std_t;
	s_mw[threadIdx.y / 32] = std_w;
	for (int offset = 16; offset > 0; offset /= 2) {
		s_mtemplate[threadIdx.y / 32] += __shfl_down_sync(mask, s_mtemplate[threadIdx.y / 32], offset);
		s_mw[threadIdx.y / 32] += __shfl_down_sync(mask, s_mw[threadIdx.y / 32], offset);
	}
	__syncthreads();
	std_t = sqrtf(s_mtemplate[0] * iroisize2);
	std_w = sqrtf(s_mw[0] * iroisize2);
	if (threadIdx.x < 32 && threadIdx.y < 32) {
		s_template[threadIdx.y][threadIdx.x] = (s_template[threadIdx.y][threadIdx.x] - mean_t) / std_t;
		s_w[threadIdx.y][threadIdx.x] = (s_w[threadIdx.y][threadIdx.x] - mean_w) / std_w;
	}
	__syncthreads();

	// Compute the dot products
	if (threadIdx.x < roi_size && threadIdx.y < roi_size) {
		s_template[threadIdx.y][threadIdx.x] =
			s_template[threadIdx.y][threadIdx.x] * s_w[threadIdx.y][threadIdx.x];
	}
	__syncthreads();
	float t = s_template[threadIdx.y][threadIdx.x];
	for (int offset = 16; offset > 0; offset /= 2) {
		t = __shfl_down_sync(mask, t, offset);
	}
	__syncthreads();
	s_mtemplate[threadIdx.y / 32] = t;
	for (int offset = 16; offset > 0; offset /= 2) {
		s_mtemplate[threadIdx.y /
#endif
}

template <typename T>
__global__ void gradX_k(T *dst, const T *src, int rows, int cols) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < rows && j < cols) {
    int index = i * cols + j;
    if (j == 0 || j == cols - 1 || i == 0 || i == rows - 1)
      dst[index] = 0;
    else {
        T gx = src[index + 1] - src[index - 1];
        gx += (src[index + cols + 1] - src[index + cols - 1]);
        gx += (src[index - cols + 1] - src[index - cols - 1]);
        dst[index] = -gx;
    }
  }
}

template <typename T>
__global__ void gradY_k(T *dst, const T *src, int rows, int cols) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < rows && j < cols) {
    int index = i * cols + j;
    if (j == 0 || j == cols - 1 || i == 0 || i == rows - 1)
      dst[index] = 0;
    else {
      T gy = src[index + cols] - src[index - cols];
      gy += (src[index + cols + 1] - src[index - cols + 1]);
      gy += (src[index + cols - 1] - src[index - cols - 1]);
      dst[index] = -gy;
    }
  }
}

namespace camera_chessboard_detector {

namespace cuda {

void pad(GpuImagePtr &dst, const GpuImagePtr &src, int pad) {
  dst->resize(src->width() + 2 * pad, src->height() + 2 * pad);

  dim3 block(32, 24);
  dim3 grid((src->width() + block.x - 1) / block.x,
            (src->height() + block.y - 1) / block.y);

  padBorder_k<float><<<grid, block>>>(dst->data(), src->data(), src->height(),
                                    src->width(), pad);
}

void nms(GpuImagePtr &dst, const GpuImagePtr &src, int R, int stride, int pad) {
  dst->resize(src->width(), src->height());

  dim3 block(32, 24);
  dim3 grid((src->width() + block.x - 1) / block.x,
            (src->height() + block.y - 1) / block.y);

  nonMaxSuppress_k<float><<<grid, block>>>(dst->data(), src->data(), src->height(),
                                    src->width(), R, stride, pad);
}

CudaConvolver::CudaConvolver() {
    NVCHK(cudaStreamCreate(&stream1_));
    NVCHK(cudaStreamCreate(&stream2_));
}
CudaConvolver::~CudaConvolver() {
    NVCHK(cudaStreamDestroy(stream1_));
    NVCHK(cudaStreamDestroy(stream2_));
}

void CudaConvolver::convolve(const GpuImagePtr &image,
                             const GpuKernelPtr &kernel, GpuImagePtr &output) {
  output->resize(image->width(), image->height());
  output->fill(0);
  const int H = image->height();
  const int W = image->width();
  const int R = kernel->height();
  const int pad = R / 2;

  // Block / grid for the float2 / 2-pixels-per-thread conv2dShared
  // kernel. Shared memory is sized for an input-tile halo even though the
  // active kernel currently only stages the small filter there — see
  // the design notes for the optimisation attempts.
  const dim3 block(16, 32);
  const dim3 grid((W + block.x * 2 - 1) / (2 * block.x),
                  (H + block.y - 1) / block.y);
  const size_t shared_size =
      (block.x * 2 + 2 * pad) * (block.y + 2 * pad) * sizeof(float);

#ifndef NDEBUG
  std::cout << "convolving: " << image->width() << "x" << image->height()
            << std::endl;
#endif

  conv2dShared<<<grid, block, shared_size, stream1_>>>(
      output->data(), image->data(), kernel->data(), H, W, R);
  cudaDeviceSynchronize();

#ifndef NDEBUG
  std::cout << "convolved: " << output->width() << "x" << output->height()
            << std::endl;
#endif
}

// static void get_corners_edge(Ptr<float, DEVICE> &corners_edge1_sin,
//                              Ptr<float, DEVICE> &corners_edge1_cos,
//                              Ptr<float, DEVICE> &corners_edge2_sin,
//                              Ptr<float, DEVICE> &corners_edge2_cos,
//                              int num_corners, const GpuImagePtr &gx,
//                              const GpuImagePtr &gy,
//                              const Ptr<float, DEVICE> &corners_x,
//                              const Ptr<float, DEVICE> &corners_y, int radius,
//                              int width, int height, bool refine) {
//   constexpr int block_size = 32;
//   cornerEdges_k<float><<<num_corners / block_size + 1, block_size>>>(
//       corners_edge1_sin->data(), corners_edge1_cos->data(),
//       corners_edge2_sin->data(), corners_edge2_cos->data(), num_corners,
//       gx->data(), gy->data(), corners_x->data(), corners_y->data(), radius,
//       width, height, refine);
// }

static void get_corners_edge(GpuImagePtr &corners_edge1_sin,
                             GpuImagePtr &corners_edge1_cos,
                             GpuImagePtr &corners_edge2_sin,
                             GpuImagePtr &corners_edge2_cos, int num_corners,
                             const GpuImagePtr &gx, const GpuImagePtr &gy,
                             const GpuImagePtr &corners_x,
                             const GpuImagePtr &corners_y, int radius,
                             int width, int height, bool refine) {
  constexpr int block_size = 32;
  cornerEdges_k<float><<<num_corners / block_size + 1, block_size>>>(
      corners_edge1_sin->data(), corners_edge1_cos->data(),
      corners_edge2_sin->data(), corners_edge2_cos->data(), num_corners,
      gx->data(), gy->data(), corners_x->data(), corners_y->data(), radius,
      width, height, refine);
}

static void refine_corners(GpuImagePtr &corners_x, GpuImagePtr &corners_y,
                           int num_corners,
                           const GpuImagePtr &corners_edge1_sin,
                           const GpuImagePtr &corners_edge1_cos,
                           const GpuImagePtr &corners_edge2_sin,
                           const GpuImagePtr &corners_edge2_cos,
                           const GpuImagePtr &gx, const GpuImagePtr &gy,
                           int radius, int width, int height) {
  constexpr int block_size = 32;
  refine_corners_k<float><<<corners_x->size() / block_size + 1, block_size>>>(
      corners_x->data(), corners_y->data(), num_corners,
      corners_edge1_sin->data(), corners_edge1_cos->data(),
      corners_edge2_sin->data(), corners_edge2_cos->data(), gx->data(),
      gy->data(), radius, width, height);
}

static GpuImagePtr sobel_x(const GpuImagePtr &src) {
  GpuImagePtr dst = std::make_shared<GpuImageF32>();
  dst->resize(src->width(), src->height());

  dim3 block(32, 24);
  dim3 grid((src->width() + block.x - 1) / block.x,
            (src->height() + block.y - 1) / block.y);

  gradX_k<float><<<grid, block>>>(dst->data(), src->data(), src->height(),
                                    src->width());

  return dst;
}

static GpuImagePtr sobel_y(const GpuImagePtr &src) {
  GpuImagePtr dst = std::make_shared<GpuImageF32>();
  dst->resize(src->width(), src->height());

  dim3 block(32, 24);
  dim3 grid((src->width() + block.x - 1) / block.x,
            (src->height() + block.y - 1) / block.y);

  gradY_k<float><<<grid, block>>>(dst->data(), src->data(), src->height(),
                                    src->width());

  return dst;
}

void gradients(GpuImagePtr &gx, GpuImagePtr &gy, const GpuImagePtr &src) {
  gx->resize(src->width(), src->height());
  gy->resize(src->width(), src->height());

  dim3 block(32, 24);
  dim3 grid((src->width() + block.x - 1) / block.x,
            (src->height() + block.y - 1) / block.y);

  gradX_k<float><<<grid, block>>>(gx->data(), src->data(), src->height(),
                                  src->width());
  gradY_k<float><<<grid, block>>>(gy->data(), src->data(), src->height(),
                                  src->width());
}

void weight(GpuImagePtr &dst, const GpuImagePtr &gx, const GpuImagePtr &gy) {
  dst->resize(gx->width(), gx->height());

  dim3 block(32, 24);
  dim3 grid((gx->width() + block.x - 1) / block.x,
            (gx->height() + block.y - 1) / block.y);

  gradientWeight_k<float><<<grid, block>>>(dst->data(), gx->data(), gy->data(),
                                   gx->height(), gx->width());
}


CudaRefiner::CudaRefiner() { NVCHK(cudaStreamCreate(&stream_)); }
CudaRefiner::~CudaRefiner() { NVCHK(cudaStreamDestroy(stream_)); }

void CudaRefiner::refine(const GpuImagePtr &image, const GpuImagePtr &gx,
                         const GpuImagePtr &gy, CornerArray &corners, int radius,
                         int refine_type) {
  assert(refine_type <= REFINE_ALL);
  bool redges = refine_type & REFINE_EDGES;
  bool rcorners = refine_type & REFINE_CORNERS;

  // Ptr<float, DEVICE> corners_x;
  // Ptr<float, DEVICE> corners_y;
  // corners_x.copy_from_host(corners.x->data(), corners.x.size());
  // corners_y.copy_from_host(corners.y->data(), corners.y.size());
  GpuImagePtr corners_x = std::make_shared<GpuImageF32>();
  GpuImagePtr corners_y = std::make_shared<GpuImageF32>();
  corners_x->resize(corners.x.size(), 1); // FIXME: Crash if 0 corners
  corners_y->resize(corners.y.size(), 1);
  corners_x->upload(corners.x.data(), corners.x.size());
  corners_y->upload(corners.y.data(), corners.y.size());

  // Get corners edge
  // Ptr<float, DEVICE> corners_edge1_sin;
  // Ptr<float, DEVICE> corners_edge1_cos;
  // Ptr<float, DEVICE> corners_edge2_sin;
  // Ptr<float, DEVICE> corners_edge2_cos;
  // corners_edge1_sin->resize(corners.x.size());
  // corners_edge1_cos->resize(corners.x.size());
  // corners_edge2_sin->resize(corners.x.size());
  // corners_edge2_cos->resize(corners.x.size());

  GpuImagePtr corners_edge1_sin = std::make_shared<GpuImageF32>();
  GpuImagePtr corners_edge1_cos = std::make_shared<GpuImageF32>();
  GpuImagePtr corners_edge2_sin = std::make_shared<GpuImageF32>();
  GpuImagePtr corners_edge2_cos = std::make_shared<GpuImageF32>();
  corners_edge1_sin->resize(corners.x.size(), 1);
  corners_edge1_cos->resize(corners.x.size(), 1);
  corners_edge2_sin->resize(corners.x.size(), 1);
  corners_edge2_cos->resize(corners.x.size(), 1);
  corners_edge1_sin->fill(0);
  corners_edge1_cos->fill(0);
  corners_edge2_sin->fill(0);
  corners_edge2_cos->fill(0);

  get_corners_edge(corners_edge1_sin, corners_edge1_cos, corners_edge2_sin,
                   corners_edge2_cos, corners.x.size(), gx, gy, corners_x,
                   corners_y, radius, image->width(), image->height(), redges);

  corners_edge1_sin->download(corners.edge1_sin.data(), corners.x.size());
  corners_edge1_cos->download(corners.edge1_cos.data(), corners.x.size());
  corners_edge2_sin->download(corners.edge2_sin.data(), corners.x.size());
  corners_edge2_cos->download(corners.edge2_cos.data(), corners.x.size());

  if (!rcorners)
    return;

  refine_corners(corners_x, corners_y, corners.x.size(), corners_edge1_sin,
                 corners_edge1_cos, corners_edge2_sin, corners_edge2_cos, gx,
                 gy, radius, image->width(), image->height());

  // corners.edge1_sin.resize(corners.x.size());
  // corners.edge1_cos.resize(corners.x.size());
  // corners.edge2_sin.resize(corners.x.size());
  // corners.edge2_cos.resize(corners.x.size());
  corners_x->download(corners.x.data(), corners.x.size());
  corners_y->download(corners.y.data(), corners.y.size());
}

void min(const GpuImagePtr &src1, const GpuImagePtr &src2, GpuImagePtr &dst) {
  assert(src1->height() == src2->height());
  assert(src1->width() == src2->width());
  dst->resize(src1->width(), src1->height());

  dim3 block(32, 24);
  dim3 grid((src1->width() + block.x - 1) / block.x,
            (src1->height() + block.y - 1) / block.y);

  min_k<float><<<grid, block>>>(dst->data(), src1->data(), src2->data(),
                                src1->height(), src1->width());
}

void min4(const GpuImagePtr &src1, const GpuImagePtr &src2,
          const GpuImagePtr &src3, const GpuImagePtr &src4, GpuImagePtr &dst) {
  assert(src1->height() == src2->height());
  assert(src1->width() == src2->width());
  assert(src1->height() == src3->height());
  assert(src1->width() == src3->width());
  assert(src1->height() == src4->height());
  assert(src1->width() == src4->width());
  dst->resize(src1->width(), src1->height());

  dim3 block(32, 24);
  dim3 grid((src1->width() + block.x - 1) / block.x,
            (src1->height() + block.y - 1) / block.y);

  min4_k<float><<<grid, block>>>(dst->data(), src1->data(), src2->data(),
                                 src3->data(), src4->data(), src1->height(),
                                 src1->width());
}

void max(const GpuImagePtr &src1, const GpuImagePtr &src2, GpuImagePtr &dst) {
  assert(src1->height() == src2->height());
  assert(src1->width() == src2->width());
  dst->resize(src1->width(), src1->height());

  dim3 block(32, 24);
  dim3 grid((src1->width() + block.x - 1) / block.x,
            (src1->height() + block.y - 1) / block.y);

  max_k<float><<<grid, block>>>(dst->data(), src1->data(), src2->data(),
                                src1->height(), src1->width());
}

void sub(const GpuImagePtr &src1, const GpuImagePtr &src2, GpuImagePtr &dst) {
  assert(src1->height() == src2->height());
  assert(src1->width() == src2->width());
  dst->resize(src1->width(), src1->height());

  dim3 block(32, 24);
  dim3 grid((src1->width() + block.x - 1) / block.x,
            (src1->height() + block.y - 1) / block.y);

  sub_k<float><<<grid, block>>>(dst->data(), src1->data(), src2->data(),
                                src1->height(), src1->width());
}

void neg(const GpuImagePtr &src, GpuImagePtr &dst) {
  dst->resize(src->width(), src->height());

  dim3 block(32, 24);
  dim3 grid((src->width() + block.x - 1) / block.x,
            (src->height() + block.y - 1) / block.y);

  neg_k<float>
      <<<grid, block>>>(dst->data(), src->data(), src->height(), src->width());
}

void mean4(const GpuImagePtr &src1, const GpuImagePtr &src2,
           const GpuImagePtr &src3, const GpuImagePtr &src4, GpuImagePtr &dst) {
  assert(src1->height() == src2->height());
  assert(src1->width() == src2->width());
  assert(src1->height() == src3->height());
  assert(src1->width() == src3->width());
  assert(src1->height() == src4->height());
  assert(src1->width() == src4->width());
  dst->resize(src1->width(), src1->height());

  dim3 block(32, 24);
  dim3 grid((src1->width() + block.x - 1) / block.x,
            (src1->height() + block.y - 1) / block.y);

  mean4_k<float><<<grid, block>>>(dst->data(), src1->data(), src2->data(),
                                  src3->data(), src4->data(), src1->height(),
                                  src1->width());
}

void likelihood(GpuImagePtr &dst, const GpuImagePtr &A,
                const GpuImagePtr &B, const GpuImagePtr &C,
                const GpuImagePtr &D) {
  assert(A->height() == B->height());
  assert(A->width() == B->width());
  assert(A->height() == C->height());
  assert(A->width() == C->width());
  assert(A->height() == D->height());
  assert(A->width() == D->width());
  dst->resize(A->width(), A->height());

  dim3 block(32, 24);
  dim3 grid((A->width() + block.x - 1) / block.x,
            (A->height() + block.y - 1) / block.y);

  combineLikelihood_k<float><<<grid, block>>>(dst->data(), A->data(), B->data(),
                                       C->data(), D->data(), A->height(),
                                       A->width());
}

void score_gradient(CornerArray &corners, const GpuImagePtr &src, int radius) {
	GpuImagePtr gx = sobel_x(src);
	GpuImagePtr gy = sobel_y(src);

	GpuImagePtr score = std::make_shared<GpuImageF32>();
	GpuImagePtr corners_x = std::make_shared<GpuImageF32>();
	GpuImagePtr corners_y = std::make_shared<GpuImageF32>();
	GpuImagePtr corners_edge1_sin = std::make_shared<GpuImageF32>();
	GpuImagePtr corners_edge1_cos = std::make_shared<GpuImageF32>();
	GpuImagePtr corners_edge2_sin = std::make_shared<GpuImageF32>();
	GpuImagePtr corners_edge2_cos = std::make_shared<GpuImageF32>();
	// corners_edge1_sin->resize(corners.x.size(), 1);
	// corners_edge1_cos->resize(corners.x.size(), 1);
	// corners_edge2_sin->resize(corners.x.size(), 1);
	// corners_edge2_cos->resize(corners.x.size(), 1);
	score->resize(corners.x.size(), 1);
	corners_x->upload(corners.x.data(), corners.x.size());
	corners_y->upload(corners.y.data(), corners.x.size());
	corners_edge1_sin->upload(corners.edge1_sin.data(), corners.x.size());
	corners_edge1_cos->upload(corners.edge1_cos.data(), corners.x.size());
	corners_edge2_sin->upload(corners.edge2_sin.data(), corners.x.size());
	corners_edge2_cos->upload(corners.edge2_cos.data(), corners.x.size());

	dim3 block(32, 32);
	dim3 grid((corners.x.size() + block.x - 1) / block.x, 1);
	gradientScore_k<<<grid, block>>>(
		score->data(),
		gx->data(), gy->data(), corners_x->data(), corners_y->data(),
		corners_edge1_sin->data(), corners_edge1_cos->data(),
		corners_edge2_sin->data(), corners_edge2_cos->data(),
		radius, corners.x.size(),
		src->width(), src->height());
	score->download(corners.score.data(), corners.x.size());
}


template <typename T>
void bgr_to_gray(T *dst, const T *src,
								 int width, int height, bool interleaved) {
  dim3 block(32, 24);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) /
  block.y);

  if (interleaved)
    bgr_to_gray_interleaved_k<T>
        <<<grid, block>>>(dst, src, width, height);
  else
    bgr_to_gray_planar_k<T>
        <<<grid, block>>>(dst, src, width, height);
}

template <typename T>
void gaussian9x9(T *dst, const T *src, float sigma, int width, int height) {
	dim3 block(32, 24);
	dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

	gaussian9x9_k<T><<<grid, block>>>(dst, src, sigma,
																		width, height);
}

// void u8_to_f32(GpuImagePtr &dst, const unsigned char *src, int width, int height) {
// 	dst->resize(width, height);
// 	dim3 block(32, 24);
// 	dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
// 	u8_to_f32_k<float><<<grid, block>>>(dst->data(), src->data(), width, height);
// }

void normalize(float *dst, const unsigned char *src, int width, int height) {
	int block_size = 256;
	int grid_size = (width * height + block_size - 1) / block_size;
	unsigned char *blocks_max, *blocks_min;
	NVCHK(cudaMalloc(&blocks_max, sizeof(char) * grid_size));
	NVCHK(cudaMalloc(&blocks_min, sizeof(char) * grid_size));
	minimum_k<unsigned char><<<grid_size, block_size>>>(blocks_min, src, width * height);
	maximum_k<unsigned char><<<grid_size, block_size>>>(blocks_max, src, width * height);
	unsigned char *h_blocks_max = new unsigned char[grid_size];
	unsigned char *h_blocks_min = new unsigned char[grid_size];
	NVCHK(cudaMemcpy(h_blocks_max, blocks_max, sizeof(char) * grid_size, cudaMemcpyDeviceToHost));
	NVCHK(cudaMemcpy(h_blocks_min, blocks_min, sizeof(char) * grid_size, cudaMemcpyDeviceToHost));
	unsigned char max = h_blocks_max[0];
	unsigned char min = h_blocks_min[0];
	std::cout << "min: " << min << ", max: " << max << std::endl;
	for (int i = 1; i < grid_size; i++) {
		if (h_blocks_max[i] > max) max = h_blocks_max[i];
		if (h_blocks_min[i] < min) min = h_blocks_min[i];
	}
	// std::cout << "min: " << min << ", max: " << max << std::endl;
	normalize_k<unsigned char><<<grid_size, block_size>>>(dst, src, min, max,
																												width * height);
	delete[] h_blocks_max;
	delete[] h_blocks_min;
	NVCHK(cudaFree(blocks_max));
	NVCHK(cudaFree(blocks_min));
}

template void bgr_to_gray<float>(float *dst, const float *src,
																 int width, int height, bool interleaved);
template void bgr_to_gray<unsigned char>(unsigned char *dst,
																				 const unsigned char *src,
																				 int width, int height,
																				 bool interleaved);

template void gaussian9x9<float>(float *dst, const float *src, float sigma,
																 int width, int height);
template void gaussian9x9<unsigned char>(unsigned char *dst,
																				 const unsigned char *src,
																				 float sigma, int width, int height);

} // namespace cuda

} // namespace camera_chessboard_detector
