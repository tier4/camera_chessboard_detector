#pragma once

#include <cuda_runtime.h>
#include "core.hpp"
#include "cuda/gpu_buffers.hpp"

namespace camera_chessboard_detector {

namespace cuda {


void pad(GpuImagePtr &dst, const GpuImagePtr &src, int pad);
void nms(GpuImagePtr &dst, const GpuImagePtr &src, int radius);
// Sobel gradients of `src` written into the (reusable) gx / gy buffers.
void gradients(GpuImagePtr &gx, GpuImagePtr &gy, const GpuImagePtr &src);
// Gradient-magnitude weight map from precomputed gradients.
void weight(GpuImagePtr &dst, const GpuImagePtr &gx, const GpuImagePtr &gy);

class CudaConvolver {
public:
  CudaConvolver();
  ~CudaConvolver();

  void convolve(const GpuImagePtr &image, const GpuKernelPtr &kernel,
                GpuImagePtr &output);

private:
  // GpuImageF32 image_;
  // GpuImageF32 output_;
  // GpuKernelF32 kernel_;
  cudaStream_t stream1_;
  cudaStream_t stream2_;
};

class CudaRefiner {
public:
  enum RefineType {
    REFINE_NONE = 0,
    REFINE_EDGES = 1,
    REFINE_CORNERS = 2,
    REFINE_ALL = 3
  };

  CudaRefiner();
  ~CudaRefiner();

  // void refine(const GpuImageF32 &image, const GpuImageF32 &likelihood_map,
  //             GpuImageF32 &output);
  void refine(const GpuImagePtr &image, const GpuImagePtr &gx,
              const GpuImagePtr &gy, CornerArray &corners, int radius,
              int type);
private:
  cudaStream_t stream_;
};

void min(const GpuImagePtr &src1, const GpuImagePtr &src2, GpuImagePtr &dst);
void max(const GpuImagePtr &src1, const GpuImagePtr &src2, GpuImagePtr &dst);
void sub(const GpuImagePtr &src1, const GpuImagePtr &src2, GpuImagePtr &dst);
void neg(const GpuImagePtr &src, GpuImagePtr &dst);
void mean4(const GpuImagePtr &src1, const GpuImagePtr &src2,
           const GpuImagePtr &src3, const GpuImagePtr &src4, GpuImagePtr &dst);
void min4(const GpuImagePtr &src1, const GpuImagePtr &src2,
          const GpuImagePtr &src3, const GpuImagePtr &src4, GpuImagePtr &dst);
void likelihood(GpuImagePtr &dst,
                const GpuImagePtr &A, const GpuImagePtr &B,
                const GpuImagePtr &C, const GpuImagePtr &D);

void score_gradient(CornerArray &corners, const GpuImagePtr &src, int radius);

template <typename T>
void bgr_to_gray(T *dst, const T *src,
                 int width, int height, bool interleaved);

template <typename T>
void gaussian9x9(T *dst, const T *src, float sigma, int width, int height);

void normalize(float *dst, const unsigned char *src, int width, int height);

} // namespace cuda

} // namespace camera_chessboard_detector
