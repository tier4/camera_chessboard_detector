#pragma once

#include "core.hpp"

#include <cassert>
#include <cstddef>
#include <cstdio>

// Abort with a CUDA error message if `err` is not cudaSuccess.
#define NVCHK(err)                                                             \
  if (err != cudaSuccess) {                                                    \
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));              \
    assert(false);                                                             \
  }

namespace camera_chessboard_detector {

namespace cuda {


template <typename T>
class GpuImage {
public:
  GpuImage();
  GpuImage(int width, int height);
  // GpuImage(int width, int height, T *data);
  // GpuImage(GpuImage<T> &&other);
  ~GpuImage();

  bool resize(int width, int height);
  void fill(T value);
  // void set(int x, int y, T value);

  std::size_t size() const;
  bool valid() const;
  int width() const;
  int height() const;
  T *data();
  // T at(int x, int y) const;

  // static GpuImage<T> from_cpu(const CpuImage<T> &other);
  void download(const CpuImage<T> &other);
  void download(T *data, std::size_t size);
  void upload(const CpuImage<T> &other);
  void upload(const T *data, std::size_t size);
  static GpuImage<T> from_cpu(const CpuImage<T> &other);
private:
  bool initialized_;
  int width_;
  int height_;
  T *data_;
};

template <typename T>
class GpuKernel : public GpuImage<T> {
public:
  GpuKernel(int radius);
  GpuKernel(const CpuKernel<T> &other);
  static GpuKernel<T> from_cpu(const CpuKernel<T> &other);
};

typedef GpuImage<float> GpuImageF32;
typedef std::shared_ptr<GpuImageF32> GpuImagePtr;
typedef GpuKernel<float> GpuKernelF32;
typedef std::shared_ptr<GpuKernelF32> GpuKernelPtr;

} // namespace cuda

} // namespace camera_chessboard_detector
