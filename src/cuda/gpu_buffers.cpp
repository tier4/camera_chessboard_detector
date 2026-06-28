#include <cassert>
#include <cuda_runtime.h>

#include "cuda/gpu_buffers.hpp"


namespace camera_chessboard_detector {

namespace cuda {

template <typename T>
GpuImage<T>::GpuImage() {
  width_ = 0;
  height_ = 0;
  data_ = nullptr;
}

template <typename T>
GpuImage<T>::GpuImage(int width, int height) {
  width_ = width;
  height_ = height;
  NVCHK(cudaMalloc(&data_, width * height * sizeof(T)));
  fill(0);
  initialized_ = true;
}

template <typename T> GpuImage<T>::~GpuImage() {
  if (data_ != nullptr) {
    NVCHK(cudaFree(data_));
    data_ = nullptr;
  }
}

template <typename T> bool GpuImage<T>::resize(int width, int height) {
  if (width == width_ && height == height_)
    return false;

  width_ = width;
  height_ = height;
  if (data_ != nullptr) {
    NVCHK(cudaFree(data_));
    data_ = nullptr;
  }
  NVCHK(cudaMalloc(&data_, width * height * sizeof(T)));
  fill(0);
  return true;
}

template <typename T> void GpuImage<T>::fill(T value) {
  assert(data_ != nullptr);
  NVCHK(cudaMemset(data_, value, width_ * height_ * sizeof(T)));
}

template <typename T> std::size_t GpuImage<T>::size() const {
  return width_ * height_;
}

template <typename T> bool GpuImage<T>::valid() const {
  return initialized_;
}

template <typename T> T *GpuImage<T>::data() { return data_; }

template <typename T> int GpuImage<T>::width() const { return width_; }
template <typename T> int GpuImage<T>::height() const { return height_; }
template <typename T>
void GpuImage<T>::download(const CpuImage<T> &other) {
  assert(other.width() == width_);
  assert(other.height() == height_);
  NVCHK(cudaMemcpy(const_cast<T*>(other.data()), data_, width_ * height_ * sizeof(T),
                   cudaMemcpyDeviceToHost));
}

template <typename T>
void GpuImage<T>::download(T *data, std::size_t size) {
  assert(size == width_ * height_);
  NVCHK(cudaMemcpy(data, data_, size * sizeof(T), cudaMemcpyDeviceToHost));
}

template <typename T>
void GpuImage<T>::upload(const CpuImage<T> &other) {
  assert(other.width() == width_);
  assert(other.height() == height_);
  NVCHK(cudaMemcpy(data_, other.data(), width_ * height_ * sizeof(T),
                   cudaMemcpyHostToDevice));
}

template <typename T>
void GpuImage<T>::upload(const T *data, std::size_t size) {
  assert(size == width_ * height_);
  NVCHK(cudaMemcpy(data_, data, size * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
GpuImage<T> GpuImage<T>::from_cpu(const CpuImage<T> &other) {
  GpuImage<T> image(other.width(), other.height());
  image.upload(other);
  return image;
}

template <typename T>
GpuKernel<T>::GpuKernel(int radius) :
  GpuImage<T>(2 * radius + 1, 2 * radius + 1) {}

template <typename T>
GpuKernel<T>::GpuKernel(const CpuKernel<T> &other) :
#define VEC2
#ifdef VEC4
  GpuImage<T>(other.width() * 4, other.height()) {
  CpuImage<T> kernel(other.width() * 4, other.height());
  for (int i = 0; i < other.width(); ++i) {
    for (int j = 0; j < other.height(); ++j) {
      kernel.set(i, j, other.at(i, j));
      kernel.set(i + other.width(), j, other.at(i, j));
      kernel.set(i + other.width() * 2, j, other.at(i, j));
      kernel.set(i + other.width() * 3, j, other.at(i, j));
    }
  }
  this->upload(kernel);
#elif defined(VEC2)
  GpuImage<T>(other.width() * 2, other.height()) {
  CpuImage<T> kernel(other.width() * 2, other.height());
  for (int i = 0; i < other.width(); ++i) {
    for (int j = 0; j < other.height(); ++j) {
      kernel.set(i, j, other.at(i, j));
      kernel.set(i + other.width(), j, other.at(i, j));
    }
  }
  this->upload(kernel);
#else
  GpuImage<T>(other.width(), other.height()) {
  this->upload(other);
#endif
}

template <typename T>
GpuKernel<T> GpuKernel<T>::from_cpu(const CpuKernel<T> &other) {
  GpuKernel<T> kernel(other.width() / 2);
  kernel.upload(other);
  return kernel;
}

template class GpuImage<float>;
template class GpuImage<unsigned char>;
template class GpuKernel<float>;

} // namespace cuda

} // namespace camera_chessboard_detector
