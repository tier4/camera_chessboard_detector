#include <cassert>
#include <cuda_runtime.h>

#include "cuda/gpu_buffers.hpp"


namespace camera_chessboard_detector {

namespace cuda {

#if 0
template <typename T, MemoryType M> Ptr<T, M>::Ptr() {
  ptr_ = nullptr;
  size_ = 0;
}

template <typename T, MemoryType M> Ptr<T, M>::Ptr(std::size_t size) {
  if (resize(size))
    fill(0);
}

template <typename T, MemoryType M>
template <MemoryType M2>
Ptr<T, M>::Ptr(T *ptr, std::size_t size) {
  if constexpr (M == M2) {
    ptr_ = ptr;
    size_ = size;
  } else {
    resize(size);
    copy_from<M2>(ptr, size);
  }
}

template <typename T, MemoryType M> Ptr<T, M>::Ptr(Ptr<T, M> &&other) {
  ptr_ = other.ptr_;
  size_ = other.size_;
  other.ptr_ = nullptr;
  other.size_ = 0;
}

template <typename T, MemoryType M>
Ptr<T, M>::Ptr(const Ptr<T, HOST> &other) {
  resize(other.size());
  copy_from<HOST>(other.data(), other.size());
}

template <typename T, MemoryType M>
Ptr<T, M>::Ptr(const Ptr<T, DEVICE> &other) {
  resize(other.size());
  copy_from<DEVICE>(other.data(), other.size());
}

template <typename T, MemoryType M> Ptr<T, M>::~Ptr() { free(); }

template <typename T, MemoryType M>

bool Ptr<T, M>::resize(std::size_t size) {
  if (size_ < size) {
    free();
    size_ = size;
    if (size_ > 0) {
      if constexpr (M == HOST)
        ptr_ = new T[size_];
      else if constexpr (M == DEVICE)
        NVCHK(cudaMalloc(&ptr_, size_ * sizeof(T)));

      return true;
    }
  } else if (size_ > size) {
    size_ = size;
  }
  return false;
}

template <typename T, MemoryType M> void Ptr<T, M>::free() {
  if (ptr_ != nullptr) {
    if constexpr (M == HOST)
      delete[] ptr_;
    else if constexpr (M == DEVICE)
      NVCHK(cudaFree(ptr_));
    ptr_ = nullptr;
    size_ = 0;
  }
}

template <typename T, MemoryType M>
void Ptr<T, M>::copy_from(const Ptr<T, HOST> &other) {
  copy_from<HOST>(other.data(), other.size());
}

template <typename T, MemoryType M>
void Ptr<T, M>::copy_from(const Ptr<T, DEVICE> &other) {
  copy_from<DEVICE>(other.data(), other.size());
}

template <typename T, MemoryType M>
template <MemoryType M2>
void Ptr<T, M>::copy_from(const T *ptr, std::size_t size) {
  assert(size > 0);
  resize(size);

  if constexpr (M == HOST && M2 == DEVICE) {
    NVCHK(cudaMemcpy(ptr_, ptr, size_ * sizeof(T), cudaMemcpyDeviceToHost));
  } else if constexpr (M == DEVICE && M2 == HOST) {
    NVCHK(cudaMemcpy(ptr_, ptr, size_ * sizeof(T), cudaMemcpyHostToDevice));
  } else if constexpr (M == HOST && M2 == HOST) {
    std::copy(ptr, ptr + size_, ptr_);
  } else if constexpr (M == DEVICE && M2 == DEVICE) {
    NVCHK(cudaMemcpy(ptr_, ptr, size_ * sizeof(T), cudaMemcpyDeviceToDevice));
  }
}

/* FIXME: memset on non-byte values behaves weirdly */
template <typename T, MemoryType M> void Ptr<T, M>::fill(T value) {
  assert(ptr_ != nullptr);
  if constexpr (M == HOST) {
    std::fill(ptr_, ptr_ + size_, value);
  } else if constexpr (M == DEVICE) {
    NVCHK(cudaMemset(ptr_, value, size_ * sizeof(T)));
  }
}

template <typename T, MemoryType M> void Ptr<T, M>::set(int index, T value) {
  assert(ptr_ != nullptr);
  assert(index < size_);
  if constexpr (M == HOST) {
    ptr_[index] = value;
  } else if constexpr (M == DEVICE) {
    NVCHK(cudaMemset(ptr_ + index, value, sizeof(T)));
  }
}

template <typename T, MemoryType M> T *Ptr<T, M>::data() const noexcept {
  assert(ptr_ != nullptr);
  return ptr_;
}

template <typename T, MemoryType M>
std::size_t Ptr<T, M>::size() const noexcept {
  return size_;
}

template <typename T, MemoryType M>
T &Ptr<T, M>::operator[](std::size_t index) noexcept {
  assert(index < size_);
  return ptr_[index];
}

template <typename T, MemoryType M>
Ptr<T, M> &Ptr<T, M>::operator=(Ptr<T, M> &&other) {
  if (this != &other) {
    ptr_ = other.ptr_;
    size_ = other.size_;
    other.ptr_ = nullptr;
    other.size_ = 0;
  }
  return *this;
}

template <typename T, MemoryType M>
Ptr<T, M> &Ptr<T, M>::operator=(const Ptr<T, HOST> &other) {
  copy_from<HOST>(other.data(), other.size());
  return *this;
}

template <typename T, MemoryType M>
Ptr<T, M> &Ptr<T, M>::operator=(const Ptr<T, DEVICE> &other) {
  copy_from<DEVICE>(other.data(), other.size());
  return *this;
}

// // Template instantiation
// #define INSTANTIATE_MEMORY(T, M) \
//   template class Memory<T, M>; \
//   template Memory<T, M>::Memory(const Memory<T, HOST> &other); \
//   template Memory<T, M>::Memory(const Memory<T, DEVICE> &other); \
//   template Memory<T, M>::Memory(Memory<T, M> &&other); \
//   template Memory<T, M>::Memory(std::size_t size); \
//   template Memory<T, M>::Memory(T *ptr, std::size_t size); \
//   template Memory<T, M>::~Memory(); \
//   template Memory<T, M> &Memory<T, M>::operator=(Memory<T, M> &&other); \
//   template void Memory<T, M>::copy_from(const Memory<T, HOST> &other); \
//   template void Memory<T, M>::copy_from(const Memory<T, DEVICE> &other); \
//   template void Memory<T, M>::copy_from(const T *ptr, std::size_t size); \
//   template void Memory<T, M>::set(T value); \
//   template T *Memory<T, M>::data() const noexcept; \
//   template std::size_t Memory<T, M>::size() const noexcept; \ template T
//   &Memory<T, M>::operator[](std::size_t index) noexcept;

// #define INSTANTIATE_MEMORY(T, M)                                                \
//   template class Memory<T, M>;                                                  \
//   template Memory<T, M>::~Memory();                                             \
//   template Memory<T, M> &Memory<T, M>::operator=(Memory<T, M> &&other);         \
//   template void Memory<T, M>::copy_from<HOST>(const Memory<T, HOST> &other);          \
//   template void Memory<T, M>::copy_from<HOST>(const Memory<T, DEVICE> &other);        \
//   template void Memory<T, M>::copy_from<HOST>(const T *ptr, std::size_t size);        \
//   template void Memory<T, M>::copy_from<DEVICE>(const Memory<T, HOST> &other);          \
//   template void Memory<T, M>::copy_from<DEVICE>(const Memory<T, DEVICE> &other);        \
//   template void Memory<T, M>::copy_from<DEVICE>(const T *ptr, std::size_t size);        \

// INSTANTIATE_MEMORY(float, HOST)
// INSTANTIATE_MEMORY(float, DEVICE)

#define INSTANTIATE_MEM2(T, M)                                                 \
  template void Ptr<T, M>::copy_from<HOST>(const T *ptr, std::size_t size); \
  template void Ptr<T, M>::copy_from<DEVICE>(const T *ptr, std::size_t size);

template class Ptr<float, HOST>;
template class Ptr<float, DEVICE>;
INSTANTIATE_MEM2(float, HOST)
INSTANTIATE_MEM2(float, DEVICE)
#endif

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
  // printf("Allocated %p\n", data_);
  fill(0);
  initialized_ = true;
}

template <typename T> GpuImage<T>::~GpuImage() {
  // printf("Freeing %p\n", data_);
  if (data_ != nullptr) {
    // FIXME: This causes a segfault for some reason
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
  // printf("Allocated %p\n", data_);
  fill(0);
  return true;
}

template <typename T> void GpuImage<T>::fill(T value) {
  assert(data_ != nullptr);
  NVCHK(cudaMemset(data_, value, width_ * height_ * sizeof(T)));
}

// template <typename T> void GpuImage<T>::set(int x, int y, T value) {
//   assert(data_ != nullptr);
//   assert(x < width_);
//   assert(y < height_);
//   NVCHK(cudaMemset(data_ + y * width_ + x, value, sizeof(T)));
// }

template <typename T> std::size_t GpuImage<T>::size() const {
  return width_ * height_;
}

template <typename T> bool GpuImage<T>::valid() const {
  return initialized_; // TODO: Remove this variable
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
