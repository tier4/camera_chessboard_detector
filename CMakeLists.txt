cmake_minimum_required(VERSION 3.18)
project(camera_chessboard_detector VERSION 0.1.0 LANGUAGES CXX)

# ---------------------------------------------------------------------------
# camera_chessboard_detector — growth-based (Geiger-style) chessboard corner detector.
#
# Recovers partially visible and strongly distorted (e.g. fisheye) boards
# where OpenCV's findChessboardCorners gives up. Detection only; no pose.
#
# A CPU pipeline is always built. A CUDA pipeline is added automatically
# when a CUDA compiler is available; no source change is needed to switch
# between the two. Default CUDA architecture is the build host's GPU
# (override with -DCMAKE_CUDA_ARCHITECTURES=<arch>).
# ---------------------------------------------------------------------------

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(CCD_BUILD_TESTS "Build the camera_chessboard_detector test suite" ON)
option(CCD_PROFILE "Print per-stage timing" OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d imgcodecs)
find_package(Eigen3 REQUIRED)

include(CheckLanguage)
check_language(CUDA)
set(CCD_WITH_CUDA OFF)
if(CMAKE_CUDA_COMPILER)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING
      "Build for the host GPU by default; override for other targets.")
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(CCD_WITH_CUDA ON)
endif()

set(camera_chessboard_detector_cpu_sources
  src/detector.cpp
  src/structure_recovery.cpp
  src/cpu/corner_detector.cpp
  src/cpu/convolution.cpp
  src/score/correlation_kernel_cache.cpp
)

set(camera_chessboard_detector_cuda_sources
  src/cuda/pipeline.cpp
  src/cuda/gpu_buffers.cpp
  src/cuda/host_image.cpp
  src/cuda/kernels.cu
  src/cuda/nms.cu
  src/cuda/score.cu
  src/cuda/separable_conv.cu
  src/cuda/preprocess.cu
)

if(CCD_WITH_CUDA)
  add_library(camera_chessboard_detector ${camera_chessboard_detector_cpu_sources} ${camera_chessboard_detector_cuda_sources})
  message(STATUS "camera_chessboard_detector: CUDA compiler found — building CPU + CUDA pipelines")
else()
  add_library(camera_chessboard_detector ${camera_chessboard_detector_cpu_sources})
  message(STATUS "camera_chessboard_detector: no CUDA compiler — CPU-only build")
endif()
add_library(camera_chessboard_detector::camera_chessboard_detector ALIAS camera_chessboard_detector)

target_compile_features(camera_chessboard_detector PUBLIC cxx_std_17)
set_target_properties(camera_chessboard_detector PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(camera_chessboard_detector
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${OpenCV_INCLUDE_DIRS}
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(camera_chessboard_detector PUBLIC ${OpenCV_LIBRARIES} Eigen3::Eigen)
target_compile_definitions(camera_chessboard_detector PRIVATE OPENCV_ENABLED)

if(CCD_WITH_CUDA)
  set_target_properties(camera_chessboard_detector PROPERTIES CUDA_STANDARD 17)
  target_compile_definitions(camera_chessboard_detector PRIVATE CUDA_ENABLED CONV_SHARED)
  target_link_libraries(camera_chessboard_detector PRIVATE CUDA::cudart)
endif()

if(CCD_PROFILE)
  target_compile_definitions(camera_chessboard_detector PRIVATE TIME_LOGGER)
endif()

include(GNUInstallDirs)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS camera_chessboard_detector EXPORT camera_chessboard_detectorTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if(CCD_BUILD_TESTS)
  enable_testing()
  find_package(GTest REQUIRED)
  add_executable(camera_chessboard_detector_regression_test test/regression_test.cpp)
  target_link_libraries(camera_chessboard_detector_regression_test
    camera_chessboard_detector ${OpenCV_LIBRARIES} GTest::gtest GTest::gtest_main)
  target_compile_definitions(camera_chessboard_detector_regression_test PRIVATE
    CCD_SAMPLE_IMAGE="${CMAKE_CURRENT_SOURCE_DIR}/sample/chessboard_image.jpg"
    CCD_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
  add_test(NAME camera_chessboard_detector_regression COMMAND camera_chessboard_detector_regression_test)

  add_executable(camera_chessboard_detector_benchmark test/benchmark.cpp)
  target_link_libraries(camera_chessboard_detector_benchmark
    camera_chessboard_detector ${OpenCV_LIBRARIES})
  target_compile_definitions(camera_chessboard_detector_benchmark PRIVATE
    CCD_SAMPLE_IMAGE="${CMAKE_CURRENT_SOURCE_DIR}/sample/chessboard_image.jpg")
endif()
