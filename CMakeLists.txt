cmake_minimum_required(VERSION 3.0)
include(FetchContent)
project(rk3588_npu_freeze)

set(CMAKE_CXX_STANDARD 17)

file(COPY assets/models DESTINATION ${PROJECT_BINARY_DIR})
file(COPY assets/images DESTINATION ${PROJECT_BINARY_DIR})

set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/3rdparty/opencv-4.9.0/build_linux/install/lib/cmake/opencv4")
find_package(OpenCV REQUIRED core imgproc video videoio highgui)


set(EIGEN_VERSION 3.4)
find_package(Eigen3 ${EIGEN_VERSION} QUIET)
if (NOT EIGEN3_FOUND)
    set(BUILD_TESTING OFF CACHE INTERNAL "")
    FetchContent_Declare(eigen
            GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
            GIT_TAG 3.4
            GIT_SHALLOW ON)
    FetchContent_MakeAvailable(eigen)
    unset(BUILD_TESTING CACHE)
endif ()


add_subdirectory(libs/rockchips)

add_executable(rk3588_npu_freeze main.cpp)
target_link_libraries(rk3588_npu_freeze ${OpenCV_LIBS} rockchips ${CMAKE_THREAD_LIBS_INIT})