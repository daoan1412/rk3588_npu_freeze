#!/bin/bash

set -e
third_party="3rdparty"
opencv_file="4.9.0.zip"

if [[ ! -d "${third_party}" ]]; then
  mkdir -p ${third_party}
fi


if [ -e "$third_party/$opencv_file" ]; then
    cd "$third_party"
    echo "Đã có source opencv"
else
    cd "$third_party"
    wget -O 4.9.0.zip https://codeload.github.com/opencv/opencv/zip/refs/tags/4.9.0
    unzip 4.9.0.zip
fi

cd opencv-4.9.0
rm -rf build_linux
mkdir build_linux
cd build_linux
cmake -DCMAKE_INSTALL_PREFIX=install \
-DCMAKE_BUILD_TYPE=Release \
-D ENABLE_NEON=ON \
-D WITH_OPENMP=OFF \
-D WITH_OPENCL=TRUE \
-DOPENCV_GENERATE_PKGCONFIG=ON \
-DWITH_FFMPEG=OFF \
-DWITH_GSTREAMER=OFF \
-DOPENCV_ENABLE_NONFREE=ON \
-D BUILD_opencv_java=OFF \
-DBUILD_opencv_world=OFF \
-D BUILD_TESTS=OFF \
-D BUILD_EXAMPLES=OFF ..
make -j4
make install

