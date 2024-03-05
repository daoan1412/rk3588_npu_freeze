#!/bin/bash

rm -rf build
cmake -S . -B build
cmake --build build --config Release --target rk3588_npu_freeze -j4
