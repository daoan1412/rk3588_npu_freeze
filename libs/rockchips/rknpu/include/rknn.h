#pragma once

#include "rknn_api.h"
#include <iostream>
#include <cstring>
#include <vector>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <fstream>

namespace RKNN::Common {
    static int core_num = -1;
    static std::mutex mtx;

    int get_core_num();

    typedef struct {
        int target_width;
        int target_height;
        int channel;
        std::vector<int32_t> n_elems;
        std::vector<int32_t> out_zps;
        std::vector<float> out_scales;
        int32_t n_output;
        int32_t n_input;
    } RKNNModelInfo;

    std::vector<char> load_data(const std::string &filename);

    int load_model(rknn_context &ctx, const char *model_path);

    RKNNModelInfo get_model_info(rknn_context &ctx);

    int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale);

    float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale);

    void dump_tensor_attr(rknn_tensor_attr *attr);

    int run_infer(rknn_context &ctx, RKNNModelInfo &rknnModelInfo, const cv::Mat &image, rknn_output *outputs);

    int output_release(rknn_context &ctx, RKNNModelInfo &rknnModelInfo, rknn_output *outputs);
};