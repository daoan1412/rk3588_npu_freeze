#include "rknn.h"

namespace RKNN::Common {

std::vector<char> load_data(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Open file " << filename << " failed.\n";
        return std::vector<char>();
    }

    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> data(size);
    if (file.read(data.data(), size)) {
        file.close();
        return data;
    } else {
        std::cerr << "Read file " << filename << " failed.\n";
        file.close();
        return std::vector<char>();
    }
}

int get_core_num() {
    std::lock_guard<std::mutex> lock(mtx);
    core_num++;
    if(core_num >= 3) {
        core_num = 0;
    }
    return core_num;
}

int dump_version(rknn_context &ctx) {
    rknn_sdk_version version;
    int ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version,
                         sizeof(rknn_sdk_version));
    if (ret < 0) {
        printf("rknn_query RKNN_QUERY_SDK_VERSION error ret=%d\n", ret);
        return -1;
    }
    printf("sdk version: %s driver version: %s\n", version.api_version,
           version.drv_version);
    return ret;
}

int load_model(rknn_context &ctx, const char *model_path) {
    printf("Loading mode %s...\n", model_path);
    auto model_data = load_data(model_path);
    int ret = rknn_init(&ctx, model_data.data(), model_data.size(), 0, NULL);
    if (ret < 0)
    {
        printf("rknn_init core error ret=%d\n", ret);
        return -1;
    }
    rknn_core_mask core_mask;
    switch (get_core_num())
    {
        case 0:
            core_mask = RKNN_NPU_CORE_0;
            break;
        case 1:
            core_mask = RKNN_NPU_CORE_1;
            break;
        case 2:
            core_mask = RKNN_NPU_CORE_2;
            break;
    }
    printf("rknn_set_core_mask core_mask num=%d\n", core_mask);
    ret = rknn_set_core_mask(ctx, core_mask);
    if (ret < 0) {
        printf("rknn_set_core_mask error ret=%d\n", ret);
        return -1;
    }
    dump_version(ctx);
    return 0;
}

int32_t __clip(float val, float min, float max) {
    float f = val <= min ? min : (val >= max ? max : val);
    return f;
}

int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale) {
    float dst_val = (f32 / scale) + zp;
    int8_t res = (int8_t) __clip(dst_val, 0, 255);
    return res;
}

float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) {
    return ((float) qnt - (float) zp) * scale;
}

inline const char *get_type_string(rknn_tensor_type type) {
    switch (type) {
        case RKNN_TENSOR_FLOAT32:
            return "FP32";
        case RKNN_TENSOR_FLOAT16:
            return "FP16";
        case RKNN_TENSOR_INT8:
            return "INT8";
        case RKNN_TENSOR_UINT8:
            return "UINT8";
        case RKNN_TENSOR_INT16:
            return "INT16";
        default:
            return "UNKNOW";
    }
}

inline const char *get_qnt_type_string(rknn_tensor_qnt_type type) {
    switch (type) {
        case RKNN_TENSOR_QNT_NONE:
            return "NONE";
        case RKNN_TENSOR_QNT_DFP:
            return "DFP";
        case RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC:
            return "AFFINE";
        default:
            return "UNKNOW";
    }
}

inline const char *get_format_string(rknn_tensor_format fmt) {
    switch (fmt) {
        case RKNN_TENSOR_NCHW:
            return "NCHW";
        case RKNN_TENSOR_NHWC:
            return "NHWC";
        default:
            return "UNKNOW";
    }
}

void dump_tensor_attr(rknn_tensor_attr *attr) {
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[3], attr->dims[2], attr->dims[1], attr->dims[0],
           attr->n_elems, attr->size, RKNN::Common::get_format_string(attr->fmt),  RKNN::Common::get_type_string(attr->type),
            RKNN::Common::get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

RKNNModelInfo get_model_info(rknn_context& ctx) {
    RKNNModelInfo rknnModelInfo;

    rknn_input_output_num io_num;
    int ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0) {
        printf("rknn_query RKNN_QUERY_IN_OUT_NUM error ret=%d\n", ret);
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input,
           io_num.n_output);

    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]),
                         sizeof(rknn_tensor_attr));
        if (ret < 0) {
            printf("rknn_query RKNN_QUERY_INPUT_ATTR error ret=%d\n", ret);
        }
        RKNN::Common::dump_tensor_attr(&(input_attrs[i]));
    }
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]),
                         sizeof(rknn_tensor_attr));
        if (ret < 0) {
            printf("rknn_query RKNN_QUERY_OUTPUT_ATTR error ret=%d\n", ret);
        }
        RKNN::Common::dump_tensor_attr(&(output_attrs[i]));
    }

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        printf("model is NCHW input fmt\n");
        rknnModelInfo.target_width = input_attrs[0].dims[0];
        rknnModelInfo.target_height = input_attrs[0].dims[1];
    } else {
        printf("model is NHWC input fmt\n");
        rknnModelInfo.target_width = input_attrs[0].dims[1];
        rknnModelInfo.target_height = input_attrs[0].dims[2];
    }
    rknnModelInfo.channel = 3;
    printf("model input height=%d, width=%d, channel=%d\n",
           rknnModelInfo.target_height, rknnModelInfo.target_width,
           rknnModelInfo.channel);

    for (int i = 0; i < io_num.n_output; ++i) {
        rknnModelInfo.out_scales.push_back(output_attrs[i].scale);
        rknnModelInfo.out_zps.push_back(output_attrs[i].zp);
        rknnModelInfo.n_elems.push_back(output_attrs[i].n_elems);
    }

    rknnModelInfo.n_output = io_num.n_output;
    rknnModelInfo.n_input = io_num.n_input;

    return rknnModelInfo;
}

int run_infer(rknn_context& ctx, RKNNModelInfo &rknnModelInfo, const cv::Mat& image, rknn_output *outputs) {
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    for (int i = 0; i < rknnModelInfo.n_output; i++) {
        outputs[i].want_float = 0;
    }
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = rknnModelInfo.target_width * rknnModelInfo.target_height * rknnModelInfo.channel;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;
    inputs[0].buf = image.data;
    rknn_inputs_set(ctx, rknnModelInfo.n_input, inputs);
    int ret = rknn_run(ctx, NULL);
    ret = rknn_outputs_get(ctx, rknnModelInfo.n_output, outputs, NULL);
    return ret;
}
int output_release(rknn_context& ctx, RKNNModelInfo &rknnModelInfo, rknn_output *outputs) {
    return rknn_outputs_release(ctx, rknnModelInfo.n_output, outputs);
}
};