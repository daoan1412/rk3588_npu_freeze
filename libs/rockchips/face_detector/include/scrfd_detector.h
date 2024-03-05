#pragma once
#include <vector>
#include "rknn_api.h"
#include <iostream>
#include <unordered_map>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include "rknn.h"

namespace rockchips {
    class ScrfdDetectInfo {
    public:
        float x1;
        float y1;
        float x2;
        float y2;
        float score;
        int label;
        std::string label_text;
        std::vector<cv::Point2f> landmarks;
        bool flag;
    };

    typedef struct
    {
        float cx;
        float cy;
        float stride;
    } SCRFDPoint;
    typedef struct
    {
        float ratio;
        int dw;
        int dh;
        bool flag;
    } SCRFDScaleParams;

    class ScrfdDetector {

    public:
        explicit ScrfdDetector();
        ~ScrfdDetector();
        int load_model(const char *model_path);


    private:
        rknn_context ctx;
        RKNN::Common::RKNNModelInfo rknnModelInfo;
        std::vector<int> input_node_dims = {1, 3, 640, 640};
        const float mean_vals[3] = {127.5f, 127.5f, 127.5f}; // RGB
        const float scale_vals[3] = {1.f / 128.f, 1.f / 128.f, 1.f / 128.f};
        std::vector<int> feat_stride_fpn = {8, 16, 32}; // steps, may [8, 16, 32, 64, 128]

        std::unordered_map<int, std::vector<SCRFDPoint>> center_points;
        bool center_points_is_update = false;
        unsigned int num_anchors = 2;
        static constexpr const unsigned int nms_pre = 1000;
        static constexpr const unsigned int max_nms = 30000;
    private:
        void resizeAndPadImage(const cv::Mat& rgbImage, int maxWidth, int maxHeight, int padding, cv::Mat &paddedImg);
        void resize_unscale(const cv::Mat &mat, cv::Mat &mat_rs,
                            int target_height, int target_width,
                            SCRFDScaleParams &scale_params);
        int get_number_points(RKNN::Common::RKNNModelInfo &rknnModelInf, unsigned int stride);
        // generate once.
        void generate_points(int target_height, int target_width);
        void generate_bboxes_kps_single_stride(const SCRFDScaleParams &scale_params,
                                               const int8_t *score_ptr,
                                               const int8_t *bbox_ptr,
                                               const int8_t *kps_ptr,
                                               int32_t score_zp, float score_scale,
                                               int32_t bbox_zp, float bbox_scale,
                                               int32_t kps_zp, float kps_scale,
                                               RKNN::Common::RKNNModelInfo &rknnModelInf,
                                               int stride,
                                               float conf_threshold,
                                               float img_height,
                                               float img_width,
                                               std::vector<ScrfdDetectInfo> &bbox_kps_collection);
        void generate_bboxes_kps(const SCRFDScaleParams &scale_params,
                                 std::vector<ScrfdDetectInfo> &bbox_kps_collection,
                                 int8_t *score_8,
                                 int8_t *score_16,
                                 int8_t *score_32,
                                 int8_t *bbox_8,
                                 int8_t *bbox_16,
                                 int8_t *bbox_32,
                                 int8_t *kps_8,
                                 int8_t *kps_16,
                                 int8_t *kps_32,
                                 RKNN::Common::RKNNModelInfo &rknnModelInf,
                                 float score_threshold, float img_height,
                                 float img_width,  std::vector<int32_t> out_zps,  std::vector<float> out_scales);
        void nms_bboxes_kps(std::vector<ScrfdDetectInfo> &input,
                            std::vector<ScrfdDetectInfo> &output,
                            float iou_threshold, unsigned int topk);

    public:
        void detect(const cv::Mat &mat, std::vector<ScrfdDetectInfo> &detected_boxes_kps,
                    float score_threshold = 0.5f, float iou_threshold = 0.45f,
                    unsigned int topk = 10);
        void detectInJpegData(const cv::Mat& rgbImage,
                              std::vector<ScrfdDetectInfo> &detected_boxes_kps,
                    float score_threshold = 0.5f);

    };
}