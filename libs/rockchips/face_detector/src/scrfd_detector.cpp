//
// Created by daoan on 10/24/23.
//

#include <cstring>
#include <opencv2/imgcodecs.hpp>
#include "scrfd_detector.h"

using namespace rockchips;

void ScrfdDetector::generate_bboxes_kps_single_stride(const SCRFDScaleParams &scale_params,
                                                      const int8_t *score_ptr,
                                                      const int8_t *bbox_ptr, const int8_t *kps_ptr,
                                                      int32_t score_zp, float score_scale,
                                                      int32_t bbox_zp, float bbox_scale,
                                                      int32_t kps_zp, float kps_scale,
                                                      RKNN::Common::RKNNModelInfo &rknnModelInf,
                                                      int stride,
                                                      float conf_threshold,
                                                      float img_height,
                                                      float img_width,
                                                      std::vector<ScrfdDetectInfo> &bbox_kps_collection) {
    unsigned int nms_pre_ = (stride / 8) * nms_pre; // 1 * 1000,2*1000,...
    nms_pre_ = nms_pre_ >= nms_pre ? nms_pre_ : nms_pre;

    int num_points = get_number_points(rknnModelInf, stride);

    float ratio = scale_params.ratio;
    int dw = scale_params.dw;
    int dh = scale_params.dh;

    unsigned int count = 0;
    auto &stride_points = center_points[stride];

    for (unsigned int i = 0; i < num_points; ++i) {
        const float cls_conf = RKNN::Common::deqnt_affine_to_f32(score_ptr[i], score_zp, score_scale);
        if (cls_conf < conf_threshold) continue; // filter
        auto &point = stride_points.at(i);
        const float cx = point.cx; // cx
        const float cy = point.cy; // cy
        const float s = point.stride; // stride

        // bbox
        const int8_t *offsets = bbox_ptr + i * 4;
        float l = RKNN::Common::deqnt_affine_to_f32(offsets[0], bbox_zp, bbox_scale); // left
        float t = RKNN::Common::deqnt_affine_to_f32(offsets[1], bbox_zp, bbox_scale); // top
        float r = RKNN::Common::deqnt_affine_to_f32(offsets[2], bbox_zp, bbox_scale); // right
        float b = RKNN::Common::deqnt_affine_to_f32(offsets[3], bbox_zp, bbox_scale); // bottom

        ScrfdDetectInfo box_kps;
        float x1 = ((cx - l) * s - (float) dw) / ratio;  // cx - l x1
        float y1 = ((cy - t) * s - (float) dh) / ratio;  // cy - t y1
        float x2 = ((cx + r) * s - (float) dw) / ratio;  // cx + r x2
        float y2 = ((cy + b) * s - (float) dh) / ratio;  // cy + b y2
        box_kps.x1 = std::max(0.f, x1);
        box_kps.y1 = std::max(0.f, y1);
        box_kps.x2 = std::min(img_width, x2);
        box_kps.y2 = std::min(img_height, y2);
        box_kps.score = cls_conf;
        box_kps.label = 1;
        box_kps.label_text = "face";
        box_kps.flag = true;

        // landmarks
        const int8_t *kps_offsets = kps_ptr + i * 10;
        for (unsigned int j = 0; j < 10; j += 2) {
            cv::Point2f kps;
            float kps_l = RKNN::Common::deqnt_affine_to_f32(kps_offsets[j], kps_zp, kps_scale);
            float kps_t = RKNN::Common::deqnt_affine_to_f32(kps_offsets[j + 1], kps_zp, kps_scale);
            float kps_x = ((cx + kps_l) * s - (float) dw) / ratio;  // cx + l x
            float kps_y = ((cy + kps_t) * s - (float) dh) / ratio;  // cy + t y
            kps.x = std::min(std::max(0.f, kps_x), img_width);
            kps.y = std::min(std::max(0.f, kps_y), img_height);
            box_kps.landmarks.push_back(kps);
        }
        box_kps.flag = true;

        bbox_kps_collection.push_back(box_kps);

        count += 1; // limit boxes for nms.
        if (count > max_nms)
            break;
    }

    if (bbox_kps_collection.size() > nms_pre_) {
        std::sort(
                bbox_kps_collection.begin(), bbox_kps_collection.end(),
                [](const ScrfdDetectInfo &a, const ScrfdDetectInfo &b) { return a.score > b.score; }
        ); // sort inplace
        // trunc
        bbox_kps_collection.resize(nms_pre_);
    }
}

int ScrfdDetector::load_model(const char *model_path) {
    int ret = RKNN::Common::load_model(ctx, model_path);
    rknnModelInfo = RKNN::Common::get_model_info(ctx);
    return ret;
}

int ScrfdDetector::get_number_points(RKNN::Common::RKNNModelInfo &rknnModelInf, unsigned int stride) {
    if (stride == 8) {
        return rknnModelInf.n_elems[0];
    }
    if (stride == 16) {
        return rknnModelInf.n_elems[1];
    }
    if (stride == 32) {
        return rknnModelInf.n_elems[2];
    }

    return 0;
}

void ScrfdDetector::generate_points(const int target_height, const int target_width) {
    if (center_points_is_update) return;
    // 8, 16, 32
    for (auto stride: feat_stride_fpn) {
        unsigned int num_grid_w = target_width / stride;
        unsigned int num_grid_h = target_height / stride;
        // y
        for (unsigned int i = 0; i < num_grid_h; ++i) {
            // x
            for (unsigned int j = 0; j < num_grid_w; ++j) {
                // num_anchors, col major
                for (unsigned int k = 0; k < num_anchors; ++k) {
                    SCRFDPoint point;
                    point.cx = (float) j;
                    point.cy = (float) i;
                    point.stride = (float) stride;
                    center_points[stride].push_back(point);
                }

            }
        }
    }

    center_points_is_update = true;
}

void ScrfdDetector::resize_unscale(const cv::Mat &mat, cv::Mat &mat_rs,
                           int target_height, int target_width,
                           SCRFDScaleParams &scale_params)
{
    if (mat.empty()) return;
    int img_height = static_cast<int>(mat.rows);
    int img_width = static_cast<int>(mat.cols);

    mat_rs = cv::Mat(target_height, target_width, CV_8UC3,
                     cv::Scalar(0, 0, 0));
    // scale ratio (new / old) new_shape(h,w)
    float w_r = (float) target_width / (float) img_width;
    float h_r = (float) target_height / (float) img_height;
    float r = std::min(w_r, h_r);
    // compute padding
    int new_unpad_w = static_cast<int>((float) img_width * r); // floor
    int new_unpad_h = static_cast<int>((float) img_height * r); // floor
    int pad_w = target_width - new_unpad_w; // >=0
    int pad_h = target_height - new_unpad_h; // >=0

    int dw = pad_w / 2;
    int dh = pad_h / 2;

    // resize with unscaling
    cv::Mat new_unpad_mat;
    // cv::Mat new_unpad_mat = mat.clone(); // may not need clone.
    cv::resize(mat, new_unpad_mat, cv::Size(new_unpad_w, new_unpad_h));
    new_unpad_mat.copyTo(mat_rs(cv::Rect(dw, dh, new_unpad_w, new_unpad_h)));

    // record scale params.
    scale_params.ratio = r;
    scale_params.dw = dw;
    scale_params.dh = dh;
    scale_params.flag = true;
}

void
ScrfdDetector::detect(const cv::Mat &mat, std::vector<ScrfdDetectInfo> &detected_boxes_kps,
                      float score_threshold,
                      float iou_threshold, unsigned int topk) {
    if (mat.empty()) return;
    SCRFDScaleParams scale_params;
    cv::Mat mat_rs;
    resize_unscale(mat, mat_rs, 640, 640, scale_params);

    // ### infer
    rknn_output outputs[rknnModelInfo.n_output];
    memset(outputs, 0, sizeof(outputs));
    RKNN::Common::run_infer(
            ctx, rknnModelInfo, mat_rs, outputs
    );

    // 3. rescale & exclude.
    std::vector<ScrfdDetectInfo> bbox_kps_collection;
    this->generate_bboxes_kps(scale_params, bbox_kps_collection,
                              (int8_t *) outputs[0].buf,
                              (int8_t *) outputs[1].buf,
                              (int8_t *) outputs[2].buf,
                              (int8_t *) outputs[3].buf,
                              (int8_t *) outputs[4].buf,
                              (int8_t *) outputs[5].buf,
                              (int8_t *) outputs[6].buf,
                              (int8_t *) outputs[7].buf,
                              (int8_t *) outputs[8].buf,
                              rknnModelInfo,
                              score_threshold, mat.rows, mat.cols, rknnModelInfo.out_zps,
                              rknnModelInfo.out_scales);
    // 4. hard nms with topk.
    this->nms_bboxes_kps(bbox_kps_collection, detected_boxes_kps, iou_threshold, topk);
    RKNN::Common::output_release(ctx, rknnModelInfo, outputs);
}

// Function to calculate the intersection area between two bounding boxes
float IntersectionArea(const ScrfdDetectInfo &box1, const ScrfdDetectInfo &box2) {
    float xA = std::max(box1.x1, box2.x1);
    float yA = std::max(box1.y1, box2.y1);
    float xB = std::min(box1.x2, box2.x2);
    float yB = std::min(box1.y2, box2.y2);

    if (xA < xB && yA < yB) {
        return (xB - xA) * (yB - yA);
    } else {
        return 0.0;
    }
}

// Function to calculate the Union area of two bounding boxes
float UnionArea(const ScrfdDetectInfo &box1, const ScrfdDetectInfo &box2) {
    float area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
    float area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
    return area1 + area2 - IntersectionArea(box1, box2);
}

// Function to calculate the Intersection over Union (IoU)
float CalculateIoU(const ScrfdDetectInfo &box1, const ScrfdDetectInfo &box2) {
    float intersection = IntersectionArea(box1, box2);
    float union_area = UnionArea(box1, box2);

    if (union_area == 0.0) {
        return 0.0;
    }

    return intersection / union_area;
}

void ScrfdDetector::nms_bboxes_kps(std::vector<ScrfdDetectInfo> &input, std::vector<ScrfdDetectInfo> &output,
                                   float iou_threshold, unsigned int topk) {
    if (input.empty()) return;
    std::sort(
            input.begin(), input.end(),
            [](const ScrfdDetectInfo &a, const ScrfdDetectInfo &b) { return a.score > b.score; }
    );
    const unsigned int box_num = input.size();
    std::vector<int> merged(box_num, 0);

    unsigned int count = 0;
    for (unsigned int i = 0; i < box_num; ++i) {
        if (merged[i]) continue;
        std::vector<ScrfdDetectInfo> buf;

        buf.push_back(input[i]);
        merged[i] = 1;

        for (unsigned int j = i + 1; j < box_num; ++j) {
            if (merged[j]) continue;

            float iou = CalculateIoU(input[i], input[j]);

            if (iou > iou_threshold) {
                merged[j] = 1;
                buf.push_back(input[j]);
            }
        }
        output.push_back(buf[0]);

        // keep top k
        count += 1;
        if (count >= topk)
            break;
    }
}

void ScrfdDetector::generate_bboxes_kps(const SCRFDScaleParams &scale_params,
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
                                        float score_threshold, float img_height, float img_width,
                                        std::vector<int32_t> out_zps, std::vector<float> out_scales
) {
    // generate center points.
    int input_height = input_node_dims.at(2); // e.g 640
    int input_width = input_node_dims.at(3); // e.g 640
    this->generate_points(input_height, input_width);
    bbox_kps_collection.clear();

    this->generate_bboxes_kps_single_stride(scale_params, score_8, bbox_8, kps_8,
                                            out_zps[0], out_scales[0],
                                            out_zps[3], out_scales[3],
                                            out_zps[6], out_scales[6],
                                            rknnModelInf,
                                            8,
                                            score_threshold,
                                            img_height, img_width,
                                            bbox_kps_collection);
    this->generate_bboxes_kps_single_stride(scale_params, score_16, bbox_16, kps_16,
                                            out_zps[1], out_scales[1],
                                            out_zps[4], out_scales[4],
                                            out_zps[7], out_scales[7],
                                            rknnModelInf,
                                            16, score_threshold,
                                            img_height, img_width, bbox_kps_collection);
    this->generate_bboxes_kps_single_stride(scale_params, score_32, bbox_32, kps_32,
                                            out_zps[2], out_scales[2],
                                            out_zps[5], out_scales[5],
                                            out_zps[8], out_scales[8],
                                            rknnModelInf,
                                            32, score_threshold,
                                            img_height, img_width, bbox_kps_collection);
}

ScrfdDetector::ScrfdDetector() {

}

ScrfdDetector::~ScrfdDetector() {
    rknn_destroy(ctx);
}

void ScrfdDetector::resizeAndPadImage(const cv::Mat &rgbImage, int maxWidth,
                                      int maxHeight, int padding, cv::Mat &paddedImg) {

    // Tính tỉ lệ và kích thước mới cho việc resize
    double scale = std::min(maxWidth / (double) rgbImage.cols, maxHeight / (double) rgbImage.rows);
    cv::Size newSize((int) (rgbImage.cols * scale), (int) (rgbImage.rows * scale));

    // Resize ảnh
    cv::Mat resizedImage;
    cv::resize(rgbImage, resizedImage, newSize);

    // Tạo ảnh mới với padding
    // Kích thước ban đầu của ảnh
    int originalWidth = resizedImage.cols;
    int originalHeight = resizedImage.rows;

    // Tính kích thước mới tối thiểu, đảm bảo padding tối thiểu padding px cho mỗi cạnh
    int minWidth = originalWidth + padding * 2;
    int minHeight = originalHeight + padding * 2;

    // Điều chỉnh kích thước để chia hết cho 16
    int newWidth = ((minWidth + 15) / 16) * 16;
    int newHeight = ((minHeight + 15) / 16) * 16;

    // Tính toán số lượng padding cần thêm vào mỗi cạnh
    int padTop = (newHeight - originalHeight) / 2;
    int padBottom = newHeight - originalHeight - padTop;
    int padLeft = (newWidth - originalWidth) / 2;
    int padRight = newWidth - originalWidth - padLeft;

    // Thêm padding vào ảnh
    cv::copyMakeBorder(resizedImage, paddedImg, padTop, padBottom, padLeft, padRight, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
}

void ScrfdDetector::detectInJpegData(const cv::Mat &rgbImage,
                                     std::vector<ScrfdDetectInfo> &detected_boxes_kps,
                                     float score_threshold) {
    cv::Mat paddedImg;
    resizeAndPadImage(rgbImage, 520, 520, 100, paddedImg);

    detect(
            paddedImg,
            detected_boxes_kps,
            score_threshold
    );
}

