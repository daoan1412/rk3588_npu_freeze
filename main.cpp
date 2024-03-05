#include <iostream>
#include "scrfd_detector.h"
#include <thread>

using namespace rockchips;

void drawBoundingBoxes(const std::vector<ScrfdDetectInfo>& detections, cv::Mat& image, const std::string& outputFilename) {
    cv::Mat imgWithBoxes = image.clone();

    for (const auto& detection : detections) {
        cv::Rect boundingBox(static_cast<int>(detection.x1),
                             static_cast<int>(detection.y1),
                             static_cast<int>(detection.x2 - detection.x1),
                             static_cast<int>(detection.y2 - detection.y1));

        // Vẽ hộp giới hạn
        cv::rectangle(imgWithBoxes, boundingBox, cv::Scalar(0, 255, 0), 2);

        // Hiển thị tọa độ box
        std::string boxCoordinates = "(" + std::to_string(detection.x1) + ", " +
                                     std::to_string(detection.y1) + ", " +
                                     std::to_string(detection.x2) + ", " +
                                     std::to_string(detection.y2) + ")";
        cv::putText(imgWithBoxes, boxCoordinates, cv::Point(boundingBox.x, boundingBox.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    }

    // Ghi ảnh với hộp giới hạn ra file JPEG
    cv::imwrite(outputFilename, imgWithBoxes);
}

void processImage(const std::string& imagePath, std::shared_ptr<ScrfdDetector> detector) {
    cv::Mat rgbImage;
    cv::Mat bgrImage = cv::imread(imagePath);

    if (bgrImage.empty()) {
        std::cerr << "Error: Could not read the image.\n";
        return;
    }

    cv::cvtColor(bgrImage, rgbImage, cv::COLOR_BGR2RGB);
    while (true) {
        std::vector<ScrfdDetectInfo> faceInfoList;
        detector->detect(rgbImage, faceInfoList);
        std::this_thread::sleep_for (std::chrono::milliseconds (10));
    }

//    drawBoundingBoxes(faceInfoList, bgrImage, "output_image.jpg");

}

int main() {
    std::vector<std::thread> threads;
    std::string imagePath = "images/9854034943_d621a062fe_o.jpg";
    // Launch threads
    for (int i = 0; i <5; i++) {
        auto scrfdDetector = std::make_shared<ScrfdDetector>();
        scrfdDetector->load_model("models/scrfd_rk3588.rknn");
        threads.emplace_back(processImage, imagePath, scrfdDetector);
    }

    // Join threads
    for (auto& thread : threads) {
        thread.join();
    }

    return 0;
}
