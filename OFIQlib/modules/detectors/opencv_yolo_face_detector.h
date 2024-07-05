/**
 * @file opencv_yolo_face_detector.h
 *
 * @copyright Copyright (c) 2024
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * @brief Implementation of a face detector using the YOLOv8 face detector CNN.
 */

#pragma once

#include "Configuration.h"
#include "detectors.h"
#include <memory>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

/**
 * @brief Provides face detector implementations.
 */
namespace OFIQ_LIB::modules::detectors
{

    /**
     * @brief Implementation of a face detector using the YOLOv8 face detector CNN.
     */
    class YoloFaceDetector : public OFIQ_LIB::FaceDetectorInterface
    {
    public:
        /**
         * @brief Constructor of the YoloFaceDetector.
         *
         * @param modelPath Path to the YOLOv8 model.
         * @param confThreshold Confidence threshold for detection.
         * @param nmsThreshold Non-maximum suppression threshold.
         */
        explicit YoloFaceDetector(const Configuration& config);

        /**
         * @brief Destructor of the YOLOv8FaceDetector.
         */
        ~YoloFaceDetector() override = default;

    protected:
        /**
         * @brief Implementation of the face detection method.
         *
         * @param session Session containing relevant information for the current task.
         * @return std::vector<OFIQ::BoundingBox> Bounding boxes of the detected faces.
         */
        std::vector<OFIQ::BoundingBox> UpdateFaces(OFIQ_LIB::Session& session) override;

    private:
        /**
         * @brief Resize the image while keeping the aspect ratio.
         *
         * @param srcImg Source image to resize.
         * @param newHeight New height of the image.
         * @param newWidth New width of the image.
         * @param padHeight Padding height.
         * @param padWidth Padding width.
         * @return cv::Mat Resized image.
         */
        cv::Mat resizeImage(const cv::Mat& srcImg, int* newHeight, int* newWidth, int* padHeight, int* padWidth);

        /**
         * @brief Apply softmax to the input data.
         *
         * @param x Input data.
         * @param y Output data after applying softmax.
         * @param length Length of the input data.
         */
        void softmax(const float* x, float* y, int length);

        /**
         * @brief Generate proposals from the model output.
         *
         * @param out Model output.
         * @param boxes Vector to store detected boxes.
         * @param confidences Vector to store confidences of the detected boxes.
         * @param landmarks Vector to store landmarks of the detected boxes.
         * @param imgHeight Height of the original image.
         * @param imgWidth Width of the original image.
         * @param ratioHeight Height ratio for resizing.
         * @param ratioWidth Width ratio for resizing.
         * @param padHeight Padding height.
         * @param padWidth Padding width.
         */
        void generateProposal(const cv::Mat& out, std::vector<cv::Rect>& boxes, std::vector<float>& confidences, std::vector<std::vector<cv::Point>>& landmarks, int imgHeight, int imgWidth, float ratioHeight, float ratioWidth, int padHeight, int padWidth);

        /**
         * @brief Draw bounding box and landmarks on the image.
         *
         * @param conf Confidence score of the detection.
         * @param left Left coordinate of the bounding box.
         * @param top Top coordinate of the bounding box.
         * @param right Right coordinate of the bounding box.
         * @param bottom Bottom coordinate of the bounding box.
         * @param frame Image on which to draw.
         * @param landmark Landmarks to draw.
         */
        void drawPrediction(float conf, int left, int top, int right, int bottom, cv::Mat& frame, const std::vector<cv::Point>& landmark);

        void detect(cv::Mat& srcImg);

        /**
         * @brief Instance of an OpenCV dnn::Net.
         */
        std::shared_ptr<cv::dnn::Net> net{nullptr};

        /**
         * @brief Confidence threshold used for face detection.
         */
        float confThreshold;

        /**
         * @brief Non-maximum suppression threshold.
         */
        float nmsThreshold;

        /**
         * @brief Flag to keep aspect ratio when resizing.
         */
        const bool keepRatio = true;

        /**
         * @brief Input width for the model.
         */
        const int inpWidth = 640;

        /**
         * @brief Input height for the model.
         */
        const int inpHeight = 640;

        /**
         * @brief Number of classes in the model.
         */
        const int numClasses = 1;

        /**
         * @brief Maximum value for the regression.
         */
        const int regMax = 16;

        /**
         * @brief Apply sigmoid function.
         *
         * @param x Input value.
         * @return float Output value after applying sigmoid.
         */
        static inline float sigmoid(float x)
        {
            return 1.0f / (1.0f + std::exp(-x));
        }
    };
}
