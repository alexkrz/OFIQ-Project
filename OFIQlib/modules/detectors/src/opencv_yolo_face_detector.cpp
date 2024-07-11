/**
 * @file opencv_yolo_face_detector.cpp
 *
 * @brief Implementation of a face detector using the YOLOv8 face detector CNN.
 */

#include "opencv_yolo_face_detector.h"
#include "OFIQError.h"
#include "utils.h"
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace OFIQ;
using namespace cv;
using namespace std;

namespace OFIQ_LIB::modules::detectors
{
    YoloFaceDetector::YoloFaceDetector(const Configuration& config)
    {
        const std::string pathPrefix = "params.detector.yolo.";
        const std::string paramModel = pathPrefix + "model_path";
        const std::string paramConfidenceThreshold = pathPrefix + "confidence_thr";
        const std::string paramNmsThreshold = pathPrefix + "nms_thr";

        this->confThreshold = config.GetNumber(paramConfidenceThreshold);
        this->nmsThreshold = config.GetNumber(paramNmsThreshold);
        const auto fileNameModel = config.getDataDir() + "/" + config.GetString(paramModel);

        // Load the YOLOv8 model
        try
        {
            /* code */
            net = std::make_shared<cv::dnn::Net>(cv::dnn::readNet(fileNameModel));
        }
        catch (const std::exception&)
        {
            throw OFIQError(
                ReturnCode::FaceDetectionError,
                "failed to initialize Yolo face detector");
        }
    }

    cv::Mat YoloFaceDetector::resizeImage(const cv::Mat& srcImg, int* newHeight, int* newWidth, int* padHeight, int* padWidth)
    {
        int srch = srcImg.rows, srcw = srcImg.cols;
        *newHeight = this->inpHeight;
        *newWidth = this->inpWidth;
        cv::Mat dstimg;
        if (this->keepRatio && srch != srcw)
        {
            float hw_scale = (float)srch / srcw;
            if (hw_scale > 1)
            {
                *newHeight = this->inpHeight;
                *newWidth = int(this->inpWidth / hw_scale);
                resize(srcImg, dstimg, cv::Size(*newWidth, *newHeight), cv::INTER_AREA);
                *padWidth = int((this->inpWidth - *newWidth) * 0.5);
                copyMakeBorder(dstimg, dstimg, 0, 0, *padWidth, this->inpWidth - *newWidth - *padWidth, cv::BORDER_CONSTANT, 0);
            }
            else
            {
                *newHeight = (int)this->inpHeight * hw_scale;
                *newWidth = this->inpWidth;
                resize(srcImg, dstimg, cv::Size(*newWidth, *newHeight), cv::INTER_AREA);
                *padHeight = (int)(this->inpHeight - *newHeight) * 0.5;
                copyMakeBorder(dstimg, dstimg, *padHeight, this->inpHeight - *newHeight - *padHeight, 0, 0, cv::BORDER_CONSTANT, 0);
            }
        }
        else
        {
            resize(srcImg, dstimg, cv::Size(*newWidth, *newHeight), cv::INTER_AREA);
        }
        return dstimg;
    }

    void YoloFaceDetector::drawPrediction(float conf, int left, int top, int right, int bottom, cv::Mat& frame, const std::vector<cv::Point>& landmark)
    {
        // Draw a rectangle displaying the bounding box
        rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(191, 0, 207), cv::LINE_AA);

        // Get the label for the class name and its confidence
        std::string label = cv::format("face:%.2f", conf);

        // Display the label at the top of the bounding box
        /*int baseLine;
        Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        top = max(top, labelSize.height);
        rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);*/
        putText(frame, label, cv::Point(left, top - 5), cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(0, 0, 0), 3);
        for (int i = 0; i < 5; i++)
        {
            circle(frame, landmark[i], 9, cv::Scalar(224, 110, 0), cv::FILLED);
        }
    }

    void YoloFaceDetector::softmax(const float* x, float* y, int length)
    {
        float sum = 0;
        int i = 0;
        for (i = 0; i < length; i++)
        {
            y[i] = exp(x[i]);
            sum += y[i];
        }
        for (i = 0; i < length; i++)
        {
            y[i] /= sum;
        }
    }

    void YoloFaceDetector::generateProposal(const cv::Mat& out, std::vector<cv::Rect>& boxes, std::vector<float>& confidences, std::vector<std::vector<cv::Point>>& landmarks, int imgHeight, int imgWidth, float ratioHeight, float ratioWidth, int padHeight, int padWidth)
    {
        const int feat_h = out.size[2];
        const int feat_w = out.size[3];
        std::cout << out.size[1] << "," << out.size[2] << "," << out.size[3] << std::endl;
        const int stride = (int)ceil((float)inpHeight / feat_h);
        const int area = feat_h * feat_w;
        float* ptr = (float*)out.data;
        float* ptr_cls = ptr + area * regMax * 4;
        float* ptr_kp = ptr + area * (regMax * 4 + numClasses);

        for (int i = 0; i < feat_h; i++)
        {
            for (int j = 0; j < feat_w; j++)
            {
                const int index = i * feat_w + j;
                int cls_id = -1;
                float max_conf = -10000;
                for (int k = 0; k < numClasses; k++)
                {
                    float conf = ptr_cls[k * area + index];
                    if (conf > max_conf)
                    {
                        max_conf = conf;
                        cls_id = k;
                    }
                }
                float box_prob = sigmoid(max_conf);
                if (box_prob > this->confThreshold)
                {
                    float pred_ltrb[4];
                    float* dfl_value = new float[regMax];
                    float* dfl_softmax = new float[regMax];
                    for (int k = 0; k < 4; k++)
                    {
                        for (int n = 0; n < regMax; n++)
                        {
                            dfl_value[n] = ptr[(k * regMax + n) * area + index];
                        }
                        softmax(dfl_value, dfl_softmax, regMax);

                        float dis = 0.f;
                        for (int n = 0; n < regMax; n++)
                        {
                            dis += n * dfl_softmax[n];
                        }

                        pred_ltrb[k] = dis * stride;
                    }
                    float cx = (j + 0.5f) * stride;
                    float cy = (i + 0.5f) * stride;
                    float xmin = std::max((cx - pred_ltrb[0] - padWidth) * ratioWidth, 0.f); /// 还原回到原图
                    float ymin = std::max((cy - pred_ltrb[1] - padHeight) * ratioHeight, 0.f);
                    float xmax = std::min((cx + pred_ltrb[2] - padWidth) * ratioWidth, float(imgWidth - 1));
                    float ymax = std::min((cy + pred_ltrb[3] - padHeight) * ratioHeight, float(imgHeight - 1));
                    cv::Rect box = cv::Rect(int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin));
                    boxes.push_back(box);
                    confidences.push_back(box_prob);

                    std::vector<cv::Point> kpts(5);
                    for (int k = 0; k < 5; k++)
                    {
                        float x = ((ptr_kp[(k * 3) * area + index] * 2 + j) * stride - padWidth) * ratioWidth; /// 还原回到原图
                        float y = ((ptr_kp[(k * 3 + 1) * area + index] * 2 + i) * stride - padHeight) * ratioHeight;
                        /// float pt_conf = sigmoid_x(ptr_kp[(k * 3 + 2)*area + index]);
                        kpts[k] = cv::Point(int(x), int(y));
                    }
                    landmarks.push_back(kpts);
                }
            }
        }
    }

    std::vector<OFIQ::BoundingBox> YoloFaceDetector::detect(cv::Mat& srcImg)
    {
        int newh = 0, neww = 0, padh = 0, padw = 0;
        cv::Mat dst = this->resizeImage(srcImg, &newh, &neww, &padh, &padw);
        cv::Mat blob;
        cv::dnn::blobFromImage(dst, blob, 1 / 255.0, cv::Size(this->inpWidth, this->inpHeight), cv::Scalar(0, 0, 0), true, false);
        this->net->setInput(blob);
        std::vector<cv::Mat> outs;
        /// net.enableWinograd(false);
        this->net->forward(outs, this->net->getUnconnectedOutLayersNames());

        /////generate proposals
        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<std::vector<cv::Point>> landmarks;
        float ratioh = (float)srcImg.rows / newh, ratiow = (float)srcImg.cols / neww;

        generateProposal(outs[0], boxes, confidences, landmarks, srcImg.rows, srcImg.cols, ratioh, ratiow, padh, padw);
        generateProposal(outs[1], boxes, confidences, landmarks, srcImg.rows, srcImg.cols, ratioh, ratiow, padh, padw);
        generateProposal(outs[2], boxes, confidences, landmarks, srcImg.rows, srcImg.cols, ratioh, ratiow, padh, padw);

        // Perform non maximum suppression to eliminate redundant overlapping boxes with
        // lower confidences
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, indices);
        std::vector<OFIQ::BoundingBox> bb;

        for (size_t i = 0; i < indices.size(); ++i)
        {
            int idx = indices[i];
            Rect box = boxes[idx];

            if (confidences[idx] >= confThreshold)
            {
                bb.push_back(OFIQ::BoundingBox(box.x, box.y, box.width, box.height, FaceDetectorType::OPENCVYOLO));
            }
        }

        return bb;
    }

    std::vector<OFIQ::BoundingBox> YoloFaceDetector::UpdateFaces(OFIQ_LIB::Session& session)
    {

        cv::Mat image = copyToCvImage(session.image());

        std::vector<OFIQ::BoundingBox> boundingBoxes = detect(image);

        // Returning an empty vector of bounding boxes
        return boundingBoxes;
    }
}
