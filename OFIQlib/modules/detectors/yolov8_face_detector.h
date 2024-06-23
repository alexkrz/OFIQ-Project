#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


class YOLOv8_face
{
    public:

        YOLOv8_face(std::string modelpath, float confThreshold, float nmsThreshold);
        void detect(cv::Mat& frame);

    private:
        cv::Mat resize_image(cv::Mat srcimg, int *newh, int *neww, int *padh, int *padw);
        const bool keep_ratio = true;
        const int inpWidth = 640;
        const int inpHeight = 640;
        float confThreshold;
        float nmsThreshold;
        const int num_class = 1;
        const int reg_max = 16;
        cv::dnn::Net net;
        void softMax(const float* x, float* y, int length);
        void evalResults(cv::Mat out, std::vector<cv::Rect>& boxes, std::vector<float>& confidences, std::vector< std::vector<cv::Point>>& landmarks, int imgh, int imgw, float ratioh, float ratiow, int padh, int padw);
        void drawResults(float conf, int left, int top, int right, int bottom, cv::Mat& frame, std::vector<cv::Point> landmark);
};