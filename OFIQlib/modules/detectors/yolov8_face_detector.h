#include "OFIQError.h"
#include "utils.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sstream>

using namespace cv;
using namespace dnn;
using namespace std;

class YOLOv8_face
{
public:
    YOLOv8_face(string modelpath, float confThreshold, float nmsThreshold);
    void detect(Mat& frame);

private:
    Mat resize_image(Mat srcimg, int* newh, int* neww, int* padh, int* padw);
    const bool keep_ratio = true;
    const int inpWidth = 640;
    const int inpHeight = 640;
    float confThreshold;
    float nmsThreshold;
    const int num_class = 1;
    const int reg_max = 16;
    Net net;
    void softmax_(const float* x, float* y, int length);
    void generate_proposal(Mat out, vector<Rect>& boxes, vector<float>& confidences, vector<vector<Point>>& landmarks, int imgh, int imgw, float ratioh, float ratiow, int padh, int padw);
    void drawPred(float conf, int left, int top, int right, int bottom, Mat& frame, vector<Point> landmark);
};
