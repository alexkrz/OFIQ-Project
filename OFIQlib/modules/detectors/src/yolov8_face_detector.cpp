#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <yolov8_face_detector.h>

static inline float sigmoid_x(float x)
{
	return static_cast<float>(1.f / (1.f + exp(-x)));
}

YOLOv8_face::YOLOv8_face(std::string modelpath, float confThreshold, float nmsThreshold)
{
	this->confThreshold = confThreshold;
	this->nmsThreshold = nmsThreshold;
	this->net = cv::dnn::readNet(modelpath);
}

cv::Mat YOLOv8_face::resize_image(cv::Mat srcimg, int *newh, int *neww, int *padh, int *padw)
{
	int srch = srcimg.rows, srcw = srcimg.cols;
	*newh = this->inpHeight;
	*neww = this->inpWidth;
	cv::Mat dstimg;
	if (this->keep_ratio && srch != srcw) {
		float hw_scale = (float)srch / srcw;
		if (hw_scale > 1) {
			*newh = this->inpHeight;
			*neww = int(this->inpWidth / hw_scale);
			resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
			*padw = int((this->inpWidth - *neww) * 0.5);
			copyMakeBorder(dstimg, dstimg, 0, 0, *padw, this->inpWidth - *neww - *padw, cv::BORDER_CONSTANT, 0);
		}
		else {
			*newh = (int)this->inpHeight * hw_scale;
			*neww = this->inpWidth;
			resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
			*padh = (int)(this->inpHeight - *newh) * 0.5;
			copyMakeBorder(dstimg, dstimg, *padh, this->inpHeight - *newh - *padh, 0, 0, cv::BORDER_CONSTANT, 0);
		}
	}
	else {
		resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
	}
	return dstimg;
}

void YOLOv8_face::drawResults(float conf, int left, int top, int right, int bottom, cv::Mat& frame, std::vector<cv::Point> landmark)   // Draw the predicted bounding box
{
	//Draw a rectangle displaying the bounding box
	cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 255), 3);

	//Get the label for the class name and its confidence
	std::string label = cv::format("face:%.2f", conf);

	putText(frame, label, cv::Point(left, top-5), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
	for (int i = 0; i < 5; i++)
	{
		circle(frame, landmark[i], 4, cv::Scalar(0, 255, 0), -1);
	}
}

void YOLOv8_face::softMax(const float* x, float* y, int length)
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

void YOLOv8_face::evalResults(cv::Mat out, std::vector<cv::Rect>& boxes, std::vector<float>& confidences, std::vector<std::vector<cv::Point>>& landmarks, int imgh,int imgw, float ratioh, float ratiow, int padh, int padw)
{
	const int feat_h = out.size[2];
	const int feat_w = out.size[3];
	std::cout << out.size[1] << "," << out.size[2] << "," << out.size[3] << std::endl;
	const int stride = (int)ceil((float)inpHeight / feat_h);
	const int area = feat_h * feat_w;
	float* ptr = (float*)out.data;
	float* ptr_cls = ptr + area * reg_max * 4;
	float* ptr_kp = ptr + area * (reg_max * 4 + num_class);

	for (int i = 0; i < feat_h; i++)
	{
		for (int j = 0; j < feat_w; j++)
		{
			const int index = i * feat_w + j;
			int cls_id = -1;
			float max_conf = -10000;
			for (int k = 0; k < num_class; k++)
			{
				float conf = ptr_cls[k*area + index];
				if (conf > max_conf)
				{
					max_conf = conf;
					cls_id = k;
				}
			}
			float box_prob = sigmoid_x(max_conf);
			if (box_prob > this->confThreshold)
			{
				float pred_ltrb[4];
				float* dfl_value = new float[reg_max];
				float* dfl_softmax = new float[reg_max];
				for (int k = 0; k < 4; k++)
				{
					for (int n = 0; n < reg_max; n++)
					{
						dfl_value[n] = ptr[(k*reg_max + n)*area + index];
					}
					softMax(dfl_value, dfl_softmax, reg_max);

					float dis = 0.f;
					for (int n = 0; n < reg_max; n++)
					{
						dis += n * dfl_softmax[n];
					}

					pred_ltrb[k] = dis * stride;
				}
				float cx = (j + 0.5f)*stride;
				float cy = (i + 0.5f)*stride;
				float xmin = std::max((cx - pred_ltrb[0] - padw)*ratiow, 0.f);
				float ymin = std::max((cy - pred_ltrb[1] - padh)*ratioh, 0.f);
				float xmax = std::min((cx + pred_ltrb[2] - padw)*ratiow, float(imgw - 1));
				float ymax = std::min((cy + pred_ltrb[3] - padh)*ratioh, float(imgh - 1));
				cv::Rect box = cv::Rect(int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin));
				boxes.push_back(box);
				confidences.push_back(box_prob);

				std::vector<cv::Point> kpts(5);
				for (int k = 0; k < 5; k++)
				{
					float x = ((ptr_kp[(k * 3)*area + index] * 2 + j) * stride - padw) * ratiow;
					float y = ((ptr_kp[(k * 3 + 1)*area + index] * 2 + i) * stride - padh) * ratioh;

					kpts[k] = cv::Point(int(x), int(y));
				}
				landmarks.push_back(kpts);
			}
		}
	}
}

void YOLOv8_face::detect(cv::Mat& srcimg)
{
	int newh = 0, neww = 0, padh = 0, padw = 0;
	cv::Mat dst = this->resize_image(srcimg, &newh, &neww, &padh, &padw);
	cv::Mat blob;
	cv::dnn::blobFromImage(dst, blob, 1 / 255.0, cv::Size(this->inpWidth, this->inpHeight), cv::Scalar(0, 0, 0), true, false);
	this->net.setInput(blob);
	std::vector<cv::Mat> outs;
	
	this->net.forward(outs, this->net.getUnconnectedOutLayersNames());

	//generate proposals
	std::vector<cv::Rect> boxes;
	std::vector<float> confidences;
	std:: vector<std::vector<cv::Point>> landmarks;
	float ratioh = (float)srcimg.rows / newh, ratiow = (float)srcimg.cols / neww;

	evalResults(outs[0], boxes, confidences, landmarks, srcimg.rows, srcimg.cols, ratioh, ratiow, padh, padw);
	evalResults(outs[1], boxes, confidences, landmarks, srcimg.rows, srcimg.cols, ratioh, ratiow, padh, padw);
	evalResults(outs[2], boxes, confidences, landmarks, srcimg.rows, srcimg.cols, ratioh, ratiow, padh, padw);

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		cv::Rect box = boxes[idx];
		this->drawResults(confidences[idx], box.x, box.y, box.x + box.width, box.y + box.height, srcimg, landmarks[idx]);
	}
}


