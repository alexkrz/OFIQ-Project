/**
 * @file OFIQImpl.cpp
 *
 * @copyright Copyright (c) 2024  Federal Office for Information Security, Germany
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
 * @author OFIQ development team
 */

#include "Configuration.h"
#include "Executor.h"
#include "FaceMeasures.h"
#include "OFIQError.h"
#include "image_io.h"
#include "ofiq_lib_impl.h"
#include "utils.h"

using namespace std;
using namespace OFIQ;
using namespace OFIQ_LIB;
using namespace OFIQ_LIB::modules::measures;

OFIQImpl::OFIQImpl() : m_emptySession({this->dummyImage, this->dummyAssement}) {}

ReturnStatus OFIQImpl::initialize(const std::string& configDir, const std::string& configFilename)
{
    try
    {
        this->config = std::make_unique<Configuration>(configDir, configFilename);
        CreateNetworks();
        m_executorPtr = CreateExecutor(m_emptySession);
    }
    catch (const OFIQError& ex)
    {
        return {ex.whatCode(), ex.what()};
    }
    catch (const std::exception& ex)
    {
        return {ReturnCode::UnknownError, ex.what()};
    }

    return ReturnStatus(ReturnCode::Success);
}

ReturnStatus OFIQImpl::scalarQuality(const OFIQ::Image& face, double& quality)
{
    FaceImageQualityAssessment assessments;

    if (auto result = vectorQuality(face, assessments, "");
        result.code != ReturnCode::Success)
        return result;

    if (assessments.qAssessments.find(QualityMeasure::UnifiedQualityScore) != assessments.qAssessments.end())
        quality = assessments.qAssessments[QualityMeasure::UnifiedQualityScore].scalar;
    else
    {
        // just as an example - the scalarQuality is an average of all valid scalar measurements
        double sumScalars = 0;
        int numScalars = 0;
        for (auto const& [measureName, aResult] : assessments.qAssessments)
        {
            if (aResult.scalar != -1)
            {
                sumScalars += aResult.scalar;
                numScalars++;
            }
        }
        quality = numScalars != 0 ? sumScalars / numScalars : 0;
    }

    return ReturnStatus(ReturnCode::Success);
}

void previewWindow(std::string title, cv::Mat& image)
{
    cv::namedWindow(title, cv::WINDOW_NORMAL);
    cv::imshow(title, image);
    cv::waitKey(0);
}

void drawBoundingBox(cv::Mat& image, OFIQ::BoundingBox& bb, cv::Scalar& color)
{
    // Corner points of bounding box
    cv::Point2f top_left = cv::Point2f(bb.xleft, bb.ytop);
    cv::Point2f top_right = cv::Point2f(bb.xleft + bb.width, bb.ytop);

    cv::Point2f bottom_left = cv::Point2f(bb.xleft, bb.ytop + bb.height);
    cv::Point2f bottom_right = cv::Point2f(bb.xleft + bb.width, bb.ytop + bb.height);

    // draw top line
    cv::line(image, top_left, top_right, color, cv::LINE_AA);

    // draw left line
    cv::line(image, top_left, bottom_left, color, cv::LINE_AA);

    // draw bottom line
    cv::line(image, bottom_left, bottom_right, color, cv::LINE_AA);

    // draw right line
    cv::line(image, top_right, bottom_right, color, cv::LINE_AA);
}

void drawLandmarkPoint(cv::Mat& image, OFIQ::LandmarkPoint& fp, cv::Scalar& color, int index)
{
    // create center point (just for readability)
    cv::Point2i center = cv::Point2i(fp.x, fp.y);
    cv::Point2i text_origin = cv::Point2i(fp.x + 15, fp.y + 15);
    // draw landmark point
    cv::circle(image, center, 9, color, cv::FILLED);
    cv::putText(image, std::to_string(index), text_origin, cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(0, 0, 0), 3);
}

void visualizeBoundingBoxes(Session& session, const std::vector<OFIQ::BoundingBox>& boxes)
{
    cv::Mat image = copyToCvImage(session.image());

    // NOTE: color values are in BGR!
    cv::Scalar purple = cv::Scalar(191, 0, 207);

    // draw all bounding boxes onto the image
    for (BoundingBox bb : boxes)
    {
        drawBoundingBox(image, bb, purple);
    }

    // open window
    previewWindow("Bounding Boxes Preview", image);
}

cv::Vec3b determineColor(int i)
{
    switch (i)
    {
    // face skin (lightblue)
    case 1:
        return cv::Vec3b(204, 153, 51);
    // left eye brow (green)
    case 2:
        return cv::Vec3b(51, 153, 0);
    // right eye brow (green)
    case 3:
        return cv::Vec3b(51, 153, 0);
    // left eye (yellow)
    case 4:
        return cv::Vec3b(0, 204, 255);
    // right eye (yellow)
    case 5:
        return cv::Vec3b(0, 204, 255);
    // eyeglasses (red)
    case 6:
        return cv::Vec3b(0, 51, 255);
    // left ear (darkblue)
    case 7:
        return cv::Vec3b(102, 0, 0);
    // right ear (darkblue)
    case 8:
        return cv::Vec3b(102, 0, 0);
    // earring (pink)
    case 9:
        return cv::Vec3b(255, 102, 255);
    // nose (purple)
    case 10:
        return cv::Vec3b(255, 0, 153);
    // mouth (green)
    case 11:
        return cv::Vec3b(51, 153, 0);
    // upper lip (lime)
    case 12:
        return cv::Vec3b(0, 255, 0);
    // lower lip (aqua)
    case 13:
        return cv::Vec3b(255, 255, 0);
    // neck (brown)
    case 14:
        return cv::Vec3b(0, 51, 102);
    // necklace (gold)
    case 15:
        return cv::Vec3b(0, 204, 255);
    // clothing (steal)
    case 16:
        return cv::Vec3b(153, 153, 102);
    // hair (coral)
    case 17:
        return cv::Vec3b(153, 153, 255);
    // head covering (red)
    case 18:
        return cv::Vec3b(0, 51, 255);
    // covering errors (white)
    default:
        return cv::Vec3b(255, 255, 255);
    }
}

void visualizeLandmarks(Session& session, const std::vector<OFIQ::FaceLandmarks>& landmarks)
{
    cv::Mat image = copyToCvImage(session.image());

    // NOTE: color values are in BGR!
    cv::Scalar lightblue = cv::Scalar(224, 110, 0);

    // go through all detected facelandmarks for each face on the image
    int index = 0;
    for (FaceLandmarks fl : landmarks)
    {
        // go through eacht specific landmark point and draw it onto the image
        for (LandmarkPoint fp : fl.landmarks)
        {
            drawLandmarkPoint(image, fp, lightblue, index);
            index++;
        }
    }
    // open window
    previewWindow("Landmark Preview", image);
}

void visualizeFaceAlignment(Session& session)
{
    cv::Mat aligned = session.getAlignedFace();
    previewWindow("Face Alignment", aligned);
}

void visualizeSegmentationMask(Session& session)
{
    double alpha, beta;
    int row, col;

    alpha = 0.7;
    beta = 1 - alpha;

    cv::Mat image, segmentation, layered;

    segmentation = session.getFaceParsingImage();
    image = session.getAlignedFace();
    cv::Mat croppedImage = image(cv::Range(0, image.rows - 60), cv::Range(30, image.cols - 30));

    cv::resize(croppedImage, croppedImage, segmentation.size(), 0.0, 0.0, cv::INTER_NEAREST);

    for (col = 0; col < segmentation.cols; col++)
    {
        for (row = 0; row < segmentation.rows; row++)
        {
            cv::Vec3b pixel = segmentation.at<cv::Vec3b>(row, col);
            cv::Vec3b color;
            // std::cout << "Pixel value at (" << row << ", " << col << "): ["
            //          << (int)pixel[0] << ", " << (int)pixel[1] << ", " << (int)pixel[2] << "]" << std::endl;
            if (pixel[0] == 0)
            {
                color = croppedImage.at<cv::Vec3b>(row, col);
            }
            else
            {
                color = determineColor(pixel[0]);
            }

            // std::cout << "Color value at (" << row << ", " << col << "): ["
            //           << (int)color[0] << ", " << (int)color[1] << ", " << (int)color[2] << "]" << std::endl;

            segmentation.at<cv::Vec3b>(row, col) = color;
        }
    }

    cv::addWeighted(segmentation, alpha, croppedImage, beta, 0.0, layered);

    previewWindow("Segmentation Face Mask", layered);
}

void visualizeOcclusionMask(Session& session)
{
    cv::Mat image, occlusion, layered;
    double alpha, beta;
    int col, row;
    alpha = 0.35;
    beta = 1 - alpha;

    occlusion = session.getFaceOcclusionSegmentationImage() * 255.0f;
    image = session.getAlignedFace();

    for (col = 0; col < occlusion.cols; col++)
    {
        for (row = 0; row < occlusion.rows; row++)
        {
            cv::Vec3b pixel = occlusion.at<cv::Vec3b>(row, col);

            if (pixel[0] > 0)
            {
                occlusion.at<cv::Vec3b>(row, col) = cv::Vec3b(255, 0, 153);
            }
            else
            {
                occlusion.at<cv::Vec3b>(row, col) = image.at<cv::Vec3b>(row, col);
            }
        }
    }

    cv::addWeighted(occlusion, alpha, image, beta, 0.0, layered);
    previewWindow("Occlusion Face Mask", layered);
}

void visualizeLandmarkRegion(Session& session)
{
    cv::Mat image, regions, layered;
    int row, col;
    double alpha, beta;
    alpha = 0.35;
    beta = 1 - alpha;

    regions = session.getAlignedFaceLandmarkedRegion() * 255.0f;
    image = session.getAlignedFace();

    cv::cvtColor(regions, regions, cv::COLOR_GRAY2BGR);

    for (col = 0; col < regions.cols; col++)
    {
        for (row = 0; row < regions.rows; row++)
        {
            cv::Vec3b pixel = regions.at<cv::Vec3b>(row, col);

            if (pixel[0] > 0)
            {
                regions.at<cv::Vec3b>(row, col) = cv::Vec3b(102, 0, 0);
            }
            else
            {
                regions.at<cv::Vec3b>(row, col) = image.at<cv::Vec3b>(row, col);
            }
        }
    }
    cv::resize(regions, regions, image.size(), 0.0, 0.0, cv::INTER_NEAREST);

    cv::addWeighted(regions, alpha, image, beta, 0.0, layered);

    previewWindow("Face Landmark Region", layered);
}

static inline float sigmoid_x(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

cv::Mat YOLOv8resizeImage(cv::Mat srcimg, int* newh, int* neww, int* padh, int* padw)
{
    int srch = srcimg.rows, srcw = srcimg.cols;
    *newh = 640;
    *neww = 640;
    cv::Mat dstimg;
    if (srch != srcw)
    {
        float hw_scale = (float)srch / srcw;
        if (hw_scale > 1)
        {
            *newh = 640;
            *neww = int(640 / hw_scale);
            resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
            *padw = int((640 - *neww) * 0.5);
            copyMakeBorder(dstimg, dstimg, 0, 0, *padw, 640 - *neww - *padw, cv::BORDER_CONSTANT, 0);
        }
        else
        {
            *newh = (int)640 * hw_scale;
            *neww = 640;
            resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
            *padh = (int)(640 - *newh) * 0.5;
            copyMakeBorder(dstimg, dstimg, *padh, 640 - *newh - *padh, 0, 0, cv::BORDER_CONSTANT, 0);
        }
    }
    else
    {
        resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
    }
    return dstimg;
}

void YOLOv8drawResults(float conf, int left, int top, int right, int bottom, cv::Mat& frame, std::vector<cv::Point> landmark) // Draw the predicted bounding box
{

    // Draw a rectangle displaying the bounding box
    cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 255), 8);

    std::cout << "bin in YOLOV, I drew a rectangle" << endl;
    ;

    // Get the label for the class name and its confidence
    std::string label = cv::format("face:%.2f", conf);

    putText(frame, label, cv::Point(left, top - 5), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 255, 0), 2);
    for (int i = 0; i < 5; i++)
    {
        circle(frame, landmark[i], 4, cv::Scalar(0, 255, 0), -1);
    }
}

void YOLOv8showWindow(cv::Mat& srcimg)
{

    static const string kWinName = "Deep learning face detection use OpenCV";
    cv::namedWindow(kWinName, cv::WINDOW_NORMAL);
    cv::imshow(kWinName, srcimg);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void YOLOv8softMax(const float* x, float* y, int length)
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

void YOLOv8evalResults(cv::Mat out, std::vector<cv::Rect>& boxes, std::vector<float>& confidences, std::vector<std::vector<cv::Point>>& landmarks, int imgh, int imgw, float ratioh, float ratiow, int padh, int padw)
{
    const int feat_h = out.size[2];
    const int feat_w = out.size[3];
    std::cout << out.size[1] << "," << out.size[2] << "," << out.size[3] << std::endl;
    const int stride = (int)ceil((float)640 / feat_h);
    const int area = feat_h * feat_w;
    float* ptr = (float*)out.data;
    float* ptr_cls = ptr + area * 16 * 4;
    float* ptr_kp = ptr + area * (16 * 4 + 1);

    std::cout << "bin in YOLOV, in evalResult: " << endl;
    ;

    for (int i = 0; i < feat_h; i++)
    {
        for (int j = 0; j < feat_w; j++)
        {
            const int index = i * feat_w + j;
            int cls_id = -1;
            float max_conf = -10000;
            for (int k = 0; k < 1; k++)
            {
                float conf = ptr_cls[k * area + index];
                if (conf > max_conf)
                {
                    max_conf = conf;
                    cls_id = k;
                }
            }
            float box_prob = sigmoid_x(max_conf);
            if (box_prob > 0.45)
            {
                float pred_ltrb[4];
                float* dfl_value = new float[16];
                float* dfl_softmax = new float[16];
                for (int k = 0; k < 4; k++)
                {
                    for (int n = 0; n < 16; n++)
                    {
                        dfl_value[n] = ptr[(k * 16 + n) * area + index];
                    }
                    YOLOv8softMax(dfl_value, dfl_softmax, 16);

                    float dis = 0.f;
                    for (int n = 0; n < 16; n++)
                    {
                        dis += n * dfl_softmax[n];
                    }

                    pred_ltrb[k] = dis * stride;
                }
                float cx = (j + 0.5f) * stride;
                float cy = (i + 0.5f) * stride;
                float xmin = std::max((cx - pred_ltrb[0] - padw) * ratiow, 0.f);
                float ymin = std::max((cy - pred_ltrb[1] - padh) * ratioh, 0.f);
                float xmax = std::min((cx + pred_ltrb[2] - padw) * ratiow, float(imgw - 1));
                float ymax = std::min((cy + pred_ltrb[3] - padh) * ratioh, float(imgh - 1));
                cv::Rect box = cv::Rect(int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin));
                boxes.push_back(box);
                confidences.push_back(box_prob);

                std::vector<cv::Point> kpts(5);
                for (int k = 0; k < 5; k++)
                {
                    float x = ((ptr_kp[(k * 3) * area + index] * 2 + j) * stride - padw) * ratiow;
                    float y = ((ptr_kp[(k * 3 + 1) * area + index] * 2 + i) * stride - padh) * ratioh;

                    kpts[k] = cv::Point(int(x), int(y));
                }
                landmarks.push_back(kpts);
            }
        }
    }
}

void YOLOv8detect(cv::Mat& srcimg, cv::dnn::Net net)
{
    std::cout << "bin in YOLOV: " << endl;
    ;

    if (srcimg.empty())
    {
        std::cerr << "Fehler: Das Eingangsbild ist leer." << std::endl;
        return;
    }
    int newh = 0, neww = 0, padh = 0, padw = 0;
    cv::Mat dst = YOLOv8resizeImage(srcimg, &newh, &neww, &padh, &padw);
    std::cout << "bin in YOLOV, habe resized: " << endl;
    ;

    cv::Mat blob;
    cv::dnn::blobFromImage(dst, blob, 1 / 255.0, cv::Size(640, 640), cv::Scalar(0, 0, 0), true, false);
    net.setInput(blob);
    std::vector<cv::Mat> outs;
    std::cout << "bin in YOLOV, habe blob erstellt und input: " << endl;
    ;

    net.forward(outs, net.getUnconnectedOutLayersNames());
    std::cout << "bin in YOLOV, habe forward: " << endl;
    ;

    // generate proposals
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<std::vector<cv::Point>> landmarks;
    float ratioh = (float)srcimg.rows / newh, ratiow = (float)srcimg.cols / neww;

    YOLOv8evalResults(outs[0], boxes, confidences, landmarks, srcimg.rows, srcimg.cols, ratioh, ratiow, padh, padw);
    YOLOv8evalResults(outs[1], boxes, confidences, landmarks, srcimg.rows, srcimg.cols, ratioh, ratiow, padh, padw);
    YOLOv8evalResults(outs[2], boxes, confidences, landmarks, srcimg.rows, srcimg.cols, ratioh, ratiow, padh, padw);
    std::cout << "bin after YOLOVevalResult " << endl;
    ;

    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, 0.45f, 0.5f, indices);
    std::cout << "bin in YOLOV, habe boxes erstellt " << endl;
    ;

    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        std::cout << "bin in YOLOV, I want to draw boxes" << endl;
        ;

        YOLOv8drawResults(confidences[idx], box.x, box.y, box.x + box.width, box.y + box.height, srcimg, landmarks[idx]);
    }
}

void OFIQImpl::performPreprocessing(Session& session, const std::string& detector)
{
    if (detector == "" || detector == "ssd")
    {
        log("\t1. detectFaces ");
        // find Bounding Boxes
        std::vector<OFIQ::BoundingBox> faces = networks->faceDetector->detectFaces(session);

        if (faces.empty())
        {
            log("\n\tNo faces were detected, abort preprocessing\n");
            throw OFIQError(ReturnCode::FaceDetectionError, "No faces were detected");
        }
        session.setDetectedFaces(faces);

        visualizeBoundingBoxes(session, faces);

        log("2. estimatePose ");
        session.setPose(networks->poseEstimator->estimatePose(session));

        log("3. extractLandmarks ");
#ifdef OFIQ_SINGLE_FACE_PRESENT_WITH_TMETRIC
        session.setLandmarksAllFaces(networks->landmarkExtractor->extractLandmarksAllFaces(session, session.getDetectedFaces()));
        if (!session.getLandmarksAllFaces().empty())
        {
            session.setLandmarks(session.getLandmarksAllFaces().front());
        }
        else
        {
            session.setLandmarks(networks->landmarkExtractor->extractLandmarks(session));
        }
#else
        session.setLandmarks(networks->landmarkExtractor->extractLandmarks(session));
#endif

        visualizeLandmarks(session, session.getLandmarksAllFaces());

        log("4. alignFaceImage ");
        // aligned face requires the landmarks of the face thus it must come after the landmark extraction.
        alignFaceImage(session);

        visualizeFaceAlignment(session);

        log("5. getSegmentationMask ");
        // segmentation results for face_parsing
        session.setFaceParsingImage(OFIQ_LIB::copyToCvImage(
            networks->segmentationExtractor->GetMask(
                session,
                OFIQ_LIB::modules::segmentations::SegmentClassLabels::face),
            false));

        visualizeSegmentationMask(session);

        log("6. getFaceOcclusionMask ");
        session.setFaceOcclusionSegmentationImage(OFIQ_LIB::copyToCvImage(
            networks->faceOcclusionExtractor->GetMask(
                session,
                OFIQ_LIB::modules::segmentations::SegmentClassLabels::face),
            false));

        visualizeOcclusionMask(session);

        static const std::string alphaParamPath = "params.measures.FaceRegion.alpha";
        double alpha = 0.0f;
        try
        {
            alpha = this->config->GetNumber(alphaParamPath);
        }
        catch (...)
        {
            alpha = 0.0f;
        }

        log("7. getAlignedFaceMask ");

        session.setAlignedFaceLandmarkedRegion(
            OFIQ_LIB::modules::landmarks::FaceMeasures::GetFaceMask(
                session.getAlignedFaceLandmarks(),
                session.getAlignedFace().rows,
                session.getAlignedFace().cols,
                (float)alpha));

        visualizeLandmarkRegion(session);

        log("\npreprocessing finished\n");
    }
    else
    {
        std::cout << "bin in else YOLO: " << endl;
        cv::Mat srcimg = copyToCvImage(session.image());
        std::filesystem::path cwd = std::filesystem::current_path();
        std::cout << "Current working directory: " << cwd << std::endl;
        cv::dnn::Net net = cv::dnn::readNet("../../data/models/face_detection/yolov8n-face.onnx");
        std::cout << "habe die Network gelesen" << endl;
        YOLOv8detect(srcimg, net);
        YOLOv8showWindow(srcimg);
    }
}

void OFIQImpl::alignFaceImage(Session& session)
{
    auto landmarks = session.getLandmarks();
    OFIQ::FaceLandmarks alignedFaceLandmarks;
    alignedFaceLandmarks.type = landmarks.type;
    cv::Mat transformationMatrix;
    cv::Mat aligned = alignImage(session.image(), landmarks, alignedFaceLandmarks, transformationMatrix);

    session.setAlignedFace(aligned);
    session.setAlignedFaceLandmarks(alignedFaceLandmarks);
    session.setAlignedFaceTransformationMatrix(transformationMatrix);
}

ReturnStatus OFIQImpl::vectorQuality(
    const OFIQ::Image& image, OFIQ::FaceImageQualityAssessment& assessments, const std::string& detector)
{
    auto session = Session(image, assessments);
    std::cout << "bin in vectorQuality: " << endl;
    ;
    try
    {
        log("perform preprocessing:\n");
        std::cout << "bin in vectorQuality try: " << endl;
        ;
        performPreprocessing(session, detector);
    }
    catch (const OFIQError& e)
    {
        log("OFIQError: " + std::string(e.what()) + "\n");
        for (const auto& measure : m_executorPtr->GetMeasures())
        {
            auto qualityMeasure = measure->GetQualityMeasure();
            switch (qualityMeasure)
            {
            case QualityMeasure::Luminance:
                session.assessment().qAssessments[QualityMeasure::LuminanceMean] =
                    {0, -1, OFIQ::QualityMeasureReturnCode::FailureToAssess};
                session.assessment().qAssessments[QualityMeasure::LuminanceVariance] =
                    {0, -1, OFIQ::QualityMeasureReturnCode::FailureToAssess};
                break;
            case QualityMeasure::CropOfTheFaceImage:
                session.assessment().qAssessments[QualityMeasure::LeftwardCropOfTheFaceImage] =
                    {0, -1, OFIQ::QualityMeasureReturnCode::FailureToAssess};
                session.assessment().qAssessments[QualityMeasure::RightwardCropOfTheFaceImage] =
                    {0, -1, OFIQ::QualityMeasureReturnCode::FailureToAssess};
                session.assessment().qAssessments[QualityMeasure::UpwardCropOfTheFaceImage] =
                    {0, -1, OFIQ::QualityMeasureReturnCode::FailureToAssess};
                session.assessment().qAssessments[QualityMeasure::DownwardCropOfTheFaceImage] =
                    {0, -1, OFIQ::QualityMeasureReturnCode::FailureToAssess};
                break;
            case QualityMeasure::HeadPose:
                session.assessment().qAssessments[QualityMeasure::HeadPoseYaw] =
                    {0, -1, OFIQ::QualityMeasureReturnCode::FailureToAssess};
                session.assessment().qAssessments[QualityMeasure::HeadPosePitch] =
                    {0, -1, OFIQ::QualityMeasureReturnCode::FailureToAssess};
                session.assessment().qAssessments[QualityMeasure::HeadPoseRoll] =
                    {0, -1, OFIQ::QualityMeasureReturnCode::FailureToAssess};
                break;
            default:
                session.assessment().qAssessments[measure->GetQualityMeasure()] =
                    {0, -1, OFIQ::QualityMeasureReturnCode::FailureToAssess};
                break;
            }
        }

        return {e.whatCode(), e.what()};
    }

    log("execute assessments:\n");
    m_executorPtr->ExecuteAll(session);

    return ReturnStatus(ReturnCode::Success);
}

OFIQ_EXPORT std::shared_ptr<Interface> Interface::getImplementation()
{
    return std::make_shared<OFIQImpl>();
}
