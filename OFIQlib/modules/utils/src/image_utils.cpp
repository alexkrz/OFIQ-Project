/**
 * @file image_utils.cpp
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

#include "image_utils.h"
#include "FaceMeasures.h"
#include "FaceParts.h"
#include "landmarks.h"
#include <array>

#include <chrono>
using hrclock = std::chrono::high_resolution_clock;

using PartExtractor = OFIQ_LIB::modules::landmarks::PartExtractor;
using FaceParts = OFIQ_LIB::modules::landmarks::FaceParts;
using FaceMeasures = OFIQ_LIB::modules::landmarks::FaceMeasures;


namespace OFIQ_LIB
{
    double ColorConvert(double x)
    {
        return x <= 0.04045 ? x / 12.92 : pow((x + 0.055) / 1.055, 2.4);
    }

    std::array<double, 256> prepare_COLOR_CVT_LUT()
    {
        std::array<double, 256> lut;
        for (size_t i = 0; i < 256; i++)
            lut[i] = ColorConvert((double)i / 255.0);
        return lut;
    }

    const static std::array<double, 256> COLOR_CVT_LUT = prepare_COLOR_CVT_LUT();

    double Cubic(double x, double k, double eps)
    {
        if (x <= eps)
        {
            return ((k * x) + 16) / 116;
        }

        return std::cbrt(x);
    }

    void ConvertBGRToCIELAB(const cv::Mat& rgbImage, double& a, double& b)
    {
        double k = 24289 / 27.0;
        double eps = 216 / 24389.0;

        std::vector<cv::Mat> channels;
        cv::split(rgbImage, channels);
        double R = mean(channels[2])[0] / 255.0;
        double G = mean(channels[1])[0] / 255.0;
        double B = mean(channels[0])[0] / 255.0;

        double R_L = ColorConvert(R);
        double G_L = ColorConvert(G);
        double B_L = ColorConvert(B);

        double X = R_L * 0.43605 + G_L * 0.38508 + B_L * 0.14309;
        double Y = R_L * 0.22249 + G_L * 0.71689 + B_L * 0.06062;
        double Z = R_L * 0.01393 + G_L * 0.09710 + B_L * 0.71419;

        double X_R = X / 0.964221;
        double Y_R = Y;
        double Z_R = Z / 0.825211;

        double F_X = Cubic(X_R, k, eps);
        double F_Y = Cubic(Y_R, k, eps);
        double F_Z = Cubic(Z_R, k, eps);

        a = 500.0 * (F_X - F_Y);
        b = 200.0 * (F_Y - F_Z);
    }

    cv::Mat GetLuminanceImageFromBGR(const cv::Mat& bgrImage)
    {
        cv::Mat L = cv::Mat::zeros(bgrImage.rows, bgrImage.cols, CV_8U);
        using Pixel = cv::Point3_<uint8_t>;
        bgrImage.forEach<Pixel>(
            [&L](const Pixel& bgr, const int coords[])
            {
                double y = 0.2126 * COLOR_CVT_LUT[bgr.z] + // R
                           0.7152 * COLOR_CVT_LUT[bgr.y] + // G
                           0.0722 * COLOR_CVT_LUT[bgr.x];  // B
                L.at<uint8_t>(coords[0], coords[1]) = (uint8_t)floor(y * 255 + 0.5);
            });
        return L;
    }

    void CalculateReferencePoints(
        const OFIQ::FaceLandmarks& landmarks,
        OFIQ::LandmarkPoint& leftEyeCenter,
        OFIQ::LandmarkPoint& rightEyeCenter,
        double& interEyeDistance,
        double& eyeMouthDistance)
    {
        auto leftEyePoints = PartExtractor::getFacePart(landmarks, FaceParts::LEFT_EYE_CORNERS);
        auto rightEyePoints = PartExtractor::getFacePart(landmarks, FaceParts::RIGHT_EYE_CORNERS);
        leftEyeCenter = FaceMeasures::GetMiddle(leftEyePoints);
        rightEyeCenter = FaceMeasures::GetMiddle(rightEyePoints);

        interEyeDistance = FaceMeasures::GetDistance(leftEyeCenter, rightEyeCenter);
        auto eyeMidPoint = FaceMeasures::GetMiddle(OFIQ::Landmarks{leftEyeCenter, rightEyeCenter});

        auto mouthCenter = FaceMeasures::GetMiddle(
            PartExtractor::getPairsForPart(landmarks, FaceParts::MOUTH_CENTER));
        eyeMouthDistance = FaceMeasures::GetDistance(eyeMidPoint, mouthCenter);
    }

    void CalculateRegionOfInterest(
        cv::Rect& leftRegionOfInterest,
        cv::Rect& rightRegionOfInterest,
        const OFIQ::LandmarkPoint& leftEyeCenter,
        const OFIQ::LandmarkPoint& rightEyeCenter,
        const double interEyeDistance,
        const double eyeMouthDistance)
    {
        auto zoneSize = static_cast<int>(interEyeDistance * 0.3);
        rightRegionOfInterest.x = rightEyeCenter.x - zoneSize;
        rightRegionOfInterest.y = rightEyeCenter.y + static_cast<int>(eyeMouthDistance / 2);
        rightRegionOfInterest.height = rightRegionOfInterest.width = zoneSize;

        leftRegionOfInterest.x = leftEyeCenter.x;
        leftRegionOfInterest.y = leftEyeCenter.y + static_cast<int>(eyeMouthDistance / 2);
        leftRegionOfInterest.height = leftRegionOfInterest.width = zoneSize;
    }

    void GetNormalizedHistogram(
        const cv::Mat& luminanceImage, const cv::Mat& maskImage, cv::Mat1f& histogram)
    {
        int histSize = 256;
        std::vector<float> range = {0, 256};

        cv::calcHist(std::vector{luminanceImage}, {0}, maskImage, histogram, {histSize}, range);

        auto pixelsInHistogram = cv::sum(histogram).val[0];

        histogram = histogram / pixelsInHistogram;
    }

    double CalculateExposure(const Session& session, const ExposureRange& exposureRange)
    {
        double quality = 0.0;

        const auto& faceOcclusionMask = session.getFaceOcclusionSegmentationImage();
        const auto& faceMask = session.getAlignedFaceLandmarkedRegion();
        cv::Mat combinedFaceMask;
        cv::bitwise_and(faceMask, faceOcclusionMask, combinedFaceMask);

        auto luminanceImage = session.getAlignedFaceLuminance();
        quality = ComputeBrightnessAspect(luminanceImage, combinedFaceMask, exposureRange);
        return quality;
    }

    double ComputeBrightnessAspect(
        const cv::Mat& luminanceImage, const cv::Mat& maskImage, const ExposureRange& exposureRange)
    {
        int histSize = 256;
        std::vector<float> range = {0, 256};
        cv::Mat1f histogram;

        cv::calcHist(std::vector{luminanceImage}, {0}, maskImage, histogram, {histSize}, range);

        auto pixelsInHistogram = cv::sum(histogram).val[0];
        if (pixelsInHistogram == 0)
            return std::nan("");
        double rawScore = 0;
        for (int i = exposureRange[0]; i <= exposureRange[1]; i++)
        {
            rawScore += histogram.at<float>(i) / pixelsInHistogram;
        }

        return rawScore;
    }
}