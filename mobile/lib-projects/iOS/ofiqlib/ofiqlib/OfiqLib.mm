/**
 * @file OfiqLib.mm
 *
 * @copyright Copyright (c) 2025  Federal Office for Information Security, Germany
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

#import <ofiq_lib_impl.h>
#import "image_io.h"
#import <Foundation/Foundation.h>
#import "ofiqlib.h"

using namespace OFIQ_LIB;

const std::map<int,std::string> measurementMapping = {
        {0x41, "UnifiedQualityScore"},
        {0x42, "BackgroundUniformity"},
        {0x43, "IlluminationUniformity"},
        {-0x44, "Luminance"},
        {0x44, "LuminanceMean"},
        {0x45, "LuminanceVariance"},
        {0x46, "UnderExposurePrevention"},
        {0x47, "OverExposurePrevention"},
        {0x48, "DynamicRange"},
        {0x49, "Sharpness"},
        {0x4a, "CompressionArtifacts"},
        {0x4b, "NaturalColour"},
        {0x4c, "SingleFacePresent"},
        {0x4d, "EyesOpen"},
        {0x4e, "MouthClosed"},
        {0x4f, "EyesVisible"},
        {0x50, "MouthOcclusionPrevention"},
        {0x51, "FaceOcclusionPrevention"},
        {0x52, "InterEyeDistance"},
        {0x53, "HeadSize"},
        {-0x54, "CropOfTheFaceImage"},
        {0x54, "LeftwardCropOfTheFaceImage"},
        {0x55, "RightwardCropOfTheFaceImage"},
        {0x56, "MarginAboveOfTheFaceImage"},
        {0x57, "MarginBelowOfTheFaceImage"},
        {-0x58, "HeadPose"},
        {0x58, "HeadPoseYaw"},
        {0x59, "HeadPosePitch"},
        {0x5a, "HeadPoseRoll"},
        {0x5b, "ExpressionNeutrality"},
        {0x5c, "NoHeadCoverings"},
        {-1, "NotSet"}
};

std::map<std::string, int> measurementMappingReverse;

void initializeMeasurementReverseMapping() {
    for (const auto& pair: measurementMapping) {
        measurementMappingReverse[pair.second] = pair.first;
    }
}

@implementation OfiqLib

std::string convertNSStringToStdString(const NSString *nsString) {
    const char *cString = [nsString UTF8String];
    std::string cppString(cString);
    return cppString;
}

std::vector<unsigned char> convertNSDataToVector(NSData *data) {
    const unsigned char *bytes = static_cast<const unsigned char *>([data bytes]);
    NSUInteger length = [data length];
    return std::vector<unsigned char>(bytes, bytes + length);
}

- (void)initOFIQNative {
    auto* implPtr = new OFIQImpl();
    ofiqImplPointer = implPtr;
}

- (instancetype)init {
    self = [super init];
    [self initOFIQNative];
    return self;
}

- (NSString *)convertStdStringToNSString:(const std::string &)cppString {
    const char *cString = cppString.c_str();
    NSString *nsString = [NSString stringWithUTF8String:cString];
    return nsString;
}

- (ReturnCode)mapCppReturnCodeToObjc:(OFIQ::ReturnCode) cppReturnCode {
    
    ReturnCode code = ReturnCode::UNKNOWN_ERROR;
    
    switch (cppReturnCode) {
        case OFIQ::ReturnCode::Success:
            code = ReturnCode::SUCCESS;
            break;
        case OFIQ::ReturnCode::ImageReadingError:
            code = ReturnCode::IMAGE_READING_ERROR;
            break;
        case OFIQ::ReturnCode::ImageWritingError:
            code = ReturnCode::IMAGE_WRITING_ERROR;
            break;
        case OFIQ::ReturnCode::MissingConfigParamError:
            code = ReturnCode::MISSING_CONFIG_PARAM_ERROR;
            break;
        case OFIQ::ReturnCode::UnknownConfigParamError:
            code = ReturnCode::UNKNOWN_CONFIG_PARAM_ERROR;
            break;
        case OFIQ::ReturnCode::FaceDetectionError:
            code = ReturnCode::FACE_DETECTION_ERROR;
            break;
        case OFIQ::ReturnCode::FaceLandmarkExtractionError:
            code = ReturnCode::FACE_LANDMARK_EXTRACTION_ERROR;
            break;
        case OFIQ::ReturnCode::FaceOcclusionSegmentationError:
            code = ReturnCode::FACE_OCCLUSION_SEGMENTATION_ERROR;
            break;
        case OFIQ::ReturnCode::FaceParsingError:
            code = ReturnCode::FACE_PARSING_ERROR;
            break;
        case OFIQ::ReturnCode::UnknownError:
            code = ReturnCode::UNKNOWN_ERROR;
            break;
        case OFIQ::ReturnCode::QualityAssessmentError:
            code = ReturnCode::QUALITY_ASSESSMENT_ERROR;
            break;
        case OFIQ::ReturnCode::NotImplemented:
            code = ReturnCode::NOT_IMPLEMENTED;
            break;
    }
    
    return code;
}

- (InitializeResult *)initialize {
    
    // check wheter the c++ instance has been deleted
    if (ofiqImplPointer == nullptr) {
        [self initOFIQNative];
    }
    
    // The config file and the models must be located in the data directory of the app.
    // The path is created here, which is passed to the ofiqlib (c++)
    NSFileManager *fm = [NSFileManager defaultManager];
    NSBundle *bundle = [NSBundle bundleForClass:[self class]];
    NSURL *resourceURL = [bundle resourceURL];
    NSURL *dataURL = [resourceURL URLByAppendingPathComponent:@"data"];
    NSString *dataPath = [dataURL path];
    
    NSURL *fileURLConfig = [dataURL URLByAppendingPathComponent:@"ofiq_config.jaxn"];
    NSString *filePathConfig = [fileURLConfig path];
    BOOL configExists = [fm fileExistsAtPath:filePathConfig];
    
    if (!configExists) {
        NSLog(@"The config file does not exist.");
    }
    
    if (ofiqImplPointer != nullptr) {
        auto implPtr = static_cast<OFIQImpl*>(ofiqImplPointer);
        auto dataPathStr = convertNSStringToStdString(dataPath);
        auto configFileStr = convertNSStringToStdString(filePathConfig);
        
        auto ret = implPtr->initialize(dataPathStr, configFileStr);
        ReturnCode retObjc = [self mapCppReturnCodeToObjc:ret.code];
        NSString *info = [self convertStdStringToNSString:ret.info];
        
        initializeResult = [[InitializeResult alloc] initWithCode:retObjc info:info];
        return initializeResult;
    }
    
    return nullptr;
}

- (QualityMetrics *)faceQa:(UIImage *)uiimage {
    if (ofiqImplPointer != nullptr && initializeResult != nullptr && initializeResult.code == SUCCESS) {
        auto implPtr = static_cast<OFIQImpl*>(ofiqImplPointer);
        
        NSData *imageData = UIImagePNGRepresentation(uiimage);
        auto imageByteArray = convertNSDataToVector(imageData);
        
        OFIQ::Image image;
        OFIQ::ReturnStatus retStatus = readImageFromByteArray(imageByteArray, image);
        
        QualityMetrics *metrics = [[QualityMetrics alloc] init];
        if (retStatus.code != OFIQ::ReturnCode::Success) {
            ReturnCode retObjc = [self mapCppReturnCodeToObjc:retStatus.code];
            metrics.code = retObjc;
            metrics.time = 0;
            NSString *info = [self convertStdStringToNSString:retStatus.info];
            metrics.info = info;
            return metrics;
        }
        
        OFIQ::FaceImageQualityAssessment assessment;
        std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
        retStatus = implPtr->vectorQuality(image, assessment);
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        int time = (int)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        ReturnCode retObjc = [self mapCppReturnCodeToObjc:retStatus.code];
        metrics.code = retObjc;
        NSString *info = [self convertStdStringToNSString:retStatus.info];
        metrics.info = info;
        
        if (retStatus.code != OFIQ::ReturnCode::Success) {
            metrics.time = 0;
            return metrics;
        }
        
        NSMutableArray<QualityMeasureAssessment *> *measurements = [[NSMutableArray alloc] init];
        for (auto const& aResult : assessment.qAssessments) {
            const OFIQ::QualityMeasureResult& qaResult = aResult.second;
            
            auto it = measurementMapping.find((int)aResult.first);
            std::string qualityMeasure = "NotSet";
            if (it != measurementMapping.end()) {
                qualityMeasure = it->second;
            }
            
            QualityMeasureAssessment *qmaObj = [[QualityMeasureAssessment alloc] init];
            NSString *qMStr = [self convertStdStringToNSString:qualityMeasure];
            qmaObj.qualityMeasure = qMStr;
            qmaObj.rawScore = (float)qaResult.rawScore;
            qmaObj.scalarScore = (float)qaResult.scalar;
            
            [measurements addObject:qmaObj];
        }
        metrics.time = time;
        metrics.measurements = measurements;
        return metrics;
    }
    
    QualityMetrics *metrics = [[QualityMetrics alloc] init];
    metrics.code = UNKNOWN_ERROR;
    metrics.time = 0;
    NSString *info = @"ofiq not successful initialized";
    metrics.info = info;
    
    return metrics;
}

- (int)getMeasurementEnumValue:(NSString *)measurementName {
    auto measurementNameStr = convertNSStringToStdString(measurementName);
    static bool isInitialized = false;
    if(!isInitialized) {
        initializeMeasurementReverseMapping();
        isInitialized = true;
    }
    auto it = measurementMappingReverse.find(measurementNameStr);
    if (it != measurementMappingReverse.end()) {
        return it->second;
    } else {
        return -1;
    }
}

- (void)destroy {
    if (ofiqImplPointer != nullptr) {
        auto implPtr = static_cast<OFIQImpl*>(ofiqImplPointer);
        delete implPtr;
        ofiqImplPointer = nullptr;
        initializeResult = nullptr;
    }
}

- (void)dealloc {
    [self destroy];
}

- (NSString *)getCoreVersion {
    int major = 0;
    int minor = 0;
    int patch = 0;
    if (ofiqImplPointer != nullptr) {
        auto implPtr = static_cast<OFIQImpl*>(ofiqImplPointer);
        implPtr->getVersion(major, minor, patch);
    }
    NSString *resultString = [NSString stringWithFormat:@"%d.%d.%d", major, minor, patch];
    return resultString;
}

- (NSString *)getFrameworkVersion {
    NSBundle *bundle = [NSBundle bundleForClass:[self class]];
    NSString *version = [bundle objectForInfoDictionaryKey:@"CFBundleShortVersionString"];
    return version ?: @"Unknown";
}

@end
