/**
 * @file ofiqlib.cpp
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

#include <jni.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <string>
#include "ofiq_lib.h"
#include <ofiq_lib_impl.h>
#include "image_io.h"
#include "DataStreamAndroid.h"
#include <chrono>
#include <map>

using namespace OFIQ;
using namespace OFIQ_LIB;

jobject mapCppReturnCodeToJavaEnum(JNIEnv* env, ReturnCode cppReturnCode);

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

// An inverted map is required for the unit tests
std::map<std::string, int> measurementMappingReverse;

void initializeMeasurementReverseMapping() {
    for (const auto& pair : measurementMapping) {
        measurementMappingReverse[pair.second] = pair.first;
    }
}

extern "C" 
JNIEXPORT void JNICALL
Java_de_bund_bsi_ofiqmobile_lib_OfiqLib_initAssetManager(JNIEnv* env, jobject thiz, jobject javaAssetManager) {
    ::AAssetManager* nativeManager = AAssetManager_fromJava(env, javaAssetManager);
    OFIQ_LIB::SetAAssetManager(nativeManager); 
}

extern "C"
JNIEXPORT void JNICALL
Java_de_bund_bsi_ofiqmobile_lib_OfiqLib_init(JNIEnv *env, jobject thiz, jobject nativePointerObj) {
    auto* implPtr = new OFIQImpl();

    // Return the c++ pointer to the instance as parameter reference
    jclass longClass = env->FindClass("java/lang/Long");
    jmethodID longConstructor = env->GetMethodID(longClass, "<init>", "(J)V");
    jobject newLongObj = env->NewObject(longClass, longConstructor, reinterpret_cast<jlong>(implPtr));
    jclass nativePointerClass = env->GetObjectClass(nativePointerObj);
    jfieldID pointerField = env->GetFieldID(nativePointerClass, "longPointer", "Ljava/lang/Long;");
    env->SetObjectField(nativePointerObj, pointerField, newLongObj);
}

extern "C" __attribute__((visibility("default"))) __attribute__((used))
JNIEXPORT void Java_de_bund_bsi_ofiqmobile_lib_OfiqLib_destroy(JNIEnv *env, jobject obj, jlong ptr) {
    auto implPtr = static_cast<OFIQImpl*>((void*)ptr);
    delete implPtr;
}

//Method name needs to include package name. If package name contains underscores, you need to use escapeletters "1" (e.g _1).
extern "C" __attribute__((visibility("default"))) __attribute__((used))
JNIEXPORT jobject JNICALL Java_de_bund_bsi_ofiqmobile_lib_OfiqLib_initialize(JNIEnv *env, jobject obj, jstring configFolder, jstring configFile, jlong ptr) {
    auto implPtr = static_cast<OFIQImpl*>((void*)ptr);
    ReturnStatus ret = implPtr->initialize(
            env->GetStringUTFChars(configFolder, nullptr),
            env->GetStringUTFChars(configFile, nullptr));

    if (ret.code != ReturnCode::Success) {
        Java_de_bund_bsi_ofiqmobile_lib_OfiqLib_destroy(env, obj, (jlong)implPtr);
        implPtr = nullptr;
    }

    // create result class object
    jclass resultClass = env->FindClass("de/bund/bsi/ofiqmobile/lib/InitializeResult");
    jmethodID resultClassConstructor = env->GetMethodID(resultClass, "<init>", "(Lde/bund/bsi/ofiqmobile/lib/ReturnCode;Ljava/lang/String;)V");

    // create return code object. enum mapping
    jobject returnCodeObj = mapCppReturnCodeToJavaEnum(env, ret.code);

    jobject resultObj = env->NewObject(resultClass, resultClassConstructor, returnCodeObj, env->NewStringUTF(ret.info.c_str()));

    return resultObj;
}

extern "C" __attribute__((visibility("default"))) __attribute__((used))
JNIEXPORT jobject JNICALL Java_de_bund_bsi_ofiqmobile_lib_OfiqLib_assessQuality(JNIEnv * env, jobject obj, jlong ptr, jbyteArray imageByteArray) {
    auto implPtr = static_cast<OFIQImpl*>((void*)ptr);

    // read image and convert it to cv::Mat
    jsize length = env->GetArrayLength(imageByteArray);
    jbyte *byteArrayElements = env->GetByteArrayElements(imageByteArray, nullptr);
    // convert to std::vector<uchar>
    std::vector<uchar> data(reinterpret_cast<uchar*>(byteArrayElements), reinterpret_cast<uchar*>(byteArrayElements) + length);
    env->ReleaseByteArrayElements(imageByteArray, byteArrayElements, JNI_ABORT);

    Image image;
    ReturnStatus retStatus = readImageFromByteArray(data, image);

    // create result class object
    jclass resultClass = env->FindClass("de/bund/bsi/ofiqmobile/lib/QualityMetrics");
    jmethodID resultClassConstructor = env->GetMethodID(resultClass, "<init>", "(Lde/bund/bsi/ofiqmobile/lib/ReturnCode;ILjava/lang/String;Ljava/util/List;)V");

    jclass listClass = env->FindClass("java/util/ArrayList");
    jmethodID listConstructor = env->GetMethodID(listClass, "<init>", "()V");
    jmethodID listAdd = env->GetMethodID(listClass, "add", "(Ljava/lang/Object;)Z");
    jobject javaList = env->NewObject(listClass, listConstructor);

    if (retStatus.code != ReturnCode::Success) {
        // image read failed
        return env->NewObject(resultClass, resultClassConstructor, mapCppReturnCodeToJavaEnum(env, retStatus.code), 0, env->NewStringUTF(retStatus.info.c_str()), javaList);
    }


    FaceImageQualityAssessment assessment;
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    retStatus = implPtr->vectorQuality(image, assessment);
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    int time = (int)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    if (retStatus.code != ReturnCode::Success) {
        return env->NewObject(resultClass, resultClassConstructor, mapCppReturnCodeToJavaEnum(env, retStatus.code), 0, env->NewStringUTF(retStatus.info.c_str()), javaList);
    }

    jclass qmaClass = env->FindClass("de/bund/bsi/ofiqmobile/lib/QualityMeasureAssessment");
    jmethodID qmaConstructor = env->GetMethodID(qmaClass, "<init>", "(Ljava/lang/String;FF)V");

    for (auto const& aResult : assessment.qAssessments) {

        const QualityMeasureResult& qaResult = aResult.second;
        auto it = measurementMapping.find((int)aResult.first);

        std::string qualityMeasure = "NotSet";
        if(it != measurementMapping.end()) {
            qualityMeasure = it->second.c_str();
        }

        jobject qmaObj = env->NewObject(qmaClass, qmaConstructor, env->NewStringUTF(qualityMeasure.c_str()), (float)qaResult.rawScore, (float)qaResult.scalar);
        env->CallBooleanMethod(javaList, listAdd, qmaObj);
    }

    return env->NewObject(resultClass, resultClassConstructor, mapCppReturnCodeToJavaEnum(env, retStatus.code), time, env->NewStringUTF(retStatus.info.c_str()), javaList);
}

jobject mapCppReturnCodeToJavaEnum(JNIEnv* env, ReturnCode cppReturnCode) {
    jclass returnCodeClass = env->FindClass("de/bund/bsi/ofiqmobile/lib/ReturnCode");
    if (returnCodeClass == nullptr) {
        return nullptr;
    }

    const char* javaEnumReturnCode = "Lde/bund/bsi/ofiqmobile/lib/ReturnCode;";
    const char* enumFieldName = nullptr;

    switch (cppReturnCode) {
        case ReturnCode::Success:
            enumFieldName = "SUCCESS";
            break;
        case ReturnCode::ImageReadingError:
            enumFieldName = "IMAGE_READING_ERROR";
            break;
        case ReturnCode::ImageWritingError:
            enumFieldName = "IMAGE_WRITING_ERROR";
            break;
        case ReturnCode::MissingConfigParamError:
            enumFieldName = "MISSING_CONFIG_PARAM_ERROR";
            break;
        case ReturnCode::UnknownConfigParamError:
            enumFieldName = "UNKNOWN_CONFIG_PARAM_ERROR";
            break;
        case OFIQ::ReturnCode::FaceDetectionError:
            enumFieldName = "FACE_DETECTION_ERROR";
            break;
        case ReturnCode::FaceLandmarkExtractionError:
            enumFieldName = "FACE_LANDMARK_EXTRACTION_ERROR";
            break;
        case ReturnCode::FaceOcclusionSegmentationError:
            enumFieldName = "FACE_OCCLUSION_SEGMENTATION_ERROR";
            break;
        case ReturnCode::FaceParsingError:
            enumFieldName = "FACE_PARSING_ERROR";
            break;
        case ReturnCode::UnknownError:
            enumFieldName = "UNKNOWN_ERROR";
            break;
        case ReturnCode::QualityAssessmentError:
            enumFieldName = "QUALITY_ASSESSMENT_ERROR";
            break;
        case ReturnCode::NotImplemented:
            enumFieldName = "NOT_IMPLEMENTED";
            break;
		default:
			break;
    }

    jfieldID enumField = env->GetStaticFieldID(returnCodeClass, enumFieldName, javaEnumReturnCode);
    if (enumField == nullptr) {
        return nullptr;
    }

    jobject javaEnumValue = env->GetStaticObjectField(returnCodeClass, enumField);
    return javaEnumValue;
}

extern "C"
JNIEXPORT jint JNICALL
Java_de_bund_bsi_ofiqmobile_lib_OfiqLib_measurementMapping(JNIEnv *env, jobject thiz, jstring enum_name) {
    const char *chars = env->GetStringUTFChars(enum_name, nullptr);
    std::string enumVal(chars);
    env->ReleaseStringUTFChars(enum_name, chars);

    static bool isInitialized = false;
    if (!isInitialized) {
        initializeMeasurementReverseMapping();
        isInitialized = true;
    }

    auto it = measurementMappingReverse.find(enumVal);
    if (it != measurementMappingReverse.end()) {
        return it->second;
    } else {
        return -1;
    }
}

extern "C"
JNIEXPORT jstring JNICALL
Java_de_bund_bsi_ofiqmobile_lib_OfiqLib_getVersion(JNIEnv *env, jobject thiz, jlong ptr) {
    auto implPtr = static_cast<OFIQImpl*>((void*)ptr);
    int major = 0;
    int minor = 0;
    int patch = 0;
    implPtr->getVersion(major, minor, patch);
    std::string version = std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch);
    return env->NewStringUTF(version.c_str());
}
