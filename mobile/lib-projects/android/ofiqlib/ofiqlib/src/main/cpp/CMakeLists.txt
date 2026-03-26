# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html.
# For more examples on how to use CMake, see https://github.com/android/ndk-samples.

# Sets the minimum CMake version required for this project.
cmake_minimum_required(VERSION 3.22.1)

# Declares the project name. The project name can be accessed via ${ PROJECT_NAME},
# Since this is the top level CMakeLists.txt, the project name is also accessible
# with ${CMAKE_PROJECT_NAME} (both CMake variables are in-sync within the top level
# build script scope).
project("ofiqlib")
set(CMAKE_CXX_STANDARD 17)

set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../../../)
set(EXTERN_SOURCE_DIR ${ROOT_DIR}/extern)
set(OFIQLIB_SOURCE_DIR ${ROOT_DIR}/OFIQlib)
set(OPENCV_INCLUDE_DIR "${EXTERN_SOURCE_DIR}/OpenCV-android-sdk/sdk/native/jni/include/")
set(ONNXRUNTIME_LIB_DIR "${EXTERN_SOURCE_DIR}/onnxruntime-android/${ANDROID_ABI}")


file(READ "${EXTERN_SOURCE_DIR}/../Version.txt" ver)

string(REGEX MATCH "VERSION_MAJOR ([0-9])" _ ${ver})
set(ver_major ${CMAKE_MATCH_1})

string(REGEX MATCH "VERSION_MINOR ([0-9])" _ ${ver})
set(ver_minor ${CMAKE_MATCH_1})

string(REGEX MATCH "VERSION_PATCH ([0-9])" _ ${ver})
set(ver_patch ${CMAKE_MATCH_1})

string(REGEX MATCH "VERSION_SUFFIX ([0-9a-zA-Z])" _ ${ver})
set(ver_suffix ${CMAKE_MATCH_1})

set(FULL_PROJECT_VERSION ${ver_major}.${ver_minor}.${ver_patch})
message("version: ${FULL_PROJECT_VERSION}")

add_compile_definitions(ANDROID)
add_compile_definitions(OFIQ_VERSION_MAJOR=${ver_major})
add_compile_definitions(OFIQ_VERSION_MINOR=${ver_minor})
add_compile_definitions(OFIQ_VERSION_PATCH=${ver_patch})

include_directories(
        ${OPENCV_INCLUDE_DIR}
        "${EXTERN_SOURCE_DIR}/libs/flatbuffers/include"
        "${EXTERN_SOURCE_DIR}/onnxruntime"
        "${EXTERN_SOURCE_DIR}/libs/spdlog/include"
        "${EXTERN_SOURCE_DIR}/libs/json/include"
        "${EXTERN_SOURCE_DIR}/libs/magic_enum/include/magic_enum"
        "${EXTERN_SOURCE_DIR}/libs/di/include"
        "${EXTERN_SOURCE_DIR}/libs/PEGTL/include"
        "${EXTERN_SOURCE_DIR}/libs/abseil-cpp"
        "${EXTERN_SOURCE_DIR}/libs/gzip-hpp-0.1.0/include"
)

set(OpenCV_DIR "${EXTERN_SOURCE_DIR}/OpenCV-android-sdk/sdk/native/jni" )
find_package(OpenCV 4.5 REQUIRED)

find_library(zlib-lib z)

add_library(onnxruntime STATIC IMPORTED)
set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so)

# Define local include pathes
file(GLOB_RECURSE include_modules LIST_DIRECTORIES true "${OFIQLIB_SOURCE_DIR}/modules/[^\.]*$")
list(FILTER include_modules EXCLUDE REGEX "/src$")
include_directories (
        ${OFIQLIB_SOURCE_DIR}/include
        ${include_modules}
)

# Find all source files

add_definitions(-DOFIQ_EXPORTS)
add_definitions(-DOFIQ_SINGLE_FACE_PRESENT_WITH_TMETRIC)

include(${ROOT_DIR}/cmake/SourceDefinition.cmake)

list(APPEND OFIQ_LINK_LIB_LIST
        onnxruntime
        android
        ${zlib-lib}
)

add_library (ofiq_objlib OBJECT
        ${module_sources}
        ${libImplementationSources})


target_link_libraries(ofiq_objlib
        PRIVATE ${OFIQ_LINK_LIB_LIST} ${OpenCV_LIBRARIES}
)

add_library(ofiq_objlib_shared SHARED $<TARGET_OBJECTS:ofiq_objlib>)

target_link_libraries(ofiq_objlib_shared
        PRIVATE ${OFIQ_LINK_LIB_LIST} ${OpenCV_LIBRARIES}
)

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.
#
# In this top level CMakeLists.txt, ${CMAKE_PROJECT_NAME} is used to define
# the target library name; in the sub-module's CMakeLists.txt, ${PROJECT_NAME}
# is preferred for the same purpose.
#
# In order to load a library into your app from Java/Kotlin, you must call
# System.loadLibrary() and pass the name of the library defined here;
# for GameActivity/NativeActivity derived applications, the same library name must be
# used in the AndroidManifest.xml file.
add_library(${CMAKE_PROJECT_NAME} SHARED
        # List C/C++ source files with relative paths to this CMakeLists.txt.
        ofiqlib.cpp)

# Specifies libraries CMake should link to your target library. You
# can link libraries from various origins, such as libraries defined in this
# build script, prebuilt third-party libraries, or Android system libraries.
target_link_libraries(${CMAKE_PROJECT_NAME}
        # List libraries link to the target library
        ofiq_objlib_shared
        android
        log)