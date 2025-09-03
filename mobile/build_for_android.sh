#!/bin/bash

# If a proxy is to be used, it must be customized here
PROXY_HOST=""
PROXY_PORT=""
if [ -n "$PROXY_HOST" ] && -n [ "$PROXY_PORT" ]; then
    PROXY="http://${PROXY_HOST}:${PROXY_PORT}"
else
    PROXY=""
fi
export http_proxy=$PROXY
export https_proxy=$PROXY

# export ANDROID_HOME=/Users/<UserName>/Library/Android/sdk
if [ -z "$ANDROID_HOME" ]; then
  echo "ANDROID_HOME is not set. Please set the ANDROID_HOME as an environment variable or here in the script."
  exit 1
else
  echo "ANDROID_HOME is set to:  $ANDROID_HOME"
fi

CURRENTDIR=$(dirname "$(readlink -f "$0")")

# Set location to subrepo
OFIQ_PROJECT_REPO="../extern"

if [ ! -d "$OFIQ_PROJECT_REPO" ]; then
    echo "Submodules $OFIQ_PROJECT_REPO is missing! Trying to create directory."
    mkdir $OFIQ_PROJECT_REPO
    if [ ! -d "$OFIQ_PROJECT_REPO" ]; then
        echo "ERROR: Could not create $OFIQ_PROJECT_REPO."
        exit 1
    fi
fi

# Create a build folder
buildfolder="build"
if [ ! -d "$buildfolder" ]; then
    echo "Create a folder for the build and to store the dependencies..."
    mkdir "$buildfolder"
fi
cd "$buildfolder"

# Create a download folder
downloadfolder="downloads"
if [ ! -d "$downloadfolder" ]; then
    echo "Create a folder to collect all downloads..."
    mkdir "$downloadfolder"
fi
cd "$downloadfolder"

# !!!!! OpenCV START !!!!!
# Download opencv-sdk for android
echo "Download opencv-sdk..."
opencvVersion="4.5.5"
url="https://sourceforge.net/projects/opencvlibrary/files/$opencvVersion/opencv-$opencvVersion-android-sdk.zip"
outputFile="OpenCV-android-sdk-$opencvVersion.zip"

if [ ! -f "$outputFile" ]; then
    curl -x "$PROXY" -L -o "$outputFile" "$url"
    if [ ! -f "$outputFile" ]; then
        echo "Download failed"
        exit 1
    fi
else
    echo "OpenCV has already been downloaded in the correct version"
fi

targetRootDir="$CURRENTDIR/$OFIQ_PROJECT_REPO"

# Before unpacking, the old folder is deleted if it already exists
openCVDestFolder="$targetRootDir/OpenCV-android-sdk"
if [ -d "$openCVDestFolder" ]; then
    rm -rf "$openCVDestFolder"
fi
echo "Unpacking openCV..."
unzip "$outputFile" -d "$targetRootDir"
# !!!!! OpenCV END !!!!!

# !!!!! onnxruntime START !!!!!
# Download onnxruntime for android from Maven Central repository
groupId="com.microsoft.onnxruntime"
artifactId="onnxruntime-android"
version="1.17.3"

# Build URL
groupPath=$(echo "$groupId" | tr '.' '/')
url="https://repo1.maven.org/maven2/$groupPath/$artifactId/$version/$artifactId-$version.aar"

# Set output path
outputFile="$artifactId-$version.aar"
zipFile="$artifactId-$version.zip"

# Download onnxruntime aar file
echo "Download onnxruntime aar from $url..."
if [ ! -f "$zipFile" ]; then
    if [ ! -f "$outputFile" ]; then
        curl -x "$PROXY" -L -o "$outputFile" "$url"

        # Check the download
        if [ ! -f "$outputFile" ]; then
            echo "Download failed"
            exit 1
        fi
    fi
else
    echo "onnxruntime has already been downloaded in the correct version"
fi

# Rename aar file to zip file
if [ ! -f "$zipFile" ]; then
    echo "Rename the aar file to a zip file"
    mv "$outputFile" "$zipFile"
fi

if [ -d "$artifactId" ]; then
    rm -rf "$artifactId"
fi
echo "Unpacking the ZIP file..."
mkdir "$artifactId"
unzip "$zipFile" -d "$artifactId"

# Copy required files
echo "Copy the required files from the unzipped directory"
sourceDirLibs="$artifactId/jni"
sourceDirHeaders="$artifactId/headers"
targetDir="$targetRootDir/$artifactId"

if [ -d "$targetDir" ]; then
    rm -rf "$targetDir"
fi
mkdir "$targetDir"
cp -r "$sourceDirLibs/"* "$targetDir/"
if [ -d "$targetRootDir/onnxruntime" ]; then
    rm -rf "$targetRootDir/onnxruntime"
fi
mv "$sourceDirHeaders" "$targetRootDir/onnxruntime"
echo "Delete onnxruntime files from download folder..."
rm -rf "$artifactId"
# !!!!! onnxruntime END !!!!!

# !!!!! copy libs START !!!!!
echo "Download the required header and lib files..."
libsDir="$targetRootDir/libs"
if [ -d "$libsDir" ]; then
    rm -rf "$libsDir"
fi
mkdir "$libsDir"

# Function to download and extract libraries
download_libs() {
    url=$1
    output=$2
    version=$3
    download="${output}_${version}.zip"
    echo "Download: $url"

    if [ ! -f "$download" ]; then
        curl -x "$PROXY" -L -o "$download" "$url"
        if [ ! -f "$download" ]; then
            echo "Download failed"
            exit 1
        fi
    else
        echo "$download has already been downloaded in the correct version"
    fi
    
    yes | unzip "$download" -d "$libsDir"

    mv "$libsDir/$output-$version" "$libsDir/$output"
}

# Download and extract libraries
download_libs "https://github.com/abseil/abseil-cpp/archive/refs/tags/20240722.0.zip" "abseil-cpp" "20240722.0"
download_libs "https://github.com/google/flatbuffers/archive/refs/tags/v24.3.25.zip" "flatbuffers" "24.3.25"
download_libs "https://github.com/gabime/spdlog/archive/refs/tags/v1.14.1.zip" "spdlog" "1.14.1"
download_libs "https://github.com/taocpp/json/archive/refs/tags/1.0.0-beta.14.zip" "json" "1.0.0-beta.14"
download_libs "https://github.com/Neargye/magic_enum/archive/refs/tags/v0.9.6.zip" "magic_enum" "0.9.6"
download_libs "https://github.com/taocpp/PEGTL/archive/refs/tags/3.2.7.zip" "PEGTL" "3.2.7"
download_libs "https://github.com/mapbox/gzip-hpp/archive/refs/tags/v0.1.0.zip" "gzip-hpp-0.1.0" "0.1.0"
# !!!!! copy libs END !!!!!

libProject="$CURRENTDIR/lib-projects/android/ofiqlib"

# !!!!! download and copy OFIQ-MODELS START !!!!!
# Download the OFIQ-models from https://standards.iso.org/iso-iec/29794/-5/ed-1/en/
echo "Download OFIQ-MODELS..."
url="https://standards.iso.org/iso-iec/29794/-5/ed-1/en/OFIQ-MODELS.zip"
ofiqModelsFile="ofiq_models.zip"

if [ ! -f "$ofiqModelsFile" ]; then
    curl -x "$PROXY" -L -o "$ofiqModelsFile" "$url"
    if [ ! -f "$ofiqModelsFile" ]; then
        echo "Download failed"
        exit 1
    fi
else
    echo "OFIQ-MODELS has already been downloaded"
fi

echo "Unpacking OFIQ-MODELS to assets folder"
assetsFolder="$libProject/ofiqlib/src/main/assets"
if [ -d "$assetsFolder" ]; then
    rm -rf "$assetsFolder"
fi
mkdir -p "$assetsFolder"
unzip "$ofiqModelsFile" -d "$assetsFolder"

echo "Rename .gz files"
find "$assetsFolder" -type f -name "*.gz" | while read -r file; do
  new_name="${file%.gz}"
  mv "$file" "$new_name"
done

echo "Copy ofiq_config file to assets folder"
cp "$targetRootDir/../data/ofiq_config.jaxn" "$assetsFolder"
# !!!!! download and copy OFIQ-MODELS END !!!!!

# !!!!! build ofiq-lib (aar file) START !!!!!
releaseFolder="$CURRENTDIR/$buildfolder"
releaseLib="ofiqlib-release.aar"
echo "Build ofiq-lib..."
cd "$libProject"
chmod +x gradlew
if [ -n "$PROXY_HOST" ] &&  [ -n "$PROXY_PORT" ]; then
    echo "INFO: Running './gradlew clean -Dhttps.proxyHost="$PROXY_HOST" -Dhttps.proxyPort="$PROXY_PORT" assembleRelease' in '$PWD'"
    ./gradlew clean -Dhttps.proxyHost="$PROXY_HOST" -Dhttps.proxyPort="$PROXY_PORT" assembleRelease
else
    echo "INFO: Running './gradlew clean assembleRelease' in '$PWD'"
    ./gradlew clean assembleRelease
fi
aarFile="ofiqlib/build/outputs/aar/$releaseLib"
if [ ! -f "$aarFile" ]; then
    echo "No result aar found!"
    exit 1
fi
cp "$aarFile" "$releaseFolder"
# !!!!! build ofiq-lib END !!!!!

# !!!!! build demonstrator app START !!!!!
echo "Build OFIQMobile app..."
demonstratorPath="$CURRENTDIR/apps/android/OFIQMobile"
if [ ! -d $demonstratorPath ]; then
    echo "ERROR: $demonstratorPath does not exist."
    exit 1
fi
cd "$demonstratorPath"
libDir="app/libs"
if [ ! -d "$libDir" ]; then
    echo "Create libs folder in OFIQMobile app..."
    mkdir -p "$libDir"
fi
echo "Copy aar to libs folder..."
if [ ! -f $releaseFolder/$releaseLib ]; then
    echo "ERROR: File $releaseFolder/$releaseLib does not exist"
    exit 1
fi
cp "$releaseFolder/$releaseLib" "$libDir/"
chmod +x gradlew
if [ -n "$PROXY_HOST" ] && [ -n "$PROXY_PORT" ]; then
    echo "INFO: Running './gradlew clean -Dhttps.proxyHost="$PROXY_HOST" -Dhttps.proxyPort="$PROXY_PORT" assembleDebug' in '$PWD'"
    ./gradlew clean -Dhttps.proxyHost="$PROXY_HOST" -Dhttps.proxyPort="$PROXY_PORT" assembleDebug
else
    echo "INFO: Running './gradlew clean assembleDebug' in '$PWD'"
    ./gradlew clean assembleDebug
fi
apkFile="app/build/outputs/apk/debug/*.apk"
if [ ! -f $apkFile ]; then
    echo "No result apk found!"
    exit 1
fi
cp $apkFile "$releaseFolder/"
# !!!!! build demonstrator app END !!!!!
