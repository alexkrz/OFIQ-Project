#!/bin/bash

# If a proxy is to be used, it must be customized here
PROXY=""
export http_proxy=$PROXY
export https_proxy=$PROXY

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
# Download opencv-sdk for iOS
echo "Download opencv-sdk..."
opencvVersion="4.5.5"
url="https://sourceforge.net/projects/opencvlibrary/files/$opencvVersion/opencv-$opencvVersion-ios-framework.zip"
outputFile="OpenCV-ios-framework-$opencvVersion.zip"

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

# create folder for all frameworks
frameworksFolder="$targetRootDir/frameworks"
rm -rf "$frameworksFolder"
echo "Create frameworks folder..."
mkdir "$frameworksFolder"

# Before unpacking, the old folder is deleted if it already exists
echo "Unpacking openCV..."
unzip "$outputFile" -d "$frameworksFolder"
# !!!!! OpenCV END !!!!!

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

libProject="$CURRENTDIR/lib-projects/iOS/ofiqlib"

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

echo "Unpacking OFIQ-MODELS to data folder"
dataFolder="$libProject/ofiqlib/data"
if [ -d "$dataFolder" ]; then
    rm -rf "$dataFolder"
fi
mkdir -p "$dataFolder"
unzip "$ofiqModelsFile" -d "$dataFolder"

echo "Copy ofiq_config file to data folder"
cp "$targetRootDir/../data/ofiq_config.jaxn" "$dataFolder"
# !!!!! download and copy OFIQ-MODELS END !!!!!

# !!!!! build ofiq-lib framework START !!!!!
echo "Build ofiq-framework..."
cd "$libProject"

# add execute rights
chmod +x readVersion.sh

app_name="ofiqlib"
projecttype="workspace"
extension="xcworkspace"

releaseFolder="$CURRENTDIR/$buildfolder"

# install pods
pod install

# read version of project
version=$(xcodebuild -showBuildSettings | grep MARKETING_VERSION | tr -d 'MARKETING_VERSION =')
echo "App-Version:"$version

# delete old release build
frameworkFolder="$releaseFolder/framework"
if [ -d "$frameworkFolder" ]; then
    rm -rf "$frameworkFolder"
fi

xcodebuild clean build \
    -workspace $app_name.$extension \
    -scheme $app_name \
    -configuration Release \
    -sdk iphoneos \
    CONFIGURATION_BUILD_DIR="$frameworkFolder" \
    SKIP_INSTALL=NO \
    BUILD_LIBRARY_FOR_DISTRIBUTION=YES

frameworkFile="$frameworkFolder/$app_name.framework"
if [ ! -d "$frameworkFile" ]; then
    echo "No result framework found!"
    exit 1
fi

# remove the files that are not needed
rm -rf "$frameworkFolder/Pods_ofiqlib.framework"
rm -rf "$frameworkFolder/ofiqlib.framework.dSYM"
# !!!!! build ofiq-lib END !!!!!

# !!!!! build demonstrator app START !!!!!
app_name="OFIQMobile"
echo "Build "$app_name" app..."
demonstratorPath="$CURRENTDIR/apps/iOS/"$app_name""
cd "$demonstratorPath"
libDir=""$app_name"/libs"
if [ ! -d "$libDir" ]; then
    echo "Create libs folder in "$app_name" app..."
    mkdir -p "$libDir"
fi
echo "Copy framework to libs folder..."
cp -r -f "$frameworkFile" "$libDir"

projecttype="project"
extension="xcodeproj"
result_folder=$app_name
result_archive="$releaseFolder/$result_folder/$app_name.xcarchive"
xcodebuild clean archive -configuration Release -sdk iphoneos -$projecttype $app_name.$extension -scheme $app_name -archivePath "$result_archive"
        
echo "Archive: "$result_archive
if [ -d "$result_archive" ]; then
    echo "create ipa file"
    xcodebuild -exportArchive -archivePath "$result_archive" -exportPath "$releaseFolder/$result_folder" -exportOptionsPlist ExportOptions.plist
else
    echo "no archive created -> Error"
    exit 1
fi

# !!!!! build demonstrator app END !!!!!
echo "Finish build of framework and demonstrator app!"
