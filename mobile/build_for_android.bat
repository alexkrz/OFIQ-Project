@echo off
setlocal enabledelayedexpansion

if not "%PROXY_HOST%"=="" if not "%PROXY_PORT%"=="" (
	set "PROXY=http://%PROXY_HOST%:%PROXY_PORT%"
) else (
	set "PROXY="
)
set http_proxy=%PROXY%
set https_proxy=%PROXY%

set CURRENTDIR=%~dp0

REM set location to subrepo
set OFIQ_PROJECT_REPO=..\extern

if not exist %OFIQ_PROJECT_REPO% (
	echo Submodules %OFIQ_PROJECT_REPO% is missing! Trying to create directory.
	mkdir %OFIQ_PROJECT_REPO%
	if not exist %OFIQ_PROJECT_REPO% (
		exit /b 1
	)
)

REM create a build folder
set buildfolder=build
if not exist %buildfolder% (
	echo Create a folder for the build and to store the dependencies...
	mkdir %buildfolder%
)
CD /D "%buildfolder%"

REM create a download folder
set downloadfolder=downloads
if not exist %downloadfolder% (
	echo Create a folder to collect all downloads...
	mkdir %downloadfolder%
)
CD /D "%downloadfolder%"


REM !!!!! OpenCV START !!!!!
REM download opencv-sdk for android
echo Download opencv-sdk...
set opencvVersion=4.5.5
@REM set "url=https://sourceforge.net/projects/opencvlibrary/files/%opencvVersion%/opencv-%opencvVersion%-android-sdk.zip"
set "url=https://github.com/opencv/opencv/releases/download/%opencvVersion%/opencv-%opencvVersion%-android-sdk.zip"
set "outputFile=opencv-%opencvVersion%-android-sdk.zip"
echo "looking for already downloaded file: %outputFile%"
if not exist %outputFile% (
	curl -x %PROXY% -L -o %outputFile% %url%
	if not exist %outputFile% (
		echo download failed
		exit /b 1
	)
) else (
	echo OpenCV has already been downloaded in the correct version
)

set targetRootDir=%CURRENTDIR%\%OFIQ_PROJECT_REPO%

REM Before unpacking, the old folder is deleted if it already exists
set "openCVDestFolder=%targetRootDir%\OpenCV-android-sdk"
if exist "%openCVDestFolder%" (
	rmdir /s /q "%openCVDestFolder%"
)
echo Unpacking openCV...
tar -xf %outputFile% -C %targetRootDir%
REM !!!!! OpenCV END !!!!!


REM !!!!! onnxruntime START !!!!!
REM download onnxruntime for android from Maven Central repository
set "groupId=com.microsoft.onnxruntime"
set "artifactId=onnxruntime-android"
set "version=1.17.3"

REM build url
set "groupPath=%groupId:.=/%"
set "url=https://repo1.maven.org/maven2/%groupPath%/%artifactId%/%version%/%artifactId%-%version%.aar"

REM set output path
set "outputFile=%artifactId%-%version%.aar"
set "zipFile=%artifactId%-%version%.zip"

REM download onnxruntime aar file
echo Download onnxruntime aar from %url%...
if not exist %zipFile% (
	if not exist %outputFile% (
		curl -x %PROXY% -L -o %outputFile% %url%

		REM check the download
		if not exist %outputFile% (
			echo Download failed
			exit /b 1
		)
	)
) else (
	echo onnxruntime has already been downloaded in the correct version
)

REM rename aar file to zip file
if not exist %zipFile% (
	echo Rename the aar file to a zip file
	rename %outputFile% %zipFile%
)

if exist "%artifactId%" (
	rmdir /s /q "%artifactId%"
)
echo Unpacking the ZIP file...
mkdir %artifactId%
tar -xf %zipFile% -C %artifactId%


REM copy required files
echo Copy the required files from the unzipped directory
set "sourceDirLibs=%artifactId%\jni"
set sourceDirHeaders=%artifactId%\headers
set targetDir=%targetRootDir%\%artifactId%
if exist %targetDir% (
	rmdir /s /q %targetDir%
)
mkdir %targetDir%
xcopy /E /I "%sourceDirLibs%" "%targetDir%"
if exist "%targetRootDir%\onnxruntime" (
	rmdir /s /q "%targetRootDir%\onnxruntime"
)
move "%sourceDirHeaders%" "%targetRootDir%\onnxruntime"
echo Delete onnxruntime files from download folder...
rmdir /s /q %artifactId%
REM !!!!! onnxruntime END !!!!!



REM !!!!! copy libs START !!!!!
echo Download the required header and lib files...
set libsDir=%targetRootDir%\libs
if exist %libsDir% (
	rmdir /s /q %libsDir%
)
mkdir %libsDir%

REM https://github.com/abseil/abseil-cpp.git
set abseilVersion=20240722.0
set abseilUrl=https://github.com/abseil/abseil-cpp/archive/refs/tags/%abseilVersion%.zip
set abseilOutputFile=abseil-cpp
call :download_libs %abseilUrl% %abseilOutputFile% %abseilVersion% true

REM https://github.com/google/flatbuffers.git
set flatbuffersVersion=24.3.25
set flatbuffersUrl=https://github.com/google/flatbuffers/archive/refs/tags/v%flatbuffersVersion%.zip
set flatbuffersOutputFile=flatbuffers
call :download_libs %flatbuffersUrl% %flatbuffersOutputFile% %flatbuffersVersion% false

REM https://github.com/gabime/spdlog
set spdlogVersion=1.14.1
set spdlogUrl=https://github.com/gabime/spdlog/archive/refs/tags/v%spdlogVersion%.zip
set spdlogOutputFile=spdlog
call :download_libs %spdlogUrl% %spdlogOutputFile% %spdlogVersion% true

REM https://github.com/taocpp/json.git
set jsonVersion=1.0.0-beta.14
set jsonUrl=https://github.com/taocpp/json/archive/refs/tags/%jsonVersion%.zip
set jsonOutputFile=json
call :download_libs %jsonUrl% %jsonOutputFile% %jsonVersion% true

REM https://github.com/Neargye/magic_enum.git
set magic_enumVersion=0.9.6
set magic_enumUrl=https://github.com/Neargye/magic_enum/archive/refs/tags/v%magic_enumVersion%.zip
set magic_enumOutputFile=magic_enum
call :download_libs %magic_enumUrl% %magic_enumOutputFile% %magic_enumVersion% true

REM https://github.com/taocpp/PEGTL.git
set pegtlVersion=3.2.7
set pegtlUrl=https://github.com/taocpp/PEGTL/archive/refs/tags/%pegtlVersion%.zip
set pegtlOutputFile=PEGTL
call :download_libs %pegtlUrl% %pegtlOutputFile% %pegtlVersion% true

REM https://github.com/mapbox/gzip-hpp.git
set gzipVersion=v0.1.0
set gzipUrl=https://github.com/mapbox/gzip-hpp/archive/refs/tags/%gzipVersion%.zip
set gzipOutputFile=gzip
call :download_libs %gzipUrl% %gzipOutputFile% %gzipVersion% true

REM !!!!! copy libs END !!!!!

set libProject=%CURRENTDIR%\lib-projects\android\ofiqlib

REM !!!!! download and copy OFIQ-MODELS START !!!!!
REM download the OFIQ-models from https://standards.iso.org/iso-iec/29794/-5/ed-1/en/
echo Download OFIQ-MODELS...
set "url=https://standards.iso.org/iso-iec/29794/-5/ed-1/en/OFIQ-MODELS.zip"
set "ofiqModelsFile=ofiq_models.zip"
if not exist %ofiqModelsFile% (
	curl -x %PROXY% -L -o %ofiqModelsFile% %url%
	if not exist %ofiqModelsFile% (
		echo download failed
		exit /b 1
	)
) else (
	echo OFIQ-MODELS has already been downloaded
)

echo Unpacking OFIQ-MODELS to assets folder
set assetsFolder=%libProject%\ofiqlib\src\main\assets
if exist %assetsFolder% (
	rmdir /s /q %assetsFolder%
)
mkdir %assetsFolder%
tar -xf %ofiqModelsFile% -C %assetsFolder%

echo Renaming .gz files
for /R "%assetsFolder%" %%f in (*.gz) do (
  if not exist "%%~dpnf" ren "%%~f" "%%~nf"
)

echo Copy ofiq_config file to assets folder
copy "%targetRootDir%\..\data\ofiq_config.jaxn" %assetsFolder%

REM !!!!! download and copy OFIQ-MODELS END !!!!!

REM !!!!! build ofiq-lib (aar file) START !!!!!
set releaseFolder="%CURRENTDIR%\%buildfolder%"
set releaseLib=ofiqlib-release.aar
echo Build ofiq-lib...
cd /D "%libProject%"
if not "%PROXY_HOST%"=="" if not "%PROXY_PORT%"=="" (
    echo "calling 'gradlew.bat clean -Dhttps.proxyHost=%PROXY_HOST% -Dhttps.proxyPort=%PROXY_PORT% assembleRelease' in directory '%cd%'"
	call gradlew.bat clean -Dhttps.proxyHost="%PROXY_HOST%" -Dhttps.proxyPort="%PROXY_PORT%" assembleRelease
) else (
	echo "calling 'gradlew.bat clean assembleRelease' in directory '%cd%'"
	call gradlew.bat clean assembleRelease
)
set aarFile=ofiqlib\build\outputs\aar\%releaseLib%
if not exist %aarFile% (
	echo No result aar found!
	exit /b 1
)
copy "%aarFile%" "%releaseFolder%"
REM !!!!! build ofiq-lib END !!!!!


REM !!!!! build demonstrator app START !!!!!
echo Build OFIQMobile app...
set demonstratorPath=%CURRENTDIR%\apps\android\OFIQMobile
cd /D "%demonstratorPath%"
set libDir=app\libs
if not exist %libDir% (
	echo Create libs folder in OFIQMobile app...
	mkdir %libDir%
)
echo Copy aar to libs folder...
copy "%releaseFolder%\%releaseLib%" "%libDir%"
if not "%PROXY_HOST%"=="" if not "%PROXY_PORT%"=="" (
	call gradlew.bat clean -Dhttps.proxyHost="%PROXY_HOST%" -Dhttps.proxyPort="%PROXY_PORT%" assembleDebug
) else (
	call gradlew.bat clean assembleDebug
)
set "apkFile=app\build\outputs\apk\debug\*.apk"
if not exist %apkFile% (
	echo No result apk found!
	exit /b 1
)
xcopy "%apkFile%" "%releaseFolder%\" /y
REM !!!!! build demonstrator app END !!!!!

echo finish
endlocal
EXIT /B %ERRORLEVEL%

:download_libs
set url=%~1
set version=%~3
set output=%~2
set download=%output%_%version%.zip
set usetar==%~4
echo Download: %url%

if not exist %download% (
	curl -x %PROXY% -L -o %download% %url%
	if not exist %download% (
		echo download failed
		exit /b 1
	)
) else (
	echo %download% has already been downloaded in the correct version
)

IF %usetar% equ true (
	tar -xf %download% -C %libsDir%
) else (
	powershell -command "Expand-Archive %download% %libsDir%"
)

rename "%libsDir%/%output%-%version%" %output%

EXIT /B 0
