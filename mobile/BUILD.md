# Compilation for mobile platforms

## Android

Different build tools are required to build the ofiq libs and the demo apps,
depending on the platform.

### Install JDK
On Ubuntu 24.04, one may install Java by running `sudo apt install openjdk-21-jdk`.
<br/><br/>
For other platforms, various Java versions can be downloaded from [here](https://jdk.java.net/archive/). Ensure for your Java installation, that all system variables are set correctly. For example, in Windows `JAVA_HOME` must be set and `Path` should contain the directory path to `javac.exe`.

### Install command-line tools

Download the Command-Line Tools for Windows from [here](https://developer.android.com/studio?hl=en). This should be a ZIP archive, e.g., of name `commandlinetools-win-13114758_latest.zip` (Windows), `commandlinetools-linux-13114758_latest.zip` (Linux) or `commandlinetools-mac-13114758_latest.zip` (MacOS).
- Unpack the ZIP archive.
- Place the cmdline-tools folder in an arbitrary location, e.g., `C:\Path\To\android-sdk\cmdline-tools\` (Windows) or `/path/to/android-sdk/cmdline-tools` (Linux/MacOS)

<br />

| Windows                                                                                               | Linux/MacOS                                                 | MacOS                                                                 |
| ----------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- | --------------------------------------------------------------------- |
| Set an `ANDROID_HOME` environment variable to android-sdk, e.g., `C:\Path\To\android-sdk\`            | Open a terminal.                                            | Open a terminal.                                                      |
| Add `C:\Path\To\android-sdk\cmdline-tools\` to the system's `Path` variable                           | Run ```export ANDROID_HOME=/path/to/android-sdk```          | Run ```export ANDROID_HOME=/path/to/android-sdk```                    |
| In a terminal, run ```sdkmanager --licenses --sdk_root=C:\Path\To\android-sdk\``` and accept licenses | Run ```cd $ANDROID_HOME```                                  | Run ```cd $ANDROID_HOME```                                            |
|                                                                                                       | Run ```sh sdkmanager --licenses --sdk_root=$ANDROID_HOME``` | Run ```sh sdkmanager --licenses --sdk_root=$ANDROID_HOME```           |
|                                                                                                       |                                                             | Run `sh sdkmanager --install "cmake;3.22.1" --sdk_root=$ANDROID_HOME` |


NOTE: On Windows, if the `sdkmanager` complains about an incompatible Java version, one may deactivate the check by running ```set SKIP_JDK_VERSION_CHECK=""``` and then repeating running the `sdkmanager` with the arguments from above.

NOTE: If required, a proxy can be specified using the `--proxy=https --proxy_host=<url> --proxy_port=<port>` syntax. 


### Download OFIQ
Check out the repository
```
git clone https://github.com/BSI-OFIQ/OFIQ-Project
```
where we assume its path being `C:\Path\To\OFIQ-Project\` on Windows and `/path/to/OFIQ-Project/` on Linux/MacOS.

### Compile OFIQ and mobile demo application

| Windows                                    | Linux/MacOS                              |
| ------------------------------------------ | ---------------------------------------- |
| Open a terminal                            | Open a terminal                          |
| Run `$ cd C:\Path\To\OFIQ-Project\mobile\` | Run `$ cd /path/to/OFIQ-Project/mobile/` |
| Run `.\build_for_android.bat`              | Run `$ sh build_for_android.sh`          |

After successful compilation, the path `C:\C:\Path\To\OFIQ-Project\mobile\build\` (Windows) or `/path/to/OFIQ-Project/mobile/build` (Linux/MacOS) contains the file
`ofiqmobile_\<version\>-release-unsigned.apk` which is a demo application for OFIQ. The file can be copied, installed and then run on an Android device.

## iOS
### Required Build-Tools

To create OFIQ for iOS, you must install the following build tools.
<ul>
 <li>Xcode (version 15 or higher)</li>
 <li>Xcode Command Line Tools (version 15 or higher)</li>
 <li>Homebrew</li>
 <li>CocoaPods</li>
</ul>

#### Xcode
Xcode can be installed in various ways. The easiest way is via the app store

#### Xcode Command Line Tools
The command line tools can be installed with the following command:
```
$ xcode-select --install
$ sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
```

#### Homebrew
Homebrew is required for the installation of CocoaPods.<br>
An alternative would be the installation via Ruby-Gem.
```
$ /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

#### CocoaPods
Installation of CocoaPods via Homebrew:
```
$ brew install cocoapods
$ pod setup
```

### Bundle Identifier and Signing
Before compiling, the bundle ID of the OFIQMobile app must be customised.<br>
To do this, using Xcode, the project must be opened under `/path/to/OFIQ-Project/mobile/apps/iOS/OFIQMobile`.<br>
The bundle ID is customised under <code>General -> Signing & Capabilities</code> in the OFIQMobile target.<br>
<br>
The team must be selected in the same section.<br>
Ensure that a certificate has been created for the app. You can create the certificate on the apple develper homepage or via Xcode.
<br><br>
There is an ExportOptions.plist file in the main directory of the OFIQMobile app. The correct teamID must be entered here.<br>
You can find the teamID on the apple developer page. The teamID is a unique 10-character string.

### Build
The build script is located in `/path/to/OFIQ-Project/mobile/build_for_ios.sh`
The script downloads dependencies from the web. If required, a proxy may be edited manually at the top of `build_for_ios.sh` using a text editor.
<br><br>
To build OFIQ, the ios framework and the OFIQMobile app, do the following.
```
 $ cd /path/to/OFIQ-Project/mobile/
 $ sh build_ios_lib_and_demoapp.sh
```
<br/>
This will create the following output.
<table>
 <tr>
  <td><b>file/directory</b></td>
  <td><b>description</b></td>
 </tr>
 <tr>
  <td>build/</td>
  <td>Folder with the iOS build including the binaries <code>ofiqlib.framework</code> and <code>OFIQMobile.ipa</code>.</td>
 </tr>
 <tr>
  <td>build/framwork/</td>
  <td>Storage location for the created iOS framework.</td>
 </tr>
 <tr>
  <td>build/OFIQMobile/</td>
  <td>OFIQMobile App, which integrates the framework and can be used for demonstration purposes.</td>
 </tr>
 <tr>
  <td>build/downloads/</td>
  <td>All third party libs are stored here.</td>
 </tr>
</table>