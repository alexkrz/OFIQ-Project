// Copyright (c) 2025  Federal Office for Information Security, Germany
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

//
//  CameraView.swift
//  OFIQMobile
//

import SwiftUI
import AVFoundation

///This view shows a camera preview screen and buttons to change the camera and to take a picture
struct CameraView: View {
    @StateObject var cameraViewModel = CameraViewModel()

    @Environment(\.horizontalSizeClass) private var horizontalSizeClass
    @Environment(\.verticalSizeClass) private var verticalSizeClass
    
    private var isLandscape: Bool { verticalSizeClass == .compact }
    
    var body: some View {
        ZStack {
            if (cameraViewModel.permissionGranted) {
                CameraPreviewView(session: cameraViewModel.session, isLandscape: isLandscape)
                    .edgesIgnoringSafeArea(.all)
                
                VStack {
                    CustomButton(
                        isDisabled: $cameraViewModel.takingImage,
                        action: {
                            cameraViewModel.switchCamera()
                        },
                        text: "changeCamera"
                    )
                    .padding(.top, isLandscape ? 20 : 80)
                    
                    Text("scanFace")
                        .foregroundColor(.black)
                        .padding(.top, 10)
                        .padding(.bottom, 10)
                        .padding(.leading, 15)
                        .padding(.trailing, 15)
                        .background(Color.gray)
                        .cornerRadius(7)
                        .padding(.top, isLandscape ? 20 : 50)
                    
                    Spacer()
                    
                    CustomButton(
                        isDisabled: $cameraViewModel.takingImage,
                        action: {
                            cameraViewModel.capturePhoto()
                        },
                        text: "takePicture"
                    )
                    .padding(.bottom)
                }
            }
        }
        .onAppear {
            cameraViewModel.requestPermission()
        }
    }
}

///This view shows a camera preview
struct CameraPreviewView: UIViewRepresentable {
    let session: AVCaptureSession
    @State var isLandscape: Bool
    
    func makeUIView(context: Context) -> UIView {
        let view = UIView(frame: .zero)
        let previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer.videoGravity = .resizeAspectFill
        previewLayer.frame = UIScreen.main.bounds
        view.layer.addSublayer(previewLayer)
        
        return view
    }
    
    func updateUIView(_ uiView: UIView, context: Context) {
        if let previewLayer = uiView.layer.sublayers?.first as? AVCaptureVideoPreviewLayer {
            updatePreviewLayer(previewLayer: previewLayer, for: UIDevice.current.orientation)
        }
    }
    
    private func updatePreviewLayer(previewLayer: AVCaptureVideoPreviewLayer, for orientation: UIDeviceOrientation) {
        previewLayer.videoGravity = .resizeAspectFill
        previewLayer.frame = UIScreen.main.bounds
        
        switch orientation {
        case .portrait:
            previewLayer.connection?.videoRotationAngle = 90
        case .landscapeRight:
            previewLayer.connection?.videoRotationAngle = 180
        case .landscapeLeft:
            previewLayer.connection?.videoRotationAngle = 0
        case .portraitUpsideDown:
            previewLayer.connection?.videoRotationAngle = 270
        default:
            previewLayer.connection?.videoRotationAngle = 90
        }
        previewLayer.frame = UIScreen.main.bounds
    }
}
