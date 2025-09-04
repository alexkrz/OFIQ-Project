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
//  CameraViewModel.swift
//  OFIQMobile
//

import SwiftUI
import AVFoundation
import os.log

///ViewModel of the CameraView. It takes pictures, changes the camera and requests the camera permission.
class CameraViewModel: NSObject, ObservableObject {
    @Published var session: AVCaptureSession
    @Published var isUsingFrontCamera = false
    @Published var capturedImage: UIImage? = nil
    @Published var capturedImageOrientation: UIDeviceOrientation? = nil
    @Published var permissionGranted = false
    @Published var takingImage: Bool = false

    private var backCamera: AVCaptureDevice?
    private var frontCamera: AVCaptureDevice?
    private var currentCameraInput: AVCaptureDeviceInput?
    
    override init() {
        self.session = AVCaptureSession()
        super.init()
        self.configure()
    }
    
    ///requests the camera permission
    func requestPermission() {
        AVCaptureDevice.requestAccess(for: .video, completionHandler: {accessGranted in
            DispatchQueue.main.async {
                self.permissionGranted = accessGranted
            }
        })
    }
    
    ///configures the camera
    func configure() {
        self.session.beginConfiguration()
        
        if let backCamera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) {
            self.backCamera = backCamera
        }
        
        if let frontCamera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front) {
            self.frontCamera = frontCamera
        }
        
        guard let defaultCamera = backCamera else {
            os_log(.debug, "No back camera available")
            return
        }
        
        do {
            let cameraInput = try AVCaptureDeviceInput(device: defaultCamera)
            if self.session.canAddInput(cameraInput) {
                self.session.addInput(cameraInput)
                self.currentCameraInput = cameraInput
            }
            
            let output = AVCapturePhotoOutput()
            if self.session.canAddOutput(output) {
                self.session.addOutput(output)
            }
            
            self.session.commitConfiguration()
            output.maxPhotoDimensions = CMVideoDimensions(width: 1920, height: 1080)
            self.session.startRunning()
            
        } catch {
            os_log(.debug, "Error setting up camera:  %{public}@", error.localizedDescription)
        }
    }
    
    ///switches the camera from back to front or from front to back
    func switchCamera() {
        guard let currentCameraInput = self.currentCameraInput else { return }
        self.session.beginConfiguration()
        self.session.removeInput(currentCameraInput)
        
        do {
            let newCamera = isUsingFrontCamera ? backCamera : frontCamera
            let newCameraInput = try AVCaptureDeviceInput(device: newCamera!)
            if self.session.canAddInput(newCameraInput) {
                self.session.addInput(newCameraInput)
                self.currentCameraInput = newCameraInput
            }
            
            isUsingFrontCamera.toggle()
            
        } catch {
            os_log(.error, "Error switching cameras: %{public}@", error.localizedDescription)
        }
        
        self.session.commitConfiguration()
    }
    
    ///Captures a photo from the camera
    func capturePhoto() {
        DispatchQueue.main.async {
            self.takingImage = true
        }

        guard let connection = (self.session.outputs.first as? AVCapturePhotoOutput)?.connection(with: .video) else {
            return
        }
        
        let photoSettings = AVCapturePhotoSettings(format: [AVVideoCodecKey: AVVideoCodecType.jpeg])
        photoSettings.maxPhotoDimensions = CMVideoDimensions(width: 1920, height: 1080)
        (self.session.outputs.first as? AVCapturePhotoOutput)?.capturePhoto(with: photoSettings, delegate: self)
    }
    
    ///cleans all fields of the ViewModel and reconfigures the camera.
    func clean() {
        self.session = AVCaptureSession()
        self.isUsingFrontCamera = false
        self.capturedImage = nil
        self.capturedImageOrientation = nil
        self.permissionGranted = false
        self.takingImage = false
        configure()
    }
}

extension CameraViewModel: AVCapturePhotoCaptureDelegate {
    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        guard error == nil, let imageData = photo.fileDataRepresentation(), let image = UIImage(data: imageData) else {
            return
        }
        
        self.capturedImage = image
        self.capturedImageOrientation = UIDevice.current.orientation
        self.session.stopRunning()
    }
}
