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
//  ImageView.swift
//  OFIQMobile
//

import SwiftUI

///This view shows the given image and lets the user accept the image or restart the image taking process
struct ImageView: View {
    @ObservedObject var imageViewModel: ImageViewModel
    let cameraImage: UIImage?
    let imageOrientation: UIDeviceOrientation?
    
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass
    @Environment(\.verticalSizeClass) private var verticalSizeClass
    
    private var isLandscape: Bool { verticalSizeClass == .compact }
    
    var body: some View {
        ZStack {
            VStack(spacing: 0) {
                TopBar(backgroundColor: Color("mainColor"), foregroundColor: Color.white)
                
                if (!isLandscape) {
                    VStack {                        
                        Image(uiImage: cameraImage ?? UIImage(systemName: "photo")!)
                            .resizable()
                            .scaledToFit()
                            .rotationEffect(.degrees(getRotation(imageOrientation: imageOrientation ?? .portrait)))
                            .frame(maxHeight: 600)
                        
                        Spacer()
                        
                        ImageViewControls(imageViewModel: imageViewModel, cameraImage: cameraImage)
                    }
                } else {
                    HStack {
                        Image(uiImage: cameraImage ?? UIImage(systemName: "photo")!)
                            .resizable()
                            .scaledToFit()
                            .rotationEffect(.degrees(getRotation(imageOrientation: imageOrientation ?? .portrait)))
                            .frame(maxWidth: 400)
                        
                        Spacer()
                        
                        ImageViewControls(imageViewModel: imageViewModel, cameraImage: cameraImage)
                    }
                }
            }

            Progress(isVisible: $imageViewModel.isAnalyzing)
        }
    }
}

///Buttons to accept or restart the image taking process
struct ImageViewControls: View {
    @ObservedObject var imageViewModel: ImageViewModel
    let cameraImage: UIImage?
    
    var body: some View {
        HStack {
            CustomButton(
                isDisabled: Binding.constant(false),
                action: {
                    imageViewModel.doRepeatScan()
                },
                text: "repeat"
            )
            
            Spacer()

            CustomButton(
                isDisabled: Binding.constant(false),
                action: {
                    imageViewModel.analyzeImage(cameraImage: cameraImage ?? UIImage(systemName: "photo")!)
                },
                text: "accept"
            )
        }
        .padding(.bottom, 20)
        .padding(.leading, 15)
        .padding(.trailing, 15)
    }
}
