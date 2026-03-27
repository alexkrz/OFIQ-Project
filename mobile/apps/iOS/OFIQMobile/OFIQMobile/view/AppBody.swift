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
//  AppBody.swift
//  OFIQMobile
//

import SwiftUI
import ofiqlib

///This view constructs all other views and controls the display of them, by observing changes in their viewModels
struct AppBody: View {
    @Binding var uiState: UiState
    
    @ObservedObject var mainViewModel: MainViewModel
    @ObservedObject var cameraViewModel: CameraViewModel
    @ObservedObject var imageViewModel: ImageViewModel
    @ObservedObject var resultViewModel: ResultViewModel
    
    var body: some View {
        ZStack {
            switch (uiState) {
                case .Initialize:
                    MainView(
                        mainViewModel: mainViewModel
                    )
                    .transition(.slide)
                case .Camera:
                    CameraView(
                        cameraViewModel: cameraViewModel
                    )
                    .transition(.slide)
                case .Image:
                    ImageView(
                        imageViewModel: imageViewModel, 
                        cameraImage: cameraViewModel.capturedImage,
                        imageOrientation: cameraViewModel.capturedImageOrientation
                    )
                    .transition(.slide)
                case .Result:
                    ResultView(
                        resultViewModel: resultViewModel,
                        cameraImage: cameraViewModel.capturedImage ?? UIImage(systemName: "photo")!,
                        imageOrientation: cameraViewModel.capturedImageOrientation,
                        qualityMetrics: imageViewModel.qualityMetrics ?? QualityMetrics()
                    )
                    .transition(.slide)
            }
        }
        .onChange(of: mainViewModel.returnCode) { oldValue, newValue in
            if (oldValue != newValue) {
                if (newValue == .SUCCESS) {
                    changeUiState(newUiState: .Camera)
                }
            }
        }
        .onChange(of: cameraViewModel.capturedImage) { oldValue, newValue in
            if (oldValue != newValue && newValue != nil) {
                changeUiState(newUiState: .Image)
            }
        }
        .onChange(of: imageViewModel.qualityMetrics) { oldValue, newValue in
            if (oldValue != newValue && newValue != nil) {
                changeUiState(newUiState: .Result)
            }
        }
        .onChange(of: imageViewModel.repeatScan) { oldValue, newValue in
            if (oldValue != newValue && newValue) {
                imageViewModel.clean()
                cameraViewModel.clean()
                changeUiState(newUiState: .Camera)
            }
        }
        .onChange(of: resultViewModel.restartProcess) { oldValue, newValue in
            if (oldValue != newValue && newValue) {
                imageViewModel.clean()
                cameraViewModel.clean()
                resultViewModel.clean()
                changeUiState(newUiState: .Camera)
            }
        }
    }
    
    private func changeUiState(newUiState: UiState) {
        DispatchQueue.main.async {
            withAnimation {
                self.uiState = newUiState
            }
        }
    }
}
