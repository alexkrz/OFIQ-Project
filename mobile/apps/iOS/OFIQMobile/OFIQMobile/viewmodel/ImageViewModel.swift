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
//  ImageViewModel.swift
//  OFIQMobile
//

import Foundation
import UIKit
import ofiqlib

///ViewModel of the ImageView. It analyzes the given image with the OFIQ library
class ImageViewModel: ObservableObject {
    private let ofiq: Ofiq
    
    @Published var isAnalyzing: Bool = false
    @Published var qualityMetrics: QualityMetrics? = nil
    @Published var repeatScan: Bool = false
    
    init(
        ofiq: Ofiq
    ) {
        self.ofiq = ofiq
    }
    
    ///analyzes the image with the OFIQ library
    func analyzeImage(cameraImage: UIImage) {
        DispatchQueue.main.async {
            self.isAnalyzing = true
        }
        
        Task.detached {
            let analyzedImage = self.ofiq.faceQa(cameraImage)
            DispatchQueue.main.async {
                self.qualityMetrics = analyzedImage
                self.isAnalyzing = false
            }
        }
    }
    
    ///Tells the AppBody to restart the camera scan
    func doRepeatScan() {
        DispatchQueue.main.async {
            self.repeatScan = true
        }
    }
    
    ///Cleans the ViewModel
    func clean() {
        DispatchQueue.main.async {
            self.isAnalyzing = false
            self.qualityMetrics = nil
            self.repeatScan = false
        }
    }
}
