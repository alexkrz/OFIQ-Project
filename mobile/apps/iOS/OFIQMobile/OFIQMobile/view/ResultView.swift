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
//  ResultView.swift
//  OFIQMobile
//

import SwiftUI
import ofiqlib

///This view shows the taken image and the analysis results
struct ResultView: View {
    @ObservedObject var resultViewModel: ResultViewModel
    let cameraImage: UIImage
    let imageOrientation: UIDeviceOrientation?
    let qualityMetrics: QualityMetrics
    
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass
    @Environment(\.verticalSizeClass) private var verticalSizeClass
    
    private var isLandscape: Bool { verticalSizeClass == .compact }
    
    var body: some View {
        VStack {
            TopBar(
                backgroundColor: Color("mainColor"),
                foregroundColor: Color.white,
                content: {
                    Image (systemName: "arrow.clockwise")
                        .rotationEffect(.degrees(90))
                        .foregroundStyle(Color.white)
                        .onTapGesture {
                            resultViewModel.doRestartProcess()
                        }
                }
            )
            
            if (!isLandscape) {
                VStack {
                    Image(uiImage: cameraImage)
                        .resizable()
                        .scaledToFit()
                        .rotationEffect(.degrees(getRotation(imageOrientation: imageOrientation ?? .portrait)))
                        .frame(maxHeight: 250)
                    
                    
                    ResultTable(
                        resultViewModel: resultViewModel,
                        qualityMetrics: qualityMetrics
                    )
                }
            } else {
                HStack {
                    Image(uiImage: cameraImage)
                        .resizable()
                        .scaledToFit()
                        .rotationEffect(.degrees(getRotation(imageOrientation: imageOrientation ?? .portrait)))
                        .frame(maxWidth: 350)
                    
                    ResultTable(
                        resultViewModel: resultViewModel,
                        qualityMetrics: qualityMetrics
                    )
                }
            }
        }
    }
}
