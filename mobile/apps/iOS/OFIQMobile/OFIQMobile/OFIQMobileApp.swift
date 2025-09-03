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
//  OFIQMobileApp.swift
//  OFIQMobile
//

import SwiftUI
import ofiqlib

///App object of the OFIQ app
@main
struct OFIQMobileApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}


private struct ContentView: View {
    
    @State private var uiState: UiState = .Initialize
    @StateObject private var mainViewModel: MainViewModel
    @StateObject private var cameraViewModel: CameraViewModel
    @StateObject private var imageViewModel: ImageViewModel
    @StateObject private var resultViewModel: ResultViewModel
    
    private let ofiq: Ofiq = OfiqLib()
    
    init() {
        let mainViewModel = MainViewModel(ofiq: ofiq)
        self._mainViewModel = StateObject(wrappedValue: mainViewModel)
        
        let cameraViewModel = CameraViewModel()
        self._cameraViewModel = StateObject(wrappedValue: cameraViewModel)
        
        let imageViewModel = ImageViewModel(ofiq: ofiq)
        self._imageViewModel = StateObject(wrappedValue: imageViewModel)
        
        let resultViewModel = ResultViewModel()
        self._resultViewModel = StateObject(wrappedValue: resultViewModel)
    }
    
    var body: some View {
        AppBody(
            uiState: $uiState,
            mainViewModel: mainViewModel,
            cameraViewModel: cameraViewModel,
            imageViewModel: imageViewModel,
            resultViewModel: resultViewModel
        )
        .onAppear {
            NotificationCenter.default.addObserver(forName: UIApplication.willTerminateNotification, object: nil, queue: .main) { _ in
                ofiq.destroy()
            }
        }
    }
}

///UI state of the app
enum UiState {
    ///Initialization of the OFIQ library
    case Initialize
    ///camera preview
    case Camera
    ///Taken image
    case Image
    ///Analyzed image and metrics
    case Result
}
