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
//  MainViewModel.swift
//  OFIQMobile
//

import Foundation
import ofiqlib

///ViewModel of the MainView, It inits the OFIQ library
class MainViewModel: ObservableObject {
    @Published var isInitializing: Bool = false
    @Published var returnCode: ReturnCode? = nil
    
    private let ofiq: Ofiq
    
    init(ofiq: Ofiq) {
        self.ofiq = ofiq
    }
    
    ///Initializes the ofiq library
    func initialize() {
        DispatchQueue.main.async {
            self.isInitializing = true
        }
        
        let initializeResult = self.ofiq.initialize()
        DispatchQueue.main.async {
            self.returnCode = initializeResult?.code
            self.isInitializing = false
        }
    }
}
