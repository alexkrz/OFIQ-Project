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
//  Progress.swift
//  OFIQMobile
//

import SwiftUI

///Dims the app and puts a spinner view in front of the dim.
public struct Progress: View {
    @Binding var isVisible: Bool
    
    public init(isVisible: Binding<Bool>) {
        self._isVisible = isVisible
    }
    
    public var body: some View {
        Dim(isVisible: $isVisible, animate: false) {
            Spinner()
        }
    }
}

struct Spinner: View {
    @State private var isAnimating = false
    @State private var rotation: Angle = .degrees(0)

    var body: some View {
        GeometryReader { geometry in
            Circle()
                .trim(from: 0, to: 0.8)
                .stroke(Color("mainColor"), lineWidth: 5)
                .frame(width: 50, height: 50)
                .rotationEffect(rotation)
                .onAppear {
                    startAnimation()
                }
                .onChange(of: geometry.size) { oldValue, newValue in
                    if (oldValue != newValue) {
                        resetAnimation()
                    }
                }
                .frame(width: geometry.size.width, height: geometry.size.height)
        }
    }

    private func startAnimation() {
        rotation = .degrees(0)
        isAnimating = true
        withAnimation(Animation.linear(duration: 1).repeatForever(autoreverses: false)) {
            rotation = .degrees(360)
        }
    }

    private func resetAnimation() {
        isAnimating = false
        rotation = .degrees(0)
    }
}
