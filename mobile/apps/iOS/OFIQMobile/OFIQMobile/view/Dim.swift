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
//  Dim.swift
//  OFIQMobile
//

import SwiftUI

///Dims the app and puts the provided content in front of the dim.
public struct Dim<Content: View>: View {
    var onClick: () -> Void = { }
    @Binding var isVisible: Bool
    @ViewBuilder var content: Content
    @State private var isVisibleWithAnimation = false
    let animate: Bool
    
    public init(
        onClick: (() -> Void)? = nil,
        isVisible: Binding<Bool>,
        animate: Bool = true,
        @ViewBuilder content: () -> Content
    ) {
        if (onClick != nil) {
            self.onClick = onClick!
        }
        
        self._isVisible = isVisible
        self.content = content()
        self.animate = animate
    }
    
    public var body: some View {
        ZStack {
            if (animate) {
                if (isVisibleWithAnimation) {
                    Color.gray
                        .opacity(0.5)
                        .onTapGesture {
                            Logger.debug(caller: self, methodName: "onTapGesture", message: "Dimmed background pressed")
                            onClick()
                        }
                    
                    content
                        .transition(.backslide)
                }
            } else if (isVisible) {
                Color.gray
                    .opacity(0.5)
                    .onTapGesture {
                        Logger.debug(caller: self, methodName: "onTapGesture", message: "Dimmed background pressed")
                        onClick()
                    }
                        
                content
            }
        }
        .ignoresSafeArea(edges: [.bottom])
        .onChange(of: isVisible) { oldValue, newValue in
            if (newValue) {
                withAnimation {
                    self.isVisibleWithAnimation = newValue
                }
            } else {
                self.isVisibleWithAnimation = newValue
            }
        }
    }
}

extension AnyTransition {
    static var backslide: AnyTransition {
        AnyTransition.asymmetric(
            insertion: .move(edge: .trailing),
            removal: .move(edge: .leading))}
}
