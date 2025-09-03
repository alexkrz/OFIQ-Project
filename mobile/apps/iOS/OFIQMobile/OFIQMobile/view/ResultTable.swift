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
//  ResultTable.swift
//  OFIQMobile
//

import SwiftUI
import ofiqlib

///This view shows the analysis results in a table, where the shown metric can be selected
struct ResultTable: View {
    @ObservedObject var resultViewModel: ResultViewModel
    let qualityMetrics: QualityMetrics
    
    var body: some View {
        VStack {
            ResultTableHeader(resultViewModel: resultViewModel)
            Divider()
            ResultTableBody(
                resultViewModel: resultViewModel,
                qualityMetrics: qualityMetrics
            )
        }
        .padding()
    }
}

struct ResultTableHeader: View {
    @ObservedObject var resultViewModel: ResultViewModel
    
    var body: some View {
        HStack {
            Text("Metric")
            Spacer()
            MetricDropDown(
                resultViewModel: resultViewModel
            )
        }
    }
}

struct ResultTableBody: View {
    @ObservedObject var resultViewModel: ResultViewModel
    let qualityMetrics: QualityMetrics
    
    var body: some View {
        ScrollView {
            VStack {
                ForEach(qualityMetrics.measurements ?? [], id: \.self.qualityMeasure) { qualityMeasureAssessment in
                    ResultTableRow(
                        resultViewModel: resultViewModel,
                        qualityMeasureAssessment: qualityMeasureAssessment
                    )
                }
            }
        }
    }
}

struct ResultTableRow: View {
    @ObservedObject var resultViewModel: ResultViewModel
    let qualityMeasureAssessment: QualityMeasureAssessment
    
    var body: some View {
        let isUnifiedQualityScore = qualityMeasureAssessment.qualityMeasure.uppercased() == "UnifiedQualityScore".uppercased()
        let textSize: CGFloat = 20 + (isUnifiedQualityScore ? 3 : 0)
        let fontWeight: Font.Weight = isUnifiedQualityScore ? .bold : .regular

        HStack {
            Text(qualityMeasureAssessment.qualityMeasure)
                .fontWeight(fontWeight)
                .font(.system(size: textSize))
            
            Spacer()
            
            if (resultViewModel.selectedMetric == .score) {
                Text("\(Int(qualityMeasureAssessment.scalarScore))")
                    .fontWeight(fontWeight)
                    .font(.system(size: textSize))
            } else {
                Text("\(qualityMeasureAssessment.rawScore)")
                    .fontWeight(fontWeight)
                    .font(.system(size: textSize))
            }
        }
    }
}

struct MetricDropDown: View {
    @ObservedObject var resultViewModel: ResultViewModel
    
    var body: some View {
        VStack {
            Menu {
                ForEach(MetricType.allCases, id: \.self) { option in
                    Button(option.label()) {
                        resultViewModel.selectedMetric = option
                    }
                }
            } label: {
                Label(resultViewModel.selectedMetric.label(), systemImage: "arrowtriangle.down.fill")
                    .frame(width: 275)
                    .padding()
                    .foregroundStyle(Color("mainColor"))
                    .background(Color.gray.opacity(0.2))
                    .cornerRadius(8)
            }
        }
    }
}
