/**
 * @file ResultTable.kt
 *
 * @copyright Copyright (c) 2025  Federal Office for Information Security, Germany
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * @author OFIQ development team
 */

@file:Suppress("kotlin:S1128")

package de.bund.bsi.ofiqmobile.composable

import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.livedata.observeAsState
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import de.bund.bsi.ofiqmobile.R
import de.bund.bsi.ofiqmobile.lib.QualityMeasureAssessment
import de.bund.bsi.ofiqmobile.lib.QualityMetrics
import de.bund.bsi.ofiqmobile.model.MetricType
import de.bund.bsi.ofiqmobile.viewmodel.ResultActivityViewModel

/**
 * Shows the analyzed metrics in a table. The shown score type can be selected in a dropdown
 */
@Composable
fun ResultTable(
    qualityMetrics: QualityMetrics,
    resultActivityViewModel: ResultActivityViewModel
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(start = 10.dp, end = 10.dp, bottom = 10.dp)
    ) {
        ResultTableHeader(
            resultActivityViewModel = resultActivityViewModel
        )

        HorizontalDivider(
            color = Color.Gray,
            thickness = 2.dp,
            modifier = Modifier
                .fillMaxWidth()
                .padding(top = 5.dp)
                .padding(bottom = 5.dp)
        )

        ResultTableBody(
            qualityMetrics = qualityMetrics,
            resultActivityViewModel = resultActivityViewModel
        )
    }
}

/**
 * Shows the header row of the result table. It includes a dropdown to select the shown score type
 */
@Composable
fun ResultTableHeader(
    resultActivityViewModel: ResultActivityViewModel
) {
    Row(
        horizontalArrangement = Arrangement.SpaceBetween,
        modifier = Modifier
            .fillMaxWidth()
    ) {
        Text(text = stringResource(R.string.metric))

        MetricDropDown(
            resultActivityViewModel = resultActivityViewModel
        )
    }
}

/**
 * Shows a dropdown that lets the user choose a metric score type
 */
@Composable
fun MetricDropDown(
    resultActivityViewModel: ResultActivityViewModel
) {
    var expanded by remember { mutableStateOf(false) }
    val selectedMetric by resultActivityViewModel.selectedMetric.observeAsState(MetricType.Score)

    Column(
        modifier = Modifier
            .width(200.dp)
    ) {
        Row(
            horizontalArrangement = Arrangement.Center,
            modifier = Modifier
                .width(200.dp)
                .border(width = 1.dp, color = MaterialTheme.colorScheme.onSurface)
                .clickable {
                    expanded = true
                }
        ) {
            Text(
                text = selectedMetric.toString(LocalContext.current)
            )
        }

        val nativeScoreLabel = stringResource(id = R.string.native_quality_measure)
        val scoreLabel = stringResource(id = R.string.score)
        DropdownMenu(
            modifier = Modifier.
            padding(top = 10.dp),
            expanded = expanded,
            onDismissRequest = {
                expanded = false
            }
        ) {
            val options = MetricType.entries.map { it.toString(LocalContext.current) }
            options.forEach { option ->
                DropdownMenuItem(
                    text = {
                        Text(text = option)
                    },
                    onClick = {
                        val metricType = when(option.uppercase()) {
                            nativeScoreLabel.uppercase() -> MetricType.RawScore
                            scoreLabel.uppercase() -> MetricType.Score
                            else -> MetricType.Score
                        }
                        resultActivityViewModel.selectedMetric.postValue(metricType)
                        expanded = false
                    }
                )
            }
        }
    }
}

/**
 * Shows the analyzed metrics table body
 */
@Composable
fun ResultTableBody(
    qualityMetrics: QualityMetrics,
    resultActivityViewModel: ResultActivityViewModel
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .verticalScroll(rememberScrollState())
    ) {
        qualityMetrics.measurements.forEach {
            ResultTableRow(
                qualityMeasureAssessment = it,
                resultActivityViewModel = resultActivityViewModel
            )
        }
    }
}

/**
 * Shows a single metric row, with the selected metric score type.
 */
@Composable
fun ResultTableRow(
    qualityMeasureAssessment: QualityMeasureAssessment,
    resultActivityViewModel: ResultActivityViewModel
) {
    val isUnifiedQualityScore = qualityMeasureAssessment.qualityMeasure.uppercase() == "UnifiedQualityScore".uppercase()
    val textSize = 16 + if (isUnifiedQualityScore) 3 else 0
    val fontWeight = if (isUnifiedQualityScore) FontWeight.Bold else FontWeight.Normal
    Row(
        horizontalArrangement = Arrangement.SpaceBetween,
        modifier = Modifier
            .fillMaxWidth()
    ) {
        Text(
            fontSize = textSize.sp,
            fontWeight = fontWeight,
            text = qualityMeasureAssessment.qualityMeasure
        )

        val selectedMetric by resultActivityViewModel.selectedMetric.observeAsState()
        val metricValue = when(selectedMetric) {
            MetricType.Score -> qualityMeasureAssessment.scalarScore.toInt().toString()
            MetricType.RawScore -> qualityMeasureAssessment.rawScore.toString()
            null -> ""
        }

        Text(
            fontSize = textSize.sp,
            fontWeight = fontWeight,
            text = metricValue
        )
    }
}