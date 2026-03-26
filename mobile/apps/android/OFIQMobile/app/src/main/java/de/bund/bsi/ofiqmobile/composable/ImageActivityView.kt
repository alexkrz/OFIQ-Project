/**
 * @file ImageActivityView.kt
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

import android.content.res.Configuration
import android.graphics.Bitmap
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.livedata.observeAsState
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import de.bund.bsi.ofiqmobile.R
import de.bund.bsi.ofiqmobile.lib.QualityMetrics
import de.bund.bsi.ofiqmobile.viewmodel.ImageActivityViewModel

/**
 * Shows the scanned image and controls to analyze the image
 */
@Composable
fun ImageActivityView(
    imageBytes: ByteArray?,
    imageBitmap: Bitmap?,
    imageActivityViewModel: ImageActivityViewModel,
    onQualityMetricsSet: (QualityMetrics?) -> Unit,
    onRepeatScan: () -> Unit
) {
    Box(
        modifier = Modifier
            .fillMaxSize()
    ) {
        val isPortrait = LocalConfiguration.current.orientation == Configuration.ORIENTATION_PORTRAIT
        if (isPortrait) {
            Column(
                verticalArrangement = Arrangement.SpaceBetween,
                horizontalAlignment = Alignment.CenterHorizontally,
                modifier = Modifier
                    .fillMaxSize()
            ) {
                OfiqImage(
                    imageBitmap = imageBitmap,
                    height = 0.75f
                )

                TakenImageControls(
                    imageBytes = imageBytes,
                    imageActivityViewModel = imageActivityViewModel,
                    onRepeatScan = onRepeatScan
                )
            }
        } else {
            Row(
                modifier = Modifier
                    .fillMaxSize()
            ) {
                Column(
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxHeight()
                ) {
                    OfiqImage(
                        imageBitmap = imageBitmap,
                        height = 1.0f
                    )
                }

                Column(
                    verticalArrangement = Arrangement.Center,
                    horizontalAlignment = Alignment.CenterHorizontally,
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxHeight()
                ) {
                    TakenImageControls(
                        imageBytes = imageBytes,
                        imageActivityViewModel = imageActivityViewModel,
                        onRepeatScan = onRepeatScan
                    )
                }
            }
        }

        val isAnalyzing by imageActivityViewModel.isAnalyzing.observeAsState()
        if (isAnalyzing == true) {
            Progress(color = R.color.main_color)
        }

        val qualityMetrics by imageActivityViewModel.qualityMetrics.observeAsState()
        if (qualityMetrics != null) {
            onQualityMetricsSet.invoke(qualityMetrics)
        }
    }
}

@Composable
fun TakenImageControls(
    imageBytes: ByteArray?,
    imageActivityViewModel: ImageActivityViewModel,
    onRepeatScan: () -> Unit
) {
    Row(
        horizontalArrangement = Arrangement.SpaceAround
    ) {
        Button(onClick = {
            onRepeatScan.invoke()
        }) {
            Text(text = stringResource(R.string.repeat_scan))
        }

        Spacer(
            modifier = Modifier
                .width(50.dp)
        )

        Button(
            modifier = Modifier
                .padding(bottom = 20.dp),
            onClick = {
                imageActivityViewModel.analyzeImage(imageBytes!!)
            }, enabled = imageBytes != null
        ) {
            Text(stringResource(R.string.accept_image))
        }
    }
}