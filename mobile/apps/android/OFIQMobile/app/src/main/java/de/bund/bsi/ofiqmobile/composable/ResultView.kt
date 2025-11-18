/**
 * @file ResultView.kt
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
import android.util.Log
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Refresh
import androidx.compose.material3.Icon
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
import de.bund.bsi.ofiqmobile.viewmodel.ResultActivityViewModel

/**
 * Shows the scanned image and the corresponding analyzed metrics
 */
@Composable
fun ResultView(
    qualityMetrics: QualityMetrics?,
    imageBitmap: Bitmap?,
    resultActivityViewModel: ResultActivityViewModel
) {
    OfiqScaffold(
        barIcons = {
            Icon(
                Icons.Outlined.Refresh,
                contentDescription = stringResource(R.string.restart_icon_content_description),
                modifier = Modifier
                    .clickable {
                        resultActivityViewModel.restartProcess.postValue(true)
                    }
                    .padding(end = 5.dp)
            )
        }
    ) {

        val isPortrait = LocalConfiguration.current.orientation == Configuration.ORIENTATION_PORTRAIT
        if (isPortrait) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(top = 20.dp),
                verticalArrangement = Arrangement.Top,
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                OfiqImage(
                    imageBitmap = imageBitmap,
                    height = 0.3f
                )

                ResultTableContainer(
                    qualityMetrics = qualityMetrics,
                    resultActivityViewModel = resultActivityViewModel
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
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxHeight()
                ) {
                    ResultTableContainer(
                        qualityMetrics = qualityMetrics,
                        resultActivityViewModel = resultActivityViewModel
                    )
                }
            }
        }

        val showBackDialog by resultActivityViewModel.showBackDialog.observeAsState(false)
        if (showBackDialog) {
            YesNoDialog(
                text = stringResource(R.string.restart_process_dialog_title),
                onAccept = {
                    resultActivityViewModel.restartProcess.postValue(true)
                    resultActivityViewModel.showBackDialog.postValue(false)
                },
                modifier = Modifier
                    .height(200.dp)
            ) {
                resultActivityViewModel.showBackDialog.postValue(false)
            }
        }
    }
}

/**
 * Shows the result table of the analyzed metrics
 */
@Composable
fun ResultTableContainer(
    qualityMetrics: QualityMetrics?,
    resultActivityViewModel: ResultActivityViewModel
) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .padding(top = 20.dp)
    ) {
        if (qualityMetrics != null) {
            ResultTable(
                qualityMetrics = qualityMetrics,
                resultActivityViewModel = resultActivityViewModel
            )
        } else {
            Log.e("ResultView", "ResultView: QualityMetrics are not set!")
            resultActivityViewModel.restartProcess.postValue(true)
        }
    }
}