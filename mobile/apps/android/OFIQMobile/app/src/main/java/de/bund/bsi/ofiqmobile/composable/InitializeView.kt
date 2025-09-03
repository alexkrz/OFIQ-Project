/**
 * @file InitializeView.kt
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

import android.util.Log
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.livedata.observeAsState
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import de.bund.bsi.ofiqmobile.R
import de.bund.bsi.ofiqmobile.lib.ReturnCode
import de.bund.bsi.ofiqmobile.viewmodel.MainActivityViewModel

/**
 * Shows the initializing screen of the app
 */
@Composable
fun InitializeView(
    mainActivityViewModel: MainActivityViewModel,
    onInitialized: () -> Unit
) {
    Box(Modifier.fillMaxSize()) {
        Text(
            text = stringResource(R.string.initializing),
            modifier = Modifier
                .align(Alignment.TopCenter)
                .padding(top = 50.dp)
        )

        val isInitializing by mainActivityViewModel.isInitializing.observeAsState(true)
        if (isInitializing) {
            Progress(
                color = R.color.main_color
            )
        }

        val returnCode by mainActivityViewModel.returnCode.observeAsState()
        val showBackDialog by mainActivityViewModel.showBackDialog.observeAsState(false)
        if (ReturnCode.SUCCESS == returnCode && !showBackDialog) {
            onInitialized()
        } else if (ReturnCode.SUCCESS != returnCode && !isInitializing) {
            Log.e("InitializeView", "InitializeView: returnCode $returnCode detected.")
            InitFailedDialog(
                mainActivityViewModel = mainActivityViewModel,
                returnCode = returnCode
            )
        }

        if (showBackDialog) {
            YesNoDialog(
                text = stringResource(R.string.close_app_dialog),
                onAccept = {
                    mainActivityViewModel.closeApp.postValue(true)
                    mainActivityViewModel.showBackDialog.postValue(false)
                },
                modifier = Modifier
                    .height(200.dp)
            ) {
                mainActivityViewModel.showBackDialog.postValue(false)
            }
        }
    }
}

/**
 * Shows a dialog, that informs the user of the failed initialization
 */
@Composable
fun InitFailedDialog(
    mainActivityViewModel: MainActivityViewModel,
    returnCode: ReturnCode?
) {
    DialogWithDimBackground(
        onDismiss = {
            //should not be reached
        },
        showCloseButton = false,
        modifier = Modifier
            .height(250.dp)
    ) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text(text = stringResource(R.string.could_not_initialize, returnCode.toString()))
            Button(onClick = {
                mainActivityViewModel.closeApp.postValue(true)
            }) {
                Text(text = stringResource(id = R.string.close))
            }
        }
    }
}