/**
 * @file CameraActivityView.kt
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

import android.content.Context
import android.widget.Toast
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.livedata.observeAsState
import androidx.compose.ui.Modifier
import de.bund.bsi.ofiqmobile.R
import de.bund.bsi.ofiqmobile.viewmodel.CameraActivityViewModel

/**
 * Shows the camera preview and camera controls
 */
@Composable
fun CameraActivityView(
    cameraActivityViewModel: CameraActivityViewModel,
    context: Context,
    onImageTaken: (ByteArray) -> Unit
) {
    val showCameraPreview by cameraActivityViewModel.showCamera.observeAsState(false)
    Box(modifier = Modifier.fillMaxSize()) {
        if (showCameraPreview) {
            Box(modifier = Modifier.fillMaxSize()) {
                CameraPreviewScreen(cameraActivityViewModel)
            }
        }

        val errorMessage by cameraActivityViewModel.errorMessage.observeAsState()
        if (errorMessage != null) {
            Toast.makeText(context, "Error: $errorMessage", Toast.LENGTH_LONG).show()
            cameraActivityViewModel.clearError()
        }

        val imageBytes by cameraActivityViewModel.imageBytes.observeAsState()
        if (imageBytes != null) {
            onImageTaken.invoke(imageBytes!!)
        }

        val isCapturing by cameraActivityViewModel.isCapturing.observeAsState(false)
        if (isCapturing) {
            Progress(R.color.main_color)
        }
    }
}