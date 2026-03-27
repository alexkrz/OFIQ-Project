/**
 * @file CameraPreview.kt
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
import android.content.res.Configuration
import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.core.resolutionselector.ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.livedata.observeAsState
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import de.bund.bsi.ofiqmobile.R
import de.bund.bsi.ofiqmobile.viewmodel.CameraActivityViewModel
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.util.concurrent.Executor
import kotlin.coroutines.resume
import kotlin.coroutines.suspendCoroutine

/**
 * Camera preview, that shows the live images of the camera
 */
@Composable
fun CameraPreviewScreen(
    cameraActivityViewModel: CameraActivityViewModel
) {
    val isFrontCameraSelected by cameraActivityViewModel.isFrontCameraSelected.observeAsState(false)
    val lifecycleOwner = LocalLifecycleOwner.current
    val context = LocalContext.current
    val preview = Preview.Builder().build()
    val previewView = remember {
        PreviewView(context)
    }
    val imageCapture = remember {
        val resolutionSelector = ResolutionSelector.Builder()
        resolutionSelector.setResolutionStrategy(ResolutionStrategy(Size(1920, 1080), FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER))
        ImageCapture.Builder().setResolutionSelector(resolutionSelector.build())
            .build()
    }
    val cameraxSelector = CameraSelector.Builder().requireLensFacing(if (isFrontCameraSelected) CameraSelector.LENS_FACING_FRONT else CameraSelector.LENS_FACING_BACK).build()
    LaunchedEffect(if (isFrontCameraSelected) CameraSelector.LENS_FACING_FRONT else CameraSelector.LENS_FACING_BACK) {
        val cameraProvider = context.getCameraProvider()
        cameraProvider.unbindAll()
        cameraProvider.bindToLifecycle(lifecycleOwner, cameraxSelector, preview, imageCapture)
        preview.setSurfaceProvider(previewView.surfaceProvider)
    }

    Box(modifier = Modifier.fillMaxSize()) {
        AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())

        val isPortrait = LocalConfiguration.current.orientation == Configuration.ORIENTATION_PORTRAIT
        val isCapturing by cameraActivityViewModel.isCapturing.observeAsState(false)
        Button(
            onClick = {
                cameraActivityViewModel.setCamera(!isFrontCameraSelected)
            },
            enabled = !isCapturing,
            modifier = Modifier
                .align(Alignment.TopCenter)
                .padding(top = if (isPortrait) 100.dp else 20.dp)
        ) {
            Text(text = stringResource(R.string.switch_camera))
        }

        Text(
            text = stringResource(R.string.scan_face_explanation),
            modifier = Modifier
                .align(Alignment.TopCenter)
                .padding(top = if (isPortrait) 200.dp else 100.dp)
                .background(
                    color = Color.Gray,
                    shape = RoundedCornerShape(7.dp)
                )
                .padding(8.dp)
        )

        Button(
            onClick = {
                cameraActivityViewModel.captureImage(
                    imageCapture = { executor, callback -> imageCapture.takePicture(executor, callback) },
                    isPortrait = isPortrait
                )
                CoroutineScope(Dispatchers.Main).launch {
                    context.getCameraProvider().unbind(preview)
                }
            },
            enabled = !isCapturing,
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(bottom = if (isPortrait) 100.dp else 20.dp)
        ) {
            Text(text = stringResource(R.string.capture))
        }
    }
}

private suspend fun Context.getCameraProvider(): ProcessCameraProvider =
    suspendCoroutine { continuation ->
        ProcessCameraProvider.getInstance(this).also { cameraProvider ->
            cameraProvider.addListener({
                continuation.resume(cameraProvider.get())
            }, ContextCompat.getMainExecutor(this))
        }
    }