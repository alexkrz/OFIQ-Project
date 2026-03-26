/**
 * @file CameraActivityViewModel.kt
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

package de.bund.bsi.ofiqmobile.viewmodel

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Log
import androidx.annotation.NonNull
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.core.content.ContextCompat
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import de.bund.bsi.ofiqmobile.model.Utils.Companion.toByteArray
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.util.concurrent.Executor
import java.util.concurrent.Executors

/**
 * View model of the camera activity
 */
class CameraActivityViewModel(
    private val context: Context
) {
    companion object {
        private val TAG = CameraActivityViewModel::class.simpleName
    }

    private val _imageBytes = MutableLiveData<ByteArray?>(null)
    /**
     * Scanned image as byte array
     */
    val imageBytes: LiveData<ByteArray?> = _imageBytes

    private val _errorMessage = MutableLiveData<String?>(null)
    /**
     * Error message if the camera process fails
     */
    val errorMessage: LiveData<String?> = _errorMessage

    private val _isFrontCameraSelected = MutableLiveData(false)
    /**
     * Is set to true, if the front camera is selected and false if the back camera is selected
     */
    val isFrontCameraSelected: LiveData<Boolean> = _isFrontCameraSelected

    private val _isCapturing = MutableLiveData(false)
    /**
     * Is set to true, if the camera is capturing images
     */
    val isCapturing: LiveData<Boolean> = _isCapturing

    /**
     * Is set to true, if the camera is started
     */
    val showCamera = MutableLiveData(false)

    fun interface IImageCapture {
        fun takePicture(
            executor: Executor,
            callback: ImageCapture.OnImageCapturedCallback
        )
    }

    /**
     * Captures an image using the camera
     */
    fun captureImage(
        imageCapture: IImageCapture,
        isPortrait: Boolean
    ) {
        Log.d(TAG, "captureImage: begin capturing image")
        _isCapturing.postValue(true)
        imageCapture.takePicture(Executors.newSingleThreadExecutor(), object: ImageCapture.OnImageCapturedCallback() {
            override fun onCaptureSuccess(image: ImageProxy) {
                super.onCaptureSuccess(image)

                Log.d(TAG, "onCaptureSuccess: image successfully captured. Image width: ${image.width}, height: ${image.height}. Beginning conversion to bitmap")

                CoroutineScope(Dispatchers.Default).launch {
                    val rotatedBitmap = when(isFrontCameraSelected.value) {
                        true -> if (isPortrait) rotateBitmap(image.toBitmap(), 270f) else rotateBitmap(image.toBitmap(), 180f)
                        false -> if (isPortrait) rotateBitmap(image.toBitmap(), 90f) else rotateBitmap(image.toBitmap(), 180f)
                        null -> image.toBitmap()
                    }

                    Log.d(TAG, "onCaptureSuccess: bitmap created")

                    image.close()

                    _imageBytes.postValue(toByteArray(rotatedBitmap))
                    _isCapturing.postValue(false)

                    Log.d(TAG, "onCaptureSuccess: finished")
                }
            }

            override fun onError(exception: ImageCaptureException) {
                super.onError(exception)

                Log.e(TAG, "onError: ", exception)

                _isCapturing.postValue(false)
                _errorMessage.postValue(exception.message)
            }
        })
    }

    /**
     * Sets the front or back camera
     */
    fun setCamera(isFrontCamera: Boolean) {
        _isFrontCameraSelected.postValue(isFrontCamera)
    }

    /**
     * Removes the error message
     */
    fun clearError() {
        _errorMessage.postValue(null)
    }

    /**
     * Starts the camera, or asks for permissions
     */
    fun setupCamera(
        onCallCameraPermissionRequest: () -> Unit
    ) {
        when (PackageManager.PERMISSION_GRANTED) {
            ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.CAMERA
            ) -> {
                startCamera()
            }

            else -> {
                onCallCameraPermissionRequest.invoke()
            }
        }
    }

    /**
     * Starts the camera and its preview
     */
    fun startCamera() {
        showCamera.postValue(true)
    }

    private fun rotateBitmap(bitmap: Bitmap, degrees: Float): Bitmap {
        val matrix = Matrix().apply { postRotate(degrees) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }
}