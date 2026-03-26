/**
 * @file CameraActivity.kt
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

package de.bund.bsi.ofiqmobile.activity

import android.Manifest
import android.content.Intent
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.addCallback
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import de.bund.bsi.ofiqmobile.OfiqMobileDemoAppApplication
import de.bund.bsi.ofiqmobile.composable.CameraActivityView
import de.bund.bsi.ofiqmobile.ui.theme.OFIQMobileDemoAppTheme
import de.bund.bsi.ofiqmobile.viewmodel.CameraActivityViewModel

/**
 * Takes an image with the camera
 */
class CameraActivity: ComponentActivity() {
    companion object {
        private val TAG = ResultActivity::class.simpleName
    }

    private val cameraActivityViewModel = CameraActivityViewModel(this)
    private val cameraPermissionRequest =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
            if (isGranted) {
                cameraActivityViewModel.startCamera()
            } else {
                startActivity(Intent(this, MainActivity::class.java))
                finish()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val context = this

        enableEdgeToEdge()
        setContent {
            OFIQMobileDemoAppTheme {
                CameraActivityView(
                    cameraActivityViewModel = cameraActivityViewModel,
                    context = context
                ) { imageBytes ->
                    (application as OfiqMobileDemoAppApplication).imageBytes = imageBytes
                    startActivity(Intent(context, ImageActivity::class.java))
                    finish()
                }
            }
        }

        onBackPressedDispatcher.addCallback {
            //do nothing
        }

        cameraActivityViewModel.setupCamera {
            cameraPermissionRequest.launch(Manifest.permission.CAMERA)
        }
    }
}

