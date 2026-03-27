/**
 * @file MainActivity.kt
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

package de.bund.bsi.ofiqmobile.activity

import android.content.Intent
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.addCallback
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.getValue
import androidx.compose.runtime.livedata.observeAsState
import androidx.compose.ui.Modifier
import de.bund.bsi.ofiqmobile.OfiqMobileDemoAppApplication
import de.bund.bsi.ofiqmobile.composable.InitializeView
import de.bund.bsi.ofiqmobile.composable.OfiqScaffold
import de.bund.bsi.ofiqmobile.ui.theme.OFIQMobileDemoAppTheme
import de.bund.bsi.ofiqmobile.viewmodel.MainActivityViewModel
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

/**
 * Entry activity of the app. Initializes the OFIQ library and starts the face scan afterwards
 */
class MainActivity: ComponentActivity() {
    companion object {
        private val TAG = ResultActivity::class.simpleName
    }

    private lateinit var mainActivityViewModel: MainActivityViewModel

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        mainActivityViewModel = MainActivityViewModel((application as OfiqMobileDemoAppApplication).ofiq)
        val context = this

        setContent {
            OFIQMobileDemoAppTheme {
                OfiqScaffold {
                    Box(modifier = Modifier.fillMaxSize()) {
                        InitializeView(mainActivityViewModel = mainActivityViewModel) {
                            startActivity(Intent(context, CameraActivity::class.java))
                        }

                        val closeApp by mainActivityViewModel.closeApp.observeAsState(false)
                        if (closeApp) {
                            closeApp()
                        }
                    }
                }
            }
        }

        onBackPressedDispatcher.addCallback {
            mainActivityViewModel.showBackDialog.postValue(true)
        }

        CoroutineScope(Dispatchers.IO).launch {
            mainActivityViewModel.initialize(context)
        }
    }

    private fun closeApp() {
        finishAffinity()
    }
}
