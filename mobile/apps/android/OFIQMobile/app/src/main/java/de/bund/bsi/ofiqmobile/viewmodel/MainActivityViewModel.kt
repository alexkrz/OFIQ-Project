/**
 * @file MainActivityViewModel.kt
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

import android.content.Context
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import de.bund.bsi.ofiqmobile.lib.IOfiq
import de.bund.bsi.ofiqmobile.lib.ReturnCode
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

/**
 * Initializes the OFIQ library
 */
class MainActivityViewModel(private val ofiq: IOfiq) {
    private val coroutineScope = CoroutineScope(Dispatchers.Default)

    private val _isInitializing = MutableLiveData(true)
    /**
     * Is set to true, if the lib is being initialized
     */
    val isInitializing: LiveData<Boolean> = _isInitializing

    private val _returnCode = MutableLiveData<ReturnCode?>(null)
    /**
     * The ReturnCode of the initialization. Is set, once the initialization is finished.
     */
    val returnCode: LiveData<ReturnCode?> = _returnCode

    /**
     * Is set to true, if the user pressed on the back button.
     */
    val showBackDialog = MutableLiveData(false)

    /**
     * Is set to true, of the app should be closed.
     */
    val closeApp = MutableLiveData(false)

    /**
     * Initializes the OFIQ library
     */
    fun initialize(context: Context) {
        _isInitializing.postValue(true)
        coroutineScope.launch {
            val initializeResult = ofiq.init(context)
            _returnCode.postValue(initializeResult?.code)
            _isInitializing.postValue(false)
        }
    }
}