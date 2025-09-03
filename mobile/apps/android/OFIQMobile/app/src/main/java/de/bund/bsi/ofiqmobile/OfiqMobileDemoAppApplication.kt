/**
 * @file OfiqMobileDemoAppApplication.kt
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

package de.bund.bsi.ofiqmobile

import android.app.Application
import de.bund.bsi.ofiqmobile.lib.IOfiq
import de.bund.bsi.ofiqmobile.lib.OfiqLib
import de.bund.bsi.ofiqmobile.lib.QualityMetrics

/**
 * Stores scan and analysis results for the app
 */
class OfiqMobileDemoAppApplication: Application() {
    val ofiq: IOfiq = OfiqLib()
    /**
     * Bytes of the scanned image
     */
    var imageBytes: ByteArray? = null

    /**
     * Analysis result
     */
    var qualityMetrics: QualityMetrics? = null

    /**
     * Cleans the analysis and scan results
     */
    fun clean() {
        imageBytes = null
        qualityMetrics = null
    }

    override fun onTerminate() {
        ofiq.destroy()

        super.onTerminate()
    }
}