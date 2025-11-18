/**
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

package de.bund.bsi.ofiqmobile.lib

import android.content.Context
import android.content.res.AssetManager
import de.bund.bsi.ofiqmobile.lib.BuildConfig
import java.io.File

class OfiqLib: IOfiq {

    private var ofiqImpl: NativePointer? = null
    private var initializeResult: InitializeResult? = null

    private external fun init(instancePointer: NativePointer);

    private external fun initAssetManager(assetManager: AssetManager)

    private external fun initialize(pathToModels: String, pathToConfig: String, ptr: Long): InitializeResult

    private external fun assessQuality(ptr: Long, image: ByteArray): QualityMetrics

    private external fun destroy(ptr: Long)

    private external fun getVersion(ptr: Long): String

    private external fun measurementMapping(enumName: String): Int

    init {
        initNativeLib()
    }

    private fun initNativeLib() {
        if (ofiqImpl != null && ofiqImpl!!.longPointer != null) {
            // destroy old instance
            destroy()
        }
        ofiqImpl = NativePointer(null)
        init(ofiqImpl!!)
    }

    /**
     *  Method to start Image quality check.
     * @return Int value based on result.
     */
    override fun init(context: Context): InitializeResult {
        // check whether the c++ instance has been deleted
        if (ofiqImpl == null || ofiqImpl!!.longPointer == null)
            initNativeLib()

        // initialize AssetManager
        val assetManager = context.assets
        initAssetManager(assetManager)

        val configFileName = "ofiq_config.jaxn"

        initializeResult = initialize("", configFileName, ofiqImpl!!.longPointer!!)
        return initializeResult!!
    }

    /**
     * Evaluates the quality of a facial image and returns the corresponding quality metrics.
     *
     * @param image A byte array representing the facial image to be assessed.
     * @return An instance of {@link QualityMetrics} containing the evaluation results,
     *         including the return code, processing time, additional information, and a list of individual quality measurements.
     */
    override fun faceQa(image: ByteArray): QualityMetrics {
        if (ofiqImpl != null && ofiqImpl!!.longPointer != null && initializeResult != null && initializeResult!!.code == ReturnCode.SUCCESS) {
            return assessQuality(ofiqImpl!!.longPointer!!, image)
        }
        return QualityMetrics(ReturnCode.UNKNOWN_ERROR, 0, "ofiq not successful initialized", ArrayList())
    }

    override fun getCoreVersion(): String {
        if (ofiqImpl != null && ofiqImpl!!.longPointer != null) {
            return getVersion(ofiqImpl!!.longPointer!!)
        }
        return "0.0.0"
    }

    override fun getModuleVersion(): String {
        return BuildConfig.VERSION_NAME;
    }

    override fun destroy() {
        if (ofiqImpl != null && ofiqImpl!!.longPointer != null) {
            initializeResult = null
            ofiqImpl!!.longPointer?.let { destroy(it) }
            ofiqImpl = null
        }
    }

    /**
     * This function is only needed for the unit test to get the value from the measurementMapping.
     * The map is implemented in ofiqlib.cpp.
     */
    fun getMeasurementEnumValue(measurementName: String): Int {
        return measurementMapping(measurementName)
    }

    companion object {
        // Used to load the 'ofiqlib' library on application startup.
        init {
            System.loadLibrary("ofiqlib")
        }
    }
}