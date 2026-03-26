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

interface IOfiq {
    /**
     * Method to start Image quality check.
     * @param context The context is required to access the files in the assets directory
     * @return Int value based on result.
     */
    fun init(context: Context): InitializeResult?

    /**
     * Evaluates the quality of a facial image and returns the corresponding quality metrics.
     *
     * @param image A byte array representing the facial image to be assessed.
     * @return An instance of [QualityMetrics] containing the evaluation results,
     * including the return code, processing time, additional information, and a list of individual quality measurements.
     */
    fun faceQa(image: ByteArray): QualityMetrics

    /**
     * Returns the version of the ofiq lib (shared c++ code)
     * @return Version in format (MAJOR.MINOR.PATCH)
     */
    fun getCoreVersion(): String

    /**
     * Return the version of the java/c++ module
     * @return Version in format (MAJOR.MINOR.PATCH)
     */
    fun getModuleVersion(): String

    /**
     * Deletes the reference to the C++ ofid object
     * After a destroy, the lib can be initialised again with init.
     */
    fun destroy()
}