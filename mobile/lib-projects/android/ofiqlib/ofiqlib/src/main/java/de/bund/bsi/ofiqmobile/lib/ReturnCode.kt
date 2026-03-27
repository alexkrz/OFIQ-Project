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

/**
* Result code of OFIQ-Call
*/
enum class ReturnCode {
    /**
     * Success
     */
    SUCCESS,

    /** Failed to read an image.   */
    IMAGE_READING_ERROR,

    /** Failed to write an image to disk  */
    IMAGE_WRITING_ERROR,

    /** A required config parameter is missing  */
    MISSING_CONFIG_PARAM_ERROR,

    /** A required config parameter is missing  */
    UNKNOWN_CONFIG_PARAM_ERROR,

    /** Unable to detect face in the image  */
    FACE_DETECTION_ERROR,

    /** Unable to extract landmarks from face  */
    FACE_LANDMARK_EXTRACTION_ERROR,

    /** Unable to extract occlusion segments from face  */
    FACE_OCCLUSION_SEGMENTATION_ERROR,

    /** Unable to parse face  */
    FACE_PARSING_ERROR,

    /** Catch-all error  */
    UNKNOWN_ERROR,

    /** Failure to generate a quality score on the input image  */
    QUALITY_ASSESSMENT_ERROR,

    /** Function is not implemented  */
    NOT_IMPLEMENTED
}