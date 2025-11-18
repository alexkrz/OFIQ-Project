/**
 * @file ofiqlib.h
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

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

//! Project version number for ofiqlib.
FOUNDATION_EXPORT double ofiqlibVersionNumber;

//! Project version string for ofiqlib.
FOUNDATION_EXPORT const unsigned char ofiqlibVersionString[];

#import <ofiqlib/Version.h>
#import <ofiqlib/InitializeResult.h>
#import <ofiqlib/QualityMetrics.h>
#import <ofiqlib/QualityMeasureAssessment.h>

@protocol Ofiq <NSObject>
/// Method to start Image quality check.
/// @return Int value based on result.
- (InitializeResult *)initialize;

/// Evaluates the quality of a facial image and returns the corresponding quality metrics.
/// @param image A byte array representing the facial image to be assessed.
/// @return An instance of [QualityMetrics] containing the evaluation results,
/// including the return code, processing time, additional information, and a list of individual quality measurements.
- (QualityMetrics *)faceQa:(UIImage *)image;

/// Deletes the reference to the C++ ofiq object
/// After a destroy, the lib can be initialised again with init
- (void)destroy;

/// Return the version of the ios framework
/// @return Version in format: MAJOR.MINOR.PATCH
- (NSString *)getFrameworkVersion;

/// Returns the version of the ofiq lib (shared c++ code)
/// @return Version in format: MAJOR.MINOR.PATCH
- (NSString *)getCoreVersion;
@end

@interface OfiqLib : NSObject <Ofiq> {
    void *ofiqImplPointer;
    InitializeResult *initializeResult;
}
- (instancetype)init;
- (int)getMeasurementEnumValue:(NSString *)measurementName;
@end
