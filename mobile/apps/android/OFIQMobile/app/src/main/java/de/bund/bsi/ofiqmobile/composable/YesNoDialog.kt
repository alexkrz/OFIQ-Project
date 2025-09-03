/**
 * @file YesNoDialog.kt
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

package de.bund.bsi.ofiqmobile.composable

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.wrapContentSize
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Alignment.Companion.CenterHorizontally
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import de.bund.bsi.ofiqmobile.R

/**
 * Shows a dialog with an accept and an abort button above a dimmed background
 */
@Composable
fun YesNoDialog(
    modifier: Modifier = Modifier,
    text: String,
    onAccept: () -> Unit,
    onDismiss: () -> Unit
) {
    DialogWithDimBackground(
        modifier = modifier
            .wrapContentSize(),
        showCloseButton = false,
        onDismiss = onDismiss
    ) {
        Text(
            modifier = Modifier
                .align(CenterHorizontally)
                .padding(top = 20.dp),
            text = text
        )
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(top = 10.dp),
            contentAlignment = Alignment.Center
        ) {
            Row (
                modifier = Modifier
                    .fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                val buttonWidth = 135.dp
                Button(
                    onClick = onDismiss,
                    modifier = Modifier
                        .width(buttonWidth),
                ) {
                    Text(text = stringResource(R.string.cancel))
                }

                Button(
                    onClick = onAccept,
                    modifier = Modifier
                        .width(buttonWidth),
                ) {
                    Text(text = stringResource(R.string.accept))
                }
            }
        }
    }
}

@Preview
@Composable
fun YesNoDialogPreview() {
    Box(
        modifier = Modifier
            .fillMaxSize()
    ) {
        YesNoDialog(
            text = "TestText",
            onAccept = {
                //do nothing
            }
        ) {
            //do nothing
        }
    }
}