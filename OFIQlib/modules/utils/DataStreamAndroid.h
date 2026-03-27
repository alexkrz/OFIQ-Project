/**
 * @file DataStreamAndroid.h
 *
 * @copyright Copyright (c) 2024  Federal Office for Information Security, Germany
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
 * @brief Provides the Android-specific file interface implementation for streaming model data.
 * @author OFIQ development team
 */

#pragma once

#include <istream>
#include <streambuf>
#include <ios>

struct AAsset;
struct AAssetDir;
struct AAssetManager;

namespace OFIQ_LIB
{
    void SetAAssetManager(::AAssetManager* assetManager);

    class asset_streambuf: public std::streambuf
    {
    public:
        asset_streambuf(::AAsset* asset);
        virtual ~asset_streambuf() = default;
    protected:
        virtual pos_type seekoff(off_type off, std::ios_base::seekdir dir, std::ios_base::openmode which) override;
        virtual pos_type seekpos(pos_type pos, std::ios_base::openmode which) override;
    };

    class asset_istream: public std::istream
    {
    public:
        asset_istream(std::string const& filePath, std::ios_base::openmode mode = std::ios_base::in);
        virtual ~asset_istream();
    private:
        ::AAsset* asset;
        asset_streambuf streambuf;
    };

    using DataStream = asset_istream;
}