/**
 * @file DataStreamAndroid.cpp
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
 * @author OFIQ development team
 */

#if defined(ANDROID)

#include "DataStreamAndroid.h"
#include <android/asset_manager.h>
#include <android/log.h>
#include <istream>
#include <string>

static AAssetManager* g_assetManager;

namespace OFIQ_LIB
{
  using pos_type = std::streambuf::pos_type;
  using off_type = std::streambuf::off_type;

  void SetAAssetManager(::AAssetManager* assetManager)
  {
    g_assetManager = assetManager;
  }

  asset_streambuf::asset_streambuf(::AAsset* asset)
  {
    char* beg(const_cast<char*>(static_cast<char const*>(::AAsset_getBuffer(asset))));
    char* end(beg + ::AAsset_getLength(asset));
    __android_log_print(ANDROID_LOG_INFO, __FUNCTION__, "streambuf for asset %p: %p .. %p", asset, beg, end);

    setg(beg, beg, end);
  }

  pos_type asset_streambuf::seekoff(off_type off, std::ios_base::seekdir dir, std::ios_base::openmode which)
  {
    pos_type result(off_type(-1));
    if ((which & (std::ios_base::in | std::ios_base::out)) == std::ios_base::in)
    {
      switch(dir)
      {
        case std::ios::beg:
          result = off;
	  break;
        case std::ios::end:
	  result = (egptr() - eback()) - (off + 1);
          break;
    	case std::ios::cur:
	  result = (gptr() - eback()) + off;
	  break;
        default:
	  return result;
      }
      return(seekpos(result, which)); 
    }
    return result; 
  }

  pos_type asset_streambuf::seekpos(pos_type pos, std::ios_base::openmode which)
  {
    if ((which & (std::ios_base::in | std::ios_base::out)) == std::ios_base::in)
    {
      char* p(eback() + pos);
      if((p >= eback()) && (p < egptr()))
      {
        setg(eback(), p, egptr());
        return pos;
      }
    }
    return pos_type(off_type(-1));
  }

  asset_istream::asset_istream(std::string const& filePath, std::ios_base::openmode mode)
   : std::istream(&streambuf)
   , asset(::AAssetManager_open(g_assetManager, filePath.c_str(), AASSET_MODE_BUFFER))
   , streambuf(asset)
  {
    __android_log_print(ANDROID_LOG_INFO, __FUNCTION__, "opened asset file \"%s\": %p", filePath.c_str(), asset);	
  }

  asset_istream::~asset_istream()
  {
    ::AAsset_close(asset);
    __android_log_print(ANDROID_LOG_INFO, __FUNCTION__, "closed asset: %p", asset);	
  }
}

#endif