// This file is part of the reference implementation for the paper 
//   Bayesian Collaborative Denoising for Monte-Carlo Rendering
//   Malik Boughida and Tamy Boubekeur.
//   Computer Graphics Forum (Proc. EGSR 2017), vol. 36, no. 4, p. 137-153, 2017
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE.txt file.

#ifndef IMAGE_IO_H
#define IMAGE_IO_H

#include <vector>

namespace bcd
{

	template<typename T> class DeepImage;

	/// @brief Class used as an interface for loading .exr files as DeepImage
	class ImageIO
	{
	private:
		ImageIO() {}

	public:
		static bool loadEXR(DeepImage<float>& o_rImage, const char* i_pFilePath);
		static bool loadMultiChannelsEXR(DeepImage<float>& o_rImage, const char* i_pFilePath);

		/// @brief Writes a .exr image with 1 or 3 channels (if 1, converted to 3)
		static bool writeEXR(const DeepImage<float>& i_rImage, const char* i_pFilePath);
		static bool writeMultiChannelsEXR(const DeepImage<float>& i_rImage, const char* i_pFilePath);

	private:
		static void reorderDataForWritingEXR(std::vector<float>& o_rData, const DeepImage<float>& i_rImage);
	};

}

#endif // IMAGE_IO_H
