// This file is part of the reference implementation for the paper 
//   Bayesian Collaborative Denoising for Monte-Carlo Rendering
//   Malik Boughida and Tamy Boubekeur.
//   Computer Graphics Forum (Proc. EGSR 2017), vol. 36, no. 4, p. 137-153, 2017
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE.txt file.

#include "ImageIO.h"

#include "DeepImage.h"

#include "io_exr.h"

using namespace std;

namespace bcd
{

	typedef DeepImage<float> Deepimf;

	bool ImageIO::loadEXR(Deepimf& o_rImage, const char* i_pFilePath)
	{
		cout << "Loading " << i_pFilePath << endl;
		int width, height, depth;
		float* aValues = nullptr;
		{
			int intWidth, intHeight;
			aValues = readImageEXR(i_pFilePath, &intWidth, &intHeight);
			width = int(intWidth);
			height = int(intHeight);
		}

		if (!aValues) {
			cerr << "error :: '" << i_pFilePath << "' not found  or not a correct exr image" << endl;
			return false;
		}

		depth = 3;
		int nbOfPixels = width * height;

		// test if image is really a color image even if it has more than one channel
		{

			// dc equals 3
			int i=0;
			while (i < nbOfPixels && aValues[i] == aValues[nbOfPixels + i] && aValues[i] == aValues[2 * nbOfPixels + i ])
				i++;
			if (i == nbOfPixels)
				depth = 1;
		}

		o_rImage.resize(width, height, depth);

		for(int l = 0; l < height; l++)
			for(int c = 0; c < width; c++)
				for (int z = 0; z < depth; z++)
					o_rImage.set(l, c, z, aValues[z * nbOfPixels + l * width + c]);

		delete[] aValues;

		return true;
	}


	bool ImageIO::loadMultiChannelsEXR(Deepimf& o_rImage, const char* i_pFilePath)
	{
		cout << "Loading " << i_pFilePath << endl;
		int width, height, depth, nbOfPixels;
		float* pValues = nullptr;
		pValues = readMultiImageEXR(i_pFilePath, &width, &height, &depth);
		nbOfPixels = width * height;

		o_rImage.resize(width, height, depth);

		for(int l = 0; l < height; l++)
			for(int c = 0; c < width; c++)
				for (int z = 0; z < depth; z++)
					o_rImage.set(l, c, z, pValues[z * nbOfPixels + l * width + c]);

		free(pValues);

		return true;
	}


	bool ImageIO::writeEXR(const Deepimf& i_rImage, const char* i_pFilePath)
	{
		if(i_rImage.getDepth() == 1)
		{
			int width = i_rImage.getWidth();
			int height = i_rImage.getHeight();
			Deepimf triChannelImage(width, height, 3);
			float val;
			for(int l = 0; l < height; l++)
				for(int c = 0; c < width; c++)
				{
					val = i_rImage.get(l, c, 0);
					triChannelImage.set(l, c, 0, val);
					triChannelImage.set(l, c, 1, val);
					triChannelImage.set(l, c, 2, val);
				}
			vector<float> data;
			reorderDataForWritingEXR(data, triChannelImage);
			writeImageEXR(i_pFilePath, data.data(),
					static_cast<int>(triChannelImage.getWidth()),
					static_cast<int>(triChannelImage.getHeight()));
		}
		else
		{
			vector<float> data;
			reorderDataForWritingEXR(data, i_rImage);
			writeImageEXR(i_pFilePath, data.data(),
					static_cast<int>(i_rImage.getWidth()),
					static_cast<int>(i_rImage.getHeight()));
		}
		return true;
	}


	bool ImageIO::writeMultiChannelsEXR(const Deepimf& i_rImage, const char* i_pFilePath)
	{
		vector<float> data;
		reorderDataForWritingEXR(data, i_rImage);
		writeMultiImageEXR(i_pFilePath, data.data(),
				static_cast<int>(i_rImage.getWidth()),
				static_cast<int>(i_rImage.getHeight()),
				static_cast<int>(i_rImage.getDepth()));
		return true;
	}

	void ImageIO::reorderDataForWritingEXR(std::vector<float>& o_rData, const Deepimf& i_rImage)
	{
		int width = i_rImage.getWidth();
		int height = i_rImage.getHeight();
		int depth = i_rImage.getDepth();
		o_rData.resize(width*height*depth);
		for(int l = 0; l < height; l++)
			for(int c = 0; c < width; c++)
				for (int z = 0; z < depth; z++)
					o_rData[(z*height + l)*width + c] = i_rImage.get(l, c, z);
	}

} // namespace bcd
