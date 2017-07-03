// This file is part of the reference implementation for the paper 
//   Bayesian Collaborative Denoising for Monte-Carlo Rendering
//   Malik Boughida and Tamy Boubekeur.
//   Computer Graphics Forum (Proc. EGSR 2017), vol. 36, no. 4, p. 137-153, 2017
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE.txt file.

#include "Utils.h"

#include "DeepImage.h"

#include <cstring>

using namespace std;

namespace bcd
{

	bool Utils::separateNbOfSamplesFromHistogram(
			Deepimf& o_rHistImage,
			Deepimf& o_rNbOfSamplesImage,
			const Deepimf& i_rHistAndNbOfSamplesImage)
	{
		int w = i_rHistAndNbOfSamplesImage.getWidth();
		int h = i_rHistAndNbOfSamplesImage.getHeight();
		int d = i_rHistAndNbOfSamplesImage.getDepth() - 1;

		o_rHistImage.resize(w, h, d);
		o_rNbOfSamplesImage.resize(w, h, 1);

		size_t histDataSize = d * sizeof(float);
		ImfIt histIt = o_rHistImage.begin();
		ImfIt nbOfSamplesIt = o_rNbOfSamplesImage.begin();
		ImfConstIt histAndNbOfSamplesIt = i_rHistAndNbOfSamplesImage.begin();
		ImfConstIt histAndNbOfSamplesItEnd = i_rHistAndNbOfSamplesImage.end();

		for( ; histAndNbOfSamplesIt != histAndNbOfSamplesItEnd; ++histIt, ++nbOfSamplesIt, ++histAndNbOfSamplesIt)
		{
			memcpy(*histIt, *histAndNbOfSamplesIt, histDataSize);
			nbOfSamplesIt[0] = histAndNbOfSamplesIt[d];
		}
		return true;
	}

}
