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

	string Utils::extractFolderPath(const string& i_rFilePath)
	{
		cout << "extracting folder path from file: '" << i_rFilePath << "': ";
		const char sep = '/';
		size_t pos = i_rFilePath.rfind(sep);
		if(pos == string::npos)
			return "";
		return i_rFilePath.substr(0, pos + 1);
	}

	string Utils::getRelativePathFromFolder(const string& i_rFileAbsolutePath, const string& i_rFolderAbsolutePath)
	{
		const char sep = '/';
		size_t l1 = i_rFileAbsolutePath.length();
		size_t l2 = i_rFolderAbsolutePath.length();
		size_t l = (l1 > l2 ? l2 : l1);

		size_t posAfterLastCommonSep = 0;
		for(size_t i = 0; i < l; ++i)
		{
			char c = i_rFileAbsolutePath[i];
			if(c != i_rFolderAbsolutePath[i])
				break;
			if(c == sep)
				posAfterLastCommonSep = i + 1;
		}

		string relativePath = "";
		for(size_t i = posAfterLastCommonSep; i < l2; ++i)
			if(i_rFolderAbsolutePath[i] == sep)
				relativePath += "../";

		relativePath += i_rFileAbsolutePath.substr(posAfterLastCommonSep);

		return relativePath;
	}

}
