// This file is part of the reference implementation for the paper 
//   Bayesian Collaborative Denoising for Monte-Carlo Rendering
//   Malik Boughida and Tamy Boubekeur.
//   Computer Graphics Forum (Proc. EGSR 2017), vol. 36, no. 4, p. 137-153, 2017
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE.txt file.

#ifndef UTILS_H
#define UTILS_H

#include <string>

namespace bcd
{

	template<typename T> class DeepImage;

	/// @brief Class for various useful functions
	class Utils
	{
	private:
		Utils() {}

	public:
		static bool separateNbOfSamplesFromHistogram(
				DeepImage<float>& o_rHistImage,
				DeepImage<float>& o_rNbOfSamplesImage,
				const DeepImage<float>& i_rHistAndNbOfSamplesImage
		);

		static std::string extractFolderPath(const std::string& i_rFilePath);

		static std::string getRelativePathFromFolder(const std::string& i_rFileAbsolutePath, const std::string& i_rFolderAbsolutePath);

	};

}

#endif // UTILS_H
