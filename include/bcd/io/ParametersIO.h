// This file is part of the reference implementation for the paper 
//   Bayesian Collaborative Denoising for Monte-Carlo Rendering
//   Malik Boughida and Tamy Boubekeur.
//   Computer Graphics Forum (Proc. EGSR 2017), vol. 36, no. 4, p. 137-153, 2017
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE.txt file.

#ifndef PARAMETERS_IO_H
#define PARAMETERS_IO_H

#include "IDenoiser.h"

#include <string>

namespace bcd
{

	/// @brief Struct to store the absolute paths to the input files
	struct InputFileNames
	{
		std::string m_colors;
		std::string m_histograms;
		std::string m_covariances;
	};

	struct PrefilteringParameters
	{
		PrefilteringParameters() : m_performSpikeRemoval(true), m_spikeRemovalThresholdStDevFactor(1.5f) {}
		bool m_performSpikeRemoval;
		float m_spikeRemovalThresholdStDevFactor;
	};

	struct MultiscaleDenoiserParameters
	{
		MultiscaleDenoiserParameters() : m_nbOfScales(3), m_monoscaleParameters() {}
		int m_nbOfScales;
		DenoiserParameters m_monoscaleParameters;
	};

	struct PipelineParameters
	{
		InputFileNames m_inputFileNames;
		PrefilteringParameters m_prefilteringParameters;
		MultiscaleDenoiserParameters m_denoiserParameters;
	};

	struct PipelineParametersSelector
	{
		PipelineParametersSelector() :
				m_inputFileNames(true),
				m_prefilteringParameters(true),
				m_denoiserParameters(true) {}
		bool m_inputFileNames;
		bool m_prefilteringParameters;
		bool m_denoiserParameters;
	};

	/// @brief Class used as an interface for loading and writing .bcd.json files
	class ParametersIO
	{
	private:
		ParametersIO() {}

	public:

		static const std::string& getPipelineParametersFileExtension()
		{
			static const std::string extension = "bcd.json";
			return extension;
		}

		static bool load(PipelineParameters& o_rParams, const std::string& i_rFilePath,
				PipelineParametersSelector i_selector = PipelineParametersSelector());

		static bool write(const PipelineParameters& i_rParams, const std::string& i_rFilePath,
				PipelineParametersSelector i_selector = PipelineParametersSelector());
	};

}

#endif // PARAMETERS_IO_H
