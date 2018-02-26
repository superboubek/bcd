// This file is part of the reference implementation for the paper
//   Bayesian Collaborative Denoising for Monte-Carlo Rendering
//   Malik Boughida and Tamy Boubekeur.
//   Computer Graphics Forum (Proc. EGSR 2017), vol. 36, no. 4, p. 137-153, 2017
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE.txt file.

#include "ParametersIO.h"

#include "Utils.h"

#include "json.hpp"

#include <iostream>
#include <fstream>

using namespace std;

using Json = nlohmann::json;

namespace bcd
{

	void testJson()
	{
		Json j;
		j["pi"] = 3.141;
		j["happy"] = true;
		j["name"] = "Niels";

		cout << setw(2) << j << endl;
		cout << setw(6) << j << endl;
	}

	bool ParametersIO::load(PipelineParameters& o_rParams, const string& i_rFilePath,
			PipelineParametersSelector i_selector)
	{
		if(i_rFilePath == "")
		{
			cerr << "Couldn't load parameters: empty file name" << endl;
			return false;
		}
		ifstream file(i_rFilePath);
		if(!file)
		{
			cerr << "Error: couldn't open file '" << i_rFilePath << "'" << endl;
			return false;
		}
		string folderPath = Utils::extractFolderPath(i_rFilePath);
		Json jsonObject;
		file >> jsonObject;

		Json::const_iterator it;
		Json::const_iterator notFound = jsonObject.cend();
		if(i_selector.m_inputFileNames)
		{
			string newFilePath;

			it = jsonObject.find("inputColorFile");
			if(it != notFound)
			{
				newFilePath = folderPath;
				newFilePath += it->get<string>();
				o_rParams.m_inputFileNames.m_colors = newFilePath;
			}

			it = jsonObject.find("inputHistoFile");
			if(it != notFound)
			{
				newFilePath = folderPath;
				newFilePath += it->get<string>();
				o_rParams.m_inputFileNames.m_histograms = newFilePath;
			}

			it = jsonObject.find("inputCovarFile");
			if(it != notFound)
			{
				newFilePath = folderPath;
				newFilePath += it->get<string>();
				o_rParams.m_inputFileNames.m_covariances = newFilePath;
			}
		}

		if(i_selector.m_prefilteringParameters)
		{
			if((it = jsonObject.find("performSpikeRemovalPrefiltering")) != notFound)
				o_rParams.m_prefilteringParameters.m_performSpikeRemoval = it.value();
			if((it = jsonObject.find("spikeRemovalThresholdStDevFactor")) != notFound)
				o_rParams.m_prefilteringParameters.m_spikeRemovalThresholdStDevFactor = it.value();
		}

		if(i_selector.m_denoiserParameters)
		{
			if((it = jsonObject.find("nbOfScales")) != notFound)
				o_rParams.m_denoiserParameters.m_nbOfScales = it.value();
			DenoiserParameters& rParams = o_rParams.m_denoiserParameters.m_monoscaleParameters;
			if((it = jsonObject.find("histoDistanceThreshold")) != notFound)
				rParams.m_histogramDistanceThreshold = it.value();
			if((it = jsonObject.find("useCuda")) != notFound)
				rParams.m_useCuda = it.value();
			if((it = jsonObject.find("nbOfCores")) != notFound)
				rParams.m_nbOfCores = it.value();
			if((it = jsonObject.find("patchRadius")) != notFound)
				rParams.m_patchRadius = it.value();
			if((it = jsonObject.find("searchWindowRadius")) != notFound)
				rParams.m_searchWindowRadius = it.value();
			if((it = jsonObject.find("randomPixelOrder")) != notFound)
				rParams.m_useRandomPixelOrder = it.value();
			if((it = jsonObject.find("markedPixelsSkippingProbability")) != notFound)
				rParams.m_markedPixelsSkippingProbability = it.value();
			if((it = jsonObject.find("minEigenValue")) != notFound)
				rParams.m_minEigenValue = it.value();
		}

		return true;
	}

	bool ParametersIO::write(const PipelineParameters& i_rParams, const string& i_rFilePath,
			PipelineParametersSelector i_selector)
	{
		if(i_rFilePath == "")
		{
			cerr << "Couldn't save parameters: empty file name" << endl;
			return false;
		}
		ofstream file(i_rFilePath);
		if(!file)
		{
			cerr << "Error: couldn't write file '" << i_rFilePath << "'" << endl;
			return false;
		}
		string folderPath = Utils::extractFolderPath(i_rFilePath);

		Json jsonObject;
		if(i_selector.m_inputFileNames)
		{
			const InputFileNames& rParams = i_rParams.m_inputFileNames;
			jsonObject["inputColorFile"] = Utils::getRelativePathFromFolder(rParams.m_colors, folderPath);
			jsonObject["inputHistoFile"] = Utils::getRelativePathFromFolder(rParams.m_histograms, folderPath);
			jsonObject["inputCovarFile"] = Utils::getRelativePathFromFolder(rParams.m_covariances, folderPath);
		}
		if(i_selector.m_prefilteringParameters)
		{
			jsonObject["performSpikeRemovalPrefiltering"] = i_rParams.m_prefilteringParameters.m_performSpikeRemoval;
			jsonObject["spikeRemovalThresholdStDevFactor"] = i_rParams.m_prefilteringParameters.m_spikeRemovalThresholdStDevFactor;
		}
		if(i_selector.m_denoiserParameters)
		{
			jsonObject["nbOfScales"] = i_rParams.m_denoiserParameters.m_nbOfScales;
			const DenoiserParameters& rParams = i_rParams.m_denoiserParameters.m_monoscaleParameters;
			jsonObject["histoDistanceThreshold"] = rParams.m_histogramDistanceThreshold;
			jsonObject["useCuda"] = rParams.m_useCuda;
			jsonObject["nbOfCores"] = rParams.m_nbOfCores;
			jsonObject["patchRadius"] = rParams.m_patchRadius;
			jsonObject["searchWindowRadius"] = rParams.m_searchWindowRadius;
			jsonObject["randomPixelOrder"] = rParams.m_useRandomPixelOrder;
			jsonObject["markedPixelsSkippingProbability"] = rParams.m_markedPixelsSkippingProbability;
			jsonObject["minEigenValue"] = rParams.m_minEigenValue;
		}
		file << jsonObject.dump(4);
	}


} // namespace bcd
