// --------------------------------------------------------------------------------------------------
// Source code for the paper no.36 "Bayesian Collaborative Denoising
// for Monte-Carlo Rendering", submitted to the ACM SIGGRAPH 2016
// conference technical paper program.
//
// For review only. Please, do not distribute outside the review context.
//
// Final open source licence and comments to be inserted if published.
//
// Copyright(C) 2014-2016
// The Authors
//
// All rights reserved.
// ---------------------------------------------------------------------------------------------------


#include "SamplesAccumulator.h"
#include "ImageIO.h"
#include "DeepImage.h"

#include "CovarianceMatrix.h"
#include "Utils.h"

#include <iostream>
#include <ctime>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <memory>
#include <numeric>

using namespace std;

namespace bcd
{

	static const char* g_pProgramName = "raw2bcd";
	static const char* g_pColorSuffix = "";
	static const char* g_pHistogramSuffix = "_hist";
	static const char* g_pCovarianceSuffix = "_cov";
	static const char* g_pDeepImageExtension = ".exr";

	const float g_satureLevelGamma = 2.f; // used in histogram accumulation

	class ProgramArguments
	{
	public:
		ProgramArguments() :
				m_histogramNbOfBins(20),
				m_histogramGamma(2.2f),
				m_histogramMaxValue(2.5f)
		{}

	public:
		string m_inputFilePath; ///< File path to the input image
		string m_outputColorFilePath; ///< File path to the output color image
		string m_outputHistogramFilePath; ///< File path to the output histogram image
		string m_outputCovarianceFilePath; ///< File path to the output covariance image
		float m_histogramNbOfBins;
		float m_histogramGamma;
		float m_histogramMaxValue;
	};

	typedef struct
	{
		int32_t version;
		int32_t width;
		int32_t height;
		int32_t nbOfSamples;
		int32_t nbOfChannels;
	} RawFileHeader;


	/// @brief Error/Exit print a message and exit.
	/// @param[in] msg Message to print before exiting program
	static void printError(const char *msg)
	{
		cerr << "Error in program '" << g_pProgramName << "': " << msg << endl;
	}


	static void printUsage()
	{
		ProgramArguments defaultProgramArgs;
		cout << "raw2bcd" << endl << endl;
		cout << "Usage: " << g_pProgramName << " <input> <outputPrefix>" << endl;
		cout << "Converts a raw file with all samples into the inputs for the BayesianCollaborativeDenoiser program" << endl;
		cout << "Required arguments list:" << endl;
		cout << "    <input>           The file path to the input image" << endl;
		cout << "    <outputPrefix>    The file path to the output image, without .exr extension" << endl;
	}


	bool parseProgramArguments(int argc, const char** argv, ProgramArguments& o_rProgramArguments)
	{
		if(argc != 3)
		{
			printUsage();
			return false;
		}

		o_rProgramArguments.m_inputFilePath = argv[1];
		if(!ifstream(o_rProgramArguments.m_inputFilePath))
		{
			cerr << "ERROR in program arguments: cannot open input file '" << o_rProgramArguments.m_inputFilePath << "'" << endl;
			return false;
		}

		string outputFilePathPrefix = argv[2];
		o_rProgramArguments.m_outputColorFilePath = outputFilePathPrefix + g_pColorSuffix + g_pDeepImageExtension;
		o_rProgramArguments.m_outputHistogramFilePath = outputFilePathPrefix + g_pHistogramSuffix + g_pDeepImageExtension;
		o_rProgramArguments.m_outputCovarianceFilePath = outputFilePathPrefix + g_pCovarianceSuffix + g_pDeepImageExtension;

		if(!ofstream(o_rProgramArguments.m_outputColorFilePath))
		{
			cerr << "ERROR in program arguments: cannot write output file '" << o_rProgramArguments.m_outputColorFilePath << "'" << endl;
			return false;
		}
		if(!ofstream(o_rProgramArguments.m_outputHistogramFilePath))
		{
			cerr << "ERROR in program arguments: cannot write output file '" << o_rProgramArguments.m_outputHistogramFilePath << "'" << endl;
			return false;
		}
		if(!ofstream(o_rProgramArguments.m_outputCovarianceFilePath))
		{
			cerr << "ERROR in program arguments: cannot write output file '" << o_rProgramArguments.m_outputCovarianceFilePath << "'" << endl;
			return false;
		}

		return true;
	}

	void printHeader(const RawFileHeader& i_rHeader)
	{
		cout << "Version: " << i_rHeader.version << endl;
		cout << "Resolution: " << i_rHeader.width << "x" << i_rHeader.height << endl;
		cout << "Nb of samples: " << i_rHeader.nbOfSamples << endl;
		cout << "Nb of channels: " << i_rHeader.nbOfChannels << endl;
	}

	int convertRawToBcd(int argc, const char** argv)
	{
		ProgramArguments programArgs;
		if(!parseProgramArguments(argc, argv, programArgs))
			return 1;

		ifstream inputFile(programArgs.m_inputFilePath.c_str(), ios::binary);
		RawFileHeader header;
		inputFile.read(reinterpret_cast<char*>(&header), sizeof(RawFileHeader));
		printHeader(header);

		HistogramParameters histoParams;
		histoParams.m_nbOfBins = programArgs.m_histogramNbOfBins;
		histoParams.m_gamma = programArgs.m_histogramGamma;
		histoParams.m_maxValue = programArgs.m_histogramMaxValue;
		SamplesAccumulator samplesAccumulator(header.width, header.height, histoParams);

		float sample[4]; // assumes nbOfChannels can be only 3 or 4
		streamsize sampleSize = header.nbOfChannels * sizeof(float);
		for(int32_t line = 0; line < header.height; ++line)
		{
			for(int32_t col = 0; col < header.width; ++col)
			{
				for(int32_t sampleIndex = 0; sampleIndex < header.nbOfSamples; ++sampleIndex)
				{
					inputFile.read(reinterpret_cast<char*>(sample), sampleSize);
					samplesAccumulator.addSample(line, col, sample[0], sample[1], sample[2]);
				}
			}
		}

		SamplesStatisticsImages samplesStats = samplesAccumulator.extractSamplesStatistics();
		Deepimf histoAndNbOfSamplesImage = Utils::mergeHistogramAndNbOfSamples(samplesStats.m_histoImage, samplesStats.m_nbOfSamplesImage);

		samplesStats.m_histoImage.clearAndFreeMemory();
		samplesStats.m_nbOfSamplesImage.clearAndFreeMemory();

		ImageIO::writeEXR(samplesStats.m_meanImage, programArgs.m_outputColorFilePath.c_str());
		ImageIO::writeMultiChannelsEXR(samplesStats.m_covarImage, programArgs.m_outputCovarianceFilePath.c_str());
		ImageIO::writeMultiChannelsEXR(histoAndNbOfSamplesImage, programArgs.m_outputHistogramFilePath.c_str());

		return 0;
	}

	void pauseBeforeExit()
	{
		cout << "Exiting program!" << endl;
	//	cout << endl << "Press the Enter key to exit program." << endl;
	//	cin.get();
	}

} // namespace bcd

int main(int argc, const char** argv)
{
	int rc = bcd::convertRawToBcd(argc, argv); // return code

	bcd::pauseBeforeExit();
	return rc;
}
