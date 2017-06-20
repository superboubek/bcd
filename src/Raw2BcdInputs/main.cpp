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


#include "ImageIO.h"
#include "DeepImage.h"

#include "CovarianceMatrix.h"
#include "Utils.h"

#include "io_exr.h"

#include <iostream>
#include <ctime>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <memory>
#include <numeric>

using namespace std;

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

	Deepimf outputColor(header.width, header.height, 3);
	Deepimf outputHistogram(header.width, header.height, 3 * programArgs.m_histogramNbOfBins + 1); // + 1 for nb of samples as last channel
	Deepimf outputCovariance(header.width, header.height, 6);
	outputColor.fill(0.f);

	int floorBinIndex;
	int ceilBinIndex;
	float binFloatIndex;
	float floorBinWeight;
	float ceilBinWeight;

	float sample[4]; // assumes nbOfChannels can be only 3 or 4
	streamsize sampleSize = header.nbOfChannels * sizeof(float);
	for(int32_t line = 0; line < header.height; ++line)
	{
		for(int32_t col = 0; col < header.width; ++col)
		{
			for(int32_t sampleIndex = 0; sampleIndex < header.nbOfSamples; ++sampleIndex)
			{
				inputFile.read(reinterpret_cast<char*>(sample), sampleSize);
				for(int32_t channelIndex = 0; channelIndex < header.nbOfChannels; ++channelIndex)
				{
					outputColor.get(line, col, channelIndex) += sample[channelIndex];
					{ // fill histogram; code refactored from Ray Histogram Fusion PBRT code
						float value = sample[channelIndex];
						value = value > 0 ? value : 0;
						if(programArgs.m_histogramGamma > 1)
							value = pow(value, 1.f / programArgs.m_histogramGamma); // exponential scaling
						if(programArgs.m_histogramMaxValue > 0)
							value  = value / programArgs.m_histogramMaxValue; // normalize to the maximum value
						value = value > g_satureLevelGamma ? g_satureLevelGamma : value;

						binFloatIndex = value * (programArgs.m_histogramNbOfBins - 2);
						floorBinIndex = int(binFloatIndex);

						if(floorBinIndex < programArgs.m_histogramNbOfBins - 2) // in bounds
						{
							ceilBinIndex = floorBinIndex + 1;
							ceilBinWeight = binFloatIndex - floorBinIndex;
							floorBinWeight = 1.0f - ceilBinWeight;
						}
						else
						{ //out of bounds... v >= 1
							floorBinIndex = programArgs.m_histogramNbOfBins - 2;
							ceilBinIndex = floorBinIndex + 1;
							ceilBinWeight = (value - 1.0f) / (g_satureLevelGamma - 1.f);
							floorBinWeight = 1.0f - ceilBinWeight;
						}
						outputHistogram.get(line, col, channelIndex * programArgs.m_histogramNbOfBins + floorBinIndex) += floorBinWeight;
						outputHistogram.get(line, col, channelIndex * programArgs.m_histogramNbOfBins + ceilBinIndex) += ceilBinWeight;
					}
				}
				outputCovariance.get(line, col, static_cast<int>(ESymmetricMatrix3x3Data::e_xx)) += sample[0] * sample[0];
				outputCovariance.get(line, col, static_cast<int>(ESymmetricMatrix3x3Data::e_yy)) += sample[1] * sample[1];
				outputCovariance.get(line, col, static_cast<int>(ESymmetricMatrix3x3Data::e_zz)) += sample[2] * sample[2];
				outputCovariance.get(line, col, static_cast<int>(ESymmetricMatrix3x3Data::e_yz)) += sample[1] * sample[2];
				outputCovariance.get(line, col, static_cast<int>(ESymmetricMatrix3x3Data::e_xz)) += sample[0] * sample[2];
				outputCovariance.get(line, col, static_cast<int>(ESymmetricMatrix3x3Data::e_xy)) += sample[0] * sample[1];
			}
		}
	}

	float nbOfSamples = header.nbOfSamples;
	float invNbOfSamples = 1.f / nbOfSamples;
	float biasCorrectionFactor = nbOfSamples / (nbOfSamples - 1);
	float colorValue[3];

	outputColor.isotropicalScale(invNbOfSamples);

	for(int32_t line = 0; line < header.height; ++line)
	{
		for(int32_t col = 0; col < header.width; ++col)
		{
			outputHistogram.set(line, col, 3 * programArgs.m_histogramNbOfBins, nbOfSamples);

			colorValue[0] = outputColor.get(line, col, 0);
			colorValue[1] = outputColor.get(line, col, 1);
			colorValue[2] = outputColor.get(line, col, 2);
			{
				float& covValue = outputCovariance.get(line, col, static_cast<int>(ESymmetricMatrix3x3Data::e_xx));
				covValue = (covValue * invNbOfSamples - colorValue[0] * colorValue[0]) * biasCorrectionFactor;
			}
			{
				float& covValue = outputCovariance.get(line, col, static_cast<int>(ESymmetricMatrix3x3Data::e_yy));
				covValue = (covValue * invNbOfSamples - colorValue[1] * colorValue[1]) * biasCorrectionFactor;
			}
			{
				float& covValue = outputCovariance.get(line, col, static_cast<int>(ESymmetricMatrix3x3Data::e_zz));
				covValue = (covValue * invNbOfSamples - colorValue[2] * colorValue[2]) * biasCorrectionFactor;
			}
			{
				float& covValue = outputCovariance.get(line, col, static_cast<int>(ESymmetricMatrix3x3Data::e_yz));
				covValue = (covValue * invNbOfSamples - colorValue[1] * colorValue[2]) * biasCorrectionFactor;
			}
			{
				float& covValue = outputCovariance.get(line, col, static_cast<int>(ESymmetricMatrix3x3Data::e_xz));
				covValue = (covValue * invNbOfSamples - colorValue[0] * colorValue[2]) * biasCorrectionFactor;
			}
			{
				float& covValue = outputCovariance.get(line, col, static_cast<int>(ESymmetricMatrix3x3Data::e_xy));
				covValue = (covValue * invNbOfSamples - colorValue[0] * colorValue[1]) * biasCorrectionFactor;
			}
		}
	}

	ImageIO::writeEXR(outputColor, programArgs.m_outputColorFilePath.c_str());
	ImageIO::writeMultiChannelsEXR(outputCovariance, programArgs.m_outputCovarianceFilePath.c_str());
	ImageIO::writeMultiChannelsEXR(outputHistogram, programArgs.m_outputHistogramFilePath.c_str());

	return 0;
}

void pauseBeforeExit()
{
	cout << "Exiting program!" << endl;
//	cout << endl << "Press the Enter key to exit program." << endl;
//	cin.get();
}

int main(int argc, const char** argv)
{
	int rc = convertRawToBcd(argc, argv); // return code

	pauseBeforeExit();
	return rc;
}
