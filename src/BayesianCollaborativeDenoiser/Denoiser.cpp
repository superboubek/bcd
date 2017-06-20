// This file is part of the reference implementation for the paper 
//   Bayesian Collaborative Denoising for Monte-Carlo Rendering
//   Malik Boughida and Tamy Boubekeur.
//   Computer Graphics Forum (Proc. EGSR 2017), vol. 36, no. 4, p. 137-153, 2017
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE.txt file.

#include "Denoiser.h"

#ifdef FOUND_CUDA
#include "CudaHistogramDistance.h"
#endif

#include <iostream>
#include <algorithm>
#include <random>
#include <chrono>

#include <cassert>

#define USE_ATOMIC
#ifndef USE_ATOMIC
#define USE_CRITICAL
#endif


using namespace std;
using namespace Eigen;


#ifdef COMPUTE_DENOISING_STATS

DenoisingStatistics::DenoisingStatistics() :
		m_nbOfManagedPixels(0),
		m_nbOfDenoiseOnlyMainPatch(0),
		m_chronoElapsedTimes(),
		m_chronometers()
{
	m_chronoElapsedTimes.fill(0.f);
}

DenoisingStatistics& DenoisingStatistics::operator+=(const DenoisingStatistics& i_rStats)
{
	m_nbOfManagedPixels += i_rStats.m_nbOfManagedPixels;
	m_nbOfDenoiseOnlyMainPatch += i_rStats.m_nbOfDenoiseOnlyMainPatch;
	for(int i = 0; i < static_cast<int>(EChronometer::e_nb); ++i)
		m_chronoElapsedTimes[i] += i_rStats.m_chronoElapsedTimes[i];
	return *this;
}

void DenoisingStatistics::storeElapsedTimes()
{
	for(int i = 0; i < static_cast<int>(EChronometer::e_nb); ++i)
		m_chronoElapsedTimes[i] = m_chronometers[i].getElapsedTime();
}

void DenoisingStatistics::print()
{
	cout << "Number of pixels with fall back to simple average: "
		 << m_nbOfDenoiseOnlyMainPatch << " / " << m_nbOfManagedPixels << endl;
	cout << "Chronometers:" << endl;
	cout << "  denoisePatchAndSimilarPatches:      " << Chronometer::getStringFromTime(m_chronoElapsedTimes[static_cast<size_t>(EChronometer::e_denoisePatchAndSimilarPatches     )]) << endl;
	cout << "    selectSimilarPatches              " << Chronometer::getStringFromTime(m_chronoElapsedTimes[static_cast<size_t>(EChronometer::e_selectSimilarPatches              )]) << endl;
	cout << "    denoiseSelectedPatches            " << Chronometer::getStringFromTime(m_chronoElapsedTimes[static_cast<size_t>(EChronometer::e_denoiseSelectedPatches            )]) << endl;
	cout << "      computeNoiseCovPatchesMean      " << Chronometer::getStringFromTime(m_chronoElapsedTimes[static_cast<size_t>(EChronometer::e_computeNoiseCovPatchesMean        )]) << endl;
	cout << "      denoiseSelectedPatchesStep1     " << Chronometer::getStringFromTime(m_chronoElapsedTimes[static_cast<size_t>(EChronometer::e_denoiseSelectedPatchesStep1       )]) << endl;
	cout << "      denoiseSelectedPatchesStep2     " << Chronometer::getStringFromTime(m_chronoElapsedTimes[static_cast<size_t>(EChronometer::e_denoiseSelectedPatchesStep2       )]) << endl;
	cout << "      aggregateOutputPatches          " << Chronometer::getStringFromTime(m_chronoElapsedTimes[static_cast<size_t>(EChronometer::e_aggregateOutputPatches            )]) << endl;
	cout << endl;
}

#endif



// to shorten notations
const size_t g_xx = static_cast<size_t>(ESymmetricMatrix3x3Data::e_xx);
const size_t g_yy = static_cast<size_t>(ESymmetricMatrix3x3Data::e_yy);
const size_t g_zz = static_cast<size_t>(ESymmetricMatrix3x3Data::e_zz);
const size_t g_yz = static_cast<size_t>(ESymmetricMatrix3x3Data::e_yz);
const size_t g_xz = static_cast<size_t>(ESymmetricMatrix3x3Data::e_xz);
const size_t g_xy = static_cast<size_t>(ESymmetricMatrix3x3Data::e_xy);

void Denoiser::ompTest()
{
	m_width = m_inputs.m_pColors->getWidth();
	m_height = m_inputs.m_pColors->getHeight();
	m_nbOfPixels = m_width * m_height;
	vector<PixelPosition> pixelSet(m_nbOfPixels);
	{
		int k = 0;
		for (int line = 0; line < m_height; line++)
			for (int col = 0; col < m_width; col++)
				pixelSet[k++] = PixelPosition(line, col);
	}
	reorderPixelSet(pixelSet);

	m_outputs.m_pDenoisedColors->fill(0.f);
	int radius = 2;

#pragma omp parallel for ordered schedule(dynamic)
	for (int pixelIndex = 0; pixelIndex < m_nbOfPixels; pixelIndex++)
	{
		int l0, c0, li, ci, lf, cf, l, c;
		pixelSet[pixelIndex].get(l0, c0);
//		cout << "(" << l0 << "," << c0 << ")" << endl;
		li = max(0, l0 - radius);
		ci = max(0, c0 - radius);
		lf = min(m_width - 1, l0 + radius);
		cf = min(m_height - 1, c0 + radius);
#ifdef USE_CRITICAL
#pragma omp critical(accumulateOutput)
#endif
		{
			for (l = li; l <= lf; l++)
				for (c = ci; c <= cf; c++)
				{
#ifdef USE_ATOMIC
					float& rValue = m_outputs.m_pDenoisedColors->get(l, c, 0);
#pragma omp atomic
					rValue += 0.04f;
#endif
#ifdef USE_CRITICAL
					m_outputs.m_pDenoisedColors->get(l, c, 0) += 0.04f;
#endif
				}
		}
	}

}

bool Denoiser::denoise()
{
	if(!inputsOutputsAreOk())
		return false;

	m_width = m_inputs.m_pColors->getWidth();
	m_height = m_inputs.m_pColors->getHeight();
	m_nbOfPixels = m_width * m_height;
	int widthWithoutBorder = m_width - 2 * m_parameters.m_patchRadius;
	int heightWithoutBorder = m_height - 2 * m_parameters.m_patchRadius;
	int nbOfPixelsWithoutBorder = widthWithoutBorder * heightWithoutBorder;

	computeNbOfSamplesSqrt();
	computePixelCovFromSampleCov();

#ifdef FOUND_CUDA
	if(m_parameters.m_useCuda)
		cout << "Parallelizing computations using Cuda" << endl;
	else
		cout << "Cuda parallelization has been disabled" << endl;
#else
	if(m_parameters.m_useCuda)
		cout << "WARNING: Cuda parallelization has been disabled because the program has not been built with Cuda" << endl;
	else
		cout << "The program has not been built with Cuda" << endl;
#endif

	if(m_parameters.m_nbOfCores > 0)
		omp_set_num_threads(m_parameters.m_nbOfCores);

#pragma omp parallel
#pragma omp master
	{
		m_parameters.m_nbOfCores = omp_get_num_threads();
		// now m_parameters.m_nbOfCores is set to the actual number of threads
		// even if it was set to the default value 0
		if(m_parameters.m_nbOfCores > 1)
			cout << "Parallelizing computations with " << m_parameters.m_nbOfCores << " threads using OpenMP" << endl;
		else
			cout << "No parallelization using OpenMP" << endl;
	}

	vector<PixelPosition> pixelSet(nbOfPixelsWithoutBorder);
	{
		int k = 0;
		int lMin = m_parameters.m_patchRadius;
		int lMax = m_height - m_parameters.m_patchRadius - 1;
		int cMin = m_parameters.m_patchRadius;
		int cMax = m_width - m_parameters.m_patchRadius - 1;
		for (int l = lMin; l <= lMax; l++)
			for (int c = cMin; c <= cMax; c++)
				pixelSet[k++] = PixelPosition(l, c);
	}
	reorderPixelSet(pixelSet);

	m_outputSummedColorImages.resize(m_parameters.m_nbOfCores);
	m_outputSummedColorImages[0].resize(m_width, m_height, m_inputs.m_pColors->getDepth());
	m_outputSummedColorImages[0].fill(0.f);
	for(int i = 1; i < m_parameters.m_nbOfCores; ++i)
		m_outputSummedColorImages[i] = m_outputSummedColorImages[0];

	m_estimatesCountImages.resize(m_parameters.m_nbOfCores);
	m_estimatesCountImages[0].resize(m_width, m_height, 1);
	m_estimatesCountImages[0].fill(0);
	for(int i = 1; i < m_parameters.m_nbOfCores; ++i)
		m_estimatesCountImages[i] = m_estimatesCountImages[0];

	m_isCenterOfAlreadyDenoisedPatchImage.resize(m_width, m_height, 1);
	m_isCenterOfAlreadyDenoisedPatchImage.fill(false);

	int chunkSize; // nb of pixels a thread has to treat before dynamically asking for more work
	chunkSize = widthWithoutBorder * (2 * m_parameters.m_searchWindowRadius);

	int nbOfPixelsComputed = 0;
	int currentPercentage = 0, newPercentage = 0;
#pragma omp parallel
	{
		DenoisingUnit denoisingUnit(*this);
#pragma omp for ordered schedule(dynamic, chunkSize)
		for (int pixelIndex = 0; pixelIndex < nbOfPixelsWithoutBorder; pixelIndex++)
		{
//			if(pixelIndex != 2222)
//				continue;
			PixelPosition mainPatchCenter = pixelSet[pixelIndex];
			denoisingUnit.denoisePatchAndSimilarPatches(mainPatchCenter);
#pragma omp atomic
			++nbOfPixelsComputed;
			if(omp_get_thread_num() == 0)
			{
				newPercentage = (nbOfPixelsComputed * 100) / nbOfPixelsWithoutBorder;
				if(newPercentage != currentPercentage)
				{
					currentPercentage = newPercentage;
					cout << "\r" << currentPercentage << " %" << flush;
				}
			}

		}
#pragma omp master
		cout << endl << endl;
#pragma omp barrier
#ifdef COMPUTE_DENOISING_STATS
#pragma omp critical
		{
			denoisingUnit.m_uStats->storeElapsedTimes();
			denoisingUnit.m_uStats->print();
		}
#endif
	}

	m_outputs.m_pDenoisedColors->fill(0.f);
	finalAggregation();

	return true;
}

bool Denoiser::inputsOutputsAreOk()
{
	if(m_parameters.m_patchRadius != 1 && m_parameters.m_useCuda)
	{
		m_parameters.m_useCuda = false;
		cout << "Warning: disabling Cuda, that cannot be used for patch radius " << m_parameters.m_patchRadius << " > 1" << endl;
	}
	return true;
}

void Denoiser::computeNbOfSamplesSqrt()
{
	m_nbOfSamplesSqrtImage = *(m_inputs.m_pNbOfSamples);
	for(float* pPixelValues : m_nbOfSamplesSqrtImage)
		pPixelValues[0] = sqrt(pPixelValues[0]);
}

void Denoiser::computePixelCovFromSampleCov()
{
	m_pixelCovarianceImage = *(m_inputs.m_pSampleCovariances);
	ImfIt covIt = m_pixelCovarianceImage.begin();
	float nbOfSamplesInv;
	for(const float* pPixelNbOfSamples : *(m_inputs.m_pNbOfSamples))
	{
		nbOfSamplesInv = 1.f / *pPixelNbOfSamples;
		covIt[0] *= nbOfSamplesInv;
		covIt[1] *= nbOfSamplesInv;
		covIt[2] *= nbOfSamplesInv;
		covIt[3] *= nbOfSamplesInv;
		covIt[4] *= nbOfSamplesInv;
		covIt[5] *= nbOfSamplesInv;
		++covIt;
	}
}

void Denoiser::reorderPixelSet(vector<PixelPosition>& io_rPixelSet) const
{
	if(m_parameters.m_useRandomPixelOrder)
		reorderPixelSetShuffle(io_rPixelSet);
	else if(m_parameters.m_nbOfCores > 1)
		reorderPixelSetJumpNextStrip(io_rPixelSet);
}
void Denoiser::reorderPixelSetJumpNextStrip(vector<PixelPosition>& io_rPixelSet) const
{
	int widthWithoutBorder = m_width - 2 * m_parameters.m_patchRadius;
	int heightWithoutBorder = m_height - 2 * m_parameters.m_patchRadius;
	int nbOfPixelsWithoutBorder = widthWithoutBorder * heightWithoutBorder;
	assert(nbOfPixelsWithoutBorder == io_rPixelSet.size());
	int chunkSize = widthWithoutBorder * (2 * m_parameters.m_searchWindowRadius);
	// chunkSize is the number of pixels of a strip of 2*searchWindowRadius lines
	reorderPixelSetJumpNextChunk(io_rPixelSet, chunkSize);
}

void Denoiser::reorderPixelSetJumpNextChunk(vector<PixelPosition>& io_rPixelSet, int i_chunkSize)
{
	int doubleChunkSize = 2 * i_chunkSize;
	int nbOfFullChunks = io_rPixelSet.size() / i_chunkSize;

	vector<PixelPosition> pixelSetCopy(io_rPixelSet);
	vector<PixelPosition>::iterator inputIt = pixelSetCopy.begin();
	vector<PixelPosition>::iterator outputIt = io_rPixelSet.begin();

	for(int chunkIndexStart = 0; chunkIndexStart < 2; ++chunkIndexStart)
	{
		inputIt = pixelSetCopy.begin() + chunkIndexStart * i_chunkSize;
		for(int chunkIndex = chunkIndexStart; chunkIndex < nbOfFullChunks; )
		{
			copy(inputIt, inputIt + i_chunkSize, outputIt);
			outputIt += i_chunkSize;
			chunkIndex += 2;
			if(chunkIndex < nbOfFullChunks)
				inputIt += doubleChunkSize;
		}
	}
}

void reorderPixelSetShuffleCPP11(vector<PixelPosition>& io_rPixelSet)
{
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	shuffle (io_rPixelSet.begin(), io_rPixelSet.end(), std::default_random_engine(seed));
}

void Denoiser::reorderPixelSetShuffle(vector<PixelPosition>& io_rPixelSet)
{
	reorderPixelSetShuffleCPP11(io_rPixelSet);
	/*
	PixelPosition* pPixelSetStart = io_rPixelSet.data();
	for(int i = io_rPixelSet.size(); i > 0; --i)
		swap(pPixelSetStart[rand() % i], pPixelSetStart[i-1]);
	*/
}



void Denoiser::finalAggregation()
{
	int nbOfImages = static_cast<int>(m_outputSummedColorImages.size());

	for(int bufferIndex = 1; bufferIndex < nbOfImages; ++bufferIndex)
	{
		ImfIt it = m_outputSummedColorImages[bufferIndex].begin();
		for(float* pSum : m_outputSummedColorImages[0])
		{
			pSum[0] += it[0];
			pSum[1] += it[1];
			pSum[2] += it[2];
			++it;
		}
	}
	for(int bufferIndex = 1; bufferIndex < nbOfImages; ++bufferIndex)
	{
		DeepImage<int>::iterator it = m_estimatesCountImages[bufferIndex].begin();
		for(int* pSum : m_estimatesCountImages[0])
		{
			pSum[0] += it[0];
			++it;
		}
	}
	ImfIt sumIt = m_outputSummedColorImages[0].begin();
	DeepImage<int>::iterator countIt = m_estimatesCountImages[0].begin();
	float countInv;
	for(float* pFinalColor : *(m_outputs.m_pDenoisedColors))
	{
		countInv = 1.f / countIt[0];
		pFinalColor[0] = countInv * sumIt[0];
		pFinalColor[1] = countInv * sumIt[1];
		pFinalColor[2] = countInv * sumIt[2];
		++sumIt;
		++countIt;
	}
}

DenoisingUnit::DenoisingUnit(Denoiser& i_rDenoiser) :
		m_rDenoiser(i_rDenoiser),
		m_width(i_rDenoiser.getImagesWidth()),
		m_height(i_rDenoiser.getImagesHeight()),
		
		m_histogramDistanceThreshold(i_rDenoiser.getParameters().m_histogramDistanceThreshold),
		m_patchRadius(i_rDenoiser.getParameters().m_patchRadius),
		m_searchWindowRadius(i_rDenoiser.getParameters().m_searchWindowRadius),
		
		m_nbOfPixelsInPatch((2 * i_rDenoiser.getParameters().m_patchRadius + 1)
						  * (2 * i_rDenoiser.getParameters().m_patchRadius + 1)),
		m_maxNbOfSimilarPatches((2 * i_rDenoiser.getParameters().m_searchWindowRadius + 1)
						 * (2 * i_rDenoiser.getParameters().m_searchWindowRadius + 1)),
		m_colorPatchDimension(3 * m_nbOfPixelsInPatch),
		
		m_pColorImage(i_rDenoiser.getInputs().m_pColors),
		m_pNbOfSamplesImage(i_rDenoiser.getInputs().m_pNbOfSamples),
		m_pHistogramImage(i_rDenoiser.getInputs().m_pHistograms),
		m_pCovarianceImage(&(i_rDenoiser.getPixelCovarianceImage())),

		m_pNbOfSamplesSqrtImage(&(i_rDenoiser.getNbOfSamplesSqrtImage())),
		m_pOutputSummedColorImage(&(i_rDenoiser.getOutputSummedColorImage(omp_get_thread_num()))),
		m_pEstimatesCountImage(&(i_rDenoiser.getEstimatesCountImage(omp_get_thread_num()))),
		m_pIsCenterOfAlreadyDenoisedPatchImage(&(i_rDenoiser.getIsCenterOfAlreadyDenoisedPatchImage())),

		m_nbOfBins(i_rDenoiser.getInputs().m_pHistograms->getDepth()),

		m_mainPatchCenter(),
		m_similarPatchesCenters(m_maxNbOfSimilarPatches),
		
		m_nbOfSimilarPatches(0),
		m_nbOfSimilarPatchesInv(0.f),
		
		m_noiseCovPatchesMean(static_cast<size_t>(m_nbOfPixelsInPatch)),

		m_colorPatches(m_maxNbOfSimilarPatches, VectorXf(m_colorPatchDimension)),
		m_colorPatchesMean(m_colorPatchDimension),
		m_centeredColorPatches(m_maxNbOfSimilarPatches, VectorXf(m_colorPatchDimension)),
		m_colorPatchesCovMat(m_colorPatchDimension, m_colorPatchDimension),
		m_clampedCovMat(m_colorPatchDimension, m_colorPatchDimension),
		m_inversedCovMat(m_colorPatchDimension, m_colorPatchDimension),
		m_denoisedColorPatches(m_maxNbOfSimilarPatches, VectorXf(m_colorPatchDimension)),

		m_eigenSolver(m_colorPatchDimension),

		m_tmpNoiseCovPatch(static_cast<size_t>(m_nbOfPixelsInPatch)),
		m_tmpVec(m_colorPatchDimension),
		m_tmpMatrix(m_colorPatchDimension, m_colorPatchDimension)
#ifdef FOUND_CUDA
		, m_uCudaHistogramDistance(nullptr)
		, m_distancesToNeighborPatches(m_maxNbOfSimilarPatches)
#endif
#ifdef COMPUTE_DENOISING_STATS
		, m_uStats(new DenoisingStatistics())
#endif
{
#ifdef FOUND_CUDA
	if(m_rDenoiser.getParameters().m_useCuda)
		m_uCudaHistogramDistance = unique_ptr<CudaHistogramDistance>(new CudaHistogramDistance(
				m_pHistogramImage->getDataPtr(), m_pNbOfSamplesImage->getDataPtr(),
				m_pHistogramImage->getWidth(), m_pHistogramImage->getHeight(), m_pHistogramImage->getDepth(),
				m_patchRadius, m_searchWindowRadius));
#endif
}

DenoisingUnit::~DenoisingUnit()
{
}

void DenoisingUnit::denoisePatchAndSimilarPatches(const PixelPosition& i_rMainPatchCenter)
{
	startChrono(EChronometer::e_denoisePatchAndSimilarPatches);
#ifdef COMPUTE_DENOISING_STATS
	++m_uStats->m_nbOfManagedPixels;
#endif
	m_mainPatchCenter = i_rMainPatchCenter;
	{
		float skippingProbability = m_rDenoiser.getParameters().m_markedPixelsSkippingProbability;
		if(skippingProbability != 0)
			if(m_pIsCenterOfAlreadyDenoisedPatchImage->getValue(m_mainPatchCenter, 0))
				if(skippingProbability == 1 || rand() < static_cast<int>(skippingProbability * RAND_MAX))
				{
					stopChrono(EChronometer::e_denoisePatchAndSimilarPatches);
					return;
				}
	}
#ifdef FOUND_CUDA
	if(m_rDenoiser.getParameters().m_useCuda)
		selectSimilarPatchesUsingCuda();
	else
		selectSimilarPatches();
#else
	selectSimilarPatches();
#endif
	if(m_nbOfSimilarPatches < m_colorPatchDimension + 1) // cannot inverse covariance matrix: fallback to simple average ; + 1 for safety
	{
		denoiseOnlyMainPatch();
#ifdef COMPUTE_DENOISING_STATS
		++m_uStats->m_nbOfDenoiseOnlyMainPatch;
#endif
		// m_pIsCenterOfAlreadyDenoisedPatchImage->set(m_mainPatchCenter, 0, true); // useless
		stopChrono(EChronometer::e_denoisePatchAndSimilarPatches);
		return;
	}
	denoiseSelectedPatches();
	stopChrono(EChronometer::e_denoisePatchAndSimilarPatches);
}

void DenoisingUnit::selectSimilarPatches()
{
	startChrono(EChronometer::e_selectSimilarPatches);
	m_nbOfSimilarPatches = 0;
	PixelWindow searchWindow(
			m_width, m_height, m_mainPatchCenter,
			m_searchWindowRadius,
			m_patchRadius);
//	float normalizedThreshold = m_histogramDistanceThreshold * m_nbOfPixelsInPatch;
	m_similarPatchesCenters.resize(m_maxNbOfSimilarPatches);
	for(PixelPosition neighborPixel : searchWindow)
	{
//		if(histogramPatchSummedDistanceBad(m_mainPatchCenter, neighborPixel) <= normalizedThreshold)
		if(histogramPatchDistance(m_mainPatchCenter, neighborPixel) <= m_histogramDistanceThreshold)
			m_similarPatchesCenters[m_nbOfSimilarPatches++] = neighborPixel;
	}
	assert(m_nbOfSimilarPatches > 0);
	m_nbOfSimilarPatchesInv = 1.f / m_nbOfSimilarPatches;
	m_similarPatchesCenters.resize(m_nbOfSimilarPatches);
//	m_colorPatches.resize(m_nbOfSimilarPatches);
//	m_centeredColorPatches.resize(m_nbOfSimilarPatches);
//	m_denoisedColorPatches.resize(m_nbOfSimilarPatches);
	stopChrono(EChronometer::e_selectSimilarPatches);
}

#ifdef FOUND_CUDA
void DenoisingUnit::selectSimilarPatchesUsingCuda()
{
	startChrono(EChronometer::e_selectSimilarPatches);
	m_nbOfSimilarPatches = 0;

	m_uCudaHistogramDistance->computeDistances(
			&(*(m_distancesToNeighborPatches.begin())),
			m_mainPatchCenter.m_line, m_mainPatchCenter.m_column);

	PixelWindow searchWindow(
			m_width, m_height, m_mainPatchCenter,
			m_searchWindowRadius,
			m_patchRadius);

	m_similarPatchesCenters.resize(m_maxNbOfSimilarPatches);
	int neighborIndex = 0;
	for(PixelPosition neighborPixel : searchWindow)
	{
		// Begin of temporary test
		/*
		if((rand() % 10000) == 0)
		{
			cout << "(line,column) = (" << neighborPixel.m_line << "," << neighborPixel.m_column << ")" << endl;
			cout << "  distance computed by CUDA: " << m_distancesToNeighborPatches[neighborIndex] << endl;
			cout << "  distance computed on the CPU: " << histogramPatchDistance(m_mainPatchCenter, neighborPixel) << endl << endl;
		}
		*/
		// End of temporary test
		if(m_distancesToNeighborPatches[neighborIndex++] <= m_histogramDistanceThreshold)
			m_similarPatchesCenters[m_nbOfSimilarPatches++] = neighborPixel;
	}
	assert(m_nbOfSimilarPatches > 0);
	m_nbOfSimilarPatchesInv = 1.f / m_nbOfSimilarPatches;
	m_similarPatchesCenters.resize(m_nbOfSimilarPatches);
//	m_colorPatches.resize(m_nbOfSimilarPatches);
//	m_centeredColorPatches.resize(m_nbOfSimilarPatches);
//	m_denoisedColorPatches.resize(m_nbOfSimilarPatches);
	stopChrono(EChronometer::e_selectSimilarPatches);
}
#endif // FOUND_CUDA

inline
float DenoisingUnit::histogramPatchSummedDistanceBad(
		const PixelPosition& i_rPatchCenter1,
		const PixelPosition& i_rPatchCenter2)
{
	float summedDistance = 0;
	PixelPatch pixPatch1(m_width, m_height, i_rPatchCenter1, m_patchRadius);
	PixelPatch pixPatch2(m_width, m_height, i_rPatchCenter2, m_patchRadius);

	assert(pixPatch1.getSize() == pixPatch2.getSize());

	PixPatchIt pixPatch1It = pixPatch1.begin();
	PixPatchIt pixPatch2It = pixPatch2.begin();
	PixPatchIt pixPatch1ItEnd = pixPatch1.end();
	for( ; pixPatch1It != pixPatch1ItEnd; ++pixPatch1It, ++pixPatch2It)
	{
		summedDistance += pixelHistogramDistanceBad(*pixPatch1It, *pixPatch2It);
	}
	return summedDistance;
}

inline
float DenoisingUnit::pixelHistogramDistanceBad(
		const PixelPosition& i_rPixel1,
		const PixelPosition& i_rPixel2)
{
	int nbOfNonBoth0Bins = 0;
	const float* pHistogram1Val = &(m_pHistogramImage->get(i_rPixel1, 0));
	const float* pHistogram2Val = &(m_pHistogramImage->get(i_rPixel2, 0));
	float binValue1, binValue2;
	float nbOfSamples1 = m_pNbOfSamplesImage->get(i_rPixel1, 0);
	float nbOfSamples2 = m_pNbOfSamplesImage->get(i_rPixel2, 0);;
	float diff;
	float sum = 0.f;
	for(int binIndex = 0; binIndex < m_nbOfBins; ++binIndex)
	{
		binValue1 = *pHistogram1Val++;
		binValue2 = *pHistogram2Val++;
		if(binValue1 == 0.f && binValue2 == 0.f)
			continue;
		++nbOfNonBoth0Bins;
		diff = nbOfSamples2 * binValue1 - nbOfSamples1 * binValue2;
		sum += diff * diff / (nbOfSamples1 * nbOfSamples2 * (binValue1 + binValue2));
	}
	return sum / nbOfNonBoth0Bins;
}

float DenoisingUnit::pixelHistogramDistanceBad2(
		const PixelPosition& i_rPixel1,
		const PixelPosition& i_rPixel2)
{
	int nbOfNonBoth0Bins = 0;
	const float* pHistogram1Val = &(m_pHistogramImage->get(i_rPixel1, 0));
	const float* pHistogram2Val = &(m_pHistogramImage->get(i_rPixel2, 0));
	float binValue1, binValue2;
	float diff;
	float sum = 0.f;
	float nbOfSamplesSqrtQuotient = m_pNbOfSamplesSqrtImage->get(i_rPixel1, 0)
			/ m_pNbOfSamplesSqrtImage->get(i_rPixel2, 0);
	float nbOfSamplesSqrtQuotientInv = 1.f / nbOfSamplesSqrtQuotient;
	for(int binIndex = 0; binIndex < m_nbOfBins; ++binIndex)
	{
		binValue1 = *pHistogram1Val++;
		binValue2 = *pHistogram2Val++;
		if(binValue1 == 0.f && binValue2 == 0.f)
			continue;
		++nbOfNonBoth0Bins;
		diff = binValue1 * nbOfSamplesSqrtQuotientInv - binValue2 * nbOfSamplesSqrtQuotient;
		sum += diff * diff / (binValue1 + binValue2);
	}
	return sum / nbOfNonBoth0Bins;
}

inline
float DenoisingUnit::histogramPatchDistance(
		const PixelPosition& i_rPatchCenter1,
		const PixelPosition& i_rPatchCenter2)
{
	float summedDistance = 0;
	PixelPatch pixPatch1(m_width, m_height, i_rPatchCenter1, m_patchRadius);
	PixelPatch pixPatch2(m_width, m_height, i_rPatchCenter2, m_patchRadius);

	assert(pixPatch1.getSize() == pixPatch2.getSize());

	PixPatchIt pixPatch1It = pixPatch1.begin();
	PixPatchIt pixPatch2It = pixPatch2.begin();
	PixPatchIt pixPatch1ItEnd = pixPatch1.end();
	int totalNbOfNonBoth0Bins = 0;
	int nbOfNonBoth0Bins = 0;
	for( ; pixPatch1It != pixPatch1ItEnd; ++pixPatch1It, ++pixPatch2It)
	{
		summedDistance += pixelSummedHistogramDistance(nbOfNonBoth0Bins, *pixPatch1It, *pixPatch2It);
		totalNbOfNonBoth0Bins += nbOfNonBoth0Bins;
	}
	return summedDistance / totalNbOfNonBoth0Bins;
}

inline
float DenoisingUnit::pixelSummedHistogramDistance(
		int& i_rNbOfNonBoth0Bins,
		const PixelPosition& i_rPixel1,
		const PixelPosition& i_rPixel2)
{
	i_rNbOfNonBoth0Bins = 0;
	const float* pHistogram1Val = &(m_pHistogramImage->get(i_rPixel1, 0));
	const float* pHistogram2Val = &(m_pHistogramImage->get(i_rPixel2, 0));
	float binValue1, binValue2;
	float nbOfSamples1 = m_pNbOfSamplesImage->get(i_rPixel1, 0);
	float nbOfSamples2 = m_pNbOfSamplesImage->get(i_rPixel2, 0);;
	float diff;
	float sum = 0.f;
	for(int binIndex = 0; binIndex < m_nbOfBins; ++binIndex)
	{
		binValue1 = *pHistogram1Val++;
		binValue2 = *pHistogram2Val++;
//		if(binValue1 == 0.f && binValue2 == 0.f) // TEMPORARILY COMMENTED
		if(binValue1 + binValue2 <= 1.f) // TEMPORARY
			continue;
		++i_rNbOfNonBoth0Bins;
		diff = nbOfSamples2 * binValue1 - nbOfSamples1 * binValue2;
		sum += diff * diff / (nbOfSamples1 * nbOfSamples2 * (binValue1 + binValue2));
	}
	return sum;
}

void DenoisingUnit::denoiseSelectedPatches()
{
	startChrono(EChronometer::e_denoiseSelectedPatches);

	computeNoiseCovPatchesMean();
	denoiseSelectedPatchesStep1();
	denoiseSelectedPatchesStep2();
	aggregateOutputPatches();

	stopChrono(EChronometer::e_denoiseSelectedPatches);
}

void DenoisingUnit::computeNoiseCovPatchesMean()
{
	startChrono(EChronometer::e_computeNoiseCovPatchesMean);

	CovMat3x3 zero3x3;
	zero3x3.m_data.fill(0.f);
	fill(m_noiseCovPatchesMean.m_blocks.begin(), m_noiseCovPatchesMean.m_blocks.end(), zero3x3);

	for(PixelPosition similarPatchCenter : m_similarPatchesCenters)
	{
		size_t patchPixelIndex = 0;
		ConstPatch patch(*m_pCovarianceImage, similarPatchCenter, m_patchRadius);
		for(const float* pPixelCovData : patch)
			m_tmpNoiseCovPatch.m_blocks[patchPixelIndex++].copyFrom(pPixelCovData);
		m_noiseCovPatchesMean += m_tmpNoiseCovPatch;
	}
	m_noiseCovPatchesMean *= m_nbOfSimilarPatchesInv;

	stopChrono(EChronometer::e_computeNoiseCovPatchesMean);
}

void DenoisingUnit::denoiseSelectedPatchesStep1()
{
	startChrono(EChronometer::e_denoiseSelectedPatchesStep1);

	pickColorPatchesFromColorImage(m_colorPatches);
	empiricalMean(m_colorPatchesMean, m_colorPatches, m_nbOfSimilarPatches);
	centerPointCloud(m_centeredColorPatches, m_colorPatchesMean, m_colorPatches, m_nbOfSimilarPatches);
	empiricalCovarianceMatrix(m_colorPatchesCovMat, m_centeredColorPatches, m_nbOfSimilarPatches);
	substractCovMatPatchFromMatrix(m_colorPatchesCovMat, m_noiseCovPatchesMean);
	clampNegativeEigenValues(m_clampedCovMat, m_colorPatchesCovMat);
	addCovMatPatchToMatrix(m_clampedCovMat, m_noiseCovPatchesMean);
	inverseSymmetricMatrix(m_inversedCovMat, m_clampedCovMat);
	finalDenoisingMatrixMultiplication(m_denoisedColorPatches, m_colorPatches, m_noiseCovPatchesMean, m_inversedCovMat, m_centeredColorPatches);

	stopChrono(EChronometer::e_denoiseSelectedPatchesStep1);
}

void DenoisingUnit::denoiseSelectedPatchesStep2()
{
	startChrono(EChronometer::e_denoiseSelectedPatchesStep2);

	empiricalMean(m_colorPatchesMean, m_denoisedColorPatches, m_nbOfSimilarPatches);
	centerPointCloud(m_centeredColorPatches, m_colorPatchesMean, m_denoisedColorPatches, m_nbOfSimilarPatches);
	empiricalCovarianceMatrix(m_colorPatchesCovMat, m_centeredColorPatches, m_nbOfSimilarPatches);
//	clampNegativeEigenValues(m_clampedCovMat, m_colorPatchesCovMat); // maybe to uncomment to ensure a stricly positive minimum for eigen values... In that case comment next line
	m_clampedCovMat = m_colorPatchesCovMat;
	addCovMatPatchToMatrix(m_clampedCovMat, m_noiseCovPatchesMean);
	inverseSymmetricMatrix(m_inversedCovMat, m_clampedCovMat);
	centerPointCloud(m_centeredColorPatches, m_colorPatchesMean, m_colorPatches, m_nbOfSimilarPatches);
	finalDenoisingMatrixMultiplication(m_denoisedColorPatches, m_colorPatches, m_noiseCovPatchesMean, m_inversedCovMat, m_centeredColorPatches);

	stopChrono(EChronometer::e_denoiseSelectedPatchesStep2);
}

void DenoisingUnit::denoiseOnlyMainPatch()
{
	m_colorPatchesMean.fill(0.f);
	int patchDataIndex = 0;
	for(const PixelPosition& rSimilarPatchCenter : m_similarPatchesCenters)
	{
		patchDataIndex = 0;
		ConstPatch patch(*m_pColorImage, rSimilarPatchCenter, m_patchRadius);
		for(const float* pPixelColorData : patch)
		{
			m_colorPatchesMean(patchDataIndex++) += pPixelColorData[0];
			m_colorPatchesMean(patchDataIndex++) += pPixelColorData[1];
			m_colorPatchesMean(patchDataIndex++) += pPixelColorData[2];
		}
	}
	Patch outputMainPatch(*m_pOutputSummedColorImage, m_mainPatchCenter, m_patchRadius);
	patchDataIndex = 0;
	for(float* pPixelOutputColorData : outputMainPatch)
	{
		pPixelOutputColorData[0] += m_nbOfSimilarPatchesInv * m_colorPatchesMean(patchDataIndex++);
		pPixelOutputColorData[1] += m_nbOfSimilarPatchesInv * m_colorPatchesMean(patchDataIndex++);
		pPixelOutputColorData[2] += m_nbOfSimilarPatchesInv * m_colorPatchesMean(patchDataIndex++);
	}
	ImageWindow<int> estimatesCountPatch(*m_pEstimatesCountImage, m_mainPatchCenter, m_patchRadius);
	for(int* pPixelEstimateCount : estimatesCountPatch)
		++(pPixelEstimateCount[0]);
}

void DenoisingUnit::pickColorPatchesFromColorImage(vector<VectorXf>& o_rColorPatches) const
{
	int patchIndex = 0;
	for(const PixelPosition& rSimilarPatchCenter : m_similarPatchesCenters)
	{
		VectorXf& rColorPatch = o_rColorPatches[patchIndex++];
		int patchDataIndex = 0;
		ConstPatch patch(*m_pColorImage, rSimilarPatchCenter, m_patchRadius);
		for(const float* pPixelColorData : patch)
		{
			rColorPatch(patchDataIndex++) = pPixelColorData[0];
			rColorPatch(patchDataIndex++) = pPixelColorData[1];
			rColorPatch(patchDataIndex++) = pPixelColorData[2];
		}
	}
}

void DenoisingUnit::empiricalMean(
		VectorXf& o_rMean, 
		const vector<VectorXf>& i_rPointCloud, 
		int i_nbOfPoints) const
{
	o_rMean.fill(0.f);
	for(int i = 0; i < i_nbOfPoints; ++i)
		o_rMean += i_rPointCloud[i];
	o_rMean *= (1.f / i_nbOfPoints);
}

void DenoisingUnit::centerPointCloud(
		vector<VectorXf>& o_rCenteredPointCloud,
		VectorXf& o_rMean, 
		const vector<VectorXf>& i_rPointCloud, 
		int i_nbOfPoints) const
{
	vector<VectorXf>::iterator it = o_rCenteredPointCloud.begin();
	for(int i = 0; i < i_nbOfPoints; ++i)
		*it++ = i_rPointCloud[i] - o_rMean;
}

void DenoisingUnit::empiricalCovarianceMatrix(
		MatrixXf& o_rCovMat, 
		const vector<VectorXf>& i_rCenteredPointCloud, 
		int i_nbOfPoints) const
{
	int d = static_cast<int>(o_rCovMat.rows());
	assert(d == o_rCovMat.cols());
	assert(d == i_rCenteredPointCloud[0].rows());
	o_rCovMat.fill(0.f);
	for(int i = 0; i < i_nbOfPoints; ++i)
		for(int c = 0; c < d; ++c)
			for(int r = 0; r < d; ++r)
				o_rCovMat(r, c) += i_rCenteredPointCloud[i](r) * i_rCenteredPointCloud[i](c);
	o_rCovMat *= (1.f / (i_nbOfPoints - 1));
}

void DenoisingUnit::addCovMatPatchToMatrix(MatrixXf& io_rMatrix, const CovMatPatch& i_rCovMatPatch) const
{
	int blockXIndex = 0, blockYIndex = 1, blockZIndex = 2;
	for(const CovMat3x3& rCovMat3x3 : i_rCovMatPatch.m_blocks)
	{
		io_rMatrix(blockXIndex, blockXIndex) += rCovMat3x3.m_data[g_xx];
		io_rMatrix(blockYIndex, blockYIndex) += rCovMat3x3.m_data[g_yy];
		io_rMatrix(blockZIndex, blockZIndex) += rCovMat3x3.m_data[g_zz];
		io_rMatrix(blockYIndex, blockZIndex) += rCovMat3x3.m_data[g_yz];
		io_rMatrix(blockZIndex, blockYIndex) += rCovMat3x3.m_data[g_yz];
		io_rMatrix(blockXIndex, blockZIndex) += rCovMat3x3.m_data[g_xz];
		io_rMatrix(blockZIndex, blockXIndex) += rCovMat3x3.m_data[g_xz];
		io_rMatrix(blockXIndex, blockYIndex) += rCovMat3x3.m_data[g_xy];
		io_rMatrix(blockYIndex, blockXIndex) += rCovMat3x3.m_data[g_xy];
		blockXIndex += 3;
		blockYIndex += 3;
		blockZIndex += 3;
	}
}

void DenoisingUnit::substractCovMatPatchFromMatrix(MatrixXf& io_rMatrix, const CovMatPatch& i_rCovMatPatch) const
{
	int blockXIndex = 0, blockYIndex = 1, blockZIndex = 2;
	for(const CovMat3x3& rCovMat3x3 : i_rCovMatPatch.m_blocks)
	{
		io_rMatrix(blockXIndex, blockXIndex) -= rCovMat3x3.m_data[g_xx];
		io_rMatrix(blockYIndex, blockYIndex) -= rCovMat3x3.m_data[g_yy];
		io_rMatrix(blockZIndex, blockZIndex) -= rCovMat3x3.m_data[g_zz];
		io_rMatrix(blockYIndex, blockZIndex) -= rCovMat3x3.m_data[g_yz];
		io_rMatrix(blockZIndex, blockYIndex) -= rCovMat3x3.m_data[g_yz];
		io_rMatrix(blockXIndex, blockZIndex) -= rCovMat3x3.m_data[g_xz];
		io_rMatrix(blockZIndex, blockXIndex) -= rCovMat3x3.m_data[g_xz];
		io_rMatrix(blockXIndex, blockYIndex) -= rCovMat3x3.m_data[g_xy];
		io_rMatrix(blockYIndex, blockXIndex) -= rCovMat3x3.m_data[g_xy];
		blockXIndex += 3;
		blockYIndex += 3;
		blockZIndex += 3;
	}
}

void DenoisingUnit::inverseSymmetricMatrix(MatrixXf& o_rInversedMatrix, const MatrixXf& i_rSymmetricMatrix)
{
	float minEigenVal = m_rDenoiser.getParameters().m_minEigenValue;

	int d = static_cast<int>(i_rSymmetricMatrix.rows());
	assert(d == i_rSymmetricMatrix.cols());
	assert(d == o_rInversedMatrix.rows());
	assert(d == o_rInversedMatrix.cols());
	assert(d == m_tmpMatrix.rows());
	assert(d == m_tmpMatrix.cols());

	m_eigenSolver.compute(i_rSymmetricMatrix); // Decomposes i_rSymmetricMatrix into V D V^T
	const MatrixXf& rEigenVectors = m_eigenSolver.eigenvectors(); // Matrix V
	const VectorXf& rEigenValues = m_eigenSolver.eigenvalues(); // Matrix D is rEigenValues.asDiagonal()

	float diagValue;
	for(int r = 0; r < d; ++r)
	{
//		if(rEigenValues(r) <= minEigenVal)
//			cout << "Warning: eigen value below minimum during inversion:" << endl;
		diagValue = 1.f / max(minEigenVal, rEigenValues(r));
		for(int c = 0; c < d; ++c)
			m_tmpMatrix(r, c) = diagValue * rEigenVectors(c, r);
	}
	// now m_tmpMatrix equals (D^-1) V^T
	o_rInversedMatrix = rEigenVectors * m_tmpMatrix;
}

void DenoisingUnit::clampNegativeEigenValues(MatrixXf& o_rClampedMatrix, const MatrixXf& i_rSymmetricMatrix)
{
	float minEigenVal = 0;

	int d = static_cast<int>(i_rSymmetricMatrix.rows());
	assert(d == i_rSymmetricMatrix.cols());
	assert(d == o_rClampedMatrix.rows());
	assert(d == o_rClampedMatrix.cols());
	assert(d == m_tmpMatrix.rows());
	assert(d == m_tmpMatrix.cols());

	m_eigenSolver.compute(i_rSymmetricMatrix); // Decomposes i_rSymmetricMatrix into V D V^T
	const MatrixXf& rEigenVectors = m_eigenSolver.eigenvectors(); // Matrix V
	const VectorXf& rEigenValues = m_eigenSolver.eigenvalues(); // Matrix D is rEigenValues.asDiagonal()

	float diagValue;
	for(int r = 0; r < d; ++r)
	{
		diagValue = max(minEigenVal, rEigenValues(r));
		for(int c = 0; c < d; ++c)
			m_tmpMatrix(r, c) = diagValue * rEigenVectors(c, r);
	}
	// now m_tmpMatrix equals (D^-1) V^T
	o_rClampedMatrix = rEigenVectors * m_tmpMatrix;
}

/// @brief o_rVector and i_rVector might be the same
void DenoisingUnit::multiplyCovMatPatchByVector(VectorXf& o_rVector, const CovMatPatch& i_rCovMatPatch, const VectorXf& i_rVector) const
{
	int blockXIndex = 0, blockYIndex = 1, blockZIndex = 2;
	for(const CovMat3x3& rCovMat3x3 : i_rCovMatPatch.m_blocks)
	{
		o_rVector(blockXIndex) = 
				rCovMat3x3.m_data[g_xx] * i_rVector(blockXIndex) +
				rCovMat3x3.m_data[g_xy] * i_rVector(blockYIndex) +
				rCovMat3x3.m_data[g_xz] * i_rVector(blockZIndex);
		o_rVector(blockYIndex) = 
				rCovMat3x3.m_data[g_xy] * i_rVector(blockXIndex) +
				rCovMat3x3.m_data[g_yy] * i_rVector(blockYIndex) +
				rCovMat3x3.m_data[g_yz] * i_rVector(blockZIndex);
		o_rVector(blockZIndex) = 
				rCovMat3x3.m_data[g_xz] * i_rVector(blockXIndex) +
				rCovMat3x3.m_data[g_yz] * i_rVector(blockYIndex) +
				rCovMat3x3.m_data[g_zz] * i_rVector(blockZIndex);
		blockXIndex += 3;
		blockYIndex += 3;
		blockZIndex += 3;
	}
}

void DenoisingUnit::finalDenoisingMatrixMultiplication(
		std::vector<Eigen::VectorXf>& o_rDenoisedColorPatches,
		const std::vector<Eigen::VectorXf>& i_rNoisyColorPatches,
		const CovMatPatch& i_rNoiseCovMatPatch,
		const Eigen::MatrixXf& i_rInversedCovMat,
		const std::vector<Eigen::VectorXf>& i_rCenteredNoisyColorPatches)
{
	for(int i = 0; i < m_nbOfSimilarPatches; ++i)
	{
		m_tmpVec = i_rInversedCovMat * i_rCenteredNoisyColorPatches[i];
		m_tmpVec *= -1.f;
		multiplyCovMatPatchByVector(o_rDenoisedColorPatches[i], i_rNoiseCovMatPatch, m_tmpVec);
		o_rDenoisedColorPatches[i] += i_rNoisyColorPatches[i];
	}
}

void DenoisingUnit::aggregateOutputPatches()
{
	startChrono(EChronometer::e_aggregateOutputPatches);
	int patchIndex = 0;
	for(const PixelPosition& rSimilarPatchCenter : m_similarPatchesCenters)
	{
		VectorXf& rColorPatch = m_denoisedColorPatches[patchIndex++];
		int patchDataIndex = 0;
		Patch outputPatch(*m_pOutputSummedColorImage, rSimilarPatchCenter, m_patchRadius);
		for(float* pPixelColorData : outputPatch)
		{
			pPixelColorData[0] += rColorPatch(patchDataIndex++);
			pPixelColorData[1] += rColorPatch(patchDataIndex++);
			pPixelColorData[2] += rColorPatch(patchDataIndex++);
		}
		ImageWindow<int> estimatesCountPatch(*m_pEstimatesCountImage, rSimilarPatchCenter, m_patchRadius);
		for(int* pPixelEstimateCount : estimatesCountPatch)
			++(pPixelEstimateCount[0]);
		m_pIsCenterOfAlreadyDenoisedPatchImage->set(rSimilarPatchCenter, 0, true);
	}
	stopChrono(EChronometer::e_aggregateOutputPatches);
}
