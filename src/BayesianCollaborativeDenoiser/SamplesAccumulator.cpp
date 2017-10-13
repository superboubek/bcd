// This file is part of the reference implementation for the paper 
//   Bayesian Collaborative Denoising for Monte-Carlo Rendering
//   Malik Boughida and Tamy Boubekeur.
//   Computer Graphics Forum (Proc. EGSR 2017), vol. 36, no. 4, p. 137-153, 2017
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE.txt file.

#include "SamplesAccumulator.h"

#include <cassert>

using namespace std;

namespace bcd
{

	SamplesStatisticsImages::SamplesStatisticsImages(int i_width, int i_height, int i_nbOfBins) :
			m_nbOfSamplesImage(i_width, i_height, 1),
			m_meanImage(i_width, i_height, 3),
			m_covarImage(i_width, i_height, 6),
			m_histoImage(i_width, i_height, 3 * i_nbOfBins)
	{
	}

	SamplesAccumulator::SamplesAccumulator(
			int i_width, int i_height,
			const HistogramParameters& i_rHistogramParameters) :
			m_width(i_width), m_height(i_height),
			m_histogramParameters(i_rHistogramParameters),
			m_samplesStatisticsImages(i_width, i_height, i_rHistogramParameters.m_nbOfBins),
			m_isValid(true)
	{
	}

	void SamplesAccumulator::addSample(
			int i_line, int i_column,
			float i_sampleR, float i_sampleG, float i_sampleB,
			float i_weight)
	{
		assert(m_isValid);

		PixelPosition p(i_line, i_column);
		DeepImage<float>& rSum = m_samplesStatisticsImages.m_meanImage;
		rSum.get(i_line, i_column, 0) += i_sampleR;
		rSum.get(i_line, i_column, 1) += i_sampleG;
		rSum.get(i_line, i_column, 2) += i_sampleB;

		// TODO


	}


	SamplesStatisticsImages SamplesAccumulator::getSamplesStatistics() const
	{
		SamplesStatisticsImages stats(m_width, m_height, m_histogramParameters.m_nbOfBins);

		// TODO: fill stats

		return stats;
	}

	SamplesStatisticsImages SamplesAccumulator::extractSamplesStatistics()
	{
		// TODO: modify m_samplesStatisticsImages

		return move(m_samplesStatisticsImages);
	}

	void SamplesAccumulatorThreadSafe::addSampleThreadSafely(
			int i_line, int i_column,
			float i_sampleR, float i_sampleG, float i_sampleB,
			float i_weight)
	{
		// lock

		addSample(i_line, i_column, i_sampleR, i_sampleG, i_sampleB, i_weight);

	}

} // namespace bcd
