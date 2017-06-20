// This file is part of the reference implementation for the paper 
//   Bayesian Collaborative Denoising for Monte-Carlo Rendering
//   Malik Boughida and Tamy Boubekeur.
//   Computer Graphics Forum (Proc. EGSR 2017), vol. 36, no. 4, p. 137-153, 2017
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE.txt file.

#ifndef CHRONOMETER_H
#define CHRONOMETER_H

#include <chrono>

#include <string>

#include <iostream>


/// @brief Class to implement a cross-platform chronometer
///
/// Warning : uses C++11
class Chronometer
{

public:
	Chronometer();

	void reset();
	void start();
	void stop();
	float getElapsedTime(); ///< Returns elapsed time in seconds
	static std::string getStringFromTime(float i_timeInSeconds);
	void printElapsedTime(std::ostream& o_stream = std::cout);

private:
	bool m_isRunning;
	float m_elapsedTime;
	std::chrono::high_resolution_clock::time_point m_startTime;
};



#endif // CHRONOMETER_H
