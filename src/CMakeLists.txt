# This file is part of the reference implementation for the paper 
#   Bayesian Collaborative Denoising for Monte-Carlo Rendering
#   Malik Boughida and Tamy Boubekeur.
#   Computer Graphics Forum (Proc. EGSR 2017), vol. 36, no. 4, p. 137-153, 2017
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE.txt file.

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

PROJECT(BayesianCollaborativeDenoising)

OPTION(BCD_USE_CUDA "Use CUDA for accelerating computations" ON)

#C++11
IF(NOT WIN32)
	SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11")
ENDIF(NOT WIN32)

# Add CMake folder to the  module path : contains CMake specific files *.cmake
SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake" ${CMAKE_MODULE_PATH} )

# Set the default build to Release.  Note this doesn't do anything for the VS
# default build target which defaults to Debug when you first start it.
IF(NOT CMAKE_BUILD_TYPE)
	SET(CMAKE_BUILD_TYPE "Release" CACHE STRING
		"Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
		FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)


IF(WIN32)
	ADD_DEFINITIONS(-DOPENEXR_DLL)
ENDIF(WIN32)

FIND_PACKAGE(OpenEXR)
IF(NOT OPENEXR_FOUND)
	MESSAGE(WARNING "Could not find OpenEXR library")
ENDIF(NOT OPENEXR_FOUND)

FIND_PACKAGE(OpenMP)
IF(NOT OPENMP_FOUND)
	MESSAGE(WARNING "Could not find OpenMP library")
ENDIF(NOT OPENMP_FOUND)

FIND_PACKAGE(Eigen3)
IF(NOT EIGEN3_FOUND)
	MESSAGE(WARNING "Could not find Eigen library")
ENDIF(NOT EIGEN3_FOUND)

IF(BCD_USE_CUDA)

	SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
	FIND_PACKAGE(CUDA)

	IF(NOT CUDA_FOUND)
		MESSAGE(WARNING "Could not find CUDA")
	ELSE(NOT CUDA_FOUND)
		SET(config_types DEBUG RELEASE MINSIZEREL RELWITHDEBINFO)

		# We have to manually propagate all host flags but C++11...
		FOREACH(config ${config_types})
			SET(CUDA_NVCC_FLAGS_${config} "${CUDA_NVCC_FLAGS_${config}} -Xcompiler ")
	#		SET(CUDA_NVCC_FLAGS_${config} "${CUDA_NVCC_FLAGS_${config}} -g -lineinfo -Xcompiler ") #use this line instead of the previous one for debugging and line info (for nvprof for example)
			IF(OPENMP_FOUND)
				SET(CUDA_NVCC_FLAGS_${config} "${CUDA_NVCC_FLAGS_${config}},\"${OpenMP_CXX_FLAGS}\"")
			ENDIF(OPENMP_FOUND)
			STRING(REPLACE " " ";" FLAGS_LIST ${CMAKE_CXX_FLAGS_${config}})
			FOREACH(flag ${FLAGS_LIST})
				SET(CUDA_NVCC_FLAGS_${config} "${CUDA_NVCC_FLAGS_${config}},\"${flag}\"")
			ENDFOREACH()
		ENDFOREACH()
	ENDIF(NOT CUDA_FOUND)

	# Verbose option: put ON to see the whole build command lines
	IF(CUDA_FOUND)
		SET(CUDA_VERBOSE_BUILD ON)
	ENDIF(CUDA_FOUND)

ENDIF(BCD_USE_CUDA)

# Verbose option: put ON to see the whole build command lines
SET(CMAKE_VERBOSE_MAKEFILE ON)

SET(bcd_folder_name "BayesianCollaborativeDenoiser")
SET(bcd_folder "${CMAKE_SOURCE_DIR}/${bcd_folder_name}")
SET(common_folder "${CMAKE_SOURCE_DIR}/Common")
SET(exr_folder "${common_folder}/exr")

SET(bcd_library "")

ADD_SUBDIRECTORY(${bcd_folder_name})
ADD_SUBDIRECTORY(bcd_cli)
ADD_SUBDIRECTORY(Raw2BcdInputs)

#INCLUDE(CPack)

