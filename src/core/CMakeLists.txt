# --------------------------------------------------------------------------------------------------
# Source code for the paper no.36 "Bayesian Collaborative Denoising
# for Monte-Carlo Rendering", submitted to the ACM SIGGRAPH 2016
# conference technical paper program.
#
# For review only. Please, do not distribute outside the review context.
#
# Final open source licence and comments to be inserted if published.
#
# Copyright(C) 2014-2016
# The Authors
#
# All rights reserved.
# ---------------------------------------------------------------------------------------------------


SET(target_name ${lib_bcd_core})


# First we check whether optional and required libraries have been found

SET(required_library_is_missing FALSE)
SET(missing_required_libraries "")

SET(optional_library_is_missing FALSE)
SET(missing_optional_libraries "")


if (NOT OPENMP_FOUND)
	SET(optional_library_is_missing TRUE)
	SET(missing_optional_libraries ${missing_optional_libraries} OpenMP)
endif(NOT OPENMP_FOUND) 

if (NOT CUDA_FOUND)
	SET(optional_library_is_missing TRUE)
	SET(missing_optional_libraries ${missing_optional_libraries} "CUDA")
endif(NOT CUDA_FOUND)

if(optional_library_is_missing)
	message(WARNING "missing optional libraries ${missing_optional_libraries} for target ${target_name}")
endif(optional_library_is_missing)

if(required_library_is_missing)
	message(WARNING "disabling target ${target_name} which requires libraries ${missing_required_libraries}")
	return()
endif(required_library_is_missing)



SET(bcd_core_include_folder "${bcd_include_folder}/core")


SET(src_files
	Chronometer.cpp
	CovarianceMatrix.cpp
	Denoiser.cpp
	DenoisingUnit.cpp
	MultiscaleDenoiser.cpp
	SamplesAccumulator.cpp
	SpikeRemovalFilter.cpp
	Utils.cpp
)
SET(header_files
	${bcd_core_include_folder}/Chronometer.h
	${bcd_core_include_folder}/CovarianceMatrix.h
	${bcd_core_include_folder}/DeepImage.h
	${bcd_core_include_folder}/Denoiser.h
	${bcd_core_include_folder}/DenoisingUnit.h
	${bcd_core_include_folder}/IDenoiser.h
	${bcd_core_include_folder}/MultiscaleDenoiser.h
	${bcd_core_include_folder}/SamplesAccumulator.h
	${bcd_core_include_folder}/SpikeRemovalFilter.h
	${bcd_core_include_folder}/Utils.h
)
SET(header_implementation_files
	${bcd_core_include_folder}/DeepImage.hpp
)


IF(CUDA_FOUND)
	MESSAGE(STATUS "Using cuda's nvcc for generating ${target_name}")
	list(APPEND CUDA_NVCC_FLAGS "-arch=sm_20")
	SET(src_files ${src_files} CudaHistogramDistance.cu)
	SET(header_files
		${header_files}
		${bcd_core_include_folder}/CudaHistogramDistance.h
		${bcd_core_include_folder}/CudaUtils.h
	)
	CUDA_INCLUDE_DIRECTORIES(${bcd_core_include_folder}) # sadly, the TARGET_INCLUDE_DIRECTORIES a few lines below does not work with CUDA, see https://cmake.org/Bug/view.php?id=14201
	CUDA_ADD_LIBRARY(${target_name} ${src_files} ${header_files} ${header_implementation_files} STATIC)
	TARGET_COMPILE_DEFINITIONS(${target_name} PRIVATE FOUND_CUDA)
ELSE(CUDA_FOUND)
	ADD_LIBRARY(${target_name} STATIC ${src_files} ${header_files} ${header_implementation_files})
ENDIF(CUDA_FOUND)


TARGET_INCLUDE_DIRECTORIES(${target_name} PUBLIC ${bcd_core_include_folder})

SET_TARGET_PROPERTIES(${target_name} PROPERTIES CXX_STANDARD 11)
if (OPENMP_FOUND)
TARGET_LINK_LIBRARIES(${target_name} PUBLIC OpenMP)
endif (OPENMP_FOUND)
TARGET_LINK_LIBRARIES(${target_name} PRIVATE eigen)

INSTALL(TARGETS ${target_name} ARCHIVE DESTINATION lib)


