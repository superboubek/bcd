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


SET(target_name ${lib_bcd_io})


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



SET(exr_folder "exr")
SET(bcd_io_include_folder "${bcd_include_folder}/io")

SET(src_files
	ImageIO.cpp
	ParametersIO.cpp
	${exr_folder}/io_exr.cpp
)
SET(header_files
	${bcd_io_include_folder}/ImageIO.h
	${bcd_io_include_folder}/ParametersIO.h
	${exr_folder}/io_exr.h
)
SET(header_implementation_files
)

ADD_LIBRARY(${target_name} STATIC ${src_files} ${header_files} ${header_implementation_files})


TARGET_INCLUDE_DIRECTORIES(${target_name} PUBLIC "${bcd_include_folder}/io")
TARGET_INCLUDE_DIRECTORIES(${target_name} PRIVATE ${exr_folder})


SET_TARGET_PROPERTIES(${target_name} PROPERTIES CXX_STANDARD 11)
TARGET_LINK_LIBRARIES(${target_name} PUBLIC ${lib_bcd_core})
TARGET_LINK_LIBRARIES(${target_name} PRIVATE OpenExr)
TARGET_LINK_LIBRARIES(${target_name} PRIVATE Json)

INSTALL(TARGETS ${target_name} ARCHIVE DESTINATION lib)

