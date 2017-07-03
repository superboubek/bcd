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


SET(target_name bcd_nanogui)


# First we check whether optional and required libraries have been found

SET(required_library_is_missing FALSE)
SET(missing_required_libraries "")

SET(optional_library_is_missing FALSE)
SET(missing_optional_libraries "")


if (NOT OPENMP_FOUND)
	SET(optional_library_is_missing TRUE)
	SET(missing_optional_libraries ${missing_optional_libraries} OpenMP)
endif(NOT OPENMP_FOUND) 

if (NOT OPENEXR_FOUND)
	SET(required_library_is_missing TRUE)
	SET(missing_required_libraries ${missing_required_libraries} OpenEXR)
endif(NOT OPENEXR_FOUND)

if (NOT EIGEN3_FOUND)
	SET(required_library_is_missing TRUE)
	SET(missing_required_libraries ${missing_required_libraries} Eigen)
endif(NOT EIGEN3_FOUND)

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



# Then we add include directories, libraries and definitions related to the dependencies

SET(libs ${bcd_library})

if (OPENMP_FOUND)
	SET (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else(OPENMP_FOUND)
	SET (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unknown-pragmas") # to avoid warnings ; drawback : that could hide warnings unrelated to OpenMP...
	SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas") # to avoid warnings ; drawback : that could hide warnings unrelated to OpenMP...
endif(OPENMP_FOUND)

if(OPENEXR_FOUND)
	INCLUDE_DIRECTORIES(${OPENEXR_INCLUDE_PATHS})
	SET(libs ${libs} ${OPENEXR_LIBRARIES})
endif(OPENEXR_FOUND)

IF(EIGEN3_FOUND)
	INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})
ENDIF(EIGEN3_FOUND)

# Finally we compile the target

INCLUDE_DIRECTORIES(${common_folder})
INCLUDE_DIRECTORIES(${exr_folder})
INCLUDE_DIRECTORIES(${bcd_folder})

INCLUDE_DIRECTORIES(../ext/nanogui/include)
INCLUDE_DIRECTORIES(../ext/nanogui/ext/eigen)
INCLUDE_DIRECTORIES(../ext/nanogui/ext/glfw/include)
INCLUDE_DIRECTORIES(../ext/nanogui/ext/nanovg/src)

SET(libs ${libs} nanogui)


SET(src_files
	main.cpp
	GuiWindow.cpp
)
SET(header_files
	GuiWindow.h
)
SET(header_implementation_files
)

IF(CUDA_FOUND)
	MESSAGE(STATUS "Using cuda's nvcc for generating ${target_name}")
	list(APPEND CUDA_NVCC_FLAGS "-arch=sm_20")
	ADD_DEFINITIONS(-DFOUND_CUDA)
	CUDA_ADD_EXECUTABLE(${target_name} ${src_files} ${header_files} ${header_implementation_files})
ELSE(CUDA_FOUND)
	ADD_EXECUTABLE(${target_name} ${src_files} ${header_files} ${header_implementation_files})
ENDIF(CUDA_FOUND)

#MESSAGE(AUTHOR_WARNING "for bcd_cli, libs = '${libs}'")

TARGET_LINK_LIBRARIES(${target_name} ${libs})

INSTALL(TARGETS ${target_name} RUNTIME DESTINATION bin)
