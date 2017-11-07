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


SET(target_name raw2bcd)


# First we check whether optional and required libraries have been found

SET(required_library_is_missing FALSE)
SET(missing_required_libraries "")

SET(optional_library_is_missing FALSE)
SET(missing_optional_libraries "")


if(optional_library_is_missing)
	message(WARNING "missing optional libraries ${missing_optional_libraries} for target ${target_name}")
endif(optional_library_is_missing)

if(required_library_is_missing)
	message(WARNING "disabling target ${target_name} which requires libraries ${missing_required_libraries}")
	return()
endif(required_library_is_missing)



# Then we add include directories, libraries and definitions related to the dependencies



# Finally we compile the target

SET(src_files
	main.cpp
)
SET(header_files
)
SET(header_implementation_files
)

ADD_EXECUTABLE(${target_name} ${src_files} ${header_files} ${header_implementation_files})

TARGET_LINK_LIBRARIES(${target_name} PRIVATE ${bcd_library})
TARGET_LINK_LIBRARIES(${target_name} PRIVATE OpenExr)


set_target_properties(${target_name} PROPERTIES CXX_STANDARD 11)

INSTALL(TARGETS ${target_name} RUNTIME DESTINATION bin)

