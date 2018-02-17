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


SET(target_name ${exe_bcd_raw_converter})

SET(src_files
	main.cpp
)
SET(header_files
)
SET(header_implementation_files
)

ADD_EXECUTABLE(${target_name} ${src_files} ${header_files} ${header_implementation_files})

TARGET_LINK_LIBRARIES(${target_name} PRIVATE ${lib_bcd_core})
TARGET_LINK_LIBRARIES(${target_name} PRIVATE ${lib_bcd_io})

set_target_properties(${target_name} PROPERTIES CXX_STANDARD 11)

INSTALL(TARGETS ${target_name} RUNTIME DESTINATION bin)

