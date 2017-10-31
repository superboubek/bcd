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


SET(target_name ${exe_bcd_gui})

SET(src_files
	main.cpp
	GuiWindow.cpp
)
SET(header_files
	GuiWindow.h
)
SET(header_implementation_files
)

ADD_EXECUTABLE(${target_name} ${src_files} ${header_files} ${header_implementation_files})

SET_TARGET_PROPERTIES(${target_name} PROPERTIES CXX_STANDARD 11)

TARGET_LINK_LIBRARIES(${target_name} PRIVATE ${lib_bcd_core})
TARGET_LINK_LIBRARIES(${target_name} PRIVATE ${lib_bcd_io})
TARGET_LINK_LIBRARIES(${target_name} PRIVATE NanoguiTarget)

INSTALL(TARGETS ${target_name} RUNTIME DESTINATION bin)
