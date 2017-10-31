# This file is part of the reference implementation for the paper 
#   Bayesian Collaborative Denoising for Monte-Carlo Rendering
#   Malik Boughida and Tamy Boubekeur.
#   Computer Graphics Forum (Proc. EGSR 2017), vol. 36, no. 4, p. 137-153, 2017
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE.txt file.

SET(target_name ${exe_bcd_cli})

SET(src_files
	main.cpp
)
SET(header_files
)
SET(header_implementation_files
)

ADD_EXECUTABLE(${target_name} ${src_files} ${header_files} ${header_implementation_files})

SET_TARGET_PROPERTIES(${target_name} PROPERTIES CXX_STANDARD 11)

TARGET_LINK_LIBRARIES(${target_name} PRIVATE ${lib_bcd_core})
TARGET_LINK_LIBRARIES(${target_name} PRIVATE ${lib_bcd_io})

INSTALL(TARGETS ${target_name} RUNTIME DESTINATION bin)
