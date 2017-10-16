
# fake FindZlib to deceive OpenEXR


#set(ZLIB_LIBRARIES "unfoundable_zlib")
set(ZLIB_LIBRARIES "zlibstatic")

set(ZLIB_INCLUDE_DIR "")

set(ZLIB_FOUND TRUE)
