if(DEFINED TRIBITS_PACKAGE)
  include(pkg_tribits.cmake)
  return()
endif()

# Package options
option(PCU_COMPRESS "Enable SMB compression using libbzip2 [ON|OFF]" OFF)
message(STATUS "PCU_COMPRESS: " ${PCU_COMPRESS})

# Package sources
set(SOURCES
  pcu.c
  pcu_aa.c
  pcu_coll.c
  pcu_io.c
  pcu_buffer.c
  pcu_mem.c
  pcu_mpi.c
  pcu_msg.c
  pcu_order.c
  pcu_pmpi.c
  pcu_util.c
  noto/noto_malloc.c
  reel/reel.c
)

# Package headers
set(HEADERS
  PCU.h
  pcu_io.h
  pcu_util.h
  reel/reel.h
)

# Add the pcu library
add_library(pcu ${SOURCES})
# this compiler definition is needed to silence warnings caused by the openmpi CXX
# bindings that are depreciated. This is needed on gcc 8 forward.
# see: https://github.com/open-mpi/ompi/issues/5157
target_compile_definitions(pcu PUBLIC OMPI_SKIP_MPICXX)

# Include directories
target_include_directories(pcu
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/reel>
    $<INSTALL_INTERFACE:include>
    PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/noto>
    )

# Check for and enable compression support
if(PCU_COMPRESS)
  xsdk_add_tpl(BZIP2)
  find_package(BZip2 REQUIRED)
  target_include_directories(pcu PRIVATE ${BZIP2_INCLUDE_DIR})
  target_link_libraries(pcu PRIVATE ${BZIP2_LIBRARIES})
  target_compile_definitions(pcu PRIVATE "-DPCU_BZIP")
endif()

scorec_export_library(pcu)

bob_end_subdir()
