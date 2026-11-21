include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_triton_library(TritonKunlunXPUC
  lib/XPUC/Emitter.cpp

  LINK_LIBS PUBLIC
  TritonIR
  MLIRIR
)

if(TRITON_BUILD_PYTHON_MODULE)
  add_triton_plugin(TritonKunlun triton_kunlun.cc LINK_LIBS TritonKunlunXPUC)
endif()
