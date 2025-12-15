cmake_minimum_required(VERSION 3.15)
project(VeloxDB VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

add_library(veloxdb_core SHARED
    src/index.cpp
    src/metrics.cpp
)

if(MSVC)
    target_compile_options(veloxdb_core PRIVATE /arch:AVX2 /O2)
else()
    #-O3 for max optimization, -mavx2 for SIMD, -mfma for Fused Multiply-Add
    target_compile_options(veloxdb_core PRIVATE -O3 -mavx2 -mfma)
endif()

target_include_directories(veloxdb_core PUBLIC include)

pybind11_add_module(veloxdb 
    bindings/python_bindings.cpp
)

target_link_libraries(veloxdb PRIVATE veloxdb_core)

message(STATUS "Build setup for VeloxDB complete (AVX2 Enabled).")