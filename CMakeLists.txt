cmake_minimum_required(VERSION 3.15)
project(VeloxDB VERSION 0.1.0 LANGUAGES CXX)

# setup cpp standard and position independent code (for cross-platform compatibility)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# download pybind11
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# core engine in cpp
add_library(veloxdb_core SHARED
    src/index.cpp
)
target_include_directories(veloxdb_core PUBLIC include)

# python module interface with core cpp module.
pybind11_add_module(veloxdb 
    bindings/python_bindings.cpp
)

target_link_libraries(veloxdb PRIVATE veloxdb_core)
message(STATUS "Build setup for VeloxDB complete.")