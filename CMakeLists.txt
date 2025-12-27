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

add_library(veloxdb_core STATIC
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
install(TARGETS veloxdb DESTINATION .)

enable_testing()

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(unit_tests
    tests/cpp/test_core.cpp
)

target_link_libraries(unit_tests PRIVATE
    veloxdb_core
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(unit_tests)