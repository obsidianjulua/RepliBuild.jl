# Fixture for test/test_cross_library_types_live.jl: a CMake project of three
# libraries whose APIs share types, built and wrapped in graph mode.
#
#   xc   C shared library: a POD `Pt` and an enum `Mode`, crossed by value
#   xa   C++ shared library linking xc: a polymorphic `Shape` with a factory and
#        a deleter function, a `Leaf` with a virtual destructor and a factory
#        (nothing in xa derives from it; xb does), a header-only POD `Plain`,
#        an enum `Color`, and a function taking xc's `Pt`
#   xb   C++ shared library linking xa; the TOML selects it alone, so xa and xc
#        come with it. Derives from `Shape` and `Leaf`, passes `Plain`, `Color`,
#        `Shape*`, `Pt` and `Mode` through its own API, embeds `Plain`.
cmake_minimum_required(VERSION 3.20)
project(xlib_fixture C CXX)

add_library(xc SHARED xc/xc.c xc/xc.h)
target_include_directories(xc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/xc)

add_library(xa SHARED xa/xa.cpp xa/xa.h)
target_include_directories(xa PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/xa)
target_compile_features(xa PRIVATE cxx_std_17)
target_link_libraries(xa PUBLIC xc)

add_library(xb SHARED xb/xb.cpp xb/xb.h)
target_include_directories(xb PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/xb)
target_compile_features(xb PRIVATE cxx_std_17)
target_link_libraries(xb PRIVATE xa)
