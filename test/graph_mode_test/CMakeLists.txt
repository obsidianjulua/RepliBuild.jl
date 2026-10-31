# Fixture for test/test_graph_mode_live.jl: a CMake project of several
# libraries that RepliBuild builds and wraps in graph mode.
#
#   cbase    C shared library; a counter whose one copy every library shares
#   Core     C shared library linking cbase, named like Julia's Core (its module
#            is renamed RCore), compiling against a header a build step writes
#   cxxlib   C++ shared library linking Core; the TOML selects it alone, and
#            the shared libraries it links come with it
#   extra    a shared library nothing selected links: not built
#   gm_tool  an executable: never built
cmake_minimum_required(VERSION 3.20)
project(gm_fixture C CXX)

# A build-time step writes gm_version.h; the File API marks it generated, and
# GraphGenerators runs the step before anything compiles.
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/gen/gm_version.h
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/gen
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/gm_version.h.in
          ${CMAKE_CURRENT_BINARY_DIR}/gen/gm_version.h
  DEPENDS gm_version.h.in
  VERBATIM)

add_library(cbase SHARED cbase/cbase.c cbase/cbase.h)
target_include_directories(cbase PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cbase)

add_library(Core SHARED core/core.c core/core.h ${CMAKE_CURRENT_BINARY_DIR}/gen/gm_version.h)
target_include_directories(Core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/core
                                PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/gen)
target_compile_definitions(Core PRIVATE GM_CORE_BUILD=1)
target_link_libraries(Core PUBLIC cbase)

add_library(cxxlib SHARED cxxlib/cxxlib.cpp cxxlib/cxxlib.h)
target_include_directories(cxxlib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cxxlib)
target_compile_features(cxxlib PRIVATE cxx_std_17)
target_link_libraries(cxxlib PRIVATE Core)

add_library(extra SHARED extra/extra.c)

add_executable(gm_tool tool/tool.c)
target_link_libraries(gm_tool PRIVATE cbase)
