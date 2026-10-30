# A tiny project for test/tools/gen_cmake_graph_fixtures.jl. It exercises the
# File API shapes RepliBuild.CMakeGraph reads: a PUBLIC include dir inherited
# through a link edge, a target-private define, an OBJECT library linked into a
# shared one, a source a custom command writes during the build, and a link
# library that is not a target.
cmake_minimum_required(VERSION 3.20)
project(graphfix C CXX)

# A source written during the build by a custom command attached to `core`.
set(GEN_SRC ${CMAKE_CURRENT_BINARY_DIR}/gen/table.cpp)
add_custom_command(
  OUTPUT ${GEN_SRC}
  COMMAND ${CMAKE_COMMAND} -DOUT=${GEN_SRC} -P ${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_table.cmake
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_table.cmake
  COMMENT "Generating table.cpp")

add_library(core SHARED core/core.cpp ${GEN_SRC})
target_include_directories(core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/core/include)
target_compile_definitions(core PRIVATE CORE_BUILDING=1)
target_compile_features(core PUBLIC cxx_std_17)

# A header written during the build by a custom command of its own target,
# which `ext` depends on by name.
set(GEN_HDR ${CMAKE_CURRENT_BINARY_DIR}/gen/ext_version.h)
add_custom_command(
  OUTPUT ${GEN_HDR}
  COMMAND ${CMAKE_COMMAND} -DOUT=${GEN_HDR} -P ${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_version.cmake
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_version.cmake)
add_custom_target(ext_version DEPENDS ${GEN_HDR})

add_library(helpers OBJECT ext/helpers.c)
set_target_properties(helpers PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(ext SHARED ext/ext.cpp)
target_compile_definitions(ext PRIVATE EXT_LEVEL=2)
target_include_directories(ext PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/gen)
target_link_libraries(ext PRIVATE core helpers graphfix_syslib)
add_dependencies(ext ext_version)
