# Fixture for test/test_graph_generators_live.jl. Generated sources whose steps
# run cmake only, never a compiler, in a library that links another library of
# the project; and, in a library of its own, a generated header whose step runs
# a tool the project compiles (rootcling's shape).
cmake_minimum_required(VERSION 3.20)
project(gg_fixture C)

add_library(gg_base SHARED base.c)

# One command writes two files; a second step reads the first.
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/gen/gg_gen.h ${CMAKE_CURRENT_BINARY_DIR}/gen/gg_gen.c
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/gen
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/gg_gen.h.in ${CMAKE_CURRENT_BINARY_DIR}/gen/gg_gen.h
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/gg_gen.c.in ${CMAKE_CURRENT_BINARY_DIR}/gen/gg_gen.c
  DEPENDS gg_gen.h.in gg_gen.c.in
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/gen/gg_table.h
  COMMAND ${CMAKE_COMMAND} -DIN=${CMAKE_CURRENT_BINARY_DIR}/gen/gg_gen.h
          -DOUT=${CMAKE_CURRENT_BINARY_DIR}/gen/gg_table.h
          -P ${CMAKE_CURRENT_SOURCE_DIR}/write_table.cmake
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/gen/gg_gen.h write_table.cmake
  VERBATIM)

# cmake gives every custom command of gg an order-only input on libgg_base.
add_library(gg SHARED main.c ${CMAKE_CURRENT_BINARY_DIR}/gen/gg_gen.c
            ${CMAKE_CURRENT_BINARY_DIR}/gen/gg_gen.h ${CMAKE_CURRENT_BINARY_DIR}/gen/gg_table.h)
target_include_directories(gg PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/gen)
target_link_libraries(gg PRIVATE gg_base)

# In a library of its own: cmake orders every custom command of a target after
# every target that target depends on, gg_tool included.
add_executable(gg_tool tool.c)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/tool_out.h
  COMMAND gg_tool ${CMAKE_CURRENT_BINARY_DIR}/tool_out.h
  VERBATIM)
add_library(gg_tooluser SHARED tooluser.c ${CMAKE_CURRENT_BINARY_DIR}/tool_out.h)

# An OBJECT library consumed as $<TARGET_OBJECTS:...> SOURCES of a shared
# library, not through target_link_libraries (libwebp's and curl's pattern).
# The File API lists gg_parts' object files as generated sources of gg_whole;
# they are a compile's output, so the generator step must not request them.
add_library(gg_parts OBJECT parts.c)
add_library(gg_whole SHARED whole.c $<TARGET_OBJECTS:gg_parts>)

# A VERSIONed shared dependency: gg_whole's link line names the real file
# libgg_ver.so.1.2.3, while the File API's artifact is libgg_ver.so.
add_library(gg_ver SHARED ver.c)
set_target_properties(gg_ver PROPERTIES VERSION 1.2.3 SOVERSION 1)
target_link_libraries(gg_whole PRIVATE gg_ver)
