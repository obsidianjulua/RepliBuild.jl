cmake_minimum_required(VERSION 3.20)
project(JLCS_Dialect LANGUAGES C CXX)

# Find LLVM and MLIR
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

# Add LLVM/MLIR CMake modules
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# Include directories
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# Definitions
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

#===----------------------------------------------------------------------===//
# TableGen: Generate C++ from .td files
#===----------------------------------------------------------------------===//

# Dialect
set(LLVM_TARGET_DEFINITIONS JLCSDialect.td)
mlir_tablegen(JLCSDialect.h.inc -gen-dialect-decls -dialect=jlcs)
mlir_tablegen(JLCSDialect.cpp.inc -gen-dialect-defs -dialect=jlcs)
add_public_tablegen_target(JLCSDialectIncGen)

# Operations
set(LLVM_TARGET_DEFINITIONS JLCSOps.td)
mlir_tablegen(JLCSOps.h.inc -gen-op-decls)
mlir_tablegen(JLCSOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(JLCSOpsIncGen)

# Types
set(LLVM_TARGET_DEFINITIONS JLCSTypes.td)
mlir_tablegen(JLCSTypes.h.inc -gen-typedef-decls)
mlir_tablegen(JLCSTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(JLCSTypesIncGen)

#===----------------------------------------------------------------------===//
# Build the dialect library
#===----------------------------------------------------------------------===//

add_llvm_library(JLCS SHARED
  JLCSDialect.cpp
  JLCSOps.cpp
  JLCSTypes.cpp
  JLCSCHelpers.cpp

  PARTIAL_SOURCES_INTENDED

  DEPENDS
  JLCSDialectIncGen
  JLCSOpsIncGen
  JLCSTypesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRInferTypeOpInterface
)

# Install the library
install(TARGETS JLCS
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

message(STATUS "JLCS Dialect build configured successfully")
