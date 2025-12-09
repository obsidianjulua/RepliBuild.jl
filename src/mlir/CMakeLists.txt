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
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/IR)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# Definitions
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

#===----------------------------------------------------------------------===//
# TableGen: Generate C++ from .td files
#===----------------------------------------------------------------------===//

set(LLVM_TARGET_DEFINITIONS JLCS.td)

mlir_tablegen(JLCSDialect.h.inc -gen-dialect-decls)
mlir_tablegen(JLCSDialect.cpp.inc -gen-dialect-defs)
mlir_tablegen(JLCSOps.h.inc -gen-op-decls)
mlir_tablegen(JLCSOps.cpp.inc -gen-op-defs)
mlir_tablegen(JLCSTypes.h.inc -gen-typedef-decls)
mlir_tablegen(JLCSTypes.cpp.inc -gen-typedef-defs)
mlir_tablegen(JLCSInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(JLCSInterfaces.cpp.inc -gen-type-interface-defs)

add_public_tablegen_target(JLCSIncGen)


#===----------------------------------------------------------------------===//
# Build the dialect library
#===----------------------------------------------------------------------===//

add_llvm_library(JLCS SHARED
  impl/JLCSDialect.cpp
  impl/JLCSOps.cpp
  impl/JLCSTypes.cpp
  impl/JLCSCHelpers.cpp
  impl/JLCSPasses.cpp

  DEPENDS
  JLCSIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRInferTypeOpInterface
  MLIRFuncDialect
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRControlFlowDialect
  MLIRLLVMDialect
  MLIRPass
  MLIRTransforms
  MLIRExecutionEngine
  MLIRCAPIIR
  MLIRCAPIExecutionEngine
  MLIRTargetLLVMIRExport
  MLIRToLLVMIRTranslation
)

# Install the library
install(TARGETS JLCS
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

message(STATUS "JLCS Dialect build configured successfully")
