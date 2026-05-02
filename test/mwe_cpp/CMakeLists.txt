cmake_minimum_required(VERSION 3.20)
project(mlir_jit_mwe C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "LLVM version: ${LLVM_PACKAGE_VERSION}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

set(MWE_LIBS
    ${dialect_libs}
    ${conversion_libs}
    ${translation_libs}
    MLIRExecutionEngine
    MLIRParser
    MLIRPass
    MLIRIR
    MLIRSupport
)

add_executable(mwe mwe.cpp)
target_link_libraries(mwe PRIVATE ${MWE_LIBS})

add_executable(mwe_dual_llvm mwe_dual_llvm.cpp)
target_link_libraries(mwe_dual_llvm PRIVATE ${MWE_LIBS} dl)
