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

# C++ exceptions are enabled per-target below (after LLVM's -fno-exceptions)

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

add_public_tablegen_target(JLCSIncGen)


#===----------------------------------------------------------------------===//
# Build the dialect library
#===----------------------------------------------------------------------===//

add_llvm_library(JLCS SHARED
  impl/JLCSDialect.cpp
  impl/JLCSOps.cpp
  impl/JLCSTypes.cpp
  impl/JLCSPasses.cpp
  impl/JLCSCAPIWrappers.cpp
  impl/JlcsJIT.cpp
  impl/AOTCompiler.cpp

  DEPENDS
  JLCSIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRCAPIIR
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
  MLIRCAPIExecutionEngine
  MLIRToLLVMIRTranslationRegistration
  MLIRArithToLLVM
  MLIRReconcileUnrealizedCasts
)

# Enable C++ exceptions for our code (must come after LLVM's -fno-exceptions)
target_compile_options(JLCS PRIVATE -fexceptions)

# === JIT & Execution Engine Linking ===
# We must link these explicitly. -Wl,--whole-archive is strictly required on Linux
# to force the linker to include the JIT registration symbols (Target/ExecutionEngine).
target_link_libraries(JLCS PRIVATE
  -Wl,--whole-archive
  MLIRExecutionEngine
  MLIRTargetLLVMIRExport      # Required for converting MLIR -> LLVM IR
  MLIRLLVMToLLVMIRTranslation # Required for lowering LLVM Dialect -> LLVM IR
  MLIRBuiltinToLLVMIRTranslation
  MLIRFuncToLLVM
  -Wl,--no-whole-archive
  
  # Standard link dependencies
  MLIRParser
  MLIROptLib
)

#===----------------------------------------------------------------------===//
# Test only: LLVM's ABI lowering library as a SysV oracle
#===----------------------------------------------------------------------===//
#
# test/test_sysv_abi_oracle.jl diffs classifySysVStruct + buildCallShape against
# LLVM's own x86-64 SysV classifier (llvm/ABI, impl/JLCSAbiOracle.cpp). That
# classifier ships from LLVM 23: 22 has the type system alone and 21 has no ABI
# library. So it is optional. Found, the oracle is compiled in and libJLCS
# exports jlcs_abi_oracle_sysv; not found, libJLCS is built exactly as before and
# the test skips.
#
# Decided by compiling the calls the oracle makes rather than by version: the
# library is new and its API may still move.
#
# The component can be in two places. The static archive (libLLVMABI.a, what
# apt.llvm.org and a default LLVM install ship) is linked where it exists. Its
# own dependencies (Core, Support) come from the LLVM libJLCS already links.
# With LLVM as one shared library that is the dylib, so only the archive FILE
# is linked: the LLVMABI target would add static copies of LLVMSupport, and two
# copies of Support in one process register every cl::opt twice. With no
# archive, a distro that ships LLVM as one shared library may have built the
# component into it (Arch's llvm 23.1 does; that any did is not something to
# assume), so the probe is then LINKED against the dylib libJLCS links anyway:
# resolved, the oracle is on and nothing new is linked.
#
# Nothing here may outlive the LLVM it was found in. build.sh reconfigures the
# same build directory after an LLVM upgrade, and a cached archive path would
# link one LLVM's library into a build against another. So the searches run
# on every configure, and each check's result is keyed on the version.
set(JLCS_LLVM_ABI_ORACLE OFF)
set(JLCS_LLVM_ABI_SOURCE "")
unset(JLCS_LLVM_ABI_INCLUDE_DIR CACHE)
unset(JLCS_LLVM_ABI_LIBRARY CACHE)
find_path(JLCS_LLVM_ABI_INCLUDE_DIR llvm/ABI/TargetInfo.h
  PATHS ${LLVM_INCLUDE_DIRS} NO_DEFAULT_PATH)
find_library(JLCS_LLVM_ABI_LIBRARY NAMES LLVMABI
  PATHS ${LLVM_LIBRARY_DIR} ${LLVM_LIBRARY_DIRS} NO_DEFAULT_PATH)
# The calls the oracle makes: one copy for both checks.
set(JLCS_LLVM_ABI_PROBE [=[
    #include "llvm/ABI/FunctionInfo.h"
    #include "llvm/ABI/TargetInfo.h"
    #include "llvm/ABI/Types.h"
    #include "llvm/Support/JSON.h"
    int jlcs_abi_probe() {
      llvm::BumpPtrAllocator A;
      llvm::abi::TypeBuilder TB(A);
      auto TI = llvm::abi::createX86_64TargetInfo(
          TB, llvm::abi::X86AVXABILevel::None, true, llvm::abi::ABICompatInfo());
      const llvm::abi::Type *I = TB.getIntegerType(32, llvm::Align(4), true);
      llvm::abi::FieldInfo F(I, 0, true, 3);
      const llvm::abi::Type *R = TB.getUnionType({F}, llvm::TypeSize::getFixed(32),
          llvm::Align(4), llvm::abi::StructPacking::Default,
          llvm::abi::RecordFlags::CanPassInRegisters);
      auto FI = llvm::abi::FunctionInfo::create(llvm::CallingConv::C, R, {I});
      TI->computeInfo(*FI);
      return FI->getReturnInfo().isDirect() && FI->arguments()[0].Info.isExtend();
    }
]=])
string(MAKE_C_IDENTIFIER "JLCS_LLVM_ABI_X86_64_COMPILES_${LLVM_PACKAGE_VERSION}"
  JLCS_LLVM_ABI_CHECK)
string(MAKE_C_IDENTIFIER "JLCS_LLVM_ABI_X86_64_IN_DYLIB_${LLVM_PACKAGE_VERSION}"
  JLCS_LLVM_ABI_DYLIB_CHECK)
if(JLCS_LLVM_ABI_INCLUDE_DIR)
  include(CheckCXXSourceCompiles)
  set(CMAKE_REQUIRED_INCLUDES ${LLVM_INCLUDE_DIRS})
  set(CMAKE_REQUIRED_DEFINITIONS ${LLVM_DEFINITIONS_LIST})
  if(JLCS_LLVM_ABI_LIBRARY)
    set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
    check_cxx_source_compiles("${JLCS_LLVM_ABI_PROBE}" ${JLCS_LLVM_ABI_CHECK})
    unset(CMAKE_TRY_COMPILE_TARGET_TYPE)
    if(${JLCS_LLVM_ABI_CHECK})
      set(JLCS_LLVM_ABI_ORACLE ON)
      set(JLCS_LLVM_ABI_SOURCE "${JLCS_LLVM_ABI_LIBRARY}")
    endif()
  elseif(LLVM_LINK_LLVM_DYLIB AND TARGET LLVM)
    set(CMAKE_REQUIRED_LIBRARIES LLVM)
    check_cxx_source_compiles(
      "${JLCS_LLVM_ABI_PROBE}\n    int main() { return jlcs_abi_probe(); }\n"
      ${JLCS_LLVM_ABI_DYLIB_CHECK})
    unset(CMAKE_REQUIRED_LIBRARIES)
    if(${JLCS_LLVM_ABI_DYLIB_CHECK})
      set(JLCS_LLVM_ABI_ORACLE ON)
      set(JLCS_LLVM_ABI_SOURCE "the LLVM shared library, which carries llvm/ABI")
    endif()
  endif()
  unset(CMAKE_REQUIRED_INCLUDES)
  unset(CMAKE_REQUIRED_DEFINITIONS)
endif()

if(JLCS_LLVM_ABI_ORACLE)
  target_sources(JLCS PRIVATE impl/JLCSAbiOracle.cpp)
  if(JLCS_LLVM_ABI_LIBRARY)
    if(LLVM_LINK_LLVM_DYLIB)
      target_link_libraries(JLCS PRIVATE ${JLCS_LLVM_ABI_LIBRARY})
    else()
      target_link_libraries(JLCS PRIVATE LLVMABI)
    endif()
  endif()
  message(STATUS "JLCS: SysV ABI oracle ON (${JLCS_LLVM_ABI_SOURCE})")
else()
  message(STATUS "JLCS: SysV ABI oracle OFF: no LLVM ABI library with an x86-64 "
                 "classifier (LLVM 23+); test_sysv_abi_oracle.jl will skip")
endif()

# Install the library
install(TARGETS JLCS
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

message(STATUS "JLCS Dialect build configured successfully")
