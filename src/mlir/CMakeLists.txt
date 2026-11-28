cmake_minimum_required(VERSION 3.20)
project(JLCS_Dialect LANGUAGES C CXX)

# Find LLVM and MLIR
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

# A cross build cannot run the mlir-tblgen its MLIR install names: that one is
# a target binary. JLCS_MLIR_TABLEGEN names one that runs on the build machine.
set(JLCS_MLIR_TABLEGEN "" CACHE FILEPATH "mlir-tblgen to run instead of the MLIR install's (cross builds)")
if(JLCS_MLIR_TABLEGEN)
  set(MLIR_TABLEGEN_EXE "${JLCS_MLIR_TABLEGEN}")
  set(MLIR_TABLEGEN_TARGET "")
endif()

message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

# A self-contained libJLCS: LLVM and MLIR linked in as archives, and only the
# C API exported (jlcs_exports.map). This is what the BinaryBuilder recipe
# (deps/jlcs_jll/build_tarballs.jl) builds, so the library needs no LLVM or
# MLIR on the machine that loads it. OFF, the default and what build.sh
# builds, links whatever the install offers, usually its shared libraries.
#
# An install built with LLVM_LINK_LLVM_DYLIB (LLVM_full_jll, most distros)
# exports MLIR's archives with the LLVM shared library as their dependency,
# so asking for the archives by name is not enough: every imported MLIR
# target is pointed at LLVM's component archives instead. The linker takes
# only the members something references.
option(JLCS_STATIC_LLVM "Link LLVM and MLIR statically and export only the C API" OFF)
if(JLCS_STATIC_LLVM)
  if(WIN32 OR APPLE)
    message(FATAL_ERROR "JLCS_STATIC_LLVM is ELF only: the export list is a GNU version script")
  endif()
  set(LLVM_LINK_LLVM_DYLIB OFF)
  set(_jlcs_llvm_archives "")
  foreach(_lib ${LLVM_AVAILABLE_LIBS})
    if(TARGET ${_lib})
      get_target_property(_type ${_lib} TYPE)
      # A few tool-support archives (LLVMOptDriver, LLVMDebuginfod, ...) are
      # themselves exported against the shared library; nothing here needs
      # them, and one on the link line would bring it back.
      get_target_property(_lib_deps ${_lib} INTERFACE_LINK_LIBRARIES)
      if(_type STREQUAL "STATIC_LIBRARY" AND NOT "LLVM" IN_LIST _lib_deps)
        list(APPEND _jlcs_llvm_archives ${_lib})
      endif()
    endif()
  endforeach()
  if(NOT _jlcs_llvm_archives)
    message(FATAL_ERROR "JLCS_STATIC_LLVM: this LLVM install ships no component archives")
  endif()
  add_library(JLCSStaticLLVM INTERFACE)
  target_link_libraries(JLCSStaticLLVM INTERFACE ${_jlcs_llvm_archives})
  get_directory_property(_jlcs_imported IMPORTED_TARGETS)
  foreach(_t ${_jlcs_imported})
    if(_t MATCHES "^MLIR")
      get_target_property(_deps ${_t} INTERFACE_LINK_LIBRARIES)
      if(_deps)
        list(TRANSFORM _deps REPLACE "^LLVM$" "JLCSStaticLLVM")
        list(TRANSFORM _deps REPLACE "^\\$<LINK_ONLY:LLVM>$" "$<LINK_ONLY:JLCSStaticLLVM>")
        set_target_properties(${_t} PROPERTIES INTERFACE_LINK_LIBRARIES "${_deps}")
      endif()
    endif()
  endforeach()
  message(STATUS "JLCS: static LLVM/MLIR, C API exports only")
endif()

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

# The dialect's ABI version, one number in one file. libJLCS exports it
# (jlcs_abi_version) and RepliBuild reads the same file, and refuses a library
# that reports a different one.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/ABI_VERSION" JLCS_ABI_VERSION LIMIT_COUNT 1)
string(STRIP "${JLCS_ABI_VERSION}" JLCS_ABI_VERSION)
if(NOT JLCS_ABI_VERSION MATCHES "^[0-9]+$")
  message(FATAL_ERROR "src/mlir/ABI_VERSION must hold one integer, not '${JLCS_ABI_VERSION}'")
endif()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/ABI_VERSION")
target_compile_definitions(JLCS PRIVATE JLCS_ABI_VERSION=${JLCS_ABI_VERSION})
message(STATUS "JLCS: dialect ABI version ${JLCS_ABI_VERSION}")

# The soname is the dialect's, not LLVM's. add_llvm_library gives a shared
# library LLVM's own version as its soname (libJLCS.so.23.1, and
# libJLCS.so.23.1jl from LLVM_full_jll), so the name followed the LLVM the
# library was built with and moved when that did, with the dialect unchanged.
# What a thunks library linked against libJLCS has to match is the dialect's
# ABI: the soname is libJLCS.so.<ABI_VERSION>, and the file carries the
# binary's own version (src/mlir/JLCS_VERSION), whose major is that number. ELF
# only, as add_llvm_library's is: a DLL's name carries no version.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/JLCS_VERSION" JLCS_VERSION LIMIT_COUNT 1)
string(STRIP "${JLCS_VERSION}" JLCS_VERSION)
if(NOT JLCS_VERSION MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+$")
  message(FATAL_ERROR "src/mlir/JLCS_VERSION must hold one version of three numbers, not '${JLCS_VERSION}'")
endif()
if(NOT JLCS_VERSION MATCHES "^${JLCS_ABI_VERSION}\\.")
  message(FATAL_ERROR "src/mlir/JLCS_VERSION (${JLCS_VERSION}) must have the dialect ABI version (${JLCS_ABI_VERSION}) as its major: the soname is libJLCS.so.${JLCS_ABI_VERSION}")
endif()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/JLCS_VERSION")
if(UNIX AND NOT APPLE)
  set_target_properties(JLCS PROPERTIES
    SOVERSION ${JLCS_ABI_VERSION}
    VERSION ${JLCS_VERSION})
endif()
message(STATUS "JLCS: version ${JLCS_VERSION}")

if(JLCS_STATIC_LLVM)
  target_link_libraries(JLCS PRIVATE JLCSStaticLLVM)
  target_link_options(JLCS PRIVATE
    "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/jlcs_exports.map"
    # Nothing LLVM or MLIR may be left to the loader. A compiler other than
    # the one that built the archives can ask for a symbol they do not define
    # (clang calls the C2 constructor where GCC's archives carry only C1),
    # and without this that is a library that fails only under RTLD_NOW.
    "-Wl,-z,defs")
  set_target_properties(JLCS PROPERTIES
    LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/jlcs_exports.map")
endif()

# Another host's struct ABI rules, built into THIS host's libJLCS for
# inspection: the result cannot run anything (the JIT would emit that
# convention's calls into this process), so build it to a scratch directory,
# never over the working library. The decision tables need neither: the probe
# seams (jlcs_probe_sysv_call_shapes, jlcs_probe_aapcs64_call_shapes) take an
# explicit target on any host. These exist for diffing the LOWERING (the
# allocas, loads and call the pass emits) against clang's for the same
# signatures. See kHostAbi in impl/JLCSPasses.cpp.
option(JLCS_FORCE_ABI_WIN64 "Build the Win64 (Microsoft x64) struct ABI rules as the host's (inspection only)" OFF)
option(JLCS_FORCE_ABI_AAPCS64 "Build the AArch64 AAPCS64 struct ABI rules as the host's (inspection only)" OFF)
if(JLCS_FORCE_ABI_WIN64 AND JLCS_FORCE_ABI_AAPCS64)
  message(FATAL_ERROR "JLCS_FORCE_ABI_WIN64 and JLCS_FORCE_ABI_AAPCS64 name two hosts; pick one")
endif()
if(JLCS_FORCE_ABI_WIN64)
  target_compile_definitions(JLCS PRIVATE JLCS_FORCE_ABI_WIN64)
  message(WARNING "JLCS: forcing the Win64 struct ABI rules; this libJLCS cannot run thunks on this host")
endif()
if(JLCS_FORCE_ABI_AAPCS64)
  target_compile_definitions(JLCS PRIVATE JLCS_FORCE_ABI_AAPCS64)
  message(WARNING "JLCS: forcing the AAPCS64 struct ABI rules; this libJLCS cannot run thunks on this host")
endif()

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
