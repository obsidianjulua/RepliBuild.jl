# Note that this script can accept some limited command-line arguments, run
# `julia build_tarballs.jl --help` to see a usage message.
using BinaryBuilder, Pkg

name = "JLCS"
version = v"2.0.0"

# The JLCS MLIR dialect of RepliBuild.jl: the library its Tier 2 (C++ methods,
# by-value aggregates, virtual dispatch, exceptions) lowers and JITs thunks
# with. Its sources live in RepliBuild.jl's own repository, under src/mlir.
# LLVM and MLIR are linked in as archives and only the C API is exported, so
# the library shares no LLVM symbol with Julia's own libLLVM.
sources = [
    GitSource("https://github.com/obsidianjulua/RepliBuild.jl.git",
              "0000000000000000000000000000000000000000"),
]

# Bash recipe for building across all platforms
script = raw"""
cd ${WORKSPACE}/srcdir/RepliBuild.jl/src/mlir

# The target's MLIR names its own mlir-tblgen as the generator, and that is a
# target binary; JLCS_MLIR_TABLEGEN names the host's, which runs here.
cmake -S . -B ${WORKSPACE}/jlcs-build -GNinja \
    -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TARGET_TOOLCHAIN} \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${prefix} \
    -DLLVM_DIR=${prefix}/lib/cmake/llvm \
    -DMLIR_DIR=${prefix}/lib/cmake/mlir \
    -DJLCS_MLIR_TABLEGEN=${host_prefix}/tools/mlir-tblgen \
    -DJLCS_STATIC_LLVM=ON
cmake --build ${WORKSPACE}/jlcs-build -j${nproc}
cmake --install ${WORKSPACE}/jlcs-build

install_license ${WORKSPACE}/srcdir/RepliBuild.jl/LICENSE
"""

# Linux only. The export list is a GNU version script, and RepliBuild's Windows
# host (MSYS2 CLANG64, libc++) is not BinaryBuilder's mingw (GCC's runtime).
platforms = expand_cxxstring_abis([
    Platform("x86_64", "linux"; libc = "glibc"),
    Platform("aarch64", "linux"; libc = "glibc"),
])

products = [
    LibraryProduct("libJLCS", :libJLCS),
]

# The LLVM the dialect is built from; its archives are linked in, so it is a
# build dependency only. The build number is part of the version: Pkg resolves
# a JLL exactly.
llvm_version = v"23.1.4+1"

dependencies = [
    HostBuildDependency(PackageSpec(name = "LLVM_full_jll", version = llvm_version)),
    BuildDependency(PackageSpec(name = "LLVM_full_jll", version = llvm_version)),
    # LLVMSupport's compression, the only libraries the result still loads
    # beyond the C and C++ runtimes.
    Dependency("Zlib_jll"),
    Dependency("Zstd_jll"),
]

# GCC, like LLVM_full_jll's own build: its archives define the constructor
# variants GCC asks for, not the ones clang does.
build_tarballs(ARGS, name, version, sources, script, platforms, products, dependencies;
               julia_compat = "1.10", preferred_gcc_version = v"13")
