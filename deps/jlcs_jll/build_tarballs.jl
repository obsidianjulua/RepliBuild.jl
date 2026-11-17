# BinaryBuilder recipe for the JLCS MLIR dialect (libJLCS).
#
# Builds a self-contained libJLCS: LLVM and MLIR come from LLVM_full_jll as
# archives and are linked in (`-DJLCS_STATIC_LLVM=ON`), and only the C API is
# exported (src/mlir/jlcs_exports.map). The result needs no LLVM or MLIR on
# the machine that loads it, and it shares no LLVM symbol with Julia's own
# libLLVM.
#
# Run it from an environment that has BinaryBuilder (it is not a dependency
# of RepliBuild):
#
#     julia --project=<env with BinaryBuilder> deps/jlcs_jll/build_tarballs.jl --verbose
#     julia ... deps/jlcs_jll/build_tarballs.jl --verbose x86_64-linux-gnu-cxx11
#
# Tarballs land in ./products. `--deploy=local` also writes a JLCS_jll
# package to ~/.julia/dev/JLCS_jll.
#
# The source is this checkout's src/mlir less what git ignores (tracked files
# and new ones not yet added), staged into a temporary directory: a
# DirectorySource of src/mlir itself would ship the gitignored build/ tree,
# CMake cache and all.
#
# Linux only. Windows (PE has no version script, and BinaryBuilder's mingw is
# GCC's runtime where the supported host is MSYS2 CLANG64's libc++) and macOS
# are not built here.

using BinaryBuilder, Pkg, TOML

const REPO = normpath(joinpath(@__DIR__, "..", ".."))

name = "JLCS"
version = VersionNumber(TOML.parsefile(joinpath(REPO, "Project.toml"))["version"])

# The LLVM the dialect is built from. The SysV ABI oracle (test only) needs 23.
# The build number is part of the version: Pkg resolves a JLL exactly.
llvm_version = v"23.1.4+1"

function staged_dialect_source()
    files = split(read(`git -C $REPO ls-files -z -co --exclude-standard -- src/mlir`, String), '\0'; keepempty = false)
    isempty(files) && error("no source files under src/mlir in $REPO")
    stage = mktempdir()
    for f in files
        dest = joinpath(stage, "mlir", relpath(f, joinpath("src", "mlir")))
        mkpath(dirname(dest))
        cp(joinpath(REPO, f), dest)
    end
    return stage
end

sources = [DirectorySource(staged_dialect_source())]

script = raw"""
cd ${WORKSPACE}/srcdir/mlir

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
"""

platforms = expand_cxxstring_abis([
    Platform("x86_64", "linux"; libc = "glibc"),
    Platform("aarch64", "linux"; libc = "glibc"),
])

products = [
    LibraryProduct("libJLCS", :libJLCS),
]

dependencies = [
    HostBuildDependency(PackageSpec(name = "LLVM_full_jll", version = llvm_version)),
    BuildDependency(PackageSpec(name = "LLVM_full_jll", version = llvm_version)),
    # LLVMSupport's compression, the only libraries the result still loads
    # beyond the C and C++ runtimes.
    Dependency("Zlib_jll"),
    Dependency("Zstd_jll"),
]

build_tarballs(ARGS, name, version, sources, script, platforms, products, dependencies;
               julia_compat = "1.12", preferred_gcc_version = v"13")
