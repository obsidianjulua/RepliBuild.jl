# The JLCS dialect's BinaryBuilder recipe, run from this checkout.
#
# It is yggdrasil/build_tarballs.jl (the recipe that goes to
# JuliaPackaging/Yggdrasil as J/JLCS/build_tarballs.jl, where it builds from a
# pinned commit of this repository) with one difference: the source is this
# working tree, so a change to src/mlir can be built before it is committed.
# The script, platforms, products and dependencies are the same text, and
# test_jlcs_abi_version.jl holds them equal. See README.md beside this file.
#
# Run it from an environment that has BinaryBuilder (not a dependency of
# RepliBuild), on a Julia BinaryBuilder supports:
#
#     julia +1.12 --project=<env with BinaryBuilder> deps/jlcs_jll/build_tarballs.jl --verbose
#     julia ... deps/jlcs_jll/build_tarballs.jl --verbose x86_64-linux-gnu-cxx11
#
# Run it from this directory: tarballs land in ./products, and
# write_artifacts_toml.jl turns them into the package's Artifacts.toml entries.

using BinaryBuilder, Pkg

const REPO = normpath(joinpath(@__DIR__, "..", ".."))

name = "JLCS"
# The dialect's own version: src/mlir/JLCS_VERSION, not RepliBuild's.
version = VersionNumber(strip(read(joinpath(REPO, "src", "mlir", "JLCS_VERSION"), String)))

# src/mlir less what git ignores (tracked files and new ones not yet added),
# and the licence, staged where a clone of the repository would have them: a
# DirectorySource of src/mlir itself would ship the gitignored build/ tree.
function staged_source()
    files = split(read(`git -C $REPO ls-files -z -co --exclude-standard -- src/mlir LICENSE`, String),
                  '\0'; keepempty = false)
    any(f -> startswith(f, "src/mlir/"), files) || error("no source files under src/mlir in $REPO")
    stage = mktempdir()
    for f in files
        dest = joinpath(stage, "RepliBuild.jl", f)
        mkpath(dirname(dest))
        cp(joinpath(REPO, f), dest)
    end
    return stage
end

sources = [DirectorySource(staged_source())]

script = raw"""
cd ${WORKSPACE}/srcdir/RepliBuild.jl/src/mlir

# JLCS_STATIC_LLVM links LLVM's and MLIR's archives in and exports only the
# library's C API, so it loads beside Julia's own libLLVM, of another version,
# sharing no symbol with it.
# MLIR's CMake package names its own mlir-tblgen as the generator, and that is
# a target binary; JLCS_MLIR_TABLEGEN names the host's, which runs here.
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
# LLVM and MLIR are linked in
install_license ${prefix}/share/licenses/LLVM_full/LICENSE.TXT
"""

platforms = supported_platforms()
# The dialect lowers a call the way the host's C calling convention passes it,
# and has those rules for x86-64 and AArch64 only: its sources `#error` on any
# other architecture.
filter!(p -> arch(p) in ("x86_64", "aarch64"), platforms)
# The export list is a GNU version script, and the build refuses
# JLCS_STATIC_LLVM where the object format is not ELF.
filter!(p -> !Sys.iswindows(p) && !Sys.isapple(p), platforms)
# The library is linked with `-z defs`, and FreeBSD's libc leaves `environ`,
# which LLVMSupport refers to, to the executable: `ld.lld: error: undefined
# symbol: environ`.
filter!(!Sys.isfreebsd, platforms)
platforms = expand_cxxstring_abis(platforms)

products = [
    LibraryProduct("libJLCS", :libJLCS),
]

# One build for every Julia: the LLVM it is built from is linked in, so it is
# a build dependency only and the library is not tied to Julia's LLVM version.
llvm_version = v"23.1.4+1"

dependencies = [
    HostBuildDependency(PackageSpec(name="LLVM_full_jll", version=llvm_version)),
    BuildDependency(PackageSpec(name="LLVM_full_jll", version=llvm_version)),
    # libstdc++ and libgcc_s
    Dependency("CompilerSupportLibraries_jll"; compat="1.0.5"),
    # LLVM_full's archives are built with zlib and zstd enabled, so the
    # library loads libz and libzstd.
    Dependency("Zlib_jll"; compat="1.2.12"),
    Dependency("Zstd_jll"; compat="1.5.7"),
]

# GCC 10 is what LLVM_full_jll 23 is built with, and its archives are linked in.
build_tarballs(ARGS, name, version, sources, script, platforms, products, dependencies;
               julia_compat="1.10", preferred_gcc_version=v"10")
