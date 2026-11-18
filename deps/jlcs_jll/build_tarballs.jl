# The JLCS dialect's BinaryBuilder recipe, run from this checkout.
#
# It is yggdrasil/build_tarballs.jl (the recipe that goes to
# JuliaPackaging/Yggdrasil as J/JLCS/build_tarballs.jl, where it builds from a
# pinned commit of this repository) with one difference: the source is this
# working tree, so a change to src/mlir can be built before it is committed.
# The script, platforms, products and dependencies are the same text, and
# test_jlcs_recipes.jl holds them equal. See README.md beside this file.
#
# Run it from an environment that has BinaryBuilder (not a dependency of
# RepliBuild), on a Julia BinaryBuilder supports:
#
#     julia +1.12 --project=<env with BinaryBuilder> deps/jlcs_jll/build_tarballs.jl --verbose
#     julia ... deps/jlcs_jll/build_tarballs.jl --verbose x86_64-linux-gnu-cxx11
#
# Tarballs land in ./products; write_artifacts_toml.jl turns them into the
# package's Artifacts.toml entries.

using BinaryBuilder, Pkg

const REPO = normpath(joinpath(@__DIR__, "..", ".."))

name = "JLCS"
# The dialect's own version: src/mlir/VERSION, not RepliBuild's.
version = VersionNumber(strip(read(joinpath(REPO, "src", "mlir", "VERSION"), String)))

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
