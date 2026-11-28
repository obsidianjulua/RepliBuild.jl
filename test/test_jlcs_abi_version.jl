# test_jlcs_abi_version.jl — one dialect ABI version, and a library that
# reports another is refused.
#
# libJLCS can come from somewhere other than this checkout now: the prebuilt
# artifact, a JLL, a local build left from an older checkout. What Julia
# generates (the ops, their attributes, the thunk conventions) and what it
# calls (the C API) are for exactly one dialect, and a mismatch is a wrong
# call at run time, not an error. So the dialect has a version, one integer in
# src/mlir/ABI_VERSION that CMake compiles into `jlcs_abi_version()` and
# RepliBuild reads from the same file, and RepliBuild asks a library for it
# before using it (JLCSLibrary.verify_abi!).
#
# Also here: the two BinaryBuilder recipes (this checkout's, and the one for
# Yggdrasil) are the same build.
#
# No toolchain. test_jlcs_abi_live.jl asks real libraries.

using Test
using RepliBuild

const _AV = RepliBuild.JLCSLibrary
const _AV_ROOT = dirname(@__DIR__)
const _AV_MLIR = joinpath(_AV_ROOT, "src", "mlir")

@testset "one number, in one file, read by both sides" begin
    text = strip(read(joinpath(_AV_MLIR, "ABI_VERSION"), String))
    @test occursin(r"^[0-9]+$", text)
    @test _AV.ABI_VERSION == parse(Int, text)
    @test _AV.ABI_VERSION >= 1
    @test _AV.ABI_VERSION_FILE == joinpath(_AV_MLIR, "ABI_VERSION")

    # The dialect is compiled with that file's number and exports it.
    cmake = read(joinpath(_AV_MLIR, "CMakeLists.txt"), String)
    @test occursin("file(STRINGS \"\${CMAKE_CURRENT_SOURCE_DIR}/ABI_VERSION\" JLCS_ABI_VERSION", cmake)
    @test occursin("target_compile_definitions(JLCS PRIVATE JLCS_ABI_VERSION=\${JLCS_ABI_VERSION})", cmake)
    capi = read(joinpath(_AV_MLIR, "impl", "JLCSCAPIWrappers.cpp"), String)
    @test occursin(r"int jlcs_abi_version\(void\)\s*\{\s*return JLCS_ABI_VERSION;", capi)
    # No literal of its own: a second copy is how two sides come to disagree.
    @test !occursin(r"#define\s+JLCS_ABI_VERSION", capi)
    # And the static build's export list lets it out.
    exports = read(joinpath(_AV_MLIR, "jlcs_exports.map"), String)
    @test occursin("jlcs*;", exports)
end

@testset "a library of another dialect is refused, and told how to be fixed" begin
    want = _AV.ABI_VERSION
    loc(source) = _AV.Location("/somewhere/libJLCS.so", source, "")

    # The same version, or a library that does not load (the loader's story).
    for source in (:local, :artifact, :override)
        @test _AV.abi_problem(loc(source); got = want) === nothing
        @test _AV.abi_problem(loc(source); got = nothing) === nothing
    end
    # Nothing to ask when there is no library.
    @test _AV.abi_problem(loc(:missing); got = want + 1) === nothing

    # Another version, either way round.
    for got in (want + 1, want - 1 == 0 ? want + 7 : want - 1)
        msg = _AV.abi_problem(loc(:local); got)
        @test msg !== nothing
        @test occursin("speaks dialect ABI $got", msg)
        @test occursin("this RepliBuild speaks dialect ABI $want", msg)
        @test occursin("/somewhere/libJLCS.so", msg)
    end
    # A library from before the check has no version at all.
    old = _AV.abi_problem(loc(:artifact); got = 0)
    @test old !== nothing && occursin("has no dialect ABI version", old)

    # The fix depends on where the library came from.
    @test occursin("build.sh", _AV.abi_problem(loc(:local); got = 0))
    @test occursin("REPLIBUILD_JLCS=artifact", _AV.abi_problem(loc(:local); got = 0))
    @test occursin("update RepliBuild", _AV.abi_problem(loc(:artifact); got = 0))
    @test occursin("REPLIBUILD_JLCS names it", _AV.abi_problem(loc(:override); got = 0))

    # `expected` is the comparison's other side, not a constant inside it.
    @test _AV.abi_problem(loc(:local); expected = 5, got = 5) === nothing
    @test _AV.abi_problem(loc(:local); expected = 5, got = want) !== nothing || want == 5

    # verify_abi! raises nothing for a library that is not there.
    @test _AV.verify_abi!(loc(:missing)) === nothing
end

@testset "the two recipes are one build" begin
    dir = joinpath(_AV_ROOT, "deps", "jlcs_jll")
    here = read(joinpath(dir, "build_tarballs.jl"), String)
    ygg = read(joinpath(dir, "yggdrasil", "build_tarballs.jl"), String)

    # The shell script, and everything after it: platforms, products,
    # dependencies, the build_tarballs call.
    function build(text)
        a = findfirst("script = raw\"\"\"", text)
        @test a !== nothing
        return text[first(a):end]
    end
    @test build(here) == build(ygg)
    @test occursin("-DJLCS_STATIC_LLVM=ON", build(ygg))
    @test occursin("install_license", build(ygg))

    # The dialect's version is its own, and the Yggdrasil literal is it.
    version = strip(read(joinpath(_AV_MLIR, "JLCS_VERSION"), String))
    @test VersionNumber(version) isa VersionNumber
    @test occursin("version = v\"$version\"", ygg)
    @test occursin("joinpath(REPO, \"src\", \"mlir\", \"JLCS_VERSION\")", here)

    # Yggdrasil builds from a pinned commit of the repository, never a path:
    # the commit of the dialect's own release tag, which its build checks the
    # hash against (`tag=`, computed from `version`, as Yggdrasil asks).
    m = match(r"GitSource\(\"(https://github\.com/[^\"]+)\",\s*\"([0-9a-f]{40})\";\s*tag\s*=\s*\"jlcs-v\$\(version\)\"\)", ygg)
    @test m !== nothing
    @test m.captures[1] == "https://github.com/obsidianjulua/RepliBuild.jl.git"
    @test m.captures[2] != "0"^40
    @test !occursin("@__DIR__", ygg)
    @test !occursin("DirectorySource", ygg)
    # This checkout's builds the working tree, laid out as a clone is.
    @test occursin("DirectorySource(staged_source())", here)
    @test occursin("joinpath(stage, \"RepliBuild.jl\", f)", here)

    for text in (here, ygg)
        ex = Meta.parseall(text)
        @test !(ex.head in (:incomplete, :error))
        @test !any(a -> a isa Expr && a.head in (:incomplete, :error), ex.args)
    end
end
