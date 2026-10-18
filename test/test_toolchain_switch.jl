# Switching toolchains: what a build depends on, and what a wrapper assumes.
#
# A C project's compiler is the JLL clang pinned to Julia's own libLLVM, and its
# Tier 1 slices are IR printed by that libLLVM. Julia 1.12 is LLVM 18 and 1.13
# is LLVM 20, and newer IR does not parse in an older LLVM: a zlib wrapper built
# under 1.13 and run under 1.12 failed 77 of its deep tests with "Failed to parse
# LLVM assembly: expected icmp predicate" (`icmp samesign`). Three things carried
# that across a switch:
#
#   * the project hash had no toolchain in it, so a switch rebuilt only because
#     `Base.hash` happens to differ between 1.12 and 1.13 (it is not stable
#     across Julia releases, so it is not a cache key either);
#   * the slice cache keyed on module text, which a wrap without a rebuild does
#     not change, so it served the other Julia's slices;
#   * the wrapper's kernels never asked which LLVM printed their slices.
#
# Toolchain-free: the last testset runs a real kernel against the C runtime's
# `abs`, with a slice that makes llvmcall and ccall give different answers.

using Test
using Libdl
using RepliBuild

const _TS_C  = RepliBuild.Compiler
const _TS_CM = RepliBuild.ConfigurationManager
const _TS_W  = RepliBuild.Wrapper
const _TS_S  = RepliBuild.Slicer

_ts_path(p) = replace(p, '\\' => '/')

function _ts_project(dir, language)
    mkpath(joinpath(dir, "src"))
    write(joinpath(dir, "src", "main.c"), "int f(void) { return 0; }\n")
    write(joinpath(dir, "replibuild.toml"), """
    [project]
    name = "probe"
    root = "$(_ts_path(dir))"

    [compile]
    source_files = ["$(_ts_path(joinpath(dir, "src", "main.c")))"]

    [cache]
    enabled = true
    directory = ".replibuild_cache"

    [wrap]
    language = "$language"
    """)
    return _TS_CM.load_config(joinpath(dir, "replibuild.toml"))
end

_is_sha256_hex(s) = length(s) == 64 && all(c -> c in "0123456789abcdef", s)

@testset "Toolchain switching" begin

    @testset "the toolchain identity names the compiler a project uses" begin
        mktempdir() do d
            c   = _ts_project(joinpath(d, "c"), "c")
            cpp = _ts_project(joinpath(d, "cpp"), "cpp")
            # C: Julia's own libLLVM, which the JLL clang is pinned to.
            @test startswith(_TS_C._toolchain_identity(c), "libllvm $(Base.libllvm_version) | ")
            # C++: the system clang. Julia's libLLVM plays no part, so a Julia
            # switch must not rebuild a C++ project.
            @test startswith(_TS_C._toolchain_identity(cpp), "system ")
            @test !occursin("libllvm", _TS_C._toolchain_identity(cpp))
        end
    end

    @testset "the project hash moves with the toolchain, and only with it" begin
        mktempdir() do d
            cfg = _ts_project(d, "c")
            a = _TS_C.compute_project_hash(cfg; toolchain="libllvm 18.1.7 | x")
            b = _TS_C.compute_project_hash(cfg; toolchain="libllvm 20.1.8 | x")
            @test a != b
            @test _TS_C.compute_project_hash(cfg; toolchain="libllvm 18.1.7 | x") == a
            # A build takes the identity itself.
            @test _TS_C.compute_project_hash(cfg) ==
                  _TS_C.compute_project_hash(cfg; toolchain=_TS_C._toolchain_identity(cfg))
            # SHA-256, not Base.hash: the key is written to disk and read back
            # by whatever Julia runs next.
            @test _is_sha256_hex(a)
        end
    end

    @testset "the per-file IR fingerprint carries the same identity" begin
        mktempdir() do d
            cfg = _ts_project(d, "c")
            a = _TS_C.compute_compile_fingerprint(cfg; toolchain="libllvm 18.1.7 | x")
            @test a != _TS_C.compute_compile_fingerprint(cfg; toolchain="libllvm 20.1.8 | x")
            @test _TS_C.compute_compile_fingerprint(cfg) ==
                  _TS_C.compute_compile_fingerprint(cfg; toolchain=_TS_C._toolchain_identity(cfg))
            @test _is_sha256_hex(a)
        end
    end

    @testset "the slice cache is per libLLVM major" begin
        bytes = Vector{UInt8}("define i32 @f() { ret i32 0 }\n")
        k18 = _TS_S._slice_cache_key(bytes; llvm_major=18)
        @test k18 == _TS_S._slice_cache_key(bytes; llvm_major=18)
        @test k18 != _TS_S._slice_cache_key(bytes; llvm_major=20)
        @test endswith(k18, "-v$(_TS_S.SLICER_VERSION)-llvm18")
        @test endswith(_TS_S._slice_cache_key(bytes), "-llvm$(Base.libllvm_version.major)")
    end

    @testset "a kernel keeps its ccall body under another libLLVM major" begin
        major = Base.libllvm_version.major
        reg = _TS_W._tier1_registry_chunk(["abs"], Dict("abs" => String[]))
        @test occursin("const TIER1_LIBLLVM_MAJOR = $major", reg)
        kern = _TS_W._tier1_kernel_chunk("_SLICE_abs", "_TIER1_abs", "abs", "Cint",
                                         ["Cint"], ["x"])
        guard = findfirst("Base.libllvm_version.major != TIER1_LIBLLVM_MAJOR", kern)
        @test guard !== nothing
        # Part of the generation-time doubt, so before the slice is read.
        @test guard.start < findfirst("ir = read(", kern).start

        # Run it. The slice defines `abs` to return 777, so llvmcall answers 777
        # and the ccall body (the C runtime's abs) answers 5.
        crt = Sys.iswindows() ? "msvcrt" : "libc.so.6"
        mktempdir() do d
            slice = joinpath(d, "abs.ll")
            write(slice, "define i32 @abs(i32 %0) {\n  ret i32 777\n}\n")
            ran = 0
            for (recorded, want) in ((major, 777), (major + 1, 5))
                m = Module(:ToolchainSwitchProbe)
                Core.eval(m, :(using Libdl))
                Core.eval(m, :(const LIBRARY_PATH = $crt))
                Core.eval(m, :(const LIB_HANDLE = Ref(Libdl.dlopen($crt))))
                Core.eval(m, Meta.parseall(replace(reg,
                    r"const TIER1_LIBLLVM_MAJOR = \d+" => "const TIER1_LIBLLVM_MAJOR = $recorded")))
                Core.eval(m, Meta.parseall(replace(kern,
                    r"const _SLICE_abs = .*" => "const _SLICE_abs = $(repr(slice))")))
                @test Base.invokelatest(getfield(m, :_TIER1_abs), Cint(-5)) == want
                ran += 1
            end
            @test ran == 2
        end
    end
end
