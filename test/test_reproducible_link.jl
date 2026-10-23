# test_reproducible_link.jl — a relink of unchanged IR gives the same library.
#
# lld writes the link time into a PE image twice, in the COFF file header and in
# the debug directory (its `.buildid` record), so every relink of an unchanged
# project gave a different DLL. A wrapper's `LIBRARY_SHA` is the SHA-256 of that
# file: a rebuilt llamacpp's 60336-line wrapper differed from the last one in
# that line alone, and a CI job that judges a Hub bump by diffing regenerated
# wrappers would see it on every run. Every link now passes
# `Compiler._reproducible_link_flags()`.
#
# Each fixture is built, its project hash deleted (the per-file IR cache stays,
# so the second build relinks the same IR, as a bump with unchanged sources
# does), and built again in a later wall-clock second. The two libraries must be
# the same bytes. On PE both link-time fields must also read 0, which does not
# depend on the timing.
#
# The C bucket (Clang_unified_jll, linked in process) and C++ (system clang++,
# llvm-link and opt) reach create_library with different toolchains; an
# executable goes through create_executable. ELF has no link time, so on Linux
# this asserts the invariant rather than the fix. The AOT thunks library is
# checked in test_aot_thunks.jl.

using Test
using RepliBuild

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

const _RL_C  = RepliBuild.Compiler
const _RL_CM = RepliBuild.ConfigurationManager

_rl_path(p) = replace(p, '\\' => '/')

"A one-file project in `dir`: a static and two exported functions."
function _rl_project(dir::String, language::String)
    ext = language == "c" ? "c" : "cpp"
    mkpath(joinpath(dir, "src"))
    src = joinpath(dir, "src", "rlprobe.$ext")
    write(src, """
        static int rl_counter;
        int rl_bump(int by) { rl_counter += by; return rl_counter; }
        double rl_scale(double x, int n) { return x * n + rl_counter; }
        """)
    toml = joinpath(dir, "replibuild.toml")
    write(toml, """
        [project]
        name = "rlprobe_$language"
        root = "$(_rl_path(dir))"

        [compile]
        source_files = ["$(_rl_path(src))"]

        [link]
        enable_lto = false

        [cache]
        enabled = true
        directory = ".replibuild_cache"

        [wrap]
        language = "$language"
        """)
    return toml
end

"Return once the wall clock has reached a later second than when called."
function _rl_next_second()
    t = floor(Int, time())
    while floor(Int, time()) <= t
        sleep(0.05)
    end
end

"First few byte offsets at which `a` and `b` differ, for the failure log."
_rl_first_diffs(a, b) = first(findall(i -> a[i] != b[i], 1:min(length(a), length(b))), 8)

"""
Build `toml`, delete its project hash, build again a second later, and return
both libraries' bytes. Asserts that the second build relinked: the per-file IR
was reused and the library was written again.
"""
function _rl_build_twice(toml::String)
    root = dirname(toml)
    lib = RepliBuild.build(toml)
    @test isfile(lib)
    first_bytes, first_mtime = read(lib), mtime(lib)

    build_dir = joinpath(root, "build")
    per_tu() = Dict(f => mtime(joinpath(build_dir, f)) for f in readdir(build_dir)
                    if startswith(f, "rlprobe.") && endswith(f, ".ll"))
    ir_before = per_tu()
    @test length(ir_before) == 1

    rm(joinpath(root, ".replibuild_cache", "project_hash"))
    _rl_next_second()
    @test RepliBuild.build(toml) == lib
    @test per_tu() == ir_before           # the same IR, not a recompile
    @test mtime(lib) > first_mtime        # linked again
    second_bytes = read(lib)
    first_bytes == second_bytes ||
        @info "relinked library differs at byte offsets $(_rl_first_diffs(first_bytes, second_bytes))" lib length(first_bytes) length(second_bytes)
    return (lib, first_bytes, second_bytes)
end

@testset "Reproducible link" begin

    @testset "C: a relink of the same IR gives the same library" begin
        mktempdir() do dir
            (lib, a, b) = _rl_build_twice(_rl_project(dir, "c"))
            @test a == b
            if Sys.iswindows()
                t = pe_link_times(lib)
                @test t.header == 0
                @test all(iszero, t.debug)
            end
        end
    end

    if requires("Reproducible link (C++)", :clangxx)
        @testset "C++: a relink of the same IR gives the same library" begin
            mktempdir() do dir
                (lib, a, b) = _rl_build_twice(_rl_project(dir, "cpp"))
                @test a == b
                if Sys.iswindows()
                    t = pe_link_times(lib)
                    @test t.header == 0
                    @test all(iszero, t.debug)
                end
            end
        end
    end

    @testset "an executable links to the same bytes" begin
        mktempdir() do dir
            config = _RL_CM.load_config(_rl_project(dir, "c"))
            main_c = joinpath(dir, "src", "main.c")
            write(main_c, "int main(void) { return 0; }\n")
            ll = joinpath(dir, "main.ll")
            (out, code) = _RL_C._clang_for_c_bucket("clang", ["-S", "-emit-llvm", "-g", "-o", ll, main_c])
            code == 0 || @info "main.c -> IR failed" out
            @test code == 0

            # clang names the file exactly as asked; the .exe is ours to add.
            name = Sys.iswindows() ? "rlprobe.exe" : "rlprobe"
            exe = _RL_C.create_executable(config, ll, name)
            a = read(exe)
            _rl_next_second()
            @test _RL_C.create_executable(config, ll, name) == exe
            b = read(exe)
            a == b || @info "relinked executable differs at byte offsets $(_rl_first_diffs(a, b))" exe
            @test a == b
            if Sys.iswindows()
                t = pe_link_times(exe)
                @test t.header == 0
                @test all(iszero, t.debug)
            end
        end
    end
end
