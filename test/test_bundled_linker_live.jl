# test_bundled_linker_live.jl — a C library linked by the bundled LLD.
#
# Builds a one-file C project three ways:
#
#   * as a build does by default: the library's `.comment` names LLD, and it
#     loads and computes;
#   * with REPLIBUILD_TOOLS=system: the driver's own linker, no LLD;
#   * in a child whose PATH is an empty directory: the compiler, the linker,
#     the DWARF dumper and the symbol reader all come from the package's
#     dependencies, so the build and the wrap finish with nothing found by
#     name. (The C runtime's start files and headers still come from the
#     system: clang finds those by its own search, not through PATH.)
#
# Linux, C bucket only.

using Test
using RepliBuild
import Libdl

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

haskey(ENV, "REPLIBUILD_HOME") || (ENV["REPLIBUILD_HOME"] = mktempdir())

const _BLL_C = RepliBuild.Compiler

function _bll_project(dir::String)
    mkpath(joinpath(dir, "src"))
    src = joinpath(dir, "src", "blprobe.c")
    write(src, """
        #include <math.h>
        static int bl_counter;
        int bl_bump(int by) { bl_counter += by; return bl_counter; }
        double bl_root(double x) { return sqrt(x) + bl_counter; }
        int bl_twice(int by) { return bl_bump(by) + bl_bump(by); }
        """)
    toml = joinpath(dir, "replibuild.toml")
    write(toml, """
        [project]
        name = "blprobe"
        root = "$dir"

        [compile]
        source_files = ["$src"]

        [link]
        enable_lto = false
        link_libraries = ["m"]

        [cache]
        enabled = true
        directory = ".replibuild_cache"

        [wrap]
        language = "c"
        """)
    return toml
end

# What linked a library, as the linker itself wrote it.
function _bll_comment(lib::String)
    out, code = RepliBuild.BuildBridge.execute("readelf", ["-p", ".comment", lib])
    @test code == 0
    return out
end

if !Sys.islinux()
    @testset "bundled linker, live" begin
        @info "skipped: the bundled linker is Linux's"
        @test_skip false
    end
elseif _BLL_C._bundled_linker(:c) === nothing
    @testset "bundled linker, live" begin
        @info "skipped: no bundled linker on this host (no JLL clang or no JLL lld)"
        @test_skip false
    end
else
    @testset "bundled linker, live" begin
        @testset "a default C build is linked by LLD" begin
            mktempdir() do dir
                lib = RepliBuild.build(_bll_project(dir))
                @test occursin("Linker: LLD", _bll_comment(lib))
                h = Libdl.dlopen(lib)
                @test ccall(Libdl.dlsym(h, :bl_twice), Cint, (Cint,), 3) == 9
                @test ccall(Libdl.dlsym(h, :bl_root), Cdouble, (Cdouble,), 81.0) == 15.0
                # Its own symbols are its own under LLD too.
                dyn, code = RepliBuild.BuildBridge.execute("readelf", ["-d", lib])
                @test code == 0
                @test occursin("SYMBOLIC", dyn)
            end
        end

        @testset "REPLIBUILD_TOOLS=system links with the driver's own linker" begin
            mktempdir() do dir
                withenv("REPLIBUILD_TOOLS" => "system") do
                    if have(:nm) && Sys.which("ld") !== nothing
                        lib = RepliBuild.build(_bll_project(dir))
                        @test !occursin("Linker: LLD", _bll_comment(lib))
                    else
                        @info "skipped: no system binutils to link with"
                    end
                end
            end
        end

        @testset "a C build and wrap with nothing on PATH" begin
            mktempdir() do dir
                toml = _bll_project(dir)
                empty = mktempdir()
                code = """
                using RepliBuild
                lib = RepliBuild.build($(repr(toml)))
                println("PROBE build: ", isfile(lib) ? "PASS" : "FAIL")
                wrapper = RepliBuild.wrap($(repr(toml)))
                println("PROBE wrap: ", isfile(wrapper) ? "PASS" : "FAIL")
                M = Module(:BLHost)
                Base.include(M, wrapper)
                W = getfield(M, :Blprobe)
                r = Base.invokelatest(W.bl_twice, 5)
                println("PROBE call: ", r == 15 ? "PASS" : "FAIL \$r")
                srcs = [RepliBuild.Tools.tool_source(n) for n in ("readelf", "nm", "ld.lld")]
                println("PROBE sources: ", all(==(:jll), srcs) ? "PASS" : "FAIL \$srcs")
                println("PROBE_DONE")
                """
                r = withenv("PATH" => empty, "REPLIBUILD_JULIA_ENV" => "none") do
                    run_isolated(`-e $code`; timeout = 600)
                end
                v = probe_verdicts(r.output)
                @test r.ok
                @test haskey(v, "PROBE_DONE")
                for name in ("build", "wrap", "call", "sources")
                    @test startswith(get(v, name, "no PROBE line"), "PASS")
                end
            end
        end
    end
end
