# test_jlcs_soname_live.jl — libJLCS's soname, read off the library in use,
# and the loader made to bring in a second copy.
#
# A thunks library imports libJLCS by the soname of the one it was linked
# against. When the libJLCS in the process has another soname, the loader does
# not take it for that import: it searches the thunks library's RUNPATH, and
# an older libJLCS still on disk there is loaded BESIDE the one RepliBuild
# uses, with no error. Each copy has its own exception buffer, so a C++
# exception a thunk catches is written where Julia never reads.
#
# Three libraries stand for the three cases, each linked as ThunkBuilder links
# a thunks library (the libJLCS by path, its directory as RUNPATH):
#
#   good   against the libJLCS in use                    -> opens, one copy
#   stale  against a libJLCS of another soname, on disk  -> refused: wrap again
#   gone   against one whose file is no longer there     -> the loader's error,
#                                                           and wrap again
#
# The opens run in a child: a refused library has already put a second
# libJLCS-named file into the process by the time it is refused.
#
# ELF only. A DLL is imported by its name alone and has no RUNPATH.

using Test
using RepliBuild
import Libdl

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

haskey(ENV, "REPLIBUILD_HOME") || (ENV["REPLIBUILD_HOME"] = mktempdir())

const _SNL = RepliBuild.JLCSLibrary

"The soname GNU readelf prints for the ELF file at `path`, or `nothing`."
function _snl_soname(path::AbstractString)
    (out, code) = RepliBuild.BuildBridge.execute("readelf", ["-d", String(path)])
    code == 0 || return nothing
    m = match(r"\(SONAME\)\s+Library soname: \[([^\]]+)\]", out)
    return m === nothing ? nothing : String(m.captures[1])
end

if !Sys.islinux()
    @info "Skipping libJLCS soname, live — sonames and RUNPATH are ELF's"
    @testset "libJLCS soname, live (skipped — ELF only)" begin
        @test_skip false
    end
elseif requires("libJLCS soname, live", :libJLCS, :clang)
    @testset "libJLCS soname, live" begin
        loc = _SNL.locate(; install = false)
        @test loc.source !== :missing

        @testset "the library in use is named for the dialect" begin
            soname = _snl_soname(loc.path)
            want = "libJLCS.so.$(_SNL.ABI_VERSION)"
            if loc.source === :override
                # REPLIBUILD_JLCS names any file: nothing here built it.
                @info "libJLCS is an override; its soname is not this checkout's" loc.path soname
                @test soname !== nothing
            else
                soname == want ||
                    @warn "libJLCS has the soname of an older build; build it again " *
                          "(src/mlir/build.sh), or update the artifact" loc.path soname want
                @test soname == want
            end
        end

        mktempdir() do dir
            function cc(source::String, out::String, args::Vector{String})
                c = joinpath(dir, splitext(basename(out))[1] * ".c")
                write(c, source)
                (said, code) = RepliBuild.BuildBridge.execute(
                    "clang", vcat(["-shared", "-fPIC", c, "-o", out], args))
                code == 0 || println(said)
                return code == 0
            end
            # A stand-in for a libJLCS of another soname, in its own directory.
            function other_jlcs(sub::String, n::Int)
                d = mkpath(joinpath(dir, sub))
                lib = joinpath(d, "libJLCS.so.$n")
                ok = cc("int jlcs_other_$n(void) { return $n; }\n", lib, ["-Wl,-soname,libJLCS.so.$n"])
                return (ok, lib)
            end
            # A thunks library is linked with the libJLCS by path and its
            # directory as RUNPATH (ThunkBuilder.build_aot_thunks).
            thunks(name, source, jlcs) =
                cc(source, joinpath(dir, "lib$(name)_thunks.so"),
                   [String(jlcs), "-Wl,-rpath," * dirname(jlcs)])

            jlcs = RepliBuild.MLIRNative.libJLCS_path()
            (ok_old, old) = other_jlcs("old", 99)
            (ok_gone, gone) = other_jlcs("gone", 98)
            @test ok_old && ok_gone
            @test thunks("good", "int jlcs_abi_version(void);\nint good_abi(void) { return jlcs_abi_version(); }\n", jlcs)
            @test thunks("stale", "int jlcs_other_99(void);\nint stale(void) { return jlcs_other_99(); }\n", old)
            @test thunks("gone", "int jlcs_other_98(void);\nint gone(void) { return jlcs_other_98(); }\n", gone)
            # The fixtures are what they stand for: each records its libJLCS's soname.
            needed(name) = first(RepliBuild.BuildBridge.execute(
                "readelf", ["-d", joinpath(dir, "lib$(name)_thunks.so")]))
            @test occursin("[libJLCS.so.99]", needed("stale"))
            @test occursin("[libJLCS.so.98]", needed("gone"))
            @test occursin("[$(something(_snl_soname(jlcs), "no soname"))]", needed("good"))
            rm(gone)

            code = """
            using RepliBuild
            import Libdl
            JM = RepliBuild.JITManager
            dir = $(repr(dir))
            jlcs = RepliBuild.MLIRNative.libJLCS_path()
            lib(name) = joinpath(dir, "lib" * name * "_thunks.so")
            refusal(name) = try
                JM.open_thunks_library(lib(name))
                "opened"
            catch e
                sprint(showerror, e)
            end

            h = try
                JM.open_thunks_library(lib("good"))
            catch e
                println("good threw: ", sprint(showerror, e))
                C_NULL
            end
            println("PROBE good_opens: ", h != C_NULL ? "PASS" : "FAIL")
            got = h == C_NULL ? -1 : ccall(Libdl.dlsym(h, :good_abi), Cint, ())
            want = RepliBuild.JLCSLibrary.ABI_VERSION
            println("PROBE good_reaches_the_one_in_use: ", got == want ? "PASS" : "FAIL \$got, not \$want")
            println("PROBE good_is_one_copy: ",
                    isempty(JM._foreign_jlcs(jlcs)) ? "PASS" : "FAIL \$(JM._foreign_jlcs(jlcs))")

            said = refusal("stale")
            ok = occursin("another libJLCS", said) && occursin(joinpath(dir, "old", "libJLCS.so.99"), said) &&
                 occursin(lib("stale"), said) && occursin(jlcs, said) && occursin("RepliBuild.wrap", said)
            println("PROBE stale_is_refused: ", ok ? "PASS" : "FAIL " * replace(said, "\\n" => " | "))

            said = refusal("gone")
            ok = occursin("libJLCS.so.98", said) && occursin("RepliBuild.wrap", said) && occursin(jlcs, said)
            println("PROBE gone_says_wrap_again: ", ok ? "PASS" : "FAIL " * replace(said, "\\n" => " | "))

            # A library that is simply not there says nothing about wrapping
            # against another libJLCS: that advice is for the case it names.
            said = refusal("absent")
            ok = said != "opened" && !occursin("RepliBuild.wrap", said)
            println("PROBE absent_is_the_plain_error: ", ok ? "PASS" : "FAIL " * replace(said, "\\n" => " | "))
            println("PROBE_DONE")
            """
            r = run_isolated(`-e $code`; timeout = 300)
            v = probe_verdicts(r.output)
            @test r.ok
            @test haskey(v, "PROBE_DONE")
            cases = ("good_opens", "good_reaches_the_one_in_use", "good_is_one_copy",
                     "stale_is_refused", "gone_says_wrap_again", "absent_is_the_plain_error")
            for name in cases
                @test startswith(get(v, name, "no PROBE line"), "PASS")
            end
            @test count(k -> k != "PROBE_DONE", keys(v)) == length(cases)
        end
    end
end # requires
