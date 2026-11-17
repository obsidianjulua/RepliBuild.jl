# test_jlcs_abi_live.jl — the dialect ABI check, on real libraries.
#
# The libJLCS this RepliBuild uses reports the version RepliBuild reads from
# src/mlir/ABI_VERSION. A library that reports another one, or none, is
# refused before anything is called in it: built here as a two-line C file,
# since the check asks only for `jlcs_abi_version`.

using Test
using RepliBuild
import Libdl

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

haskey(ENV, "REPLIBUILD_HOME") || (ENV["REPLIBUILD_HOME"] = mktempdir())

const _AVL = RepliBuild.JLCSLibrary

if requires("dialect ABI version, live", :libJLCS, :clang)
    @testset "dialect ABI version, live" begin
        @testset "the library in use reports this RepliBuild's version" begin
            loc = _AVL.locate(; install = false)
            @test loc.source !== :missing
            @test _AVL.library_abi_version(loc.path) == _AVL.ABI_VERSION
            @test _AVL.abi_problem(loc) === nothing
            @test _AVL.verify_abi!(loc) === nothing
            # And through the call path: `ccall` reaches the same answer.
            @test ccall((:jlcs_abi_version, RepliBuild.MLIRNative.libJLCS), Cint, ()) == _AVL.ABI_VERSION
        end

        mktempdir() do dir
            function fake(name::String, source::String)
                c = joinpath(dir, name * ".c")
                write(c, source)
                lib = joinpath(dir, "lib" * name * "." * Libdl.dlext)
                (out, code) = RepliBuild.BuildBridge.execute("clang", ["-shared", "-fPIC", c, "-o", lib])
                @test code == 0
                code == 0 || println(out)
                return lib
            end

            @testset "a library of another version is refused" begin
                other = _AVL.ABI_VERSION + 1
                lib = fake("jlcs_other", "int jlcs_abi_version(void) { return $other; }\n")
                @test _AVL.library_abi_version(lib) == other
                loc = _AVL.Location(lib, :override, "")
                msg = _AVL.abi_problem(loc)
                @test msg !== nothing && occursin("speaks dialect ABI $other", msg)
                err = try
                    _AVL.verify_abi!(loc); nothing
                catch e
                    sprint(showerror, e)
                end
                @test err !== nothing && occursin(lib, err)
                # Asked once per library: the verdict is the same the second time.
                @test_throws ErrorException _AVL.verify_abi!(loc)
            end

            @testset "a library with no version is refused" begin
                lib = fake("jlcs_unversioned", "int jlcs_has_pending_exception(void) { return 0; }\n")
                @test _AVL.library_abi_version(lib) == 0
                msg = _AVL.abi_problem(_AVL.Location(lib, :local, ""))
                @test msg !== nothing && occursin("has no dialect ABI version", msg)
            end

            @testset "a file that does not load has no version to compare" begin
                junk = joinpath(dir, "libjunk." * Libdl.dlext)
                write(junk, "not a library")
                @test _AVL.library_abi_version(junk) === nothing
                @test _AVL.abi_problem(_AVL.Location(junk, :override, "")) === nothing
            end
        end
    end
end # requires
