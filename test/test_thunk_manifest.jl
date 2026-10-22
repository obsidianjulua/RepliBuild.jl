# test_thunk_manifest.jl — the C++ generator writes thunk_manifest.json in JIT
# mode too, and an AOT wrap compiles its thunks from the set it just derived.
#
# The manifest is the set of functions the wrapper dispatches through a Tier-2
# thunk. JITManager reads it to compile only those. Only wrap can compute it.
#
# 1c4255b moved the write ahead of the AOT assert but inside the `aot_thunks`
# branch. Generator.jl turns a wrap whose thunks library is missing (a failed AOT
# link) into a JIT wrap, so that wrap recorded nothing and the next build emitted
# the same unfiltered thunk set and failed the same way.
#
# AOT thunks used to be compiled at the end of build, before any wrap had derived
# that set: the first build guessed it, the first wrap refused or fell back to
# JIT, and only a second build and wrap converged. They are compiled by wrap's
# last stage now. The testsets below hold the stage's contract with a stand-in
# stage; test_aot_thunks.jl (devtests) runs the real one.
#
# Library-free: synthetic metadata drives the generator directly.

using Test
using RepliBuild
using Libdl
import TOML
import JSON

const _TM_W  = RepliBuild.Wrapper
const _TM_CM = RepliBuild.ConfigurationManager
# Any real file will do: the generator only names it and hashes it.
const _TM_LIB = abspath(first(filter(p -> occursin("libjulia", basename(p)), Libdl.dllist())))
const _TM_THUNKS = Set(["_Z8rb_alphav", "_Z7rb_betav"])

# A C++ function that is not noexcept, which routes it to Tier 2.
function _tm_tier2_row(mangled::String, demangled::String)
    row = only(RepliBuild.Compiler.parse_function_signatures(
        [Dict{String,Any}("mangled" => mangled, "demangled" => demangled)]))
    row["parameters"]  = Dict{String,Any}[]
    row["return_type"] = Dict{String,Any}("julia_type" => "Cint", "c_type" => "int", "size" => 4)
    row["is_vararg"]   = false
    return row
end

function _tm_config(dir::String; aot::Bool=false)
    toml = joinpath(dir, "replibuild.toml")
    compile = aot ? Dict("aot_thunks" => true) : Dict{String,Any}()
    open(toml, "w") do io
        TOML.print(io, Dict(
            "project" => Dict("name" => "manifestprobe", "root" => dir),
            "compile" => compile,
            "link"    => Dict("enable_lto" => false),
            "wrap"    => Dict("language" => "cpp"),
            "types"   => Dict("strictness" => "warn", "allow_unknown_structs" => true),
            "cache"   => Dict("enabled" => false)))
    end
    return _TM_CM.load_config(toml)
end

_tm_metadata() = Dict{String,Any}(
    "functions" => [_tm_tier2_row("_Z8rb_alphav", "rb_alpha()"),
                    _tm_tier2_row("_Z7rb_betav", "rb_beta()")],
    "globals" => Dict{String,Any}(),
    "struct_definitions" => Dict{String,Any}())

_tm_manifest_set(path) = Set{String}(String.(RepliBuild._read_json(path)["function_thunks"]))

@testset "thunk manifest is written in JIT mode" begin
    mktempdir() do dir
        cfg = _tm_config(dir)
        @test !cfg.compile.aot_thunks          # JIT is the default; this is the case under test

        (_, needed) = _TM_W.generate_introspective_module_cpp(
            cfg, _TM_LIB, _tm_metadata(), "ManifestProbe", _TM_W.create_type_registry(cfg), false)

        # Both really went to Tier 2; otherwise an empty manifest would pass below.
        @test needed == _TM_THUNKS

        path = joinpath(_TM_CM.get_output_path(cfg), "thunk_manifest.json")
        @test isfile(path)
        if isfile(path)
            @test _tm_manifest_set(path) == _TM_THUNKS
        end
    end
end

@testset "the AOT stage gets the generator's thunk set, after the manifest" begin
    mktempdir() do dir
        cfg = _tm_config(dir; aot = true)
        @test cfg.compile.aot_thunks
        manifest = joinpath(_TM_CM.get_output_path(cfg), "thunk_manifest.json")

        calls = Set{String}[]
        on_disk = Ref{Any}(nothing)
        stage = function (needed)
            push!(calls, Set{String}(needed))
            on_disk[] = isfile(manifest) ? _tm_manifest_set(manifest) : nothing
            return false                       # wrote no library
        end
        @test_throws _TM_W.AOTStageFailed _TM_W.generate_introspective_module_cpp(
            cfg, _TM_LIB, _tm_metadata(), "ManifestProbe", _TM_W.create_type_registry(cfg),
            false, joinpath(dir, "libmanifestprobe_thunks." * Libdl.dlext); aot_stage = stage)

        @test length(calls) == 1               # once per wrap, not per call site
        @test !isempty(calls) && only(calls) == _TM_THUNKS
        @test on_disk[] == _TM_THUNKS          # the manifest was written first
    end
end

@testset "wrap_library: AOT is wrap's last stage, with a JIT fallback" begin
    # The default stage is ThunkBuilder's, looked up when the wrap runs.
    @test _TM_W._thunk_builder() === RepliBuild.ThunkBuilder

    mktempdir() do dir
        cfg = _tm_config(dir; aot = true)
        out = _TM_CM.get_output_path(cfg)
        mkpath(out)
        lib = joinpath(out, "libmanifestprobe." * Libdl.dlext)
        write(lib, "stand-in library")
        open(joinpath(out, "compilation_metadata.json"), "w") do io
            JSON.print(io, _tm_metadata())
        end
        thunks = joinpath(out, "libmanifestprobe_thunks." * Libdl.dlext)

        # A stage that writes no library: the wrap goes on with JIT dispatch.
        calls = Tuple{Any,String,Set{String}}[]
        failing(c, l; needed_symbols) = (push!(calls, (c, l, needed_symbols)); false)
        wrapper = @test_logs (:warn, r"AOT thunks were not compiled") match_mode=:any begin
            _TM_W.wrap_library(cfg, lib; aot_compile = failing)
        end
        @test length(calls) == 1
        if length(calls) == 1
            (c, l, needed) = only(calls)
            @test c.compile.aot_thunks && l == lib
            @test needed == _TM_THUNKS
        end
        text = read(wrapper, String)
        @test !occursin("invoke_aot_ptr", text)
        @test !occursin("THUNKS_HANDLE", text)
        @test _tm_manifest_set(joinpath(out, "thunk_manifest.json")) == _TM_THUNKS

        # A stage that reports success without the library: the export check
        # still refuses, and the JIT wrapper written above is left in place.
        before = read(wrapper, String)
        err = try
            _TM_W.wrap_library(cfg, lib; aot_compile = (c, l; needed_symbols) -> true)
            ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("thunks library is missing", err)
        @test occursin(basename(thunks), err)
        @test read(wrapper, String) == before
    end
end
