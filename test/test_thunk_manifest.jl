# test_thunk_manifest.jl — the C++ generator writes thunk_manifest.json in JIT
# mode too, not only for AOT wraps.
#
# The manifest is the set of functions the wrapper dispatches through a Tier-2
# thunk. JITManager reads it to compile only those, and ThunkBuilder reads it to
# decide what to emit AOT. Only wrap can compute it.
#
# 1c4255b moved the write ahead of the AOT assert but inside the `aot_thunks`
# branch. Generator.jl turns a wrap whose thunks library is missing (a failed AOT
# link) into a JIT wrap, so that wrap recorded nothing and the next build emitted
# the same unfiltered thunk set and failed the same way.
#
# Library-free: synthetic metadata drives the generator directly.

using Test
using RepliBuild
using Libdl
import TOML

const _TM_W  = RepliBuild.Wrapper
const _TM_CM = RepliBuild.ConfigurationManager
# Any real file will do: the generator only names it and hashes it.
const _TM_LIB = abspath(first(filter(p -> occursin("libjulia", basename(p)), Libdl.dllist())))

# A C++ function that is not noexcept, which routes it to Tier 2.
function _tm_tier2_row(mangled::String, demangled::String)
    row = only(RepliBuild.Compiler.parse_function_signatures(
        [Dict{String,Any}("mangled" => mangled, "demangled" => demangled)]))
    row["parameters"]  = Dict{String,Any}[]
    row["return_type"] = Dict{String,Any}("julia_type" => "Cint", "c_type" => "int", "size" => 4)
    row["is_vararg"]   = false
    return row
end

@testset "thunk manifest is written in JIT mode" begin
    mktempdir() do dir
        toml = joinpath(dir, "replibuild.toml")
        open(toml, "w") do io
            TOML.print(io, Dict(
                "project" => Dict("name" => "manifestprobe", "root" => dir),
                "link"    => Dict("enable_lto" => false),
                "wrap"    => Dict("language" => "cpp"),
                "types"   => Dict("strictness" => "warn", "allow_unknown_structs" => true),
                "cache"   => Dict("enabled" => false)))
        end
        cfg = _TM_CM.load_config(toml)
        @test !cfg.compile.aot_thunks          # JIT is the default; this is the case under test

        md = Dict{String,Any}(
            "functions" => [_tm_tier2_row("_Z8rb_alphav", "rb_alpha()"),
                            _tm_tier2_row("_Z7rb_betav", "rb_beta()")],
            "globals" => Dict{String,Any}(),
            "struct_definitions" => Dict{String,Any}())
        (_, needed) = _TM_W.generate_introspective_module_cpp(
            cfg, _TM_LIB, md, "ManifestProbe", _TM_W.create_type_registry(cfg), false)

        # Both really went to Tier 2; otherwise an empty manifest would pass below.
        @test needed == Set(["_Z8rb_alphav", "_Z7rb_betav"])

        path = joinpath(_TM_CM.get_output_path(cfg), "thunk_manifest.json")
        @test isfile(path)
        if isfile(path)
            manifest = RepliBuild._read_json(path)
            @test sort(String.(manifest["function_thunks"])) == ["_Z7rb_betav", "_Z8rb_alphav"]
        end
    end
end
