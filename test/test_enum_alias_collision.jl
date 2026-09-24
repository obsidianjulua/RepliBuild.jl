#!/usr/bin/env julia
# test/test_enum_alias_collision.jl — two enums, one member set (2026-09-20,
# found by John auditing onednn 3.11.3's wrapper).
#
# DWARF records a std::map/unordered_map's member typedefs — `key_type`,
# `value_type`, `mapped_type` — as SEPARATE enum types carrying the underlying
# enum's enumerators verbatim. Both generators emitted both, and the later
# `@enum` rebound every member name. Julia permits that silently.
#
# The result was a module that parsed, loaded, and passed its smoke test while
# being wrong at the type level:
#
#     typeof(Onednn.dnnl_f32)          == key_type      (not dnnl_data_type_t)
#     typeof(Onednn.dnnl_convolution)  == value_type    (not dnnl_primitive_kind_t)
#     dnnl_memory_desc_create_with_tag(..., dnnl_f32, ...)
#       -> MethodError: Cannot convert an object of type key_type
#
# onednn's test_deep.jl was 15/15 throughout, because none of its four testsets
# passes a data type. That is the lesson worth keeping: a green suite bounds
# what was exercised, not what works.
#
# Identical member sets mean it IS the same enum under another spelling, so the
# later name is emitted as an ALIAS. The typedef stays resolvable for anything
# that references it, and the member bindings keep the canonical type.
#
# Library-free: drives both generators' emitters and EXECUTES what they produce.

using Test
using RepliBuild
using Libdl

const W = RepliBuild.Wrapper

# Any real loadable library satisfies the emitted module's load-time check;
# symbols it names are never called here. libjulia is present in every Julia
# process on every OS — the same portable anchor test_c_generator_policies uses.
const _LIBREF = abspath(first(filter(p -> occursin("libjulia", basename(p)), Libdl.dllist())))

# Julia 1.12+ has strict world-age semantics for global bindings: reading a
# binding created by `include_string` in the same world warns now and errors
# later. Every access to a probe module's bindings goes through this.
_get(m::Module, s::Symbol) = Base.invokelatest(getfield, m, s)
_has(m::Module, s::Symbol) = Base.invokelatest(isdefined, m, s)

function _enum_config(dir::String, lang::String)
    toml = joinpath(dir, "replibuild.toml")
    write(toml, """
    [project]
    name = "enumsynth"
    root = "$(escape_string(dir))"

    [link]
    enable_lto = false

    [wrap]
    language = "$lang"
    """)
    return RepliBuild.ConfigurationManager.load_config(toml)
end

# Two DWARF enum records with byte-identical enumerators, exactly as a
# std::map<dnnl_data_type_t, X> produces alongside the real enum.
function enum_fixture()
    ents = Any[Dict{String,Any}("name" => "dt_undef", "value" => 0),
               Dict{String,Any}("name" => "dt_f16",   "value" => 1),
               Dict{String,Any}("name" => "dt_f32",   "value" => 3)]
    Dict{String,Any}(
        "functions" => Any[],
        "globals" => Dict{String,Any}(),
        "struct_definitions" => Dict{String,Any}(
            "__enum__data_type_t" => Dict{String,Any}(
                "kind" => "enum", "underlying_type" => "unsigned int",
                "julia_type" => "UInt32", "enumerators" => deepcopy(ents)),
            "__enum__key_type" => Dict{String,Any}(
                "kind" => "enum", "underlying_type" => "unsigned int",
                "julia_type" => "UInt32", "enumerators" => deepcopy(ents)),
        ),
    )
end

# Emit just the enum section by running a generator over the fixture and
# keeping the lines that matter. Both generators are exercised because this is
# the failure class that recurs when only one of them gets a guard.
function enum_source(lang::Symbol; md=enum_fixture())
    dir = mktempdir()
    try
        cfg = _enum_config(dir, lang === :cpp ? "cpp" : "c")
        registry = W.create_type_registry(cfg)
        gen = lang === :cpp ? W.generate_introspective_module_cpp :
                              W.generate_introspective_module_c
        # Both entry points return (source, exported_names).
        out = gen(cfg, _LIBREF, md, "EnumSynth", registry, false)
        return out isa Tuple ? String(first(out)) : String(out)
    finally
        rm(dir; recursive=true, force=true)
    end
end

@testset "two enums, one member set" begin

    @testset "the collision is real without a guard" begin
        # Pin the Julia behaviour this depends on: a second @enum rebinding the
        # same member names WINS, silently. If Julia ever made that an error
        # the guard could be simplified, and this test would say so.
        m = Module(:EnumClobberProbe)
        Base.include_string(m, """
            @enum A::UInt32 dt_undef = 0 dt_f32 = 3
            @enum B::UInt32 dt_undef = 0 dt_f32 = 3
        """)
        @test _get(m, :dt_f32) isa _get(m, :B)
        @test !(_get(m, :dt_f32) isa _get(m, :A))
    end

    @testset "an alias keeps the member bindings on the canonical type" begin
        # What the generators now emit instead. `key_type` stays usable as a
        # type name, and dt_f32 keeps the canonical type so a ccall expecting
        # it converts.
        m = Module(:EnumAliasProbe)
        Base.include_string(m, """
            @enum data_type_t::UInt32 dt_undef = 0 dt_f32 = 3
            const key_type = data_type_t
        """)
        A = _get(m, :data_type_t)
        @test _get(m, :key_type) === A
        @test _get(m, :dt_f32) isa A
        @test Base.convert(A, _get(m, :dt_f32)) isa A
        @test Int(_get(m, :dt_f32)) == 3
    end

    @testset "both generators emit the alias, not a second @enum" begin
        for lang in (:cpp, :c)
            src = enum_source(lang)
            # Exactly one real @enum for the shared member set...
            n_enum = count(m -> true, eachmatch(r"^@enum\s+(data_type_t|key_type)\b"m, src))
            @test n_enum == 1
            # ...and the other name arrives as an alias.
            @test occursin(r"^const\s+(key_type|data_type_t)\s*=\s*(data_type_t|key_type)\s*$"m, src)
            # The member must be bound exactly once, or the clobber is back.
            @test count(m -> true, eachmatch(r"^\s*dt_f32\s*=\s*3\s*$"m, src)) == 1
        end
    end

    @testset "the emitted source loads and types the member correctly" begin
        for lang in (:cpp, :c)
            src = enum_source(lang)
            # Pull just the enum + alias lines; the rest of the module needs a
            # real library.
            lines = String[]
            keep = false
            for ln in eachsplit(src, '\n')
                if occursin(r"^@enum\s+(data_type_t|key_type)\b", ln); keep = true end
                keep && push!(lines, String(ln))
                if keep && strip(ln) == "end"; keep = false end
                occursin(r"^const\s+(key_type|data_type_t)\s*=", ln) && push!(lines, String(ln))
            end
            chunk = join(unique(lines), "\n")
            @test occursin("@enum", chunk)

            m = Module(Symbol("EnumEmitProbe_", lang))
            Base.include_string(m, chunk)
            canon = _has(m, :data_type_t) ? _get(m, :data_type_t) : nothing
            @test canon !== nothing
            @test _get(m, :dt_f32) isa canon
            if _has(m, :key_type)
                @test _get(m, :key_type) === canon
            end
        end
    end

    @testset "genuinely different enums are NOT aliased" begin
        # The guard keys on the member SET. Two enums that merely share a name
        # prefix, or share some members but not all, must both be emitted.
        md = enum_fixture()
        md["struct_definitions"]["__enum__key_type"]["enumerators"] =
            Any[Dict{String,Any}("name" => "other_a", "value" => 0),
                Dict{String,Any}("name" => "other_b", "value" => 1)]
        for lang in (:cpp, :c)
            s = enum_source(lang; md=md)
            @test occursin(r"^@enum\s+data_type_t\b"m, s)
            @test occursin(r"^@enum\s+key_type\b"m, s)
            @test !occursin(r"^const\s+key_type\s*=\s*data_type_t"m, s)
        end
    end
end
