#!/usr/bin/env julia
# test/test_enumerator_values.jl — an enumerator's value survives the metadata
# round trip on every JSON major `Project.toml` allows (2026-11-17).
#
# The build writes an enumerator of a 64-bit unsigned enum as its decimal
# value (`0xffff000000000000` is `18446462598732840960`), and wrap reads the
# metadata back through `_read_json`. JSON 1.x returns a `BigInt` for that
# literal; JSON 0.21 returns an `Int64` that wrapped to `-281474976710656`.
# The wrapper then bound `Status(-281474976710656)` on an `OpenEnum{UInt64}`
# (an InexactError at load), and a flag set whose top bit is 63 stopped being
# a flag set. `test_wide_enums_live.jl` was green or red by which JSON the
# Manifest had resolved.
#
# `Wrapper._enumerator_value` reads every DWARF enumerator in both generators:
# a negative value of an 8-byte unsigned enum is the wrapped value, and comes
# back as the original. What is pinned here, with no toolchain: the rule at
# its boundaries, then the real round trip (metadata written as the build
# writes it, read as wrap reads it) through both generators, with a signed
# enum whose negative enumerators must stay negative.

using Test
using RepliBuild
using Libdl

const EV_W = RepliBuild.Wrapper
const EV_JSON = RepliBuild.JSON
const EV_TOP = Int128(0xffff000000000000)          # 18446462598732840960

_ev_g(m, s) = Base.invokelatest(getfield, m, s)

function _ev_config(lang::AbstractString)
    dir = mktempdir()
    write(joinpath(dir, "replibuild.toml"), """
    [project]
    name = "ev$(lang)"
    root = "$(escape_string(dir))"
    [wrap]
    language = "$(lang)"
    [types]
    strictness = "warn"
    allow_unknown_structs = true
    [cache]
    enabled = false
    """)
    return dir, RepliBuild.ConfigurationManager.load_config(joinpath(dir, "replibuild.toml"))
end

# The enums of test/wide_enum_test, plus two signed ones. Values are `Int128`,
# which is what `parse_dwarf_dump` holds for a `DW_AT_const_value`.
function _ev_metadata()
    ent(n, v) = Dict{String,Any}("name" => n, "value" => Int128(v))
    enum(c, j, ents) = Dict{String,Any}("kind" => "enum", "underlying_type" => c, "julia_type" => j,
                                        "byte_size" => sizeof(getfield(Base, Symbol(j))), "enumerators" => ents)
    return Dict{String,Any}(
        "functions" => Any[], "globals" => Dict{String,Any}(), "record_abi" => Dict{String,Any}(),
        "struct_definitions" => Dict{String,Any}(
            "__enum__Status" => enum("uint64_t", "UInt64", Any[
                ent("StatusIterationMask", EV_TOP), ent("StatusIterationShift", 48),
                ent("StatusSplitMask", 0x0000ffff00000000), ent("StatusSplitShift", 32),
                ent("StatusItemMask", 0x00000000ffffffff)]),
            "__enum__Wide" => enum("uint64_t", "UInt64", Any[
                ent("WideA", Int128(1) << 60), ent("WideB", Int128(1) << 61),
                ent("WideC", Int128(1) << 62), ent("WideTop", Int128(1) << 63)]),
            "__enum__Level" => enum("int64_t", "Int64", Any[
                ent("LevelFloor", typemin(Int64)), ent("LevelBelow", -5),
                ent("LevelZero", 0), ent("LevelCeil", typemax(Int64))]),
            "__enum__Sign" => enum("int", "Cint", Any[
                ent("SignMinus", -1), ent("SignNone", 0), ent("SignPlus", 1)])))
end

@testset "Enumerator values across the metadata round trip (JSON $(pkgversion(EV_JSON)))" begin

@testset "the rule at its boundaries" begin
    ev = EV_W._enumerator_value
    # What JSON 0.21 hands back for a value above typemax(Int64): the original
    # modulo 2^64. Under an 8-byte unsigned type that is undone exactly.
    @test ev(-281474976710656, "UInt64") == EV_TOP
    @test ev(typemin(Int64), "UInt64") == Int128(1) << 63
    @test ev(-1, "UInt64") == Int128(typemax(UInt64))
    @test ev(-1, "Culonglong") == Int128(typemax(UInt64))
    # What JSON 1.x hands back: already right, and the same type comes out.
    @test ev(big"18446462598732840960", "UInt64") == EV_TOP
    @test ev(big"18446462598732840960", "UInt64") isa Int128
    @test ev(-281474976710656, "UInt64") isa Int128
    # `unsigned long` is 8 bytes on LP64 and 4 on LLP64; the rule follows the type.
    @test ev(-1, "Culong") == (sizeof(Culong) == 8 ? Int128(typemax(UInt64)) : -1)

    # A signed enum's negative enumerator is a value, at every width.
    for u in ("Int64", "Clonglong", "Cint", "Int32", "Cshort", "Int8", "Clong")
        @test ev(-5, u) === -5
        @test ev(typemin(Int64), u) === typemin(Int64)
    end
    # Only a 64-bit unsigned enum has values the read can lose: a narrower
    # one's negative value is not this defect and is left to fail at load.
    for u in ("UInt32", "Cuint", "UInt16", "Cuchar", "UInt8")
        @test ev(-1, u) === -1
    end
    # Everything that fits stays an Int, whatever the read gave.
    @test ev(48, "UInt64") === 48
    @test ev(big(48), "UInt64") === 48
    @test ev(Int128(-5), "Int64") === -5
    @test ev(typemax(Int64), "UInt64") === typemax(Int64)
    # Never an exception: an unknown underlying type, a value that is no integer.
    @test ev(-1, "NoSuchType") === -1
    @test ev(-1, "") === -1
    @test ev(nothing, "UInt64") === nothing
    @test ev(1.5, "UInt64") === 1.5
    @test ev(big(1) << 200, "UInt64") == big(1) << 200
end

# The read's own answer, stated: the two behaviours the fix is written for. A
# third (a float, a string, an exception) must fail here, not in a wrapper.
@testset "what the read gives for a value above typemax(Int64)" begin
    mktempdir() do dir
        path = joinpath(dir, "compilation_metadata.json")
        open(io -> EV_JSON.print(io, _ev_metadata(), 2), path, "w")
        @test occursin("\"value\": 18446462598732840960", read(path, String))   # the file is right
        md = RepliBuild._read_json(path)
        status = md["struct_definitions"]["__enum__Status"]["enumerators"]
        raw = only(e["value"] for e in status if e["name"] == "StatusIterationMask")
        @test raw isa Integer
        @test raw == EV_TOP || raw == -281474976710656
        @info "JSON $(pkgversion(EV_JSON)) reads 18446462598732840960 as" raw typeof(raw)
        level = md["struct_definitions"]["__enum__Level"]["enumerators"]
        @test only(e["value"] for e in level if e["name"] == "LevelFloor") == typemin(Int64)
        @test only(e["value"] for e in level if e["name"] == "LevelBelow") == -5
    end
end

# Metadata written as `compile_project` writes it and read as `wrap_library`
# reads it, then each generator. The same assertions for both: two generators,
# one answer.
function _ev_round_trip()
    dir = mktempdir()
    path = joinpath(dir, "compilation_metadata.json")
    open(io -> EV_JSON.print(io, _ev_metadata(), 2), path, "w")
    return RepliBuild._read_json(path)
end

function _ev_check_text(code::AbstractString)
    @testset "emitted text" begin
        # Not a flag set (masks beside shifts): an open enum over UInt64.
        @test occursin("primitive type Status <: RepliBuild.OpenEnums.OpenEnum{UInt64} 64 end", code)
        @test occursin("const StatusIterationMask = Status(18446462598732840960)", code)
        @test occursin(":StatusIterationMask => 18446462598732840960,", code)
        @test occursin("const StatusSplitMask = Status(281470681743360)", code)
        # A flag set through bit 63 is still a flag set.
        @test occursin("\nconst Wide = UInt64\n", code)
        @test occursin("const WideTop = Wide(9223372036854775808)", code)
        @test occursin("const WideC = Wide(4611686018427387904)", code)
        # Signed enums: negative enumerators stay negative.
        @test occursin("primitive type Level <: RepliBuild.OpenEnums.OpenEnum{Int64} 64 end", code)
        @test occursin("const LevelFloor = Level(-9223372036854775808)", code)
        @test occursin("const LevelBelow = Level(-5)", code)
        @test occursin("const LevelCeil = Level(9223372036854775807)", code)
        @test occursin("primitive type Sign <: RepliBuild.OpenEnums.OpenEnum{Cint} 32 end", code)
        @test occursin("const SignMinus = Sign(-1)", code)
        # No enumerator of an unsigned enum is bound to a negative number.
        @test !occursin(r"= (Status|Wide)\(-", code)
        @test !occursin("-281474976710656", code)
    end
end

@testset "GeneratorC, from metadata read back" begin
    _, cfg = _ev_config("c")
    libref = Sys.iswindows() ? Libdl.dlpath("msvcrt") :
             abspath(first(filter(p -> occursin("libjulia", basename(p)), Libdl.dllist())))
    code = EV_W.generate_introspective_module_c(cfg, libref, _ev_round_trip(), "EvC",
                                                EV_W.create_type_registry(cfg), false)
    _ev_check_text(code)
    # And the wrapper loads with those values: `Status(-281474976710656)` was
    # an InexactError here.
    holder = Module(:EvCHolder)
    Base.include_string(holder, code)
    M = _ev_g(holder, :EvC)
    @test UInt64(_ev_g(M, :StatusIterationMask)) === 0xffff000000000000
    @test UInt64(_ev_g(M, :StatusItemMask)) === 0x00000000ffffffff
    @test _ev_g(M, :Wide) === UInt64
    @test _ev_g(M, :WideTop) === 0x8000000000000000
    @test (_ev_g(M, :WideA) | _ev_g(M, :WideTop)) === 0x9000000000000000
    @test Int64(_ev_g(M, :LevelFloor)) === typemin(Int64)
    @test Int64(_ev_g(M, :LevelBelow)) === -5
    @test Cint(_ev_g(M, :SignMinus)) === Cint(-1)
end

@testset "GeneratorCpp, from metadata read back" begin
    _, cfg = _ev_config("cpp")
    libref = abspath(first(filter(p -> occursin("libjulia", basename(p)), Libdl.dllist())))
    res = EV_W.generate_introspective_module_cpp(cfg, libref, _ev_round_trip(), "EvCpp",
                                                 EV_W.create_type_registry(cfg), false)
    _ev_check_text(res isa Tuple ? res[1] : res)
end

end
