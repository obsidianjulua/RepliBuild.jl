#!/usr/bin/env julia
# test/test_llp64_widths.jl — the two tiers must agree on how wide a C type is.
#
# A wrapper call crosses a boundary twice. The Wrapper tier decides what Julia
# type to `ccall` with; the IRGen tier decides what MLIR type the thunk reads.
# Nothing forces those two decisions to match — they are made in different
# modules, from different tables, and a disagreement produces no error at any
# stage. The thunk simply reads a different number of bytes than the caller
# wrote, and the value is wrong.
#
# C's `long` is where this actually bites, because it is the one integer whose
# width is not settled by the word size:
#
#   Win64  is LLP64 — long is 32 bits, pointers and long long are 64
#   Unix64 is LP64  — long is 64 bits
#
# The Wrapper tier got this right for free: it emits Julia's `Clong`, which is
# already Int32 on Windows and Int64 elsewhere. The IRGen producers hardcoded
# `i64` for `long` in two separate tables (TypeUtils.map_cpp_type and
# ArrayViewGen._AV_ELEM_MLIR), so on Windows every `long` argument and return
# was passed 64 bits wide into a thunk whose C side reads 32.
#
# So this does not test that `long` is 32 bits on Windows — that would just
# restate the fix. It tests the INVARIANT the fix exists to serve: for every
# spelling both tiers know, the two tiers agree on the width. That catches the
# next type someone adds to one table and not the other, on whichever platform
# the CI happens to run.

using Test
using RepliBuild

const _WU = RepliBuild.Wrapper
const _TU = RepliBuild.JLCSIRGenerator.TypeUtils
const _AV = RepliBuild.JLCSIRGenerator.ArrayViewGen

"Byte width of an MLIR integer/float type, or `nothing` if it is not a scalar."
function _mlir_width(t::AbstractString)
    m = match(r"^i(\d+)$", t)
    m !== nothing && return max(1, Int(ceil(parse(Int, m.captures[1]) / 8)))
    t == "f32" && return 4
    t == "f64" && return 8
    return nothing
end

"Byte width of a Julia type named by the wrapper, or `nothing` if not a bits type."
function _julia_width(name::AbstractString)
    try
        T = Core.eval(Base, Meta.parse(name))
        return (T isa DataType && isbitstype(T)) ? sizeof(T) : nothing
    catch
        return nothing
    end
end

@testset "LLP64: tiers agree on C type widths" begin

    # ── The fact itself, stated once ────────────────────────────────────────
    @testset "C_LONG_MLIR tracks the platform" begin
        @test RepliBuild.C_LONG_MLIR == (Sys.iswindows() ? "i32" : "i64")
        # Julia's own Clong is the authority the wrapper side rides on; if these
        # two ever disagree the whole premise below is void.
        @test _mlir_width(RepliBuild.C_LONG_MLIR) == sizeof(Clong)
    end

    # ── The cross-tier invariant ────────────────────────────────────────────
    @testset "IRGen and Wrapper widths match" begin
        dir = mktempdir()
        write(joinpath(dir, "replibuild.toml"), """
        [project]
        name = "widthprobe"
        root = "$(escape_string(dir))"

        [wrap]
        language = "c"

        [types]
        strictness = "permissive"
        allow_unknown_structs = true
        """)
        cfg = RepliBuild.ConfigurationManager.load_config(joinpath(dir, "replibuild.toml"))
        registry = _WU.create_type_registry(cfg)

        # Every scalar C spelling the wrapper's base table knows. Driven from the
        # registry rather than a list written here, so a type added to the
        # wrapper is covered the day it is added.
        compared = 0
        mismatches = String[]

        for (c_type, julia_name) in registry.base_types
            jw = _julia_width(julia_name)
            jw === nothing && continue          # Cvoid, aggregates, complex
            mlir = _TU.map_cpp_type(c_type)
            mw = _mlir_width(mlir)
            mw === nothing && continue          # pointer/struct/unmapped
            compared += 1
            mw == jw || push!(mismatches,
                "$c_type: IRGen $mlir ($mw bytes) vs Wrapper $julia_name ($jw bytes)")
        end

        isempty(mismatches) ||
            @info "cross-tier width disagreements:\n  " * join(mismatches, "\n  ")
        @test isempty(mismatches)

        # Assert the sweep ran. Driving the loop off a dict means a rename or a
        # restructure could empty it, and an empty loop passes while testing
        # nothing — the vacuously-green failure test_symbol_hygiene.jl records
        # against itself.
        @test compared >= 15
    end

    # ── The array-view producer reads from its own table ────────────────────
    @testset "ArrayViewGen agrees with map_cpp_type" begin
        # Two tables in the same tier, filled in by hand, for the same job. They
        # drifted on `long` and nothing noticed.
        compared = 0
        for (c_type, mlir) in _AV._AV_ELEM_MLIR
            other = _TU.map_cpp_type(c_type)
            _mlir_width(other) === nothing && continue
            compared += 1
            @test mlir == other
        end
        @test compared >= 15
    end

    # ── The spellings that must NOT move with the platform ──────────────────
    # `long long`, `size_t` and the fixed-width names are 64 bits on both LLP64
    # and LP64. The fix split `long` out of a branch that held all of these, so
    # this pins that the split took only what it was supposed to.
    @testset "64-bit spellings are platform-independent" begin
        for t in ("long long", "unsigned long long", "int64_t", "uint64_t",
                  "size_t", "Csize_t", "ptrdiff_t", "intptr_t", "uintptr_t")
            @test _TU.map_cpp_type(t) == "i64"
        end
        for t in ("int", "unsigned int", "int32_t", "uint32_t")
            @test _TU.map_cpp_type(t) == "i32"
        end
    end

    # ── `long` and its aliases move together ────────────────────────────────
    @testset "every long spelling agrees" begin
        for t in ("long", "unsigned long", "long int", "unsigned long int",
                  "Clong", "Culong")
            @test _TU.map_cpp_type(t) == RepliBuild.C_LONG_MLIR
        end
    end
end
