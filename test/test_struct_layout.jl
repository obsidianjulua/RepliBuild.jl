#!/usr/bin/env julia
# test/test_struct_layout.jl — the emitted MLIR struct body must have the byte
# size DWARF recorded for the C type (2026-08-05).
#
# StructGen used to close a non-packed struct with ONE trailing filler of
# `byte_size - sum(member sizes)`. That double-counts: LLVM inserts the interior
# alignment padding itself, and DWARF reports enum members with `size = 0`, so
# the filler paid a second time for gaps the natural layout had already paid
# for. Every non-packed struct with interior padding came out LARGER than the C
# type it models.
#
# It is not a cosmetic mismatch. A MEMORY-class struct return is stored straight
# into the caller's `Ref{T}` by the `llvm.emit_c_interface` wrapper, and `Ref{T}`
# is sized from the JULIA struct — the true `byte_size`. So a Tier-2 call
# returning one wrote past a live Julia object on every invocation. Measured on
# llama.cpp: `llama_context_params` emitted 200 bytes against a native 160, and
# `llama_context_default_params()` wrote 34 bytes past a 160-byte `Ref`. Every
# member offset held the RIGHT value, which is why it presented as intermittent
# corruption (a garbage read here, a SIGSEGV there) rather than as marshalling.
#
# These assertions need no toolchain: they measure the emitted type string.

using Test
using Logging
using RepliBuild

const SG = RepliBuild.JLCSIRGenerator.StructGen

# Size of a struct as MLIR/LLVM will lay it out. Measured on the alias-free
# spelling, which is the one `_mlir_layout` can resolve end to end.
emitted_size(name, structs) =
    SG._mlir_layout(SG.get_llvm_aligned_type_string(name, structs[name], structs), structs)

mem(name, ctype, off, size) = Dict{String,Any}(
    "name" => name, "c_type" => ctype, "offset" => off, "size" => size)

@testset "Emitted struct size matches DWARF byte_size" begin

    @testset "interior padding is not double-counted" begin
        # int@0, pad 4, ptr@8, int@16, 3 bools@20..22, tail pad → 24.
        # sum(member sizes) = 4+8+4+1+1+1 = 19, so the old rule appended a
        # 5-byte filler ON TOP of a body that already laid out to 24 → 32.
        structs = Dict{String,Any}("Gap" => Dict{String,Any}(
            "kind" => "struct", "byte_size" => "0x18",
            "members" => [mem("a", "int", 0, 4), mem("p", "void*", 8, 8),
                          mem("b", "int", 16, 4), mem("f1", "bool", 20, 1),
                          mem("f2", "bool", 21, 1), mem("f3", "bool", 22, 1)]))
        @test emitted_size("Gap", structs) == (24, 8)

        body = SG.get_struct_definition_string("Gap", structs["Gap"], structs)
        # The 4 bytes between `a` and `p` are explicit now, and there is no
        # oversized tail filler.
        @test occursin("!llvm.array<4 x i8>", body)
        @test !occursin("!llvm.array<5 x i8>", body)
    end

    @testset "size-0 enum members still occupy their slot" begin
        # DWARF reports an enum member with size 0. The old rule counted it as
        # contributing nothing AND let the emitted i32 take four bytes, so the
        # struct grew by 4 per enum. This is llama.cpp's `llama_model_params`
        # shape: two enums between `n_gpu_layers` and `main_gpu`.
        structs = Dict{String,Any}(
            "__enum__Mode" => Dict{String,Any}("kind" => "enum", "byte_size" => "0x4",
                                               "underlying_type" => "unsigned int"),
            "Params" => Dict{String,Any}(
                "kind" => "struct", "byte_size" => "0x18",
                "members" => [mem("p", "void*", 0, 8), mem("n", "int", 8, 4),
                              mem("mode", "Mode", 12, 0), mem("m", "int", 16, 4),
                              mem("flag", "bool", 20, 1)]))
        @test emitted_size("Params", structs) == (24, 8)
    end

    @testset "a struct that cannot be modelled degrades to its exact size" begin
        # Overlapping members (a union arm / bitfield shape) have no
        # field-by-field LLVM body. Correct SIZE still matters more than
        # addressable fields, because size is what the ABI is built on.
        structs = Dict{String,Any}("Overlap" => Dict{String,Any}(
            "kind" => "struct", "byte_size" => "0x10",
            "members" => [mem("a", "long", 0, 8), mem("b", "long", 0, 8),
                          mem("c", "long", 8, 8)]))
        # ...and it degrades SILENTLY at warn level. StructGen runs from
        # per-library JIT init, inside a generated wrapper's __init__ — the
        # CONSUMER'S load path. `using` an app on the llamacpp wrapper printed
        # eleven paragraphs about libstdc++ internals before doing anything.
        # The report is @debug now; JULIA_DEBUG=RepliBuild opts back in.
        body = @test_logs min_level = Logging.Warn SG.get_struct_definition_string(
            "Overlap", structs["Overlap"], structs)
        @test SG._mlir_layout(body, structs)[1] == 16
        @test "Overlap" in SG._LAYOUT_WARNED        # recorded without being printed
    end

    @testset "packed structs are untouched" begin
        # is_struct_packed means sum(member sizes) == byte_size, so the body is
        # exact by construction and travels the !jlcs.c_struct marshalling path.
        # The layout pass must not rewrite it.
        structs = Dict{String,Any}("Tight" => Dict{String,Any}(
            "kind" => "struct", "byte_size" => "0x10",
            "members" => [mem("a", "long", 0, 8), mem("b", "long", 8, 8)]))
        @test SG.is_struct_packed(structs["Tight"])
        body = SG.get_struct_definition_string("Tight", structs["Tight"], structs)
        @test startswith(body, "!jlcs.c_struct<")
        @test !occursin("!llvm.array<", body)
    end

    @testset "_mlir_layout mirrors the x86-64 rules it stands in for" begin
        s = Dict{String,Any}()
        @test SG._mlir_layout("i32", s)                       == (4, 4)
        @test SG._mlir_layout("i1", s)                        == (1, 1)
        @test SG._mlir_layout("!llvm.ptr", s)                 == (8, 8)
        @test SG._mlir_layout("!llvm.array<10 x i8>", s)      == (10, 1)
        @test SG._mlir_layout("!llvm.struct<(i8, i64)>", s)   == (16, 8)   # padded
        @test SG._mlir_layout("!llvm.struct<packed (i8, i64)>", s) == (9, 1)
        @test SG._mlir_layout("!llvm.struct<(i32, !llvm.struct<(i32, i32)>)>", s) == (12, 4)
        # Unmeasurable must be `nothing`, never a silent zero — treating an
        # unknown member as zero-sized is the arithmetic that caused this bug.
        @test SG._mlir_layout("!llvm.struct<\"Nope\", opaque>", s) === nothing
        @test SG._mlir_layout("!llvm.struct<\"Absent\">", s) === nothing
    end
end
