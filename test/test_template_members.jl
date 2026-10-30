#!/usr/bin/env julia
# test/test_template_members.jl — a record member whose type is a template
# instantiation is a record held by value, not a pointer (2026-10-30).
#
# `map_cpp_type` names a record only when its spelling is a bare identifier
# (`^[A-Za-z0-9_]+$`) and answers `!llvm.ptr` for anything else, so every
# template spelling a member could have (`Vec<int, 3>`, libc++'s
# `array<int, 3ULL>` and `pair<int, float>`) became 8 bytes in StructGen's
# bodies. `_sized_member_types` passed the member's DWARF size as
# `fallback_size`, but `_resolve_struct_member_type` only rewrites
# `!llvm.struct<"…">`, so the size never applied. A spelling with a `*` inside
# its arguments (`pair<int*, int>`) took the pointer branch the same way.
#
# Found on Windows with a probe library: `H2 { Vec<int, 3> a; int tag; }`
# emitted as `[!llvm.ptr, i32]`, 12 bytes against C's 16, and `Holder
# { array<int,3>; pair<int,float>; int tag; }` as 20 against 24. Returned by
# value through Tier 2, the thunk's body put `tag` four bytes early and the
# Julia value got 0 where the callee wrote 9 and 7, with no error. The same
# body sizes the callee's return buffer.
#
# These assertions need no toolchain: they measure the emitted type strings.

using Test
using RepliBuild

const SG = RepliBuild.JLCSIRGenerator.StructGen

mem(name, ctype, off, size) = Dict{String,Any}(
    "name" => name, "c_type" => ctype, "offset" => off, "size" => size)
rec(size, members) = Dict{String,Any}("kind" => "struct", "byte_size" => size, "members" => members)

# Size as the thunk sees it: the packed spelling declares the callee and the
# aligned one is what the thunk returns to Julia. Both must be the DWARF size.
packed_size(name, structs) =
    SG._mlir_layout(SG.get_llvm_equivalent_type_string(name, structs[name], structs), structs)[1]
aligned_size(name, structs) =
    SG._mlir_layout(SG.get_llvm_aligned_type_string(name, structs[name], structs), structs)[1]
bodies(name, structs) = (SG.get_llvm_equivalent_type_string(name, structs[name], structs),
                         SG.get_llvm_aligned_type_string(name, structs[name], structs),
                         SG.get_struct_definition_string(name, structs[name], structs))

@testset "Template-spelled members keep their size" begin

    @testset "a project template member is laid out from its definition" begin
        structs = Dict{String,Any}(
            "Vec<int, 3>" => rec("0xc", [mem("v", "int[3]", 0, 12)]),
            "H2" => rec("0x10", [mem("a", "Vec<int, 3>", 0, 12), mem("tag", "int", 12, 4)]))
        @test packed_size("H2", structs) == 16
        @test aligned_size("H2", structs) == 16
        for b in bodies("H2", structs)
            @test !occursin("!llvm.ptr", b)
        end
    end

    @testset "a toolchain template with no definition is its DWARF bytes" begin
        # libc++'s `array` and `pair` are dropped by the provenance gate, so
        # the table has no entry for them: the member's own size is all there is.
        structs = Dict{String,Any}(
            "Holder" => rec("0x18", [mem("xs", "array<int, 3ULL>", 0, 12),
                                     mem("kv", "pair<int, float>", 12, 8),
                                     mem("tag", "int", 20, 4)]))
        @test packed_size("Holder", structs) == 24
        @test aligned_size("Holder", structs) == 24
        for b in bodies("Holder", structs)
            @test !occursin("!llvm.ptr", b)
            @test occursin("!llvm.array<12 x i8>", b)
            @test occursin("!llvm.array<8 x i8>", b)
        end
    end

    @testset "a `*` inside the template arguments is not a pointer member" begin
        structs = Dict{String,Any}(
            "PtrArg" => rec("0x18", [mem("p", "pair<int*, int>", 0, 16), mem("tag", "int", 16, 4)]))
        @test packed_size("PtrArg", structs) == 24
        @test aligned_size("PtrArg", structs) == 24
        @test !occursin("!llvm.ptr", SG.get_llvm_equivalent_type_string("PtrArg", structs["PtrArg"], structs))
    end

    @testset "an array of records the table lacks keeps its extent" begin
        # The template spelling, a plain name, and two dimensions. An element
        # with no table entry used to degrade to one byte, so each of these
        # held one byte per record.
        for (ctype, size) in (("pair<int, int>[2]", 16), ("Opaque[2]", 16), ("pair<int, int>[2][3]", 48))
            structs = Dict{String,Any}(
                "Arr" => rec("0x$(string(size + 4; base = 16))",
                             [mem("ps", ctype, 0, size), mem("tag", "int", size, 4)]))
            @test packed_size("Arr", structs) == size + 4
            @test aligned_size("Arr", structs) == size + 4
            @test !occursin("!llvm.ptr", SG.get_llvm_equivalent_type_string("Arr", structs["Arr"], structs))
        end
    end

    @testset "a pointer to a template is still a pointer" begin
        structs = Dict{String,Any}(
            "Vec<int, 3>" => rec("0xc", [mem("v", "int[3]", 0, 12)]),
            "PtrT" => rec("0x10", [mem("p", "Vec<int, 3>*", 0, 8), mem("tag", "int", 8, 4)]))
        @test aligned_size("PtrT", structs) == 16
        @test occursin("!llvm.ptr", SG.get_llvm_aligned_type_string("PtrT", structs["PtrT"], structs))
    end

    @testset "a non-packed template member is an alias defined before its user" begin
        # `Pad<char, int>` has interior padding, so it is not packed: the member
        # is referenced as its `!Struct_` alias, which must already be defined
        # when the enclosing record's alias is. `H3` sorts before `Pad<…>`, so
        # only the dependency edge puts them in that order.
        structs = Dict{String,Any}(
            "Pad<char, int>" => rec("0x8", [mem("c", "char", 0, 1), mem("i", "int", 4, 4)]),
            "H3" => rec("0xc", [mem("p", "Pad<char, int>", 0, 8), mem("tag", "int", 8, 4)]))
        @test aligned_size("H3", structs) == 12
        (aliases, _) = SG.generate_struct_definitions(structs)
        pad_def = findfirst("!Struct_Pad_char__int_ =", aliases)
        h3_def = findfirst("!Struct_H3 =", aliases)
        @test pad_def !== nothing
        @test h3_def !== nothing
        if pad_def !== nothing && h3_def !== nothing
            @test first(pad_def) < first(h3_def)
            h3_line = split(aliases[first(h3_def):end], '\n')[1]
            @test occursin("!Struct_Pad_char__int_", h3_line)
            @test !occursin("!llvm.ptr", h3_line)
        end
    end
end
