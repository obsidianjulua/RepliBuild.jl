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

    @testset "a record's name is an MLIR string literal" begin
        # A record carries its C++ name inside `!jlcs.c_struct<"...">` and
        # `!llvm.struct<"...">`. clang spells a `(unsigned char)3` template
        # argument as `'\x03'`, and MLIR's lexer knows `\\`, `\"`, `\n`, `\t`
        # and `\NN` only: `\x` is "unknown escape in string literal", and the
        # parse failure took every thunk in the library with it (Boost.Charconv's
        # `cache_block_count_t<true, (unsigned char)'\x03'>`, 2026-10-31). The
        # name is escaped on the way in and read back on the way out, so
        # `mlir_string_literal`/`mlir_struct_name` are the one spelling of that.
        TU = RepliBuild.JLCSIRGenerator.TypeUtils
        raw = "cache_block_count_t<true, (unsigned char)'\\x03'>"
        lit = TU.mlir_string_literal(raw)
        @test lit == "\"cache_block_count_t<true, (unsigned char)'\\\\x03'>\""
        @test TU.mlir_string_literal("a\"b\\c") == "\"a\\\"b\\\\c\""
        @test TU.mlir_string_literal("plain") == "\"plain\""
        # the inverse, on a bare ref, an opaque body and a c_struct
        @test TU.mlir_struct_name(TU.mlir_struct_ref(raw)) == raw
        @test TU.mlir_struct_name("!llvm.struct<" * lit * ", opaque>") == raw
        @test TU.mlir_struct_name("!jlcs.c_struct<" * lit * ", [i8], [[0 : i64]], packed = true>") == raw
        @test TU.mlir_struct_name(TU.mlir_struct_ref("a\"b\\c")) == "a\"b\\c"
        @test TU.mlir_struct_name("i32") === nothing
        # and the emitted bodies carry the escaped name, never a bare `\x`
        structs = Dict{String,Any}(raw => Dict{String,Any}(
            "kind" => "struct", "byte_size" => "0x8",
            "members" => [mem("p", "void*", 0, 8)]))
        body = SG.get_struct_definition_string(raw, structs[raw], structs)
        @test startswith(body, "!jlcs.c_struct<")
        @test occursin("'\\\\x03'", body)
        @test !occursin(r"[^\\]\\x03", body)
        # a member whose type is such a record: the ref is escaped and the
        # layout pass still finds the record by its real name
        holder = Dict{String,Any}("kind" => "struct", "byte_size" => "0x8",
            "members" => [mem("c", raw, 0, 8)])
        structs["Holder"] = holder
        ref = SG._member_mlir_type(raw)
        @test ref == TU.mlir_struct_ref(raw)
        @test SG._mlir_layout(ref, structs) == (8, 8)

        # Every reader takes the name back through `mlir_struct_name`. The
        # member resolver and the measure side still read the escaped capture,
        # which misses the lookup, so a record holding such a member degraded
        # to bytes where a plain name embeds the body (2026-10-31 audit). A
        # name with a quote drives the readers that excluded `"` outright.
        quoted = "parse_number_helper<true, '\"'>"
        for nm in (raw, quoted)
            s2 = Dict{String,Any}(
                nm => Dict{String,Any}("kind" => "struct", "byte_size" => "0x8",
                                       "members" => [mem("p", "void*", 0, 8)]),
                "Plain<int>" => Dict{String,Any}("kind" => "struct", "byte_size" => "0x8",
                                                 "members" => [mem("p", "void*", 0, 8)]),
                "H" => Dict{String,Any}("kind" => "struct", "byte_size" => "0x10",
                                        "members" => [mem("c", nm, 0, 8), mem("tag", "int", 8, 4)]),
                "HP" => Dict{String,Any}("kind" => "struct", "byte_size" => "0x10",
                                         "members" => [mem("c", "Plain<int>", 0, 8), mem("tag", "int", 8, 4)]))
            body_h = SG.get_struct_definition_string("H", s2["H"], s2)
            body_p = SG.get_struct_definition_string("HP", s2["HP"], s2)
            @test replace(body_h, "\"H\"" => "\"HP\"") == body_p
            @test !occursin("!llvm.array<8 x i8>", body_h)
            # the body-with-name spelling is measured too
            @test SG._mlir_layout("!llvm.struct<" * TU.mlir_string_literal(nm) * ", (i32)>", s2) == (4, 4)
            # a packed body never leaks a bare named ref (MLIR rejects one
            # outside a recursive definition): a known non-packed member record
            # is its bytes there
            s3 = Dict{String,Any}(
                nm => Dict{String,Any}("kind" => "struct", "byte_size" => "0x8",
                                       "members" => [mem("x", "int", 0, 4)]),
                "PK" => Dict{String,Any}("kind" => "struct", "byte_size" => "0x9",
                                         "members" => [mem("b", "char", 0, 1), mem("q", nm, 1, 8)]))
            pk = SG._packed_literal_string("PK", s3["PK"], s3, Set{String}())
            @test !occursin(r"!llvm\.struct<\"(?:[^\"\\]|\\.)*\">", pk)
            @test pk == "!llvm.struct<packed (i8, !llvm.array<8 x i8>)>"
        end
    end

    @testset "two records never share an alias id" begin
        # The alias id replaces every character outside [A-Za-z0-9_] by `_`,
        # which is not injective: Boost.JSON's `parse_number_helper<true, '+'>`
        # and `parse_number_helper<true, '-'>` both became
        # `!Struct_parse_number_helper_true______`, and MLIR refused the module
        # ("redefinition of type alias id"), taking every thunk in the library
        # with it (2026-10-31). A changed name carries a digest of the real
        # one; a plain identifier keeps the alias it always had.
        TU = RepliBuild.JLCSIRGenerator.TypeUtils
        plus  = TU.alias_safe_name("parse_number_helper<true, '+'>")
        minus = TU.alias_safe_name("parse_number_helper<true, '-'>")
        @test plus != minus
        @test startswith(plus, "parse_number_helper_true_______") && occursin(r"_[0-9a-f]{12}$", plus)
        @test TU.alias_safe_name("Shape") == "Shape"
        @test TU.alias_safe_name("Pad<char, int>") == TU.alias_safe_name("Pad<char, int>")   # deterministic
        structs = Dict{String,Any}(
            "parse_number_helper<true, '+'>" => Dict{String,Any}("kind" => "struct", "byte_size" => "0x10",
                "members" => [mem("a", "void*", 0, 8), mem("b", "void*", 8, 8)]),
            "parse_number_helper<true, '-'>" => Dict{String,Any}("kind" => "struct", "byte_size" => "0x10",
                "members" => [mem("a", "void*", 0, 8), mem("b", "void*", 8, 8)]))
        (aliases, _) = SG.generate_struct_definitions(structs)
        ids = [m.match for m in eachmatch(r"^!Struct_[A-Za-z0-9_]+(?= =)"m, aliases)]
        @test length(ids) == 2
        @test allunique(ids)
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
