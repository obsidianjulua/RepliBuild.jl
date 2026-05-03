#!/usr/bin/env julia
# test/dag_test/test_dag_diff.jl — DAG Diff Module Tests
#
# Comprehensive tests for the DAGDiff module: graph building, structural diff,
# transitive propagation, topo-sort, query API, and DOT visualization.
#
# Usage:  julia --project=. test/dag_test/test_dag_diff.jl
#
# No C++ toolchain required — uses synthetic metadata dicts.

using Test
using RepliBuild

const DD = RepliBuild.DAGDiff

# ── Test Metadata Builders ─────────────────────────────────────────────────

"""Aligned struct: all fields naturally aligned → no mismatch."""
function make_aligned_metadata()
    Dict(
        "struct_definitions" => Dict(
            "AlignedPoint" => Dict(
                "byte_size" => 8,
                "kind" => "struct",
                "members" => [
                    Dict("name" => "x", "c_type" => "float", "offset" => 0, "size" => 4),
                    Dict("name" => "y", "c_type" => "float", "offset" => 4, "size" => 4),
                ],
            ),
        ),
        "functions" => [
            Dict(
                "mangled" => "get_aligned",
                "name" => "get_aligned",
                "return_type" => Dict("c_type" => "AlignedPoint"),
                "parameters" => [],
            ),
            Dict(
                "mangled" => "use_aligned_ptr",
                "name" => "use_aligned_ptr",
                "return_type" => Dict("c_type" => "void"),
                "parameters" => [Dict("c_type" => "AlignedPoint*")],
            ),
        ],
    )
end

"""Packed struct: DWARF 5B vs Julia 8B (char + float with padding)."""
function make_packed_metadata()
    Dict(
        "struct_definitions" => Dict(
            "PackedPoint" => Dict(
                "byte_size" => 5,
                "kind" => "struct",
                "members" => [
                    Dict("name" => "flag", "c_type" => "char", "offset" => 0, "size" => 1),
                    Dict("name" => "value", "c_type" => "float", "offset" => 1, "size" => 4),
                ],
            ),
        ),
        "functions" => [
            Dict(
                "mangled" => "get_packed",
                "name" => "get_packed",
                "return_type" => Dict("c_type" => "PackedPoint"),
                "parameters" => [],
            ),
            Dict(
                "mangled" => "use_packed",
                "name" => "use_packed",
                "return_type" => Dict("c_type" => "void"),
                "parameters" => [Dict("c_type" => "PackedPoint")],
            ),
            Dict(
                "mangled" => "use_packed_ptr",
                "name" => "use_packed_ptr",
                "return_type" => Dict("c_type" => "void"),
                "parameters" => [Dict("c_type" => "PackedPoint*")],
            ),
        ],
    )
end

"""Containment chain: Inner (packed) → Outer (contains Inner by value)."""
function make_containment_metadata()
    Dict(
        "struct_definitions" => Dict(
            "Inner" => Dict(
                "byte_size" => 5,
                "kind" => "struct",
                "members" => [
                    Dict("name" => "flag", "c_type" => "char", "offset" => 0, "size" => 1),
                    Dict("name" => "value", "c_type" => "float", "offset" => 1, "size" => 4),
                ],
            ),
            "Outer" => Dict(
                "byte_size" => 13,
                "kind" => "struct",
                "members" => [
                    Dict("name" => "inner", "c_type" => "Inner", "offset" => 0, "size" => 5),
                    Dict("name" => "extra", "c_type" => "double", "offset" => 5, "size" => 8),
                ],
            ),
        ),
        "functions" => [
            Dict(
                "mangled" => "get_outer",
                "name" => "get_outer",
                "return_type" => Dict("c_type" => "Outer"),
                "parameters" => [],
            ),
            Dict(
                "mangled" => "get_inner",
                "name" => "get_inner",
                "return_type" => Dict("c_type" => "Inner"),
                "parameters" => [],
            ),
            Dict(
                "mangled" => "process",
                "name" => "process",
                "return_type" => Dict("c_type" => "void"),
                "parameters" => [Dict("c_type" => "Outer")],
            ),
        ],
    )
end

"""Deep containment: A → B → C, only C is packed."""
function make_deep_chain_metadata()
    Dict(
        "struct_definitions" => Dict(
            "C_packed" => Dict(
                "byte_size" => 3,
                "kind" => "struct",
                "members" => [
                    Dict("name" => "a", "c_type" => "char", "offset" => 0, "size" => 1),
                    Dict("name" => "b", "c_type" => "short", "offset" => 1, "size" => 2),
                ],
            ),
            "B_mid" => Dict(
                "byte_size" => 7,
                "kind" => "struct",
                "members" => [
                    Dict("name" => "c", "c_type" => "C_packed", "offset" => 0, "size" => 3),
                    Dict("name" => "x", "c_type" => "float", "offset" => 3, "size" => 4),
                ],
            ),
            "A_top" => Dict(
                "byte_size" => 15,
                "kind" => "struct",
                "members" => [
                    Dict("name" => "b", "c_type" => "B_mid", "offset" => 0, "size" => 7),
                    Dict("name" => "y", "c_type" => "double", "offset" => 7, "size" => 8),
                ],
            ),
        ),
        "functions" => [
            Dict(
                "mangled" => "make_a",
                "name" => "make_a",
                "return_type" => Dict("c_type" => "A_top"),
                "parameters" => [],
            ),
            Dict(
                "mangled" => "take_c",
                "name" => "take_c",
                "return_type" => Dict("c_type" => "void"),
                "parameters" => [Dict("c_type" => "C_packed")],
            ),
        ],
    )
end

"""Union type: size = max(members), all offsets 0."""
function make_union_metadata()
    Dict(
        "struct_definitions" => Dict(
            "MyUnion" => Dict(
                "byte_size" => 8,
                "kind" => "union",
                "members" => [
                    Dict("name" => "i", "c_type" => "int", "offset" => 0, "size" => 4),
                    Dict("name" => "d", "c_type" => "double", "offset" => 0, "size" => 8),
                ],
            ),
        ),
        "functions" => [
            Dict(
                "mangled" => "get_union",
                "name" => "get_union",
                "return_type" => Dict("c_type" => "MyUnion"),
                "parameters" => [],
            ),
        ],
    )
end

"""Multiple independent structs: one packed, one aligned, mixed functions."""
function make_mixed_metadata()
    Dict(
        "struct_definitions" => Dict(
            "Safe" => Dict(
                "byte_size" => 8,
                "kind" => "struct",
                "members" => [
                    Dict("name" => "x", "c_type" => "int", "offset" => 0, "size" => 4),
                    Dict("name" => "y", "c_type" => "float", "offset" => 4, "size" => 4),
                ],
            ),
            "Packed" => Dict(
                "byte_size" => 5,
                "kind" => "struct",
                "members" => [
                    Dict("name" => "tag", "c_type" => "char", "offset" => 0, "size" => 1),
                    Dict("name" => "val", "c_type" => "float", "offset" => 1, "size" => 4),
                ],
            ),
        ),
        "functions" => [
            Dict(
                "mangled" => "safe_fn",
                "name" => "safe_fn",
                "return_type" => Dict("c_type" => "Safe"),
                "parameters" => [],
            ),
            Dict(
                "mangled" => "packed_fn",
                "name" => "packed_fn",
                "return_type" => Dict("c_type" => "Packed"),
                "parameters" => [Dict("c_type" => "Safe")],
            ),
            Dict(
                "mangled" => "ptr_only",
                "name" => "ptr_only",
                "return_type" => Dict("c_type" => "Packed*"),
                "parameters" => [Dict("c_type" => "Safe*")],
            ),
        ],
    )
end

"""Empty metadata: no structs, no functions."""
function make_empty_metadata()
    Dict(
        "struct_definitions" => Dict(),
        "functions" => [],
    )
end

"""Struct with const/volatile qualifiers on member types."""
function make_qualified_metadata()
    Dict(
        "struct_definitions" => Dict(
            "QualStruct" => Dict(
                "byte_size" => 5,
                "kind" => "struct",
                "members" => [
                    Dict("name" => "flag", "c_type" => "const char", "offset" => 0, "size" => 1),
                    Dict("name" => "value", "c_type" => "volatile float", "offset" => 1, "size" => 4),
                ],
            ),
        ),
        "functions" => [
            Dict(
                "mangled" => "get_qual",
                "name" => "get_qual",
                "return_type" => Dict("c_type" => "const QualStruct"),
                "parameters" => [],
            ),
        ],
    )
end

"""Struct with hex-encoded sizes (DWARF often emits 0x...)."""
function make_hex_metadata()
    Dict(
        "struct_definitions" => Dict(
            "HexStruct" => Dict(
                "byte_size" => "0x10",
                "kind" => "struct",
                "members" => [
                    Dict("name" => "a", "c_type" => "double", "offset" => "0x0", "size" => "0x8"),
                    Dict("name" => "b", "c_type" => "double", "offset" => "0x8", "size" => "0x8"),
                ],
            ),
        ),
        "functions" => [],
    )
end

"""Polymorphic class with vtable."""
function make_vtable_metadata()
    Dict(
        "struct_definitions" => Dict(
            "Base" => Dict(
                "byte_size" => 16,
                "kind" => "class",
                "is_polymorphic" => true,
                "has_vtable" => true,
                "base_classes" => [],
                "members" => [
                    Dict("name" => "__vtable_ptr", "c_type" => "void*", "offset" => 0, "size" => 8),
                    Dict("name" => "id", "c_type" => "int", "offset" => 8, "size" => 4),
                ],
            ),
        ),
        "functions" => [
            Dict(
                "mangled" => "_ZN4Base5helloEv",
                "name" => "Base::hello",
                "return_type" => Dict("c_type" => "void"),
                "parameters" => [Dict("c_type" => "Base*")],
            ),
        ],
    )
end

"""Enum type (should be skipped by DAGDiff)."""
function make_enum_metadata()
    Dict(
        "struct_definitions" => Dict(
            "__enum__Color" => Dict(
                "byte_size" => 4,
                "kind" => "enum",
                "members" => [],
            ),
            "Pixel" => Dict(
                "byte_size" => 5,
                "kind" => "struct",
                "members" => [
                    Dict("name" => "r", "c_type" => "char", "offset" => 0, "size" => 1),
                    Dict("name" => "val", "c_type" => "float", "offset" => 1, "size" => 4),
                ],
            ),
        ),
        "functions" => [],
    )
end


# ============================================================================
# Test Suites
# ============================================================================

@testset "DAGDiff" begin

    # ── Graph Building ────────────────────────────────────────────────────

    @testset "build_cpp_graph" begin
        @testset "aligned struct" begin
            g = DD.build_cpp_graph(make_aligned_metadata())
            @test haskey(g.types, "AlignedPoint")
            t = g.types["AlignedPoint"]
            @test t.byte_size == 8
            @test length(t.members) == 2
            @test t.members[1].offset == 0
            @test t.members[2].offset == 4
            @test t.kind == :struct
            @test !t.has_vtable
        end

        @testset "packed struct" begin
            g = DD.build_cpp_graph(make_packed_metadata())
            t = g.types["PackedPoint"]
            @test t.byte_size == 5
            @test t.members[2].offset == 1  # float packed at offset 1
        end

        @testset "containment edges" begin
            g = DD.build_cpp_graph(make_containment_metadata())
            @test "Inner" in g.type_edges["Outer"]
            @test isempty(g.type_edges["Inner"])
        end

        @testset "function nodes" begin
            g = DD.build_cpp_graph(make_packed_metadata())
            @test haskey(g.functions, "get_packed")
            @test g.functions["get_packed"].return_type == "PackedPoint"
            @test haskey(g.functions, "use_packed")
            @test g.functions["use_packed"].param_types == ["PackedPoint"]
        end

        @testset "function → type edges (by-value)" begin
            g = DD.build_cpp_graph(make_packed_metadata())
            @test "PackedPoint" in g.func_type_edges["get_packed"]
            @test "PackedPoint" in g.func_type_edges["use_packed"]
        end

        @testset "pointer params don't create by-value edges" begin
            g = DD.build_cpp_graph(make_packed_metadata())
            @test isempty(g.func_type_edges["use_packed_ptr"])
        end

        @testset "enum types skipped" begin
            g = DD.build_cpp_graph(make_enum_metadata())
            @test !haskey(g.types, "__enum__Color")
            @test haskey(g.types, "Pixel")
        end

        @testset "hex-encoded sizes" begin
            g = DD.build_cpp_graph(make_hex_metadata())
            @test g.types["HexStruct"].byte_size == 16
            @test g.types["HexStruct"].members[2].offset == 8
        end

        @testset "vtable/polymorphic class" begin
            g = DD.build_cpp_graph(make_vtable_metadata())
            @test g.types["Base"].has_vtable == true
            @test g.types["Base"].kind == :class
        end

        @testset "empty metadata" begin
            g = DD.build_cpp_graph(make_empty_metadata())
            @test isempty(g.types)
            @test isempty(g.functions)
        end
    end

    @testset "build_julia_graph" begin
        @testset "aligned struct matches DWARF" begin
            g = DD.build_julia_graph(make_aligned_metadata())
            t = g.types["AlignedPoint"]
            @test t.byte_size == 8
            @test t.members[1].offset == 0
            @test t.members[2].offset == 4
        end

        @testset "packed struct gets Julia padding" begin
            g = DD.build_julia_graph(make_packed_metadata())
            t = g.types["PackedPoint"]
            # Julia: char(1) + pad(3) + float(4) = 8
            @test t.byte_size == 8
            @test t.members[1].offset == 0
            @test t.members[2].offset == 4  # padded from 1 → 4
        end

        @testset "union: all offsets 0, size = max member" begin
            g = DD.build_julia_graph(make_union_metadata())
            t = g.types["MyUnion"]
            @test t.byte_size == 8
            @test all(m.offset == 0 for m in t.members)
        end

        @testset "deep chain: each level gets Julia alignment" begin
            g = DD.build_julia_graph(make_deep_chain_metadata())
            c = g.types["C_packed"]
            # char(1) + pad(1) + short(2) = 4
            @test c.byte_size == 4
            @test c.members[2].offset == 2  # short aligned to 2
        end

        @testset "qualified types" begin
            g = DD.build_julia_graph(make_qualified_metadata())
            @test haskey(g.types, "QualStruct")
            t = g.types["QualStruct"]
            @test t.byte_size == 8  # same padding as PackedPoint
        end
    end

    # ── Structural Diff ──────────────────────────────────────────────────

    @testset "diff_graphs" begin
        @testset "aligned struct → no mismatches" begin
            meta = make_aligned_metadata()
            cpp = DD.build_cpp_graph(meta)
            jl = DD.build_julia_graph(meta)
            mismatches = DD.diff_graphs(cpp, jl)
            @test isempty(mismatches)
        end

        @testset "packed struct → layout mismatch" begin
            meta = make_packed_metadata()
            cpp = DD.build_cpp_graph(meta)
            jl = DD.build_julia_graph(meta)
            mismatches = DD.diff_graphs(cpp, jl)

            type_m = filter(m -> m.kind == DD.LAYOUT_MISMATCH, mismatches)
            @test !isempty(type_m)

            # Size mismatch: 5 DWARF vs 8 Julia → delta = -3
            size_m = filter(m -> m.delta != 0 && m.symbol == "PackedPoint", type_m)
            @test !isempty(size_m)
            @test size_m[1].delta == -3  # DWARF(5) - Julia(8)

            # Offset mismatch on 'value' member
            offset_m = filter(m -> m.dwarf_offset == UInt64(1), type_m)
            @test !isempty(offset_m)
        end

        @testset "packed return → RETURN_CONV_MISMATCH" begin
            meta = make_packed_metadata()
            cpp = DD.build_cpp_graph(meta)
            jl = DD.build_julia_graph(meta)
            mismatches = DD.diff_graphs(cpp, jl)

            ret_m = filter(m -> m.kind == DD.RETURN_CONV_MISMATCH, mismatches)
            @test !isempty(ret_m)
            @test ret_m[1].symbol == "get_packed"
        end

        @testset "packed param → PARAM_CONV_MISMATCH" begin
            meta = make_packed_metadata()
            cpp = DD.build_cpp_graph(meta)
            jl = DD.build_julia_graph(meta)
            mismatches = DD.diff_graphs(cpp, jl)

            param_m = filter(m -> m.kind == DD.PARAM_CONV_MISMATCH, mismatches)
            @test !isempty(param_m)
            @test param_m[1].symbol == "use_packed"
        end

        @testset "pointer to packed struct → no function mismatch" begin
            meta = make_packed_metadata()
            cpp = DD.build_cpp_graph(meta)
            jl = DD.build_julia_graph(meta)
            mismatches = DD.diff_graphs(cpp, jl)

            ptr_m = filter(m -> m.symbol == "use_packed_ptr", mismatches)
            @test isempty(ptr_m)
        end

        @testset "transitive propagation through containment" begin
            meta = make_containment_metadata()
            cpp = DD.build_cpp_graph(meta)
            jl = DD.build_julia_graph(meta)
            mismatches = DD.diff_graphs(cpp, jl)

            # Both Inner and Outer should be flagged as layout mismatches
            mismatched_types = Set(m.symbol for m in mismatches if m.kind == DD.LAYOUT_MISMATCH)
            @test "Inner" in mismatched_types
            @test "Outer" in mismatched_types

            # All functions touching these types should be flagged
            func_symbols = Set(m.symbol for m in mismatches if m.kind in (DD.RETURN_CONV_MISMATCH, DD.PARAM_CONV_MISMATCH))
            @test "get_outer" in func_symbols
            @test "get_inner" in func_symbols
            @test "process" in func_symbols
        end

        @testset "propagation-only (hand-crafted graphs)" begin
            # Construct graphs where Wrapper is NOT directly mismatched
            # but contains Inner which IS mismatched — tests Pass 2 path
            inner_cpp = DD.TypeNode("Inner", 5,
                [DD.MemberLayout("f", "char", 0, 1, 0, 0), DD.MemberLayout("v", "float", 1, 4, 0, 0)],
                :struct, false, String[], true)
            wrapper_cpp = DD.TypeNode("Wrapper", 16,
                [DD.MemberLayout("inner", "Inner", 0, 5, 0, 0), DD.MemberLayout("pad", "char", 5, 11, 0, 0)],
                :struct, false, String[], false)

            inner_jl = DD.TypeNode("Inner", 8,
                [DD.MemberLayout("f", "char", 0, 1, 0, 0), DD.MemberLayout("v", "float", 4, 4, 0, 0)],
                :struct, false, String[], false)
            # Wrapper: same size/offsets as cpp → no direct mismatch
            wrapper_jl = DD.TypeNode("Wrapper", 16,
                [DD.MemberLayout("inner", "Inner", 0, 5, 0, 0), DD.MemberLayout("pad", "char", 5, 11, 0, 0)],
                :struct, false, String[], false)

            cpp = DD.IRGraph(
                Dict("Inner" => inner_cpp, "Wrapper" => wrapper_cpp),
                Dict{String, DD.FunctionNode}(),
                Dict("Inner" => Set{String}(), "Wrapper" => Set(["Inner"])),
                Dict{String, Set{String}}(),
                Dict{String, Set{String}}())
            jl = DD.IRGraph(
                Dict("Inner" => inner_jl, "Wrapper" => wrapper_jl),
                Dict{String, DD.FunctionNode}(),
                Dict("Inner" => Set{String}(), "Wrapper" => Set(["Inner"])),
                Dict{String, Set{String}}(),
                Dict{String, Set{String}}())

            mismatches = DD.diff_graphs(cpp, jl)
            symbols = Set(m.symbol for m in mismatches)
            @test "Inner" in symbols    # direct mismatch
            @test "Wrapper" in symbols  # propagated only

            # Wrapper's mismatch should mention "contains"
            prop_m = filter(m -> m.symbol == "Wrapper" && contains(m.detail, "contains"), mismatches)
            @test !isempty(prop_m)
            @test contains(prop_m[1].detail, "Inner")
        end

        @testset "deep chain propagation A → B → C" begin
            meta = make_deep_chain_metadata()
            cpp = DD.build_cpp_graph(meta)
            jl = DD.build_julia_graph(meta)
            mismatches = DD.diff_graphs(cpp, jl)

            mismatched_types = Set(m.symbol for m in mismatches if m.kind == DD.LAYOUT_MISMATCH)
            @test "C_packed" in mismatched_types
            @test "B_mid" in mismatched_types
            @test "A_top" in mismatched_types
        end

        @testset "union → no size mismatch (Julia agrees)" begin
            meta = make_union_metadata()
            cpp = DD.build_cpp_graph(meta)
            jl = DD.build_julia_graph(meta)
            mismatches = DD.diff_graphs(cpp, jl)

            # Union size = max(members) = 8 on both sides
            type_m = filter(m -> m.kind == DD.LAYOUT_MISMATCH && m.delta != 0, mismatches)
            @test isempty(type_m)
        end

        @testset "mixed safe/unsafe: only unsafe flagged" begin
            meta = make_mixed_metadata()
            cpp = DD.build_cpp_graph(meta)
            jl = DD.build_julia_graph(meta)
            mismatches = DD.diff_graphs(cpp, jl)

            symbols = Set(m.symbol for m in mismatches)
            @test "Packed" in symbols
            @test "packed_fn" in symbols
            @test !("Safe" in symbols)
            @test !("safe_fn" in symbols)
            @test !("ptr_only" in symbols)
        end

        @testset "empty metadata → no mismatches" begin
            meta = make_empty_metadata()
            cpp = DD.build_cpp_graph(meta)
            jl = DD.build_julia_graph(meta)
            @test isempty(DD.diff_graphs(cpp, jl))
        end
    end

    # ── Topological Sort ─────────────────────────────────────────────────

    @testset "topo_sort_mismatches" begin
        @testset "types before functions" begin
            meta = make_packed_metadata()
            result = DD.dag_diff(meta)

            # In the lowering order, PackedPoint should come before get_packed/use_packed
            idx_type = findfirst(==("PackedPoint"), result.lowering_order)
            idx_get = findfirst(==("get_packed"), result.lowering_order)
            idx_use = findfirst(==("use_packed"), result.lowering_order)

            @test idx_type !== nothing
            @test idx_get !== nothing
            @test idx_use !== nothing
            @test idx_type < idx_get
            @test idx_type < idx_use
        end

        @testset "containment chain: Inner before Outer before functions" begin
            meta = make_containment_metadata()
            result = DD.dag_diff(meta)

            idx_inner = findfirst(==("Inner"), result.lowering_order)
            idx_outer = findfirst(==("Outer"), result.lowering_order)
            idx_get_outer = findfirst(==("get_outer"), result.lowering_order)
            idx_process = findfirst(==("process"), result.lowering_order)

            @test idx_inner !== nothing
            @test idx_outer !== nothing
            @test idx_inner < idx_outer
            if idx_get_outer !== nothing
                @test idx_outer < idx_get_outer
            end
            if idx_process !== nothing
                @test idx_outer < idx_process
            end
        end

        @testset "deep chain: C before B before A" begin
            meta = make_deep_chain_metadata()
            result = DD.dag_diff(meta)

            idx_c = findfirst(==("C_packed"), result.lowering_order)
            idx_b = findfirst(==("B_mid"), result.lowering_order)
            idx_a = findfirst(==("A_top"), result.lowering_order)

            @test idx_c !== nothing
            @test idx_b !== nothing
            @test idx_a !== nothing
            @test idx_c < idx_b
            @test idx_b < idx_a
        end

        @testset "empty mismatches → empty order" begin
            meta = make_aligned_metadata()
            cpp = DD.build_cpp_graph(meta)
            order = DD.topo_sort_mismatches(DD.DAGMismatch[], cpp)
            @test isempty(order)
        end
    end

    # ── Top-Level dag_diff ────────────────────────────────────────────────

    @testset "dag_diff end-to-end" begin
        @testset "result types populated" begin
            result = DD.dag_diff(make_containment_metadata())
            @test result isa DD.DAGDiffResult
            @test !isempty(result.mismatches)
            @test !isempty(result.lowering_order)
            @test !isempty(result.mismatched_types)
            @test !isempty(result.mismatched_functions)
            @test result.cpp_graph isa DD.IRGraph
            @test result.julia_graph isa DD.IRGraph
        end

        @testset "aligned-only → clean result" begin
            result = DD.dag_diff(make_aligned_metadata())
            @test isempty(result.mismatches)
            @test isempty(result.lowering_order)
            @test isempty(result.mismatched_types)
            @test isempty(result.mismatched_functions)
        end

        @testset "empty metadata → clean result" begin
            result = DD.dag_diff(make_empty_metadata())
            @test isempty(result.mismatches)
            @test isempty(result.lowering_order)
        end
    end

    # ── Query API: needs_dag_thunk ────────────────────────────────────────

    @testset "needs_dag_thunk" begin
        @testset "packed return by value → true" begin
            result = DD.dag_diff(make_packed_metadata())
            @test DD.needs_dag_thunk("get_packed", result) == true
            @test DD.needs_dag_thunk("use_packed", result) == true
        end

        @testset "pointer-only → false" begin
            result = DD.dag_diff(make_packed_metadata())
            @test DD.needs_dag_thunk("use_packed_ptr", result) == false
        end

        @testset "aligned function → false" begin
            result = DD.dag_diff(make_aligned_metadata())
            @test DD.needs_dag_thunk("get_aligned", result) == false
        end

        @testset "nothing result → false (backward compat)" begin
            @test DD.needs_dag_thunk("anything", nothing) == false
        end

        @testset "unknown symbol → false" begin
            result = DD.dag_diff(make_packed_metadata())
            @test DD.needs_dag_thunk("nonexistent_fn", result) == false
        end

        @testset "transitive: function using containing type" begin
            result = DD.dag_diff(make_containment_metadata())
            @test DD.needs_dag_thunk("get_outer", result) == true
            @test DD.needs_dag_thunk("get_inner", result) == true
            @test DD.needs_dag_thunk("process", result) == true
        end
    end

    # ── Phase 2: Bitfields, Packed, Inheritance ────────────────────────

    @testset "bitfield detection" begin
        @testset "bitfield member flags mismatch" begin
            meta = Dict(
                "struct_definitions" => Dict(
                    "Flags" => Dict(
                        "byte_size" => "4",
                        "kind" => "struct",
                        "members" => [
                            Dict("name" => "a", "c_type" => "unsigned int", "offset" => "0", "size" => "4",
                                 "bit_size" => 3, "data_bit_offset" => 0),
                            Dict("name" => "b", "c_type" => "unsigned int", "offset" => "0", "size" => "4",
                                 "bit_size" => 5, "data_bit_offset" => 3),
                        ]
                    )
                ),
                "functions" => [
                    Dict("mangled" => "_Z9get_flagsv", "name" => "get_flags",
                         "return_type" => Dict("c_type" => "Flags"), "parameters" => [])
                ]
            )
            result = DD.dag_diff(meta)
            @test "Flags" in result.mismatched_types
            bf_mismatches = filter(m -> contains(m.detail, "bitfield"), result.mismatches)
            @test length(bf_mismatches) >= 1
            @test contains(bf_mismatches[1].detail, "bits")
        end

        @testset "non-bitfield struct unaffected" begin
            meta = Dict(
                "struct_definitions" => Dict(
                    "Normal" => Dict(
                        "byte_size" => "8",
                        "kind" => "struct",
                        "members" => [
                            Dict("name" => "x", "c_type" => "int", "offset" => "0", "size" => "4"),
                            Dict("name" => "y", "c_type" => "int", "offset" => "4", "size" => "4"),
                        ]
                    )
                ),
                "functions" => []
            )
            result = DD.dag_diff(meta)
            bf_mismatches = filter(m -> contains(m.detail, "bitfield"), result.mismatches)
            @test isempty(bf_mismatches)
        end
    end

    @testset "packed struct detection" begin
        @testset "stealth packed — all 1-byte fields" begin
            # All members are char (1 byte), so size matches, but layout is packed
            # with a 4-byte int crammed at offset 1 (violates 4-byte alignment)
            meta = Dict(
                "struct_definitions" => Dict(
                    "StealthPacked" => Dict(
                        "byte_size" => "5",
                        "kind" => "struct",
                        "members" => [
                            Dict("name" => "tag", "c_type" => "char", "offset" => "0", "size" => "1"),
                            Dict("name" => "val", "c_type" => "int", "offset" => "1", "size" => "4"),
                        ]
                    )
                ),
                "functions" => []
            )
            cpp = DD.build_cpp_graph(meta)
            @test cpp.types["StealthPacked"].is_packed == true

            result = DD.dag_diff(meta)
            @test "StealthPacked" in result.mismatched_types
        end
    end

    @testset "inheritance propagation" begin
        @testset "derived from packed base gets flagged" begin
            meta = Dict(
                "struct_definitions" => Dict(
                    "PackedBase" => Dict(
                        "byte_size" => "5",
                        "kind" => "struct",
                        "members" => [
                            Dict("name" => "a", "c_type" => "char", "offset" => "0", "size" => "1"),
                            Dict("name" => "b", "c_type" => "float", "offset" => "1", "size" => "4"),
                        ]
                    ),
                    "Derived" => Dict(
                        "byte_size" => "9",
                        "kind" => "class",
                        "members" => [
                            Dict("name" => "extra", "c_type" => "int", "offset" => "5", "size" => "4"),
                        ],
                        "base_classes" => [Dict("type" => "PackedBase", "accessibility" => "public")]
                    )
                ),
                "functions" => [
                    Dict("mangled" => "_Z10use_derivedv", "name" => "use_derived",
                         "return_type" => Dict("c_type" => "Derived"), "parameters" => [])
                ]
            )
            result = DD.dag_diff(meta)
            @test "PackedBase" in result.mismatched_types
            @test "Derived" in result.mismatched_types  # propagated via inheritance
            @test "_Z10use_derivedv" in result.mismatched_functions
        end

        @testset "derived from aligned base — size mismatch expected" begin
            # Derived classes always mismatch: DWARF byte_size includes base storage
            # but Julia's alignment model doesn't see the base class.  This is correct
            # behavior — the inheritance edge ensures the derived class gets a thunk.
            meta = Dict(
                "struct_definitions" => Dict(
                    "AlignedBase" => Dict(
                        "byte_size" => "8",
                        "kind" => "struct",
                        "members" => [
                            Dict("name" => "x", "c_type" => "int", "offset" => "0", "size" => "4"),
                            Dict("name" => "y", "c_type" => "int", "offset" => "4", "size" => "4"),
                        ]
                    ),
                    "CleanDerived" => Dict(
                        "byte_size" => "12",
                        "kind" => "class",
                        "members" => [
                            Dict("name" => "z", "c_type" => "int", "offset" => "8", "size" => "4"),
                        ],
                        "base_classes" => [Dict("type" => "AlignedBase", "accessibility" => "public")]
                    )
                ),
                "functions" => []
            )
            result = DD.dag_diff(meta)
            # CleanDerived is flagged because DWARF byte_size=12 (includes base)
            # vs Julia aligned size=4 (only sees declared member z).  This correctly
            # routes the derived class to a thunk.
            @test "CleanDerived" in result.mismatched_types
            # AlignedBase itself should NOT be mismatched (it's naturally aligned)
            @test !("AlignedBase" in result.mismatched_types)
        end

        @testset "base_classes Dict parsing" begin
            meta = Dict(
                "struct_definitions" => Dict(
                    "Base" => Dict(
                        "byte_size" => "4",
                        "kind" => "struct",
                        "members" => [Dict("name" => "x", "c_type" => "int", "offset" => "0", "size" => "4")]
                    ),
                    "Child" => Dict(
                        "byte_size" => "8",
                        "kind" => "class",
                        "members" => [Dict("name" => "y", "c_type" => "int", "offset" => "4", "size" => "4")],
                        "base_classes" => [Dict("type" => "Base", "accessibility" => "public")]
                    )
                ),
                "functions" => []
            )
            cpp = DD.build_cpp_graph(meta)
            # base_classes should contain "Base", not a stringified Dict
            @test cpp.types["Child"].base_classes == ["Base"]
            # Inheritance edge should exist: Child → Base (via type_edges)
            @test "Base" in cpp.type_edges["Child"]
            # Inheritance_edges should track it separately for visualization
            @test haskey(cpp.inheritance_edges, "Child")
            @test "Base" in cpp.inheritance_edges["Child"]
        end
    end

    @testset "vtable vptr synthesis" begin
        @testset "synthesizes vptr when missing" begin
            meta = Dict(
                "struct_definitions" => Dict(
                    "Polymorphic" => Dict(
                        "byte_size" => "16",
                        "kind" => "class",
                        "is_polymorphic" => true,
                        "members" => [
                            Dict("name" => "x", "c_type" => "int", "offset" => "8", "size" => "4"),
                        ]
                    )
                ),
                "functions" => []
            )
            cpp = DD.build_cpp_graph(meta)
            members = cpp.types["Polymorphic"].members
            @test members[1].name == "__vtable_ptr"
            @test members[1].size == 8
            @test members[1].offset == 0
        end

        @testset "does not duplicate existing vptr" begin
            meta = Dict(
                "struct_definitions" => Dict(
                    "HasVptr" => Dict(
                        "byte_size" => "16",
                        "kind" => "class",
                        "has_vtable" => true,
                        "members" => [
                            Dict("name" => "__vtable_ptr", "c_type" => "void *", "offset" => "0", "size" => "8"),
                            Dict("name" => "x", "c_type" => "int", "offset" => "8", "size" => "4"),
                        ]
                    )
                ),
                "functions" => []
            )
            cpp = DD.build_cpp_graph(meta)
            vptrs = filter(m -> m.name == "__vtable_ptr", cpp.types["HasVptr"].members)
            @test length(vptrs) == 1
        end
    end

    @testset "inheritance edges in DOT" begin
        meta = Dict(
            "struct_definitions" => Dict(
                "Animal" => Dict(
                    "byte_size" => "4",
                    "kind" => "struct",
                    "members" => [Dict("name" => "legs", "c_type" => "int", "offset" => "0", "size" => "4")]
                ),
                "Dog" => Dict(
                    "byte_size" => "8",
                    "kind" => "class",
                    "members" => [Dict("name" => "tricks", "c_type" => "int", "offset" => "4", "size" => "4")],
                    "base_classes" => [Dict("type" => "Animal", "accessibility" => "public")]
                )
            ),
            "functions" => []
        )
        result = DD.dag_diff(meta)
        dot = DD._generate_dot(result; side=:diff)
        @test contains(dot, "inherits")
        @test contains(dot, "steelblue") || contains(dot, "blue")
    end

    @testset "show_only_mismatches filter" begin
        meta = make_mixed_metadata()
        result = DD.dag_diff(meta)
        dot_all = DD._generate_dot(result; side=:diff)
        dot_filtered = DD._generate_dot(result; side=:diff, show_only_mismatches=true)

        # Filtered DOT should be shorter (fewer nodes)
        @test length(dot_filtered) < length(dot_all)
        # Mismatched types should still be present
        for name in result.mismatched_types
            @test contains(dot_filtered, escape_string(name))
        end
    end

    @testset "namespace clustering" begin
        meta = Dict(
            "struct_definitions" => Dict(
                "mylib::Point" => Dict(
                    "byte_size" => "8",
                    "kind" => "struct",
                    "members" => [
                        Dict("name" => "x", "c_type" => "int", "offset" => "0", "size" => "4"),
                        Dict("name" => "y", "c_type" => "int", "offset" => "4", "size" => "4"),
                    ]
                ),
                "mylib::Line" => Dict(
                    "byte_size" => "16",
                    "kind" => "struct",
                    "members" => [
                        Dict("name" => "start", "c_type" => "mylib::Point", "offset" => "0", "size" => "8"),
                        Dict("name" => "end_", "c_type" => "mylib::Point", "offset" => "8", "size" => "8"),
                    ]
                ),
                "Standalone" => Dict(
                    "byte_size" => "4",
                    "kind" => "struct",
                    "members" => [Dict("name" => "v", "c_type" => "int", "offset" => "0", "size" => "4")]
                )
            ),
            "functions" => []
        )
        result = DD.dag_diff(meta)
        dot = DD._generate_dot(result; side=:diff)
        @test contains(dot, "subgraph cluster_mylib")
        @test contains(dot, "label=\"mylib\"")
        # Standalone should NOT be in a cluster
        @test !contains(dot, "cluster_Standalone")
    end

    # ── Internal Helpers ──────────────────────────────────────────────────

    @testset "internal helpers" begin
        @testset "_parse_size" begin
            @test DD._parse_size(42) == 42
            @test DD._parse_size("16") == 16
            @test DD._parse_size("0x10") == 16
            @test DD._parse_size("0X1A") == 26
            @test DD._parse_size("") == 0
            @test DD._parse_size("garbage") == 0
            @test DD._parse_size(nothing) == 0
        end

        @testset "_strip_qualifiers" begin
            @test DD._strip_qualifiers("const int") == "int"
            @test DD._strip_qualifiers("volatile float") == "float"
            @test DD._strip_qualifiers("const volatile char") == "char"
            @test DD._strip_qualifiers("int") == "int"
        end

        @testset "_is_indirection" begin
            @test DD._is_indirection("int*") == true
            @test DD._is_indirection("const char*") == true
            @test DD._is_indirection("int&") == true
            @test DD._is_indirection("int") == false
            @test DD._is_indirection("float") == false
        end

        @testset "_transitive_closure" begin
            edges = Dict(
                "A" => Set(["B"]),
                "B" => Set(["C"]),
                "C" => Set{String}(),
            )
            closure = DD._transitive_closure(Set(["A"]), edges)
            @test "A" in closure
            @test "B" in closure
            @test "C" in closure

            # Single node, no deps
            closure2 = DD._transitive_closure(Set(["C"]), edges)
            @test closure2 == Set(["C"])

            # Empty seed
            @test isempty(DD._transitive_closure(Set{String}(), edges))
        end
    end

    # ── DOT Export ────────────────────────────────────────────────────────

    @testset "DOT visualization" begin
        @testset "export_dot diff mode" begin
            result = DD.dag_diff(make_containment_metadata())
            mktempdir() do dir
                dot_path = joinpath(dir, "test.dot")
                DD.export_dot(result, dot_path)
                @test isfile(dot_path)
                content = read(dot_path, String)

                # Basic structure
                @test contains(content, "digraph DAGDiff")
                @test contains(content, "rankdir=BT")

                # Legend present in diff mode
                @test contains(content, "cluster_legend")
                @test contains(content, "Red = layout mismatch")

                # Type nodes
                @test contains(content, "t:Inner")
                @test contains(content, "t:Outer")

                # Function nodes
                @test contains(content, "f:get_outer")
                @test contains(content, "f:get_inner")

                # Mismatch coloring
                @test contains(content, "red3")
                @test contains(content, "darkorange")

                # Containment edge
                @test contains(content, "contains")
            end
        end

        @testset "export_dot cpp-only mode (no legend)" begin
            result = DD.dag_diff(make_containment_metadata())
            mktempdir() do dir
                dot_path = joinpath(dir, "cpp.dot")
                DD.export_dot(result, dot_path; side=:cpp)
                content = read(dot_path, String)
                @test !contains(content, "cluster_legend")
            end
        end

        @testset "export_dot julia-only mode" begin
            result = DD.dag_diff(make_containment_metadata())
            mktempdir() do dir
                dot_path = joinpath(dir, "julia.dot")
                DD.export_dot(result, dot_path; side=:julia)
                content = read(dot_path, String)
                @test contains(content, "digraph DAGDiff")
                # Julia graph should still have the type nodes
                @test contains(content, "t:Inner")
            end
        end

        @testset "export_dot no members" begin
            result = DD.dag_diff(make_packed_metadata())
            mktempdir() do dir
                dot_path = joinpath(dir, "no_members.dot")
                DD.export_dot(result, dot_path; show_members=false)
                content = read(dot_path, String)
                # Should not contain member offset lines
                @test !contains(content, "+0 flag")
                @test !contains(content, "+1 value")
            end
        end

        @testset "export_dot aligned struct (no red)" begin
            result = DD.dag_diff(make_aligned_metadata())
            mktempdir() do dir
                dot_path = joinpath(dir, "aligned.dot")
                DD.export_dot(result, dot_path)
                content = read(dot_path, String)
                @test !contains(content, "fillcolor")
                @test contains(content, "gray60")
            end
        end

        @testset "export_graph_dot" begin
            g = DD.build_cpp_graph(make_containment_metadata())
            mktempdir() do dir
                dot_path = joinpath(dir, "single.dot")
                DD.export_graph_dot(g, dot_path)
                @test isfile(dot_path)
                content = read(dot_path, String)
                @test contains(content, "digraph")
                @test !contains(content, "cluster_legend")
            end
        end

        @testset "export_dot escapes special chars" begin
            # Manually test escape_dot
            @test DD.escape_dot("A<B>C") == "A&lt;B&gt;C"
            @test DD.escape_dot("a&b") == "a&amp;b"
            @test DD.escape_dot("x\"y") == "x&quot;y"
        end

        @testset "render_dot SVG" begin
            result = DD.dag_diff(make_containment_metadata())
            mktempdir() do dir
                svg_path = joinpath(dir, "test.svg")
                out = DD.render_dot(result, svg_path)
                if Sys.which("dot") !== nothing
                    @test isfile(out)
                    @test endswith(out, ".svg")
                    svg_content = read(out, String)
                    @test contains(svg_content, "<svg")
                    @test contains(svg_content, "Inner")
                    @test contains(svg_content, "Outer")
                else
                    # Falls back to DOT file
                    @test endswith(out, ".dot")
                    @test isfile(out)
                end
            end
        end

        @testset "render_dot PNG" begin
            result = DD.dag_diff(make_packed_metadata())
            mktempdir() do dir
                png_path = joinpath(dir, "test.png")
                out = DD.render_dot(result, png_path; format="png")
                if Sys.which("dot") !== nothing
                    @test isfile(out)
                    @test endswith(out, ".png")
                    @test filesize(out) > 0
                else
                    @test endswith(out, ".dot")
                end
            end
        end
    end

    # ── Interactive HTML Viewer ───────────────────────────────────────────

    @testset "render_html" begin
        result = DD.dag_diff(make_packed_metadata())

        @testset "generates valid HTML" begin
            mktempdir() do dir
                html_path = joinpath(dir, "test.html")
                out = DD.render_html(result, html_path)
                @test isfile(out)
                @test endswith(out, ".html")
                html = read(out, String)
                @test contains(html, "<!DOCTYPE html>")
                @test contains(html, "<svg")
                @test contains(html, "traceChain")
                @test contains(html, "mismatches")
            end
        end

        @testset "embeds stats" begin
            mktempdir() do dir
                out = DD.render_html(result, joinpath(dir, "stats.html"))
                html = read(out, String)
                @test contains(html, "types")
                @test contains(html, "functions")
                @test contains(html, "thunks")
            end
        end

        @testset "respects side parameter" begin
            mktempdir() do dir
                for s in (:diff, :cpp, :julia)
                    out = DD.render_html(result, joinpath(dir, "$(s).html"); side=s)
                    @test isfile(out)
                    html = read(out, String)
                    @test contains(html, "<svg")
                    @test contains(html, "traceChain")
                end
            end
        end
    end

    # ── Mismatch Detail Quality ──────────────────────────────────────────

    @testset "mismatch detail strings" begin
        result = DD.dag_diff(make_packed_metadata())

        @testset "layout mismatch detail" begin
            layout_m = filter(m -> m.kind == DD.LAYOUT_MISMATCH && m.delta != 0, result.mismatches)
            @test !isempty(layout_m)
            @test contains(layout_m[1].detail, "DWARF")
            @test contains(layout_m[1].detail, "Julia")
        end

        @testset "return conv detail" begin
            ret_m = filter(m -> m.kind == DD.RETURN_CONV_MISMATCH, result.mismatches)
            @test !isempty(ret_m)
            @test contains(ret_m[1].detail, "returns mismatched type")
        end

        @testset "param conv detail" begin
            param_m = filter(m -> m.kind == DD.PARAM_CONV_MISMATCH, result.mismatches)
            @test !isempty(param_m)
            @test contains(param_m[1].detail, "param")
            @test contains(param_m[1].detail, "mismatched type")
        end

        @testset "containment propagation detail" begin
            # Outer has a direct size mismatch in containment metadata (13 vs 16),
            # so it won't have a "contains" propagation message. Test that the
            # direct mismatch detail is correct instead.
            result2 = DD.dag_diff(make_containment_metadata())
            outer_m = filter(m -> m.symbol == "Outer" && m.kind == DD.LAYOUT_MISMATCH, result2.mismatches)
            @test !isempty(outer_m)
            # Size mismatch detail should mention DWARF vs Julia
            size_m = filter(m -> m.delta != 0, outer_m)
            @test !isempty(size_m)
            @test contains(size_m[1].detail, "DWARF")
        end
    end

    # ── Rendering Gallery (visual regression) ─────────────────────────────

    @testset "render gallery" begin
        gallery_dir = joinpath(@__DIR__, "gallery")
        mkpath(gallery_dir)

        scenarios = [
            ("aligned",      make_aligned_metadata(),      "No mismatches — all gray"),
            ("packed",        make_packed_metadata(),       "Packed struct with return/param mismatches"),
            ("containment",   make_containment_metadata(),  "Inner→Outer propagation chain"),
            ("deep_chain",    make_deep_chain_metadata(),   "A→B→C three-level propagation"),
            ("mixed",         make_mixed_metadata(),        "Safe + packed side by side"),
            ("union",         make_union_metadata(),        "Union type"),
            ("vtable",        make_vtable_metadata(),       "Polymorphic class with vtable"),
        ]

        for (name, meta, desc) in scenarios
            @testset "$name" begin
                result = DD.dag_diff(meta)
                svg_path = joinpath(gallery_dir, "$name.svg")
                out = DD.render_dot(result, svg_path)
                @test isfile(out)

                # Also export cpp-only and julia-only views for containment
                if name == "containment"
                    DD.render_dot(result, joinpath(gallery_dir, "containment_cpp.svg"); side=:cpp)
                    DD.render_dot(result, joinpath(gallery_dir, "containment_julia.svg"); side=:julia)
                    @test isfile(joinpath(gallery_dir, "containment_cpp.svg")) ||
                          isfile(joinpath(gallery_dir, "containment_cpp.dot"))
                end
            end
        end
    end

end
