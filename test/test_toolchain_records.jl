#!/usr/bin/env julia
# test/test_toolchain_records.jl — a toolchain record the library holds or passes
# by value keeps its layout (2026-10-29).
#
# parse_dwarf_dump's provenance gate kept every record declared in a toolchain
# header out of struct_definitions. A function returning `std::pair<double,
# double>` then had no layout to cross by, and both generators trapped it, as did
# a library record holding a pair by value. `_promote_toolchain_records!` puts
# back the ones the library reaches by value, when their whole by-value closure
# has a layout. An empty base (std::pair's `__pair_base`, a library `Tag`) has no
# entry and stopped the SysV classifier; it is marked and skipped. GeneratorCpp's
# STL screen named `pair<` and `basic_string_view<`, and a promoted record now
# passes it. A tiled body whose emitted members do not add up to the record's
# DWARF size is laid out by offset instead of returned short. A toolchain record
# whose bytes come from its bases (libstdc++'s tuple, optional) is laid out
# from them, and a sized `complex` member counts as laid out (2026-11-13).
#
# No toolchain needed. test_toolchain_byvalue_live.jl calls the real thing.

using Test
using RepliBuild

const _TR_C  = RepliBuild.Compiler
const _TR_FG = RepliBuild.JLCSIRGenerator.FunctionGen
const _TR_SG = RepliBuild.JLCSIRGenerator.StructGen
const _TR_W  = RepliBuild.Wrapper

_tr_mem(name, ctype, off, size) = Dict{String,Any}(
    "name" => name, "c_type" => ctype, "julia_type" => "Any", "offset" => off, "size" => size)
_tr_rec(size, members; bases = Any[], extra...) = Dict{String,Any}(
    "kind" => "struct", "byte_size" => size, "members" => members, "base_classes" => bases,
    (String(k) => v for (k, v) in extra)...)
_tr_empty_base(t) = Dict{String,Any}("type" => t, "offset" => 0, "virtual" => false, "empty" => true)

# A libstdc++-shaped toolchain table: pair derives from an empty __pair_base.
function _tr_toolchain()
    Dict{String,Dict{String,Any}}(
        "pair<double, double>" => _tr_rec("0x10", [_tr_mem("first", "double", 0, 8), _tr_mem("second", "double", 8, 8)];
                                          bases = [_tr_empty_base("__pair_base<double, double>")]),
        "pair<int, int>"       => _tr_rec("0x8", [_tr_mem("first", "int", 0, 4), _tr_mem("second", "int", 4, 4)];
                                          bases = [_tr_empty_base("__pair_base<int, int>")]),
        "pair<std::pair<int, int>, double>" =>
            _tr_rec("0x10", [_tr_mem("first", "pair<int, int>", 0, 8), _tr_mem("second", "double", 8, 8)]),
        "pair<int *, bool>"    => _tr_rec("0x10", [_tr_mem("first", "int *", 0, 8), _tr_mem("second", "bool", 8, 1)]),
        "complex<double>"      => _tr_rec("0x10", [_tr_mem("_M_value", "complex", 0, 16)]),
        "vector<int>"          => _tr_rec("0x18", [_tr_mem("_M_start", "int *", 0, 8)]),
        "div_t"                => _tr_rec("0x8", [_tr_mem("quot", "int", 0, 4), _tr_mem("rem", "int", 4, 4)]),
        "pair<int, conflicted>"=> _tr_rec("0x8", [_tr_mem("first", "int", 0, 4)]; conflict = true),
        "pair<double, vbase>"  => _tr_rec("0x10", [_tr_mem("first", "double", 8, 8)];
                                          bases = [Dict{String,Any}("type" => "vbase", "offset" => 0, "virtual" => true)]),
        "pair<double, holder>" => _tr_rec("0x10", [_tr_mem("first", "double", 0, 8), _tr_mem("second", "unlaid<int>", 8, 8)]),
    )
end
_tr_promote(seeds; library = Dict{String,Dict{String,Any}}()) = begin
    sd = Dict{String,Dict{String,Any}}(k => deepcopy(v) for (k, v) in library)
    got = _TR_C._promote_toolchain_records!(sd, _tr_toolchain(), seeds)
    (sd, got)
end

@testset "Toolchain records by value keep their layout" begin

@testset "what the library reaches by value is promoted, and marked" begin
    sd, got = _tr_promote(["pair<double, double>"])
    @test got == ["pair<double, double>"]
    @test sd["pair<double, double>"]["toolchain"] == true
    # A scoped spelling (a demangled signature) finds the bare DWARF key.
    @test _tr_promote(["std::pair<double, double>"])[2] == ["pair<double, double>"]
    @test _tr_promote(["const std::pair<double, double>"])[2] == ["pair<double, double>"]
    # Whatever a promoted record holds by value comes with it.
    @test _tr_promote(["pair<std::pair<int, int>, double>"])[2] ==
          ["pair<int, int>", "pair<std::pair<int, int>, double>"]
    # An address reaches nothing by value.
    @test isempty(_tr_promote(["pair<double, double> *", "const pair<int, int> &", "pair<int, int>[2]*"])[2])
    # A `*` inside the arguments is the record's member, not the record.
    @test _tr_promote(["pair<int *, bool>"])[2] == ["pair<int *, bool>"]
    # An array of them holds them.
    @test _tr_promote(["pair<int, int>[3]"])[2] == ["pair<int, int>"]
end

@testset "a library record reached by value is walked; one that is not, is not" begin
    lib = Dict{String,Dict{String,Any}}(
        "Span" => _tr_rec("0x10", [_tr_mem("range", "pair<int, int>", 0, 8), _tr_mem("weight", "double", 8, 8)]))
    # Nothing exported crosses Span by value: its pair stays out. ggml-vulkan's
    # hidden C++ promoted 123 Vulkan-Hpp records this way before the gate.
    @test isempty(_tr_promote(String[]; library = lib)[2])
    @test isempty(_tr_promote(["Span *", "const Span &"]; library = lib)[2])
    # Returned or taken by value, what it holds comes with it.
    sd, got = _tr_promote(["Span"]; library = lib)
    @test got == ["pair<int, int>"]
    @test !haskey(sd["Span"], "toolchain")
    # A library record of the same name keeps its own entry.
    own = Dict{String,Dict{String,Any}}("pair<int, int>" => _tr_rec("0x8", [_tr_mem("a", "int", 0, 4)]))
    sd2, got2 = _tr_promote(["pair<int, int>"]; library = own)
    @test isempty(got2)
    @test sd2["pair<int, int>"]["members"][1]["name"] == "a"
end

@testset "nothing without a whole by-value layout" begin
    @test isempty(_tr_promote(["complex<double>"])[2])        # `complex` names no width
    @test isempty(_tr_promote(["tuple<int, double>"])[2])     # no entry at all
    @test isempty(_tr_promote(["pair<int, conflicted>"])[2])  # several layouts under one name
    @test isempty(_tr_promote(["pair<double, vbase>"])[2])    # a virtual base
    @test isempty(_tr_promote(["pair<double, holder>"])[2])   # holds a record with no entry
    @test isempty(_tr_promote(["vector<int>"])[2])            # a container keeps its own path
    @test isempty(_tr_promote(["div_t"])[2])                  # INTERNAL_TYPE_BLOCKLIST
    # An unmarked base with no entry is not known to be empty.
    tc = _tr_toolchain()
    tc["pair<double, double>"]["base_classes"][1]["empty"] = false
    sd = Dict{String,Dict{String,Any}}()
    @test isempty(_TR_C._promote_toolchain_records!(sd, tc, ["pair<double, double>"]))
end

# libstdc++'s shapes (2026-11-13): tuple and optional have no member of their
# own, only bases; optional's payload is a union of the value and an empty
# `_Empty_byte`; complex's member is a `complex` base type the parser sizes.
_tr_base(t, off) = Dict{String,Any}("type" => t, "offset" => off, "virtual" => false)
function _tr_from_bases()
    Dict{String,Dict{String,Any}}(
        "tuple<int, double>"          => _tr_rec("0x10", Any[]; bases = [_tr_base("_Tuple_impl<0UL, int, double>", 0)]),
        "_Tuple_impl<0UL, int, double>" => _tr_rec("0x10", Any[];
            bases = [_tr_base("_Tuple_impl<1UL, double>", 0), _tr_base("_Head_base<0UL, int, false>", 8)]),
        "_Tuple_impl<1UL, double>"    => _tr_rec("0x8", Any[]; bases = [_tr_base("_Head_base<1UL, double, false>", 0)]),
        "_Head_base<0UL, int, false>" => _tr_rec("0x4", [_tr_mem("_M_head_impl", "int", "0x0", 4)]),
        "_Head_base<1UL, double, false>" => _tr_rec("0x8", [_tr_mem("_M_head_impl", "double", "0x0", 8)]),
        "optional<double>" => _tr_rec("0x10", Any[];
            bases = [_tr_base("_Optional_base<double, true, true>", 0),
                     _tr_empty_base("_Enable_copy_move<true, true, true, true, std::optional<double> >")]),
        "_Optional_base<double, true, true>" =>
            _tr_rec("0x10", [_tr_mem("_M_payload", "_Optional_payload<double, true, true, true>", "0x0", 16)]),
        "_Optional_payload<double, true, true, true>" =>
            _tr_rec("0x10", Any[]; bases = [_tr_base("_Optional_payload_base<double>", 0)]),
        "_Optional_payload_base<double>" =>
            _tr_rec("0x10", [_tr_mem("_M_payload", "_Storage<double, true>", "0x0", 8),
                             _tr_mem("_M_engaged", "bool", "0x8", 1)]),
        "_Storage<double, true>" => _tr_rec("0x8", [_tr_mem("_M_empty", "_Empty_byte", "0x0", 1),
                                                    _tr_mem("_M_value", "double", "0x0", 8)]; kind = "union"),
        "complex<double>" => _tr_rec("0x10", [_tr_mem("_M_value", "complex double", "0x0", 16)]),
        "complex<float>"  => _tr_rec("0x8", [_tr_mem("_M_value", "complex float", "0x0", 8)]),
        # A bitfield in a base at offset 4 moves by 32 bits.
        "bits_at_4" => _tr_rec("0x8", [_tr_mem("k", "int", "0x0", 4)]; bases = [_tr_base("bits", 4)]),
        "bits" => _tr_rec("0x4", [Dict{String,Any}("name" => "f", "c_type" => "unsigned int", "julia_type" => "Cuint",
                                                   "offset" => nothing, "size" => 4, "bit_size" => 3,
                                                   "data_bit_offset" => 2)]),
    )
end
_TR_EMPTY = Set(["_Empty_byte"])
_tr_layout(d) = [(m["name"], m["c_type"], m["offset"]) for m in d["members"]]

@testset "a record whose bytes come from its bases is laid out from them" begin
    @testset "tuple: the bases' members at the bases' offsets, a shared name numbered" begin
        sd = Dict{String,Dict{String,Any}}()
        got = _TR_C._promote_toolchain_records!(sd, _tr_from_bases(), ["std::tuple<int, double>"];
                                                empty_records = _TR_EMPTY)
        # The intermediate bases are flattened into it, never promoted themselves.
        @test got == ["tuple<int, double>"]
        t = sd["tuple<int, double>"]
        @test _tr_layout(t) == [("_M_head_impl", "double", "0x0"), ("_M_head_impl_1", "int", "0x8")]
        @test !haskey(t, "base_classes")
        @test t["toolchain"] == true
        # The classifier reads it like any record: SSE, then INTEGER.
        v = _TR_FG.sysv_record_abi("tuple<int, double>", Dict{String,Any}(sd))
        @test (v.kind, v.classes) == (:regs, [:sse, :int])
    end

    @testset "optional: a union's empty member dropped, the empty base kept marked" begin
        sd = Dict{String,Dict{String,Any}}()
        got = _TR_C._promote_toolchain_records!(sd, _tr_from_bases(), ["optional<double>"];
                                                empty_records = _TR_EMPTY)
        @test got == ["_Optional_payload<double, true, true, true>", "_Storage<double, true>", "optional<double>"]
        @test _tr_layout(sd["optional<double>"]) ==
              [("_M_payload", "_Optional_payload<double, true, true, true>", "0x0")]
        @test [b["type"] for b in sd["optional<double>"]["base_classes"]] ==
              ["_Enable_copy_move<true, true, true, true, std::optional<double> >"]
        @test _tr_layout(sd["_Optional_payload<double, true, true, true>"]) ==
              [("_M_payload", "_Storage<double, true>", "0x0"), ("_M_engaged", "bool", "0x8")]
        @test _tr_layout(sd["_Storage<double, true>"]) == [("_M_value", "double", "0x0")]
        # clang returns `optional<double>` as { double, i8 }.
        v = _TR_FG.sysv_record_abi("optional<double>", Dict{String,Any}(sd))
        @test (v.kind, v.classes) == (:regs, [:sse, :int])
        # Not knowing `_Empty_byte` is empty, the union holds a record with no
        # entry, and nothing is promoted: the old refusal.
        @test isempty(_TR_C._promote_toolchain_records!(Dict{String,Dict{String,Any}}(), _tr_from_bases(),
                                                        ["optional<double>"]))
    end

    @testset "complex: the width the parser read off byte_size" begin
        @test _TR_C._complex_base_name(8) == "complex float"
        @test _TR_C._complex_base_name(16) == "complex double"
        @test _TR_C._complex_base_name(32) == "complex long double"
        @test _TR_C._complex_base_name(12) == "complex"    # unknown: left unnamed, refused
        sd = Dict{String,Dict{String,Any}}()
        got = _TR_C._promote_toolchain_records!(sd, _tr_from_bases(), ["complex<double>", "complex<float>"])
        @test got == ["complex<double>", "complex<float>"]
        structs = Dict{String,Any}(sd)
        # Two doubles are two SSE eightbytes; two floats share one.
        @test (_TR_FG.sysv_record_abi("complex<double>", structs).classes) == [:sse, :sse]
        @test (_TR_FG.sysv_record_abi("complex<float>", structs).classes) == [:sse]
        # The name alone still names no width (the table above).
        @test isempty(_tr_promote(["complex<double>"])[2])
    end

    @testset "a bitfield in a base moves with it" begin
        sd = Dict{String,Dict{String,Any}}()
        @test _TR_C._promote_toolchain_records!(sd, _tr_from_bases(), ["bits_at_4"]) == ["bits_at_4"]
        m = sd["bits_at_4"]["members"]
        @test [x["name"] for x in m] == ["k", "f"]
        @test m[2]["data_bit_offset"] == 2 + 32 && m[2]["offset"] === nothing
    end

    @testset "a base deeper down that cannot be looked up keeps the record as it was" begin
        tc = _tr_from_bases()
        own(d) = collect(get(d, "members", ()))
        known = Set(["_Tuple_impl<1UL, double>", "_Head_base<1UL, double, false>"])
        lookup(k) = k in known ? tc[k] : nothing
        d = _TR_C._laid_out_from_bases(tc["_Tuple_impl<0UL, int, double>"], lookup, own)
        # `_Head_base<0UL, int, false>` is unknown at the top: kept as a base.
        @test _tr_layout(d) == [("_M_head_impl", "double", "0x0")]
        @test [b["type"] for b in d["base_classes"]] == ["_Head_base<0UL, int, false>"]
        # Unknown below the top: nothing is flattened, nothing is lost.
        lookup2(k) = k == "_Tuple_impl<1UL, double>" ? tc[k] : nothing
        d2 = _TR_C._laid_out_from_bases(tc["_Tuple_impl<0UL, int, double>"], lookup2, own)
        @test isempty(d2["members"])
        @test length(d2["base_classes"]) == 2
    end
end

@testset "an empty base adds nothing to the SysV class" begin
    structs = Dict{String,Any}("pair<double, double>" => _tr_toolchain()["pair<double, double>"],
                               "LP" => _tr_rec("0x10", [_tr_mem("a", "double", 0, 8), _tr_mem("b", "double", 8, 8)];
                                               bases = [_tr_empty_base("Tag")]))
    for k in ("pair<double, double>", "LP")
        v = _TR_FG.sysv_record_abi(k, structs)
        @test v.kind === :regs
        @test v.classes == [:sse, :sse]
        @test _TR_FG.sysv_trap_reason(k, structs; ret = true, abi = :sysv) === nothing
    end
    # Unmarked, the base is a name the classifier cannot read: still a trap.
    unmarked = deepcopy(structs)
    unmarked["LP"]["base_classes"][1]["empty"] = false
    @test _TR_FG.sysv_record_abi("LP", unmarked).kind === :unknown
    @test occursin("`Tag`", something(_TR_FG.sysv_trap_reason("LP", unmarked; ret = true, abi = :sysv), ""))
end

@testset "the STL screen lets a promoted record through" begin
    pr = Dict{String,Any}("pair<double, double>" => Dict{String,Any}("kind" => "struct", "scope" => "std",
                                                                     "toolchain" => true))
    @test !_TR_W._is_stl_internal_record("pair<double, double>", pr)
    @test !_TR_W._is_stl_internal_record("Ptr{pair<double, double>}", pr)
    @test !_TR_W._is_stl_internal_record("const pair<double, double>&", pr)
    pr["pair<double, double>"]["toolchain"] = false
    @test _TR_W._is_stl_internal_record("pair<double, double>", pr)
    @test _TR_W._is_stl_internal_record("pair<int, int>", pr)   # not in the table: the name decides
end

@testset "a tiled body that does not add up is laid out by offset" begin
    # `ns::Box` is not a template spelling, so the member resolves through
    # map_cpp_type, which answers `!llvm.ptr`: 8 bytes for a 16-byte record. The
    # member sizes still tile, so the body used to be returned as emitted.
    structs = Dict{String,Any}(
        "ns::Box" => _tr_rec("0x10", [_tr_mem("a", "double", 0, 8), _tr_mem("b", "double", 8, 8)]),
        "Outer" => _tr_rec("0x18", [_tr_mem("x", "double", 0, 8), _tr_mem("in", "ns::Box", 8, 16)]))
    @test _TR_SG.is_struct_packed(structs["Outer"])
    for t in (_TR_SG.get_llvm_aligned_type_string("Outer", structs["Outer"], structs),
              _TR_SG.get_llvm_equivalent_type_string("Outer", structs["Outer"], structs))
        l = _TR_SG._mlir_layout(t, structs)
        @test l !== nothing && l[1] == 24
    end
    # A body that adds up is unchanged.
    ok = Dict{String,Any}("T" => _tr_rec("0x10", [_tr_mem("a", "double", 0, 8), _tr_mem("b", "int", 8, 4),
                                                   _tr_mem("c", "int", 12, 4)]))
    @test _TR_SG.get_llvm_aligned_type_string("T", ok["T"], ok) == "!llvm.struct<(f64, i32, i32) >"
end

@testset "a symbol exclude_symbols drops seeds nothing" begin
    # glfw exports its internal state global `_glfw` (no -fvisibility=hidden) and
    # drops `_glfw*` at wrap; the walk from that global kept glibc's
    # pthread_mutex_t and friends (2026-11-08). The walk's exported set must lose
    # exactly what wrap drops, matched on the same three spellings.
    syms = [Dict{String,Any}("mangled" => m, "demangled" => d, "type" => t, "address" => "0")
            for (m, d, t) in [("_glfw", "_glfw", "B"),                          # a global
                              ("glfwInit", "glfwInit", "T"),
                              ("_glfwPlatformInit", "_glfwPlatformInit", "T"),  # mangled spelling
                              ("_ZN2vk6Device7destroyEv", "vk::Device::destroy()", "T"),  # demangled
                              ("_ZN2ns4stepEi", "ns::step(int)", "T")]]         # function name
    pats = ["_glfw*", "vk::*", "step"]
    exported = Set(s["mangled"] for s in syms)
    kept = _TR_C._without_excluded_symbols(exported, syms, pats)
    @test kept == Set(["glfwInit"])

    md = Dict{String,Any}("functions" => _TR_C.parse_function_signatures(syms),
                          "globals" => Dict{String,Any}("_glfw" => Dict{String,Any}()))
    _TR_W._apply_symbol_filter!(md, pats)
    @test Set(f["mangled"] for f in md["functions"]) == kept
    @test isempty(md["globals"])

    # A global is a data symbol, which `extract_symbols_from_binary` leaves out
    # of `symbols` (code only): it is matched on its exported name.
    code_only = filter(s -> s["type"] == "T", syms)
    @test _TR_C._without_excluded_symbols(exported, code_only, pats) == Set(["glfwInit"])

    # Nothing to drop: the same set, not a copy; a pattern matching nothing drops nothing.
    @test _TR_C._without_excluded_symbols(exported, syms, String[]) === exported
    @test _TR_C._without_excluded_symbols(exported, syms, ["nomatch_*"]) == exported
end

end
