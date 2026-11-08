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
# DWARF size is laid out by offset instead of returned short.
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

end
