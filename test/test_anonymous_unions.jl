# =============================================================================
# Anonymous struct/union support in the C generator
#
# An aggregate DIE with no DW_AT_name — a C11 `union { ... };` member, or the
# type of a `union { ... } u;` member — used to be dropped on DWARF export, and
# the member referencing it typed as `Any`. That single unresolvable member then
# failed `_resolve_exact_layout`, so the WHOLE enclosing struct degraded to an
# opaque byte blob even though DWARF carried the complete member tree.
#
# Live instance (2026-08-01): tomlc17's `toml_datum_t` (40 bytes) and
# `toml_result_t` (256 bytes) came out as `_data::NTuple{N,UInt8}` with no
# named fields at all — callers could parse a TOML document but could not read
# a single value out of the result. `-fstandalone-debug` changed nothing; the
# members were never missing, so this is NOT the box2d limited-debug-info class.
#
# Gated here over `c_abomination_test/`, which carries all four shapes:
#   §6 `TaggedValue`  — named member `u` of an anonymous union type (the
#                       toml_datum_t shape), enum tag + scalar before it
#   §1 `NightmareStruct` — nameless C11 anonymous members, nested three deep,
#                       with bitfields at the bottom
#   §7 `FloatBox`     — all-float anonymous union crossing BY VALUE, which pins
#                       the SysV SSE-vs-INTEGER region element-type rule
#
# The assertions are layered deliberately: metadata (did DWARF export keep it),
# source (did the generator emit real fields and a correctly-typed region), and
# LIVE (does the ABI actually agree with C). Only the last one can catch a
# register-class mistake — a text assertion on `NTuple{1, Float64}` would pass
# just as happily if the ccall shape were wrong elsewhere.
# =============================================================================

using Test
using JSON
using RepliBuild

const ANON_DIR = joinpath(@__DIR__, "c_abomination_test")

@testset "Anonymous struct/union support (C generator)" begin
    # Build fresh: this suite asserts on generated artifacts, so a stale
    # wrapper from an earlier fixture pass would make it test nothing.
    for d in ("build", "julia")
        p = joinpath(ANON_DIR, d)
        isdir(p) && rm(p; recursive=true, force=true)
    end
    rm(joinpath(ANON_DIR, ".replibuild_cache"); recursive=true, force=true)

    toml = RepliBuild.discover(ANON_DIR, force=true)
    RepliBuild.build(toml)
    wrapper_path = RepliBuild.wrap(toml)
    @test isfile(wrapper_path)

    meta = JSON.parsefile(joinpath(ANON_DIR, "julia", "compilation_metadata.json"))
    sd = meta["struct_definitions"]
    src = read(wrapper_path, String)

    # ── 1. DWARF export keeps anonymous aggregates ───────────────────────────
    @testset "Anonymous aggregates survive DWARF export" begin
        # Named after the member that embeds them, so the name is stable and
        # says where the type came from.
        @test haskey(sd, "TaggedValue_u")
        @test haskey(sd, "TaggedValue_u_str")
        @test haskey(sd, "NightmareStruct_anon1")
        @test haskey(sd, "NightmareStruct_anon1_complex_inner")
        @test haskey(sd, "FloatBox_v")

        # Flagged as synthesized, which is what routes them to the embeddable
        # opaque-region emission instead of the user-facing union path.
        @test get(sd["TaggedValue_u"], "anonymous", false) == true
        @test get(sd["NightmareStruct_anon1"], "anonymous", false) == true
        @test sd["TaggedValue_u"]["kind"] == "union"

        # The union member is no longer `Any`: resolve_type had no "union"
        # branch, so every union-typed member resolved to "unknown".
        tv = Dict(m["name"] => m for m in sd["TaggedValue"]["members"])
        @test Set(keys(tv)) == Set(["tag", "flags", "u"])
        @test tv["u"]["julia_type"] == "TaggedValue_u"
        @test tv["u"]["c_type"] != "unknown"
        @test tv["u"]["size"] == 16          # aggregate size, not 0
        @test tv["u"]["offset"] in ("0x8", "0x08", "8")

        # A nameless C11 member gets a synthetic FIELD name and is marked as
        # injecting its members into the enclosing scope.
        ns = Dict(m["name"] => m for m in sd["NightmareStruct"]["members"])
        @test haskey(ns, "_anon1")
        @test ns["_anon1"]["anonymous_member"] == true
        @test get(ns["id"], "anonymous_member", false) == false

        # Every anonymous aggregate carries its full member tree.
        @test Set(m["name"] for m in sd["TaggedValue_u"]["members"]) ==
              Set(["i", "d", "s", "str"])
        @test Set(m["name"] for m in sd["FloatBox_v"]["members"]) == Set(["f", "d"])
    end

    # ── 2. The enclosing struct keeps its named fields ───────────────────────
    @testset "Enclosing struct is not blobbed" begin
        # The regression this whole change exists to prevent.
        @test occursin(r"struct TaggedValue\n\s+tag::ValueTag\n\s+flags::UInt32\n\s+u::TaggedValue_u\n", src)
        @test !occursin(r"struct TaggedValue\n\s+_data::NTuple", src)
        @test !occursin(r"struct NightmareStruct\n\s+_data::NTuple", src)
        @test !occursin(r"struct FloatBox\n\s+_data::NTuple", src)

        # An enum member of a struct that DOES stay a blob still gets an
        # accessor — `@enum T::U` is a primitive type, but it is not in
        # _loadable_primitives, so enum members used to be dropped silently.
        # (toml_datum_t lost `type`, the tag saying which arm is live.)
        @test occursin("tag::ValueTag", src)
    end

    # ── 3. The region carries alignment and SysV class ───────────────────────
    @testset "Opaque region storage" begin
        # NTuple{N,UInt8} is align 1: embedding one where C wants align 8
        # under-aligns the parent and shifts every field after it.
        @test occursin(r"struct TaggedValue_u\n\s+_data::NTuple\{2, UInt64\}\n", src)
        @test occursin(r"struct NightmareStruct_anon1\n\s+_data::NTuple\{3, UInt64\}\n", src)
        # All-float union ⇒ SSE class ⇒ float element. An integer element here
        # claims INTEGER and reads the value out of the wrong register file
        # (negative-checked: forcing UInt64 makes §5 floatbox_get fail).
        @test occursin(r"struct FloatBox_v\n\s+_data::NTuple\{1, Float64\}\n", src)
        @test !occursin(r"struct FloatBox_v\n\s+_data::NTuple\{\d+, UInt", src)

        # Regions are immutable — a `mutable struct` is a REFERENCE when used as
        # a field, so the old union path could never be embedded in the parent
        # that declared it.
        @test !occursin("mutable struct TaggedValue_u", src)
        @test !occursin("mutable struct FloatBox_v", src)
    end

    # ── 4. C11 name injection ────────────────────────────────────────────────
    @testset "C11 anonymous-member name injection" begin
        # C puts a nameless member's fields in the enclosing scope: `n.x`.
        @test occursin("s === :x && return getproperty(getfield(x, :_anon1), :x)", src)
        @test occursin("s === :raw_data && return getproperty(getfield(x, :_anon1), :raw_data)", src)
        # A declared field must never be shadowed by an injected name — the
        # synthetic `_anonN` is ours, not C's, and injecting it upward would
        # make `n._anon1` resolve to the level below.
        @test !occursin("s === :_anon1 && return getproperty(getfield(x, :_anon1), :_anon1)", src)
        # `u` is a NAMED member, so nothing is injected for it.
        @test !occursin("s === :i && return getproperty(getfield(x, :u), :i)", src)
    end

    # ── 5. Live ABI agreement ────────────────────────────────────────────────
    # Subprocess: a register-class mistake can corrupt the session, and these
    # are the only assertions that can catch one.
    @testset "Live field access and by-value ABI" begin
        probe = """
        include($(repr(wrapper_path)))
        using .CAbominationTest
        const A = CAbominationTest
        using Test

        # Layout must match the C declaration exactly.
        @assert sizeof(A.TaggedValue) == 24
        @assert sizeof(A.TaggedValue_u) == 16
        @assert [fieldoffset(A.TaggedValue, i) for i in 1:fieldcount(A.TaggedValue)] == [0, 4, 8]
        @assert sizeof(A.NightmareStruct) == 48
        @assert [fieldoffset(A.NightmareStruct, i) for i in 1:fieldcount(A.NightmareStruct)] == [0, 4, 8, 32]
        @assert sizeof(A.FloatBox) == 16

        # Named member of an anonymous union type (the toml_datum_t shape).
        t = A.make_tagged_int(Int64(-99887766))
        @assert t.tag == A.TAG_INT
        @assert t.flags == 0xABCD1234
        @assert t.u.i == -99887766
        d = A.make_tagged_double(3.5)
        @assert d.tag == A.TAG_DBL && d.u.d == 3.5
        # Arms alias the same storage — that is what makes it a union.
        @assert reinterpret(Int64, d.u.d) == d.u.i
        s = "hello world"
        sv = A.make_tagged_str(s, Int32(11))
        @assert sv.u.str.len == 11
        @assert unsafe_string(Ptr{UInt8}(sv.u.str.ptr)) == "hello world"
        @assert A.tagged_is(Ref(t), A.TAG_INT) == 1
        @assert A.tagged_is(Ref(t), A.TAG_DBL) == 0

        # Nameless C11 members, three levels deep, injected into scope.
        n = A.create_nightmare(Int32(42), 1.5f0, 2.5f0, 3.5f0)
        @assert n.id == 42
        @assert (n.x, n.y, n.z) == (1.5f0, 2.5f0, 3.5f0)
        @assert n._anon1.x == 1.5f0            # explicit path still reaches it
        @assert n.complex_inner.a isa UInt8
        # raw_data aliases the {x,y,z} floats in the same union.
        @assert reinterpret(NTuple{2,Float32}, [n.raw_data[1]])[1] == (n.x, n.y)

        # All-float union crossing BY VALUE in both directions. Eightbyte 0 is
        # SSE (the union), eightbyte 1 is INTEGER (kind) — a wrong region
        # element type yields a wrong VALUE here, not a type error.
        b = A.floatbox_make(2.75, Int32(7))
        @assert b.v.d == 2.75 && b.kind == 7
        @assert A.floatbox_get(b) == 2.75
        @assert A.floatbox_kind(b) == 7

        println("ANON_UNION_PROBE_OK")
        """
        script = joinpath(mktempdir(), "anon_union_probe.jl")
        write(script, probe)
        out = try
            read(`$(Base.julia_cmd()) --project=$(dirname(@__DIR__)) $script`, String)
        catch e
            "PROBE FAILED: $e"
        end
        @test occursin("ANON_UNION_PROBE_OK", out)
    end
end
