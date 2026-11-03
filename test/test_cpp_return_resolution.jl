#!/usr/bin/env julia
# Tier 2 returns the Julia side must type.
#
# Two defects Boost.JSON's inline DOM API exposed (2026-11-01), both wrap-time
# facts about the generated text, so no toolchain is needed:
#
#   • `key_value_pair::key()` returns `const basic_string_view<char>`. The record
#     is in `struct_definitions` as `basic_string_view<char>`, is emitted as
#     `basic_string_view_char`, and crossed as a parameter; but the Tier 2
#     return resolution looked the spelling up WITH its `const`, missed, and
#     left the `Any` sentinel, which `JITManager.invoke` refuses at the call.
#     The lookups now read the cv-stripped spelling, as `by_value_crossing` does.
#   • `object::emplace<int>` returns `std::pair<key_value_pair*, bool>` by value.
#     `FunctionGen._table_entry` read the `*` inside the template arguments as
#     the type's own declarator and called the crossing `:direct`: the thunk
#     returned the 16-byte pair in a register and the stub spelled it
#     `Ptr{Cvoid}`. On Win64 that return needs a hidden pointer, and the callee
#     read `this` from the next register — an access violation. The declarator
#     test now reads `_outside_template_args`, and the crossing is refused.
#   • And the guard that was missing: a return the resolution still leaves
#     `Any` is emitted as an ABI trap stub naming the return, never as
#     `invoke(…, Any, …)`, so the refusal is where the function is defined.
#
# Negative checks: look the return up as spelled again and the `key` case emits
# `Any`; drop `_outside_template_args` from `_table_entry` and `make_pair` is a
# call; drop the guard and `lookup` is `invoke(…, Any, …)`.

using Test
using RepliBuild
using Libdl

const CRR_W = RepliBuild.Wrapper
const CRR_F = RepliBuild.JLCSIRGenerator.FunctionGen

_crr_mm(name, jt, ct, off, sz) = Dict{String,Any}("name" => name, "julia_type" => jt, "c_type" => ct,
                                                 "offset" => off, "size" => sz)
_crr_fn(name, mangled, demangled, ret_c, ret_jl) = Dict{String,Any}(
    "name" => name, "mangled" => mangled, "demangled" => demangled,
    "class" => "", "is_method" => false, "is_vararg" => false, "is_noexcept" => false,
    "exported" => true, "has_object_pointer" => nothing,
    "parameters" => Any[Dict{String,Any}("name" => "x", "c_type" => "int", "julia_type" => "Cint")],
    "return_type" => Dict{String,Any}("c_type" => ret_c, "julia_type" => ret_jl, "size" => 0))

@testset "Tier 2 returns the Julia side must type" begin
    dir = mktempdir()
    write(joinpath(dir, "replibuild.toml"), """
    [project]
    name = "retsynth"
    root = "$(escape_string(dir))"
    [wrap]
    language = "cpp"
    [types]
    strictness = "warn"
    allow_unknown_structs = true
    [cache]
    enabled = false
    """)
    cfg = RepliBuild.ConfigurationManager.load_config(joinpath(dir, "replibuild.toml"))
    libref = abspath(first(filter(p -> occursin("libjulia", basename(p)), Libdl.dllist())))
    structs = Dict{String,Any}("basic_string_view<char>" => Dict{String,Any}(
        "kind" => "class", "byte_size" => "0x10", "scope" => "boost::core",
        "members" => Any[_crr_mm("p_", "Ptr{UInt8}", "const char*", "0x0", 8),
                         _crr_mm("n_", "Csize_t", "size_t", "0x8", 8)]))
    record_abi = Dict{String,Any}("basic_string_view<char>" => Dict{String,Any}(
        "byte_size" => "0x10", "empty" => false, "pass" => "value"))
    meta = Dict{String,Any}(
        "functions" => Any[
            # the DWARF spelling of key()'s return carries a top-level const
            _crr_fn("ns::key", "_ZN2ns3keyEi", "ns::key(int)", "const basic_string_view<char>", "Any"),
            # a declared-only template record whose arguments hold a pointer
            _crr_fn("ns::make_pair", "_ZN2ns9make_pairEi", "ns::make_pair(int)",
                    "pair<int *, bool>", "pair<int *, bool>"),
            # a pointer return the metadata never mapped: the crossing is fine,
            # the Julia side has no type
            _crr_fn("ns::lookup", "_ZN2ns6lookupEi", "ns::lookup(int)", "mystery*", "Any"),
            # a reference to a type Julia cannot hold: `long double&` arrives as Ref{Any}
            let f = _crr_fn("ns::parse_ld", "_ZN2ns8parse_ldEiRe", "ns::parse_ld(int, long double&)", "int", "Cint")
                push!(f["parameters"], Dict{String,Any}("name" => "value", "c_type" => "long double&", "julia_type" => "Ref{Any}"))
                f
            end],
        "globals" => Dict{String,Any}(),
        "struct_definitions" => structs,
        "record_abi" => record_abi)
    res = CRR_W.generate_introspective_module_cpp(cfg, libref, meta, "RetSynth",
                                                  CRR_W.create_type_registry(cfg), true)
    out = res isa Tuple ? res[1] : res
    body(name) = begin
        m = match(Regex("^function \\w*$(name)\\(.*?^end\$", "ms"), out)
        m === nothing ? "" : m.match
    end
    @test count(r"^function \w*(key|make_pair|lookup|parse_ld)\("m, out) == 4     # every function is defined

    @testset "a const record return resolves to the emitted record" begin
        @test CRR_F.by_value_crossing("const basic_string_view<char>", structs, record_abi) ==
              (:layout, "basic_string_view<char>")
        k = body("key")
        @test occursin("JITManager.invoke(\"_mlir_ciface__ZN2ns3keyEi_thunk\", basic_string_view_char, x)", k)
        @test !occursin("ABI Safety Trap", k)
        @test occursin(r"^struct basic_string_view_char\n"m, out)
    end

    @testset "a pointer inside template arguments is not a pointer return" begin
        p = body("make_pair")
        @test occursin("ABI Safety Trap", p)
        @test occursin("returns `pair<int *, bool>` by value", p)
        @test !occursin("invoke(", p)
    end

    @testset "a return left Any is a trap stub, never invoke(…, Any, …)" begin
        l = body("lookup")
        @test occursin("ABI Safety Trap", l)
        @test occursin("no Julia type was resolved", l) && occursin("`mystery*`", l)
        @test !occursin("invoke(", l)
        @test !occursin(r"invoke\(\"[^\"]*\",\s*Any\b", out)
        # The stub loses a method dedup to a callable, like every other trap.
        @test CRR_W._is_trap_chunk(l)
    end

    @testset "a reference to a type Julia cannot hold is a raw pointer, never Ref{Any}" begin
        # Boost.Charconv's `long double` from_chars carried `value::Ref{Any}`: the
        # only value a caller can pass is `Ref{Any}(x)`, a pointer to a box slot the
        # callee overwrites with an 80-bit value (2026-11-01).
        # A Tier 2 signature types every pointer parameter `::Any` (a Ptr or a Ref
        # both marshal); the docstring says what it is, and the metadata's
        # `Ref{Any}` reaches neither.
        p = body("parse_ld")
        @test occursin(r"^function \w*parse_ld\(x::Integer, value::Any\)"m, p)
        @test occursin("- `value::Ptr{Cvoid}`", out)
        @test !occursin("Ref{Any}", out) && !occursin("Ptr{Any}", out)
        @test CRR_W._pointee_safe_type("Ref{Any}") == "Ptr{Cvoid}" == CRR_W._pointee_safe_type("Ptr{Any}")
        @test CRR_W._pointee_safe_type("Ref{Cint}") == "Ref{Cint}"
    end
end
