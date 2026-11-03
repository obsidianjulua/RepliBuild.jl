#!/usr/bin/env julia
# A Tier 2 argument crosses at its C width.
#
# A thunk slot holds a pointer to the argument's storage, and the thunk loads
# the C parameter's width from there. The generated body used to hand
# `JITManager.invoke` the Julia value as the caller spelled it, so for a
# parameter declared `::Integer` a narrower integer was boxed narrow:
# `Ref(Int8(-1))` is one byte, an `int` parameter read four, and pugixml's
# `as_int(attr, Int8(-1))` answered a different number on every call
# (2026-11-03). `ccall` converts to the declared C type itself; a thunk cannot,
# so the emitted body now does (`_tier2_invoke_args`): `Cint(def)` for the
# integer parameters, and the `@enum`'s base type for an emitted enum, which
# the enum value and any Integer both convert to. Floats and Bool are declared
# at their exact type; pointers, references, strings and records cross as
# they are.
#
# Wrap-time facts about the generated text, so no toolchain is needed. The
# live half is `test/cpp_routing_test`'s `scalar_widths` probe (devtests
# `test_cpp_record_routing.jl`), which calls the thunk with Int8/UInt8/Int16
# arguments and checks what the callee saw.
#
# Negative check: return `names` unchanged from `_tier2_invoke_args` and the
# first testset fails on every converted argument.

using Test
using RepliBuild
using Libdl

const TSW_W = RepliBuild.Wrapper

_tsw_param(name, ct, jt) = Dict{String,Any}("name" => name, "c_type" => ct, "julia_type" => jt)
_tsw_fn(name, mangled, demangled, params, ret_c, ret_jl) = Dict{String,Any}(
    "name" => name, "mangled" => mangled, "demangled" => demangled,
    "class" => "", "is_method" => false, "is_vararg" => false, "is_noexcept" => false,
    "exported" => true, "has_object_pointer" => nothing,
    "parameters" => Any[params...],
    "return_type" => Dict{String,Any}("c_type" => ret_c, "julia_type" => ret_jl, "size" => 8))

@testset "Tier 2 arguments cross at their C width" begin
    dir = mktempdir()
    write(joinpath(dir, "replibuild.toml"), """
    [project]
    name = "widthsynth"
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
    structs = Dict{String,Any}(
        "__enum__Kind" => Dict{String,Any}("kind" => "enum", "name" => "Kind", "byte_size" => "0x2",
            "underlying_type" => "short", "julia_type" => "Int16",
            "enumerators" => Any[Dict{String,Any}("name" => "KIND_A", "value" => 1),
                                 Dict{String,Any}("name" => "KIND_B", "value" => 2)]))
    meta = Dict{String,Any}(
        "functions" => Any[
            _tsw_fn("ns::take", "_ZN2ns4takeEijy4Kindbd", "ns::take(int, unsigned int, unsigned long long, Kind, bool, double)",
                    [_tsw_param("a", "int", "Cint"), _tsw_param("b", "unsigned int", "Cuint"),
                     _tsw_param("c", "size_t", "Csize_t"), _tsw_param("k", "Kind", "Kind"),
                     _tsw_param("f", "bool", "Bool"), _tsw_param("d", "double", "Cdouble")],
                    "long long", "Clonglong"),
            _tsw_fn("ns::name_of", "_ZN2ns7name_ofEPKcl", "ns::name_of(const char*, long)",
                    [_tsw_param("s", "const char*", "Cstring"), _tsw_param("n", "long", "Clong")],
                    "int", "Cint")],
        "globals" => Dict{String,Any}(),
        "struct_definitions" => structs,
        "record_abi" => Dict{String,Any}())
    res = TSW_W.generate_introspective_module_cpp(cfg, libref, meta, "WidthSynth",
                                                  TSW_W.create_type_registry(cfg), true)
    out = res isa Tuple ? res[1] : res
    body(name) = begin
        m = match(Regex("^function \\w*$(name)\\(.*?^end\$", "ms"), out)
        m === nothing ? "" : m.match
    end
    @test count(r"^function \w*(take|name_of)\("m, out) == 2

    @testset "integers and enums are converted in the emitted body" begin
        t = body("take")
        @test occursin(r"^function \w*take\(a::Integer, b::Integer, c::Integer, k::\w+, f::Bool, d::Cdouble\)"m, t)
        @test occursin("invoke(\"_mlir_ciface__ZN2ns4takeEijy4Kindbd_thunk\", Clonglong, Cint(a), Cuint(b), Csize_t(c), Int16(k), f, d)", t)
        @test occursin("@enum Kind::Int16", out)
        n = body("name_of")
        @test occursin(r"^function \w*name_of\(s::\w+, n::Integer\)"m, n)
        @test occursin("invoke(\"_mlir_ciface__ZN2ns7name_ofEPKcl_thunk\", Cint, s, Clong(n))", n)
    end

    @testset "the conversion spelling is one function" begin
        @test TSW_W._tier2_invoke_args(["this", "n", "k", "p"], ["Ptr{Cvoid}", "Cuint", "Kind", "Ptr{Cvoid}"],
                                       [false, true, false, false], Dict("Kind" => "Int16")) ==
              ["this", "Cuint(n)", "Int16(k)", "p"]
        # an enum the module never emitted is left alone: its name is unbound
        @test TSW_W._tier2_invoke_args(["k"], ["Other"], [false], Dict("Kind" => "Int16")) == ["k"]
    end
end
