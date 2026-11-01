#!/usr/bin/env julia
# test/test_integer_width_overloads.jl — C++ overloads that differ only in an
# integer parameter's width stay distinct Julia methods (2026-10-31).
#
# An integer parameter is annotated `::Integer` so a caller can pass any Julia
# integer. That made `to_chars(char*, char*, int, int)` and
# `to_chars(char*, char*, long long, int)` one Julia signature,
# `(Any, Any, Integer, Integer)`; `_dedup_method_chunks` kept the last and the
# other C++ entry point was unreachable (Boost.Charconv kept only the
# `unsigned long long` one of five). The generator now predicts the collision
# and keeps the C width on exactly the parameters the set disagrees on.
#
# No toolchain: the C++ generator on synthetic metadata.

using Test
using Libdl
using RepliBuild

const _IW_W = RepliBuild.Wrapper

function _iw_func(name, mangled, params; class="", is_method=false)
    Dict{String,Any}(
        "name" => name, "mangled" => mangled, "demangled" => name,
        "class" => class, "is_method" => is_method, "is_vararg" => false, "is_noexcept" => true,
        "exported" => true, "has_object_pointer" => nothing,
        "parameters" => Any[Dict{String,Any}("name" => n, "c_type" => c, "julia_type" => j) for (n, c, j) in params],
        "return_type" => Dict{String,Any}("c_type" => "void", "julia_type" => "Cvoid"))
end

@testset "Integer-width overloads stay distinct" begin
    dir = mktempdir()
    write(joinpath(dir, "replibuild.toml"), """
    [project]
    name = "intwidthsynth"
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
    ptr = ("first", "char*", "Ptr{UInt8}")
    ptr2 = ("last", "char*", "Ptr{UInt8}")
    meta = Dict{String,Any}(
        "functions" => Any[
            # five widths of one name: the value differs, the base does not
            _iw_func("boost::charconv::to_chars", "_ZN5boost8charconv8to_charsEPcS1_ii",
                     [ptr, ptr2, ("value", "int", "Cint"), ("base", "int", "Cint")]),
            _iw_func("boost::charconv::to_chars", "_ZN5boost8charconv8to_charsEPcS1_xi",
                     [ptr, ptr2, ("value", "long long", "Clonglong"), ("base", "int", "Cint")]),
            _iw_func("boost::charconv::to_chars", "_ZN5boost8charconv8to_charsEPcS1_ji",
                     [ptr, ptr2, ("value", "unsigned int", "Cuint"), ("base", "int", "Cint")]),
            # a lone integer function keeps the ergonomic annotation
            _iw_func("ns::g", "_ZN2ns1gEi", [("x", "int", "Cint")]),
            # two entry points that agree on every width are the old collapse
            _iw_func("ns::h", "_ZN2ns1hEi", [("x", "int", "Cint")]),
            _iw_func("ns::h", "_ZN2ns1hEi_alias", [("x", "int", "Cint")]),
        ],
        "globals" => Dict{String,Any}(),
        "struct_definitions" => Dict{String,Any}(),
        "record_abi" => Dict{String,Any}())
    res = _IW_W.generate_introspective_module_cpp(cfg, libref, meta, "IntWidthSynth",
                                                  _IW_W.create_type_registry(cfg), true)
    code = res isa Tuple ? res[1] : res

    sigs(name) = [m.match for m in eachmatch(Regex("^function \\w*" * name * "\\([^)]*\\)", "m"), code)]
    tc = sigs("to_chars")
    @test length(tc) == 3
    @test any(s -> occursin("value::Cint, base::Integer)", s), tc)
    @test any(s -> occursin("value::Clonglong, base::Integer)", s), tc)
    @test any(s -> occursin("value::Cuint, base::Integer)", s), tc)
    @test !any(s -> occursin("value::Integer", s), tc)
    @test !any(s -> occursin("base::Cint", s), tc)
    # the pointers stay `Any`, the lone function stays `Integer`
    @test all(s -> occursin("(first::Any, last::Any,", s), tc)
    @test sigs("g") == ["function ns_g(x::Integer)"]
    # agreeing widths are still one method (the dedup's job, unchanged)
    @test length(sigs("h")) == 1

    @testset "the prediction itself" begin
        strict = _IW_W._integer_width_overloads(meta["functions"], Set{String}(),
                                                (jt, ct) -> String(jt))
        @test strict["_ZN5boost8charconv8to_charsEPcS1_ii"] == Set([3])
        @test strict["_ZN5boost8charconv8to_charsEPcS1_xi"] == Set([3])
        @test !haskey(strict, "_ZN2ns1gEi")
        @test !haskey(strict, "_ZN2ns1hEi")
        # a varargs function is never grouped
        va = _iw_func("ns::v", "_ZN2ns1vEiz", [("x", "int", "Cint")])
        va["is_vararg"] = true
        va2 = _iw_func("ns::v", "_ZN2ns1vExz", [("x", "long long", "Clonglong")])
        va2["is_vararg"] = true
        @test isempty(_IW_W._integer_width_overloads(Any[va, va2], Set{String}(), (jt, ct) -> String(jt)))
    end

    # Julia's C aliases are not injective per host: `Cint === Clong` on LLP64
    # (Windows), `Clong === Clonglong` on LP64 (Linux). A set that differs only
    # in the spelling of one width is one dispatch signature, and defining it
    # twice is "Method overwriting is not permitted during Module
    # precompilation" in every consumer (2026-10-31 audit).
    fi = _iw_func("ns::f", "_ZN2ns1fEi", [("x", "int", "Cint")])
    fl = _iw_func("ns::f", "_ZN2ns1fEl", [("x", "long", "Clong")])
    fx = _iw_func("ns::f", "_ZN2ns1fEx", [("x", "long long", "Clonglong")])

    @testset "a width is the host's concrete type, not the C spelling" begin
        llp64 = Dict("Cint" => "Int32", "Clong" => "Int32", "Clonglong" => "Int64")
        lp64  = Dict("Cint" => "Int32", "Clong" => "Int64", "Clonglong" => "Int64")
        pred(fs, map) = _IW_W._integer_width_overloads(fs, Set{String}(), (jt, ct) -> String(jt);
                                                       concrete = t -> get(map, t, t))
        # one width on Windows, two on Linux
        @test isempty(pred(Any[fi, fl], llp64))
        @test pred(Any[fi, fl], lp64) == Dict("_ZN2ns1fEi" => Set([1]), "_ZN2ns1fEl" => Set([1]))
        # one width on Linux, two on Windows
        @test isempty(pred(Any[fl, fx], lp64))
        @test pred(Any[fl, fx], llp64) == Dict("_ZN2ns1fEl" => Set([1]), "_ZN2ns1fEx" => Set([1]))
        # three spellings, two widths: strict on both hosts, and the dedup
        # below has to see the two that coincide as one signature
        @test haskey(pred(Any[fi, fl, fx], llp64), "_ZN2ns1fEi")
        @test haskey(pred(Any[fi, fl, fx], lp64), "_ZN2ns1fEi")
        # the default is this host's answer
        host = _IW_W._integer_width_overloads(Any[fi, fl], Set{String}(), (jt, ct) -> String(jt))
        @test isempty(host) == (Cint === Clong)
    end

    @testset "the dedup compares concrete types" begin
        ki = _IW_W._method_sig_keys("function ns_f(x::Cint)\nend\n")
        kl = _IW_W._method_sig_keys("function ns_f(x::Clong)\nend\n")
        kx = _IW_W._method_sig_keys("function ns_f(x::Clonglong)\nend\n")
        @test (ki == kl) == (Cint === Clong)
        @test (kl == kx) == (Clong === Clonglong)
        @test _IW_W._method_sig_keys("function g(p::Ptr{Csize_t})\nend\n") ==
              _IW_W._method_sig_keys("function g(p::Ptr{$(string(Csize_t))})\nend\n")

        # End to end on this host: two spellings of one width stay one
        # `::Integer` method; three spellings of two widths give exactly two
        # methods, each with its own dispatch signature.
        gen(fs, mod) = begin
            m = Dict{String,Any}("functions" => fs, "globals" => Dict{String,Any}(),
                                 "struct_definitions" => Dict{String,Any}(), "record_abi" => Dict{String,Any}())
            r = _IW_W.generate_introspective_module_cpp(cfg, libref, m, mod, _IW_W.create_type_registry(cfg), true)
            c = r isa Tuple ? r[1] : r
            [String(x.captures[1]) for x in eachmatch(r"^function ns_f\(x::(\w+)\)"m, c)]
        end
        concrete(t) = t == "Integer" ? Integer : getfield(Base, Symbol(t))
        two = gen(Any[fi, fl], "IntWidthSynth2")
        if Cint === Clong
            @test two == ["Integer"]
        else
            @test sort(two) == ["Cint", "Clong"]
        end
        three = gen(Any[fi, fl, fx], "IntWidthSynth3")
        @test allunique(concrete.(three))
        @test length(three) == length(unique([Cint, Clong, Clonglong]))
        @test !("Integer" in three)
    end
end
