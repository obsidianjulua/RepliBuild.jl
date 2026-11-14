#!/usr/bin/env julia
# test/test_tier2_typed_dispatch.jl — C++ overloads that differ in a pointer's
# target, or in an enum against a pointer, stay distinct Julia methods
# (2026-11-13).
#
# A pointer parameter is annotated `::Any` so a `Managed` object, a `Ref`, a
# `String` all pass through `Base.unsafe_convert`, and an emitted enum relaxes
# the same way so an `Integer` still passes. That made `GetID(const char*)` and
# `GetID(const void*)` one Julia signature, `(this::Any, x::Any)`, and
# `_dedup_method_chunks` kept one: 52 of the Hub's 89 dropped C++ functions
# (test/tools/cpp_thunk_coverage.jl). The overload prediction that already kept
# the C width on disagreeing integer positions (`_integer_width_overloads`)
# now spells disagreeing pointer and enum positions too
# (`_strict_param_spelling`): the pointer's target as a `Union` of what callers
# pass, the enum as itself, `void*` as the `Any` it always was. A move overload
# beside its copy overload is not emitted at all (`_rvalue_shadowed`): Julia
# has no rvalue, and the kept one used to be whichever came last. And the thunk
# manifest is the set the FINAL text calls (`_thunks_named_in_text`), so a
# chunk the dedup dropped leaves no thunk behind.
#
# No toolchain: the C++ generator on synthetic metadata.
#
# Negative checks: make `_strict_param_spelling` return "Any" and the first two
# testsets fail (one method where two are expected); make `_rvalue_shadowed`
# return an empty set and the move testset fails; drop the `filter!` on
# `needed_function_thunks` and the manifest testset fails.

using Test
using Libdl
using RepliBuild

const _TD_W = RepliBuild.Wrapper

function _td_func(name, mangled, params; class="", is_method=false, ret=("void", "Cvoid"))
    Dict{String,Any}(
        "name" => name, "mangled" => mangled, "demangled" => name,
        "class" => class, "is_method" => is_method, "is_vararg" => false, "is_noexcept" => false,
        "exported" => true, "has_object_pointer" => nothing,
        "parameters" => Any[Dict{String,Any}("name" => n, "c_type" => c, "julia_type" => j) for (n, c, j) in params],
        "return_type" => Dict{String,Any}("c_type" => ret[1], "julia_type" => ret[2]))
end

_td_struct(name, members) = Dict{String,Any}(
    "name" => name, "kind" => "struct", "byte_size" => string("0x", string(8 * length(members), base = 16)),
    "members" => Any[Dict{String,Any}("name" => m, "c_type" => "long", "julia_type" => "Clong",
                                      "offset" => string("0x", string(8 * (i - 1), base = 16)), "size" => 8)
                     for (i, m) in enumerate(members)])

@testset "Tier 2 typed dispatch" begin
    dir = mktempdir()
    write(joinpath(dir, "replibuild.toml"), """
    [project]
    name = "typeddispatch"
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
    cstr = ("s", "const char*", "Ptr{UInt8}")
    structs = Dict{String,Any}(
        "Rec" => _td_struct("Rec", ["a", "b"]),
        "Other" => _td_struct("Other", ["x"]),
        "__enum__Kind" => Dict{String,Any}("kind" => "enum", "name" => "Kind", "byte_size" => "0x4",
            "underlying_type" => "int", "julia_type" => "Int32",
            "enumerators" => Any[Dict{String,Any}("name" => "KIND_A", "value" => 1),
                                 Dict{String,Any}("name" => "KIND_B", "value" => 2)]))
    meta = Dict{String,Any}(
        "functions" => Any[
            # a char pointer against a void pointer: imgui's GetID
            _td_func("ns::id", "_ZN2ns2idEPKc", [cstr]),
            _td_func("ns::id", "_ZN2ns2idEPKv", [("p", "const void*", "Ptr{Cvoid}")]),
            # an enum against a char pointer: pugixml's append_child
            _td_func("ns::child", "_ZN2ns5childE4Kind", [("k", "Kind", "Kind")]),
            _td_func("ns::child", "_ZN2ns5childEPKc", [cstr]),
            # two records' pointers: imgui's TableGcCompactTransientBuffers
            _td_func("ns::compact", "_ZN2ns7compactEP3Rec", [("r", "Rec*", "Ptr{Rec}")]),
            _td_func("ns::compact", "_ZN2ns7compactEP5Other", [("o", "Other*", "Ptr{Other}")]),
            # a char pointer against a wide one: pugixml's load_file
            _td_func("ns::load", "_ZN2ns4loadEPKc", [cstr]),
            _td_func("ns::load", "_ZN2ns4loadEPKw", [("w", "const wchar_t*", "Ptr{Cwchar_t}")]),
            # a lone pointer function and a lone enum function keep the ergonomic Any
            _td_func("ns::lone", "_ZN2ns4loneEPKc", [cstr]),
            _td_func("ns::lonek", "_ZN2ns5lonekE4Kind", [("k", "Kind", "Kind")]),
            # two spellings of one pointer type are one method, as before
            _td_func("ns::same", "_ZN2ns4sameEPKc", [cstr]),
            _td_func("ns::same", "_ZN2ns4sameEPc", [("s", "char*", "Ptr{UInt8}")]),
            # a move overload beside its copy overload: msdfgen's addContour
            _td_func("ns::add", "_ZN2ns3addERKNS_3RecE", [("r", "const Rec&", "Ref{Rec}")]),
            _td_func("ns::add", "_ZN2ns3addEONS_3RecE", [("r", "Rec&&", "Ref{Rec}")]),
            # a move overload with no lvalue twin is emitted
            _td_func("ns::take", "_ZN2ns4takeEONS_3RecE", [("r", "Rec&&", "Ref{Rec}")]),
            # two integer spellings of one width (LP64: long, long long) stay one method
            _td_func("ns::setv", "_ZN2ns4setvEl", [("v", "long", "Clong")]),
            _td_func("ns::setv", "_ZN2ns4setvEx", [("v", "long long", "Clonglong")]),
            # a receiver DWARF supplied is never a strict position: two entries of
            # one destructor attributed to a base and a derived class (tinyxml2's
            # `~XMLComment` entries) differ only in what `this` points at
            _td_func("~Rec", "_ZN2ns3RecD2Ev", [("this", "Rec*", "Ptr{Rec}")]; class = "ns::Rec", is_method = true),
            _td_func("~Rec", "_ZN2ns5OtherD2Ev", [("this", "Other*", "Ptr{Other}")]; class = "ns::Rec", is_method = true),
        ],
        "globals" => Dict{String,Any}(),
        "struct_definitions" => structs,
        "record_abi" => Dict{String,Any}())
    res = _TD_W.generate_introspective_module_cpp(cfg, libref, meta, "TypedDispatch",
                                                  _TD_W.create_type_registry(cfg), true)
    code = res[1]
    needed = res[2]

    sigs(name) = [m.match for m in eachmatch(Regex("^function ns_" * name * "\\([^\\n]*\\)", "m"), code)]
    bodies(name) = [m.captures[1] for m in eachmatch(Regex("^function ns_" * name * "\\([^\\n]*\\)\\n(.*?)^end", "ms"), code)]
    thunk_of(body) = (m = match(r"_mlir_ciface_(\S+?)_thunk", body); m === nothing ? "" : String(m.captures[1]))
    cunion = _TD_W._CHAR_POINTER_UNION

    @testset "a char pointer against a void pointer" begin
        s = sigs("id")
        @test length(s) == 2
        @test "function ns_id(s::$cunion)" in s
        @test "function ns_id(p::Any)" in s
        @test Set(thunk_of.(bodies("id"))) == Set(["_ZN2ns2idEPKc", "_ZN2ns2idEPKv"])
    end

    @testset "an enum against a char pointer" begin
        s = sigs("child")
        @test length(s) == 2
        @test "function ns_child(k::Kind)" in s
        @test "function ns_child(s::$cunion)" in s
        # the enum argument is still converted to its base type in the body
        @test any(b -> occursin("Int32(k)", b), bodies("child"))
    end

    @testset "two records' pointers" begin
        s = sigs("compact")
        @test length(s) == 2
        @test "function ns_compact(r::Union{Ptr{Rec}, Ref{Rec}, AbstractVector{Rec}})" in s
        @test "function ns_compact(o::Union{Ptr{Other}, Ref{Other}, AbstractVector{Other}})" in s
    end

    @testset "a char pointer against a wide one" begin
        s = sigs("load")
        @test length(s) == 2
        @test "function ns_load(s::$cunion)" in s
        @test "function ns_load(w::Union{Ptr{Cwchar_t}, Ref{Cwchar_t}, AbstractVector{Cwchar_t}})" in s
    end

    @testset "a lone function keeps the ergonomic Any" begin
        @test sigs("lone") == ["function ns_lone(s::Any)"]
        @test sigs("lonek") == ["function ns_lonek(k::Any)"]
    end

    @testset "one pointer type in two spellings is one method" begin
        @test length(sigs("same")) == 1
        @test sigs("same")[1] == "function ns_same(s::Any)"
    end

    @testset "a move overload beside its copy overload is not emitted" begin
        s = sigs("add")
        @test length(s) == 1
        @test thunk_of.(bodies("add")) == ["_ZN2ns3addERKNS_3RecE"]       # the lvalue one, whatever the order
        @test !occursin("_ZN2ns3addEONS_3RecE", code)
        @test length(sigs("take")) == 1                                   # no twin: emitted
        @test _TD_W._rvalue_shadowed(meta["functions"]) == Set(["_ZN2ns3addEONS_3RecE"])
    end

    @testset "a receiver is never a strict position" begin
        d = [m.match for m in eachmatch(r"^function ns_Rec_destroy_Rec\([^\n]*\)"m, code)]
        @test d == ["function ns_Rec_destroy_Rec(this::Any)"]
        strict = _TD_W._integer_width_overloads(meta["functions"], Set{String}(), (jt, ct) -> String(jt);
                                                bound = t -> t in ("Rec", "Other") || _TD_W._type_name_bound(t, Set{String}()))
        @test !haskey(strict, "_ZN2ns3RecD2Ev") && !haskey(strict, "_ZN2ns5OtherD2Ev")
    end

    @testset "two integer spellings of one width stay one method" begin
        if Clong === Clonglong
            @test length(sigs("setv")) == 1
        else
            @test length(sigs("setv")) == 2
        end
    end

    @testset "the manifest is what the final text calls" begin
        named = _TD_W._thunks_named_in_text(code, Dict{String,String}())
        @test needed == named                                             # nothing the dedup dropped
        @test "_ZN2ns2idEPKc" in needed && "_ZN2ns2idEPKv" in needed
        @test !("_ZN2ns3addEONS_3RecE" in needed)                         # never emitted
        if Clong === Clonglong
            @test length(intersect(needed, Set(["_ZN2ns4setvEl", "_ZN2ns4setvEx"]))) == 1
        end
        # the three spellings a call site has
        taken = Dict("_FPTR_x" => "_Z1xv")
        @test _TD_W._thunks_named_in_text(
                  "invoke(\"_mlir_ciface__Z1av_thunk\", a)\ninvoke_aot_ptr(_FPTR_x[], 1)\n" *
                  "ccall((:thunk__Z1vv, THUNKS_LIBRARY_PATH), Cvoid, (), )", taken) ==
              Set(["_Z1av", "_Z1xv", "_Z1vv"])
    end

    @testset "the prediction spells a strict position as the loop does" begin
        bound(t) = t in ("Rec", "Other") || _TD_W._type_name_bound(t, Set{String}())
        sp(t) = _TD_W._strict_param_spelling(t; bound, is_enum = t -> t == "Kind",
                                             owners = t -> t == "Rec" ? ["ManagedRec"] : String[])
        @test sp("Cint") == "Cint"
        @test sp("Ptr{Cvoid}") == "Any"
        @test sp("Ptr{UInt8}") == cunion && sp("Ptr{Int8}") == cunion && sp("Cstring") == cunion
        @test sp("Ptr{Rec}") == "Union{Ptr{Rec}, Ref{Rec}, AbstractVector{Rec}, ManagedRec}"
        @test sp("Ptr{UnemittedRecord}") == "Any"                         # an unbound target
        @test sp("Kind") == "Kind"
        @test sp("Cdouble") == "Cdouble"
        strict = _TD_W._integer_width_overloads(meta["functions"], Set{String}(), (jt, ct) -> String(jt);
                                                bound, is_enum = t -> t == "Kind")
        @test strict["_ZN2ns2idEPKc"] == Set([1]) && strict["_ZN2ns2idEPKv"] == Set([1])
        @test strict["_ZN2ns5childE4Kind"] == Set([1])
        @test !haskey(strict, "_ZN2ns4loneEPKc") && !haskey(strict, "_ZN2ns4sameEPKc")
    end

    # Every method defined once: what precompilation refuses is one signature twice.
    keys_all = _TD_W._method_sig_keys(code)
    @test allunique(keys_all)
    ex = Meta.parseall(code)
    @test !any(a -> a isa Expr && a.head in (:incomplete, :error), ex.args)
end
