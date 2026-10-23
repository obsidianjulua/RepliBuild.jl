#!/usr/bin/env julia
# Consumers of a function-pointer c_type with a parameter list.
#
# Until 2026-10-23 no DWARF function pointer had one: pass 2 of
# parse_dwarf_dump never collected a subroutine type's parameters, so a
# four-argument callback was `function_ptr(bool)*` (test_dwarf_attribution.jl
# pins the parser). Now a c_type can hold `;`, commas, brackets and nested
# parentheses inside its own parentheses, and each consumer here read the
# whole string for something only its outer type could mean: the first `[` as
# the array's extent, any `float` as a float member, and the text pasted raw
# into a generated docstring. And a callback whose headers spell no typedef
# now gets its `@cfunction` example from that signature (John's decision).
#
# Library-free: synthetic metadata, no toolchain.

using Test
using RepliBuild
using Libdl

const _FP_C = RepliBuild.Compiler
const _FP_W = RepliBuild.Wrapper
const _FP_LIB = abspath(first(filter(p -> occursin("libjulia", basename(p)), Libdl.dllist())))

# A callback parameter as parse_dwarf_dump records it.
_fp_param(name, sig; kw...) = Dict{String,Any}("name" => name, "julia_type" => "Ptr{Cvoid}",
                                               "c_type" => sig, "function_pointer_signature" => sig,
                                               (String(k) => v for (k, v) in kw)...)

function _fp_function(fname, params...)
    row = only(_FP_C.parse_function_signatures([Dict{String,Any}("mangled" => fname, "demangled" => fname)]))
    row["parameters"] = Any[params...]
    row["return_type"] = Dict{String,Any}("c_type" => "void", "julia_type" => "Cvoid", "size" => 0)
    row["is_noexcept"] = true
    return row
end

# One generator's module text for `functions`, with records `Rec` and
# `baremodule` (a Julia keyword, so both generators declare it `c_baremodule`),
# an enum `Mode` and the header typedef `progress_cb` in the metadata.
# `fp_parameters = false` is metadata from before function-pointer signatures
# had parameter lists.
function _fp_generate(lang, functions; fp_parameters::Bool = true)
    dir = mktempdir()
    write(joinpath(dir, "replibuild.toml"), """
    [project]
    name = "fpdocs"
    root = "$(escape_string(dir))"
    [link]
    enable_lto = false
    [wrap]
    language = "$lang"
    [types]
    strictness = "warn"
    allow_unknown_structs = true
    [cache]
    enabled = false
    """)
    cfg = RepliBuild.ConfigurationManager.load_config(joinpath(dir, "replibuild.toml"))
    member(n, off) = Dict{String,Any}("name" => n, "c_type" => "int", "julia_type" => "Cint",
                                      "offset" => off, "size" => 4)
    md = Dict{String,Any}(
        "functions" => deepcopy(Any[functions...]),
        "struct_definitions" => Dict{String,Any}(
            "Rec" => Dict{String,Any}("kind" => "struct", "byte_size" => "0x8",
                                      "members" => Any[member("x", "0x0"), member("y", "0x4")]),
            "baremodule" => Dict{String,Any}("kind" => "struct", "byte_size" => "0x4",
                                             "members" => Any[member("x", "0x0")]),
            "__enum__Mode" => Dict{String,Any}("kind" => "enum", "underlying_type" => "int",
                "julia_type" => "Cint", "byte_size" => "0x4",
                "enumerators" => Any[Dict{String,Any}("name" => "MODE_A", "value" => 0),
                                     Dict{String,Any}("name" => "MODE_B", "value" => 1)])),
        "function_pointer_typedefs" => Dict{String,Any}(
            "progress_cb" => Dict{String,Any}("return_type" => "int", "parameters" => Any["int", "void*"])))
    fp_parameters && (md["function_pointer_parameters"] = true)
    gen = lang == "cpp" ? _FP_W.generate_introspective_module_cpp : _FP_W.generate_introspective_module_c
    out = gen(cfg, _FP_LIB, md, "FpDocs", _FP_W.create_type_registry(cfg), true)
    return out isa Tuple ? String(first(out)) : String(out)
end

@testset "Function-pointer c_types with parameter lists" begin

    @testset "an array's extents are the trailing run" begin
        # The first `[` was taken as the start of the extents, so a pointer to
        # an array inside the callback's signature multiplied the count: eight
        # function pointers for two, and 64 bytes read and written where the
        # member is 16.
        @test _FP_C.cpp_to_julia_type("function_ptr(void; int[4]*)*[2]") == "NTuple{2, Ptr{Cvoid}}"
        @test _FP_C.cpp_to_julia_type("function_ptr(void; const float[16]&)*[3]") == "NTuple{3, Ptr{Cvoid}}"
        # Not only callbacks: two pointers to `int[4]` were eight ints.
        @test _FP_C.cpp_to_julia_type("int[4]*[2]") == "NTuple{2, Ptr{NTuple{4, Cint}}}"
        # What already worked still does.
        @test _FP_C.cpp_to_julia_type("function_ptr(void; int)*[2]") == "NTuple{2, Ptr{Cvoid}}"
        @test _FP_C.cpp_to_julia_type("double[3]") == "NTuple{3, Cdouble}"
        @test _FP_C.cpp_to_julia_type("int[3][4]") == "NTuple{12, Cint}"
        @test _FP_C.cpp_to_julia_type("function_ptr(void; int[4]*)*") == "Ptr{Cvoid}"
    end

    @testset "a float in a callback's signature is not a float member" begin
        # Decides the SysV SSE-class trap for a small blob passed by value. An
        # array of pointers is INTEGER class whatever it points at.
        structs = Dict{String,Any}(
            "Callbacks" => Dict{String,Any}("members" => Any[
                Dict{String,Any}("name" => "fns", "c_type" => "function_ptr(void; float)*[2]")]),
            "Pointers" => Dict{String,Any}("members" => Any[
                Dict{String,Any}("name" => "ps", "c_type" => "double*[2]")]),
            "Floats" => Dict{String,Any}("members" => Any[
                Dict{String,Any}("name" => "v", "c_type" => "float[4]")]),
            "Callback" => Dict{String,Any}("members" => Any[
                Dict{String,Any}("name" => "fn", "c_type" => "function_ptr(double; float)*")]))
        @test !_FP_W._struct_has_float_member("Callbacks", structs)
        @test !_FP_W._struct_has_float_member("Pointers", structs)
        @test !_FP_W._struct_has_float_member("Callback", structs)
        @test _FP_W._struct_has_float_member("Floats", structs)
    end

    @testset "surface reachability reads a callback's signature" begin
        # The types a callback is handed are the ones a user needs to write it.
        # What a callback gets as `void *` stays unnamed (md4c's MD_*_DETAIL).
        rec(n) = Dict{String,Any}("kind" => "struct", "byte_size" => "0x4",
                                  "members" => Any[Dict{String,Any}("name" => n, "c_type" => "int",
                                                                    "julia_type" => "Cint", "offset" => "0x0")])
        md = Dict{String,Any}(
            "functions" => Any[Dict{String,Any}(
                "name" => "set_hook",
                "parameters" => Any[Dict{String,Any}(
                    "name" => "hook", "julia_type" => "Ptr{Cvoid}",
                    "c_type" => "function_ptr(int; Detail*, Mode, void*)*")],
                "return_type" => Dict{String,Any}("c_type" => "void", "julia_type" => "Cvoid"))],
            "struct_definitions" => Dict{String,Any}(
                "Detail" => rec("x"), "Unnamed" => rec("y"),
                "__enum__Mode" => Dict{String,Any}("kind" => "enum", "underlying_type" => "int",
                    "enumerators" => Any[Dict{String,Any}("name" => "MODE_A", "value" => 0)])))
        reach = _FP_W._surface_type_keys(md)
        @test "Detail" in reach
        @test "__enum__Mode" in reach
        @test !("Unnamed" in reach)
    end

    @testset "a DWARF signature: one pointer, no varargs, lists split at depth 0" begin
        # A mapper that names every type, so these rules alone decide.
        named(c) = replace(c, r"[^A-Za-z0-9]" => "_")
        sig(s) = _FP_W._dwarf_callback_signature(s, named)
        @test sig("function_ptr(int; int, unsigned int)*") == "my_callback, int, (int, unsigned_int,)"
        @test sig("function_ptr(void)*") == "my_callback, Cvoid, ()"
        # A template's comma and a nested signature's `;` are inside brackets.
        @test sig("function_ptr(void; std::map<int, int>*, int)*") ==
              "my_callback, Cvoid, (std__map_int__int__, int,)"
        @test sig("function_ptr(function_ptr(double; float)*; int[4]*)*") ==
              "my_callback, function_ptr_double__float__, (int_4__,)"
        # No example: variadic, a pointer to a callback, a reference, and a
        # type the mapper cannot name.
        @test sig("function_ptr(int; const char*, ...)*") === nothing
        @test sig("function_ptr(void; int)**") === nothing
        @test sig("function_ptr(void; int)&") === nothing
        @test _FP_W._dwarf_callback_signature("function_ptr(void; int, float)*",
                                              c -> c == "float" ? nothing : c) === nothing
    end

    @testset "a callback without a header typedef gets DWARF's example" begin
        # One signature twice: `typed` is declared with a typedef the headers
        # spell, `raw` is not. Both examples must be the same, which is what
        # "spelled the way the wrapper spells a parameter" has to mean for the
        # types both paths can name.
        typed = _fp_param("typed", "function_ptr(int; int, void*)*"; function_pointer_typedef = "progress_cb")
        raw = _fp_param("raw", "function_ptr(int; int, void*)*")
        fns = Any[
            _fp_function("set_typed", typed),
            _fp_function("set_raw", raw),
            # A record the wrapper declares, one it declares under another name,
            # an enum, a pointer to an undeclared record, and a callback as a
            # parameter.
            _fp_function("set_rec", _fp_param("rec", "function_ptr(void; Rec*, double)*")),
            _fp_function("set_kw", _fp_param("kw", "function_ptr(void; baremodule*)*")),
            _fp_function("set_mode", _fp_param("mode", "function_ptr(_Bool; Mode, unsigned int)*")),
            _fp_function("set_opaque", _fp_param("opaque", "function_ptr(void; Undeclared*)*")),
            _fp_function("set_nested", _fp_param("nested", "function_ptr(void; function_ptr(void; int)*, double)*")),
            _fp_function("set_none", _fp_param("none", "function_ptr(void)*")),
            # No example: variadic, a pointer to a callback (an out-parameter), a
            # reference to a function, and a type the wrapper cannot spell.
            _fp_function("set_va", _fp_param("va", "function_ptr(int; const char*, ...)*")),
            _fp_function("get_out", _fp_param("out", "function_ptr(void; int)**")),
            _fp_function("set_ref", _fp_param("ref", "function_ptr(void; int)&")),
            _fp_function("set_ld", _fp_param("ld", "function_ptr(void; long double)*")),
        ]
        expected = Dict(
            "typed"  => "my_callback, Cint, (Cint, Ptr{Cvoid},)",
            "raw"    => "my_callback, Cint, (Cint, Ptr{Cvoid},)",
            "rec"    => "my_callback, Cvoid, (Ptr{Rec}, Cdouble,)",
            "kw"     => "my_callback, Cvoid, (Ptr{c_baremodule},)",
            "mode"   => "my_callback, Bool, (Mode, Cuint,)",
            "opaque" => "my_callback, Cvoid, (Ptr{Cvoid},)",
            "nested" => "my_callback, Cvoid, (Ptr{Cvoid}, Cdouble,)",
            "none"   => "my_callback, Cvoid, ()")
        checked = 0
        for lang in ("c", "cpp")
            code = _fp_generate(lang, fns)
            for (pname, sig) in sort!(collect(expected))
                m = match(Regex("\\*\\*Callback `$pname`\\*\\*: Create using `@cfunction`\\n" *
                                "```julia\\ncallback = @cfunction\\((.*)\\) Ptr\\{Cvoid\\}\\n"), code)
                @test (lang, pname, m === nothing ? nothing : m.captures[1]) == (lang, pname, sig)
                checked += 1
            end
            for pname in ("va", "out", "ref", "ld")
                @test (lang, pname, occursin("**Callback `$pname`**", code)) == (lang, pname, false)
                @test (lang, pname, occursin("- `$pname::Ptr{Cvoid}` - Callback function (signature: function_ptr(", code)) ==
                      (lang, pname, true)
                checked += 1
            end
        end
        @test checked == 2 * (length(expected) + 4)
    end

    @testset "older metadata gets no example from DWARF" begin
        # Built before signatures had parameter lists, a callback's DWARF
        # signature is `function_ptr(R)*` whatever it takes. An example from it
        # would document a four-argument callback as taking none.
        fns = Any[_fp_function("set_typed", _fp_param("typed", "function_ptr(int)*";
                                                      function_pointer_typedef = "progress_cb")),
                  _fp_function("set_raw", _fp_param("raw", "function_ptr(int)*"))]
        checked = 0
        for lang in ("c", "cpp")
            code = _fp_generate(lang, fns; fp_parameters = false)
            @test (lang, occursin("**Callback `raw`**", code)) == (lang, false)
            @test (lang, occursin("- `raw::Ptr{Cvoid}` - Callback function (signature: function_ptr(int)*)", code)) ==
                  (lang, true)
            # The header's typedef does not depend on it.
            @test (lang, occursin("@cfunction(my_callback, Cint, (Cint, Ptr{Cvoid},))", code)) == (lang, true)
            checked += 1
        end
        @test checked == 2
    end

    @testset "a DWARF signature shown as text is escaped into its docstring" begin
        # A callback with no example (this one is variadic) is documented with
        # DWARF's signature as text, and that text names every parameter type.
        # It lands inside a generated `"""` docstring, where a `\` starts an
        # escape (`\U` fails the parse) and `$` interpolates. A C++ type name
        # can hold a backslash: clang prints a `char` template argument as a
        # literal, `Tag<'\\'>`.
        sig = "function_ptr(void; Tag<'\\U'>*, cost\$t*, ...)*"
        fns = Any[_fp_function("set_hook", _fp_param("hook", sig))]
        checked = 0
        for lang in ("c", "cpp")
            code = _fp_generate(lang, fns)
            m = match(r"Callback function \(signature: (.*)\)\n", code)
            @test (lang, m !== nothing) == (lang, true)
            m === nothing && continue
            # The docstring reads back as the signature, with nothing interpolated.
            @test (lang, Meta.parse("\"" * m.captures[1] * "\"")) == (lang, sig)
            ast = Meta.parseall(code)
            @test (lang, any(a -> a isa Expr && a.head in (:incomplete, :error), ast.args)) == (lang, false)
            checked += 1
        end
        @test checked == 2
    end
end
