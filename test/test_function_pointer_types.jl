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
# into a generated docstring.
#
# Library-free: synthetic metadata, no toolchain.

using Test
using RepliBuild
using Libdl

const _FP_C = RepliBuild.Compiler
const _FP_W = RepliBuild.Wrapper
const _FP_LIB = abspath(first(filter(p -> occursin("libjulia", basename(p)), Libdl.dllist())))

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

    @testset "a DWARF signature is escaped into its docstring" begin
        # A callback declared without a header typedef is documented with
        # DWARF's own signature as text, and that text now names every
        # parameter type. It lands inside a generated `"""` docstring, where a
        # `\` starts an escape (`\U` fails the parse) and `$` interpolates. A
        # C++ type name can hold a backslash: clang prints a `char` template
        # argument as a literal, `Tag<'\\'>`.
        sig = "function_ptr(void; Tag<'\\U'>*, cost\$t*)*"
        function generate(lang)
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
            row = only(_FP_C.parse_function_signatures(
                [Dict{String,Any}("mangled" => "set_hook", "demangled" => "set_hook")]))
            row["parameters"] = Any[Dict{String,Any}("name" => "hook", "julia_type" => "Ptr{Cvoid}",
                                                     "c_type" => sig, "function_pointer_signature" => sig)]
            row["return_type"] = Dict{String,Any}("c_type" => "void", "julia_type" => "Cvoid", "size" => 0)
            row["is_noexcept"] = true
            md = Dict{String,Any}("functions" => Any[row], "struct_definitions" => Dict{String,Any}())
            gen = lang == "cpp" ? _FP_W.generate_introspective_module_cpp : _FP_W.generate_introspective_module_c
            out = gen(cfg, _FP_LIB, md, "FpDocs", _FP_W.create_type_registry(cfg), true)
            return out isa Tuple ? String(first(out)) : String(out)
        end
        checked = 0
        for lang in ("c", "cpp")
            code = generate(lang)
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
