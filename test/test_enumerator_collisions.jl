#!/usr/bin/env julia
# test/test_enumerator_collisions.jl — a scoped enum's enumerators land at module
# scope, where a record of the same bare name replaces them (2026-11-01, found
# by John in the Hub's Boost.JSON wrapper).
#
# Boost.JSON declares
#
#     enum class kind : unsigned char { null, bool_, int64, uint64, double_,
#                                       string, array, object };
#     class string; class array; class object;
#
# `@enum kind::UInt8 begin … string = 5; array = 6; object = 7 end` came first
# in the wrapper, and `struct string`, `struct array`, `struct object` came
# later. Julia 1.12+ allows the redefinition silently, so `BoostJson.object`
# was the struct and `kind::object` was reachable only as `kind(7)`; a consumer
# writing `value_kind(v) == BoostJson.object` compared an enum against a type.
# `literals::null` rebound `kind::null` the same way, enum against enum.
#
# A second symptom in the same wrap: the export list carried the raw DWARF
# spelling (`bool_`, `double_`, `true_`, `false_` — C++ keywords with Boost's
# trailing underscore) while the `@enum` bound the sanitized one (`bool`,
# `double`, `c_true`, `c_false`), so the wrap reported four names "dropped
# from `export` — the module never binds them". Two derivations of one name.
#
# Now: `_enumerator_julia_name` is the one derivation, `_render_enum_chunks`
# prefixes a colliding enumerator with its enum's name (`kind_object`, as C++
# spells it) and says so in a comment, the export list carries the names the
# `@enum` binds, and `_assert_no_duplicate_toplevel_names` refuses any wrapper
# that still binds a top-level name twice.
#
# Library-free: drives both generators over synthetic metadata and EXECUTES
# what they produce against libjulia (any loadable library satisfies the
# module's load-time check; nothing here is called).

using Test
using RepliBuild
using Libdl

const W = RepliBuild.Wrapper

const _LIBREF = abspath(first(filter(p -> occursin("libjulia", basename(p)), Libdl.dllist())))

# Julia 1.12+ world-age rules for bindings made by `include_string`: every read
# of a probe module's binding goes through invokelatest.
_get(m::Module, s::Symbol) = Base.invokelatest(getfield, m, s)
_has(m::Module, s::Symbol) = Base.invokelatest(isdefined, m, s)

function _enum_config(dir::String, lang::String)
    toml = joinpath(dir, "replibuild.toml")
    write(toml, """
    [project]
    name = "enumcollide"
    root = "$(escape_string(dir))"

    [link]
    enable_lto = false

    [wrap]
    language = "$lang"
    """)
    return RepliBuild.ConfigurationManager.load_config(toml)
end

_e(name, value) = Dict{String,Any}("name" => name, "value" => value)
_enum(underlying, jt, members...) = Dict{String,Any}(
    "kind" => "enum", "underlying_type" => underlying, "julia_type" => jt,
    "enumerators" => Any[members...])
# A record outside `std`, so the C++ STL screen leaves it alone, like
# boost::json's records.
_record() = Dict{String,Any}(
    "kind" => "struct", "byte_size" => "0x4", "scope" => "synth",
    "members" => Any[Dict{String,Any}("name" => "n", "julia_type" => "Cint", "c_type" => "int",
                                      "offset" => "0x0", "size" => 4)])
_fn(name) = Dict{String,Any}(
    "name" => name, "mangled" => name, "demangled" => name, "class" => "",
    "is_method" => false, "is_vararg" => false,
    "parameters" => Any[Dict{String,Any}("name" => "n", "julia_type" => "Cint", "c_type" => "int")],
    "return_type" => Dict{String,Any}("julia_type" => "Cint", "c_type" => "int"))

# Boost.JSON's shape: `kind` and `literals` share `null`; `kind::string`,
# `kind::array` and `kind::object` are also records; `bool_`/`double_` and
# `true_`/`false_` carry the trailing underscore. Plus a global `string` and,
# where asked, a function `array`: a global and a function are top-level
# names too.
function fixture(; with_function::Bool = false, records = ("object", "array", "string"))
    structs = Dict{String,Any}(
        "__enum__kind" => _enum("unsigned char", "UInt8",
            _e("null", 0), _e("bool_", 1), _e("int64", 2), _e("uint64", 3),
            _e("double_", 4), _e("string", 5), _e("array", 6), _e("object", 7)),
        "__enum__literals" => _enum("int", "Cint",
            _e("null", 0), _e("true_", 1), _e("false_", 2), _e("infinity", 3)),
    )
    for r in records
        structs[r] = _record()
    end
    Dict{String,Any}(
        "functions" => with_function ? Any[_fn("array")] : Any[],
        "globals" => Dict{String,Any}("string" => Dict{String,Any}("julia_type" => "Cint")),
        "struct_definitions" => structs,
    )
end

# Two bitflag enums sharing `none`: boostfilesystem's `perms` and
# `perm_options`, which the survey of existing Hub wrappers showed bound
# `const none = …` twice.
function bitflag_fixture()
    Dict{String,Any}(
        "functions" => Any[], "globals" => Dict{String,Any}(),
        "struct_definitions" => Dict{String,Any}(
            "__enum__perms" => _enum("unsigned int", "Cuint",
                _e("none", 0), _e("owner_read", 256), _e("owner_write", 128),
                _e("owner_exe", 64), _e("owner_all", 448)),
            "__enum__perm_options" => _enum("unsigned int", "Cuint",
                _e("none", 0), _e("replace", 1), _e("add", 2), _e("remove", 4)),
        ))
end

function generate(lang::Symbol, md; module_name::String = "EnumCollide")
    dir = mktempdir()
    try
        cfg = _enum_config(dir, lang === :cpp ? "cpp" : "c")
        gen = lang === :cpp ? W.generate_introspective_module_cpp :
                              W.generate_introspective_module_c
        out = gen(cfg, _LIBREF, md, module_name, W.create_type_registry(cfg), false)
        return out isa Tuple ? String(first(out)) : String(out)
    finally
        rm(dir; recursive = true, force = true)
    end
end

# `Meta.parseall` embeds a failure instead of raising it, so walk the tree.
function parses_clean(src::AbstractString)
    ex = Meta.parseall(src)
    stack = Any[ex]
    while !isempty(stack)
        node = pop!(stack)
        node isa Expr || continue
        node.head in (:error, :incomplete) && return false
        append!(stack, node.args)
    end
    return true
end

# The lines of one `@enum` block, by type name.
function enum_block(src::AbstractString, name::AbstractString)
    m = match(Regex("^@enum $(name)::\\w+ begin\\n(.*?)^end\\n", "ms"), src)
    return m === nothing ? "" : String(m.captures[1])
end

# Evaluate the whole generated module and hand its module object back.
function load_module(src::AbstractString, tag::Symbol)
    m = Module(tag)
    Base.include_string(m, src)
    return _get(m, :EnumCollide)
end

@testset "enumerators colliding with top-level names" begin

    @testset "one derivation of an enumerator's name" begin
        for s in (W._sanitize_cpp_type_name, W._sanitize_c_type_name)
            # Boost's trailing underscore goes, as the sanitizer has always done;
            # the export list now follows.
            @test W._enumerator_julia_name("bool_", "kind", s) == "bool"
            @test W._enumerator_julia_name("double_", "kind", s) == "double"
            # A Julia keyword keeps the sanitizer's escape.
            @test W._enumerator_julia_name("true_", "literals", s) == "c_true"
            @test W._enumerator_julia_name("object", "kind", s) == "object"
            # The collision prefix is the enum's name, as C++ spells `kind::object`.
            @test W._enumerator_julia_name("object", "kind", s; collides = true) == "kind_object"
            # The keyword escape comes last, so a prefixed name is never escaped.
            @test W._enumerator_julia_name("nothing", "k", s) == "var\"nothing\""
            @test W._enumerator_julia_name("nothing", "k", s; collides = true) == "k_nothing"
        end
    end

    @testset "the collision is real without a guard" begin
        # Pin the Julia behaviour this guards against: a struct named after an
        # enumerator replaces it, silently. Were Julia to refuse it, the rule
        # could be simplified and this test would say so.
        m = Module(:EnumRecordClobber)
        Base.include_string(m, """
            @enum kind::UInt8 null = 0 object = 7
            struct object; n::Cint; end
        """)
        @test _get(m, :object) isa DataType
        @test !(_get(m, :object) isa _get(m, :kind))
    end

    @testset "defined bindings carry their kind and line" begin
        b = W._defined_bindings("""
        module M
        @enum kind::UInt8 begin
            null = 0
            object = 7
        end
        struct object
            n::Cint
        end
        function object() end
        const C = 1
        x = 2
        f(a) = a
        abstract type A end
        end
        """)
        @test ("kind", :enum, 2) in b
        @test ("null", :enumerator, 3) in b
        @test ("object", :enumerator, 4) in b
        @test ("object", :struct, 6) in b
        @test ("object", :function, 9) in b
        @test ("C", :const, 10) in b
        @test ("x", :global, 11) in b
        @test ("f", :function, 12) in b
        @test ("A", :type, 13) in b
        # The flat set is the same walk.
        @test W._defined_names("module M\nconst C = 1\nf(a) = a\nend") == Set(["C", "f"])
    end

    @testset "the write-time refusal" begin
        A = W._assert_no_duplicate_toplevel_names
        # Overloads and a type with its constructor are not duplicates.
        @test A("module M\nfunction f(a) end\nfunction f(a, b) end\nf(x, y, z) = 1\nend", "M") === nothing
        @test A("module M\nstruct T\n    n::Cint\nend\nfunction T() T(0) end\nend", "M") === nothing
        @test A("module M\n@enum E::Cint a = 0\nE(x::Ptr) = E(0)\nend", "M") === nothing
        # Everything else bound twice is refused, with the line of each binding.
        for (body, name) in (
                ("@enum kind::UInt8 begin\n    object = 7\nend\nstruct object\n    n::Cint\nend", "object"),
                ("const none = 1\nconst none = 2", "none"),
                ("const x = 1\nfunction x() end", "x"),
                ("struct S end\nmutable struct S\n    p::Ptr{Cvoid}\nend", "S"),
                ("@enum A::Cint null = 0\n@enum B::Cint null = 0", "null"),
                ("f() = 1\nconst f = 2", "f"))
            err = try
                A("module M\n$body\nend", "M"); nothing
            catch e
                e
            end
            @test (name, err isa ErrorException) == (name, true)
            @test (name, occursin("`$name`:", sprint(showerror, err))) == (name, true)
            @test (name, occursin("at line", sprint(showerror, err))) == (name, true)
        end
        # An unparseable body is `_assert_wrapper_parses`'s to report.
        @test A("module Broken\nfunction f(\nend", "Broken") === nothing
        # And it runs on the wrap path, with the others.
        @test_throws ErrorException W._assert_wrapper_loadable(
            "module M\n@enum kind::UInt8 begin\n    object = 7\nend\nstruct object\n    n::Cint\nend\nend", "M")
    end

    checked = 0
    for lang in (:c, :cpp)
        @testset "$lang: colliding enumerators are prefixed, the rest keep their names" begin
            # The C++ module is loaded without the function: a C++ free function
            # routes to Tier 2, whose `__init__` needs libJLCS. Its collision is
            # checked on the text below.
            logs, src = Test.collect_test_logs() do
                generate(lang, fixture(; with_function = lang === :c))
            end
            @test parses_clean(src)
            # No derivation disagrees with another any more.
            @test !any(l -> occursin("dropped from `export`", string(l.message)), logs)

            kind = enum_block(src, "kind")
            lits = enum_block(src, "literals")
            @test !isempty(kind) && !isempty(lits)
            # The records win their bare names; the enumerators take the enum's.
            @test occursin(r"^struct object\b"m, src)
            @test occursin(r"^struct array\b"m, src)
            @test occursin(r"^struct string\b"m, src)
            @test occursin(r"^    kind_object = 7$"m, kind)
            @test occursin(r"^    kind_array = 6$"m, kind)
            @test occursin(r"^    kind_string = 5$"m, kind)
            @test !occursin(r"^    object = 7$"m, kind)
            # `null` belongs to neither enum.
            @test occursin(r"^    kind_null = 0$"m, kind)
            @test occursin(r"^    literals_null = 0$"m, lits)
            @test !occursin(r"^    null = 0$"m, src)
            # Nothing collides with these, so they keep the names they had:
            # `bool_` and `double_` lose Boost's trailing underscore as before,
            # `true_`/`false_` keep the keyword escape.
            @test occursin(r"^    bool = 1$"m, kind)
            @test occursin(r"^    int64 = 2$"m, kind)
            @test occursin(r"^    uint64 = 3$"m, kind)
            @test occursin(r"^    double = 4$"m, kind)
            @test occursin(r"^    c_true = 1$"m, lits)
            @test occursin(r"^    c_false = 2$"m, lits)
            @test occursin(r"^    infinity = 3$"m, lits)
            # Each collision is named above the `@enum`.
            @test occursin("# `object` also names a struct this module defines; the enumerator is `kind_object`.", src)
            @test occursin("# `null` also names an enumerator of `literals`; the enumerator is `kind_null`.", src)
            @test occursin("# `null` also names an enumerator of `kind`; the enumerator is `literals_null`.", src)
            # The global `string` is a function `string()` in the module; the
            # record of that name is what the comment leads with.
            @test occursin(r"^function string\(\)"m, src)

            # The export line carries the names the `@enum` binds, and only those.
            exported = W._exported_names(src)
            for n in ("kind", "literals", "kind_object", "kind_array", "kind_string", "kind_null",
                      "literals_null", "bool", "double", "int64", "c_true", "c_false", "object")
                @test (n, n in exported) == (n, true)
            end
            for n in ("bool_", "double_", "true_", "false_", "null")
                @test (n, n in exported) == (n, false)
            end

            # Neutered rename: put the bare name back and the refusal fires on
            # exactly that name. This is what every wrap now runs.
            neutered = replace(src, "    kind_object = 7" => "    object = 7")
            @test neutered != src
            err = try
                W._assert_no_duplicate_toplevel_names(neutered, "EnumCollide"); nothing
            catch e
                e
            end
            @test err isa ErrorException
            @test occursin("`object`: enumerator at line", sprint(showerror, err))
            @test W._assert_no_duplicate_toplevel_names(src, "EnumCollide") === nothing

            # The module loads, and the bindings are what the text says.
            M = load_module(src, Symbol("EnumCollideProbe_", lang))
            K = _get(M, :kind)
            L = _get(M, :literals)
            @test _get(M, :object) isa DataType && isstructtype(_get(M, :object))
            @test _get(M, :kind_object) isa K
            @test Int(_get(M, :kind_object)) == 7
            @test _get(M, :kind_array) isa K
            @test _get(M, :kind_string) isa K
            @test _get(M, :kind_null) isa K && Int(_get(M, :kind_null)) == 0
            @test _get(M, :literals_null) isa L && Int(_get(M, :literals_null)) == 0
            @test _get(M, :bool) isa K && Int(_get(M, :bool)) == 1
            @test _get(M, :double) isa K
            @test _get(M, :c_true) isa L
            @test !_has(M, :null)
            @test !_has(M, :bool_)
            # Exported under the renamed name: `using` offers it.
            @test :kind_object in names(M)
            @test :object in names(M)
            @test :bool in names(M)
            checked += 1
        end

        @testset "$lang: a function is a top-level name too" begin
            src = generate(lang, fixture(; with_function = true))
            @test parses_clean(src)
            @test occursin(r"^function array\("m, src)
            @test occursin(r"^    kind_array = 6$"m, enum_block(src, "kind"))
            @test occursin("# `array` also names a struct this module defines; the enumerator is `kind_array`.", src)
            # With the record gone the function alone still collides.
            src2 = generate(lang, fixture(; with_function = true, records = ("object",)))
            @test occursin(r"^    kind_array = 6$"m, enum_block(src2, "kind"))
            @test occursin("# `array` also names a function this module defines; the enumerator is `kind_array`.", src2)
            # And `string`, now only a global's accessor, collides with that.
            @test occursin(r"^    kind_string = 5$"m, enum_block(src2, "kind"))
            @test occursin("# `string` also names a function this module defines; the enumerator is `kind_string`.", src2)
            @test W._assert_no_duplicate_toplevel_names(src, "EnumCollide") === nothing
            @test W._assert_no_duplicate_toplevel_names(src2, "EnumCollide") === nothing
        end

        @testset "$lang: two bitflag enums sharing a member" begin
            src = generate(lang, bitflag_fixture())
            @test parses_clean(src)
            @test occursin(r"^const perms = Cuint$"m, src)
            @test occursin(r"^const perm_options = Cuint$"m, src)
            @test occursin(r"^const perms_none = perms\(0\)$"m, src)
            @test occursin(r"^const perm_options_none = perm_options\(0\)$"m, src)
            @test !occursin(r"^const none = "m, src)
            @test occursin(r"^const owner_read = perms\(256\)$"m, src)
            @test W._assert_no_duplicate_toplevel_names(src, "EnumCollide") === nothing
            exported = W._exported_names(src)
            @test "perms_none" in exported && "perm_options_none" in exported && !("none" in exported)
        end

        @testset "$lang: nothing collides, nothing moves" begin
            # One enum, records named after none of its members: the `@enum`
            # text is byte for byte what the emitters wrote before the rule.
            md = fixture(; records = ("widget",))
            delete!(md["struct_definitions"], "__enum__literals")
            delete!(md["globals"], "string")
            src = generate(lang, md)
            label = lang === :cpp ? "C++" : "C"
            @test occursin("""
                # $label enum: kind (underlying type: unsigned char)
                @enum kind::UInt8 begin
                    null = 0
                    bool = 1
                    int64 = 2
                    uint64 = 3
                    double = 4
                    string = 5
                    array = 6
                    object = 7
                end

                """, src)
            @test !occursin("also names", src)
            exported = W._exported_names(src)
            for n in ("kind", "null", "bool", "double", "object", "widget")
                @test (n, n in exported) == (n, true)
            end
            @test !("kind_object" in exported)
        end
    end
    @test checked == 2
end
