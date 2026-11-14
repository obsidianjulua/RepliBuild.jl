#!/usr/bin/env julia
# test/test_function_name_derivation.jl — the Julia function NAME is one
# derivation, and it is total (2026-09-23, Hub fmt 12.1.0).
#
# fmt instantiates `detail::write_padded` on a lambda's closure type. The GNU
# demangler spells that template argument
#
#     …::write_char<char, …>(…)::{lambda(fmt::v12::basic_appender<char>)#1}&>
#
# and both generators built the function name from a curated replace-list with
# no catch-all, so `{`, `}` and `#` reached 58 definitions:
#
#     function fmt_v12_detail_write_padded_…_constref_{lambda_…_#1}ref(out::…)
#
# `_assert_wrapper_parses` refused the module and the Hub's fmt had no wrapper
# at all. This is the FUNCTION-name spelling of the class the DWARF TYPE
# spelling `(lambda at file.cpp:L:C)` already hit in `_sanitize_cpp_type_name`
# (llama.cpp, 2026-08).
#
# The same wrap had a second, silent failure. 8 functions whose return type is
# `decltype ({parm#1}(0))` arrive from the build with name "", and were emitted
# as `function (this::Any, vis::Any)`. That is an ANONYMOUS function: valid
# syntax, so no guard fired, and it binds nothing.
#
# The list existed as four inline copies (C/C++ × main/varargs) plus a fifth
# shorter one for C++ proxy/deleter references, and they had drifted: C lacked
# `@` and `~`, the varargs paths lacked `+ = - * /`. It is now one function in
# Wrapper/Utils.jl, `_julia_function_name`.
#
# Library-free: the fmt spellings below are copied verbatim from that package's
# compilation_metadata.json and vendored here. Nothing reads the Hub.

using Test
using RepliBuild
using Libdl

const W = RepliBuild.Wrapper
const F = W._julia_function_name

# Ground truth for "valid": a parser, in the exact position the generators emit
# the name in. NOT `Base.isidentifier`, which answers true for "end" on 1.13.
function _is_fn_name(n::AbstractString)
    isempty(n) && return false
    ex = Meta.parseall("function $n(x) 1 end")
    any(a -> a isa Expr && a.head in (:incomplete, :error), ex.args) && return false
    f = ex.args[end]
    return f isa Expr && f.head === :function && f.args[1] isa Expr &&
           f.args[1].head === :call && f.args[1].args[1] === Symbol(n)
end

# ── Vendored from Hub fmt 12.1.0 (compilation_metadata.json) ─────────────────
const FMT_CLASS = "fmt::v12::detail"
const FMT_LAMBDA_NAME = "write_padded<char, (fmt::v12::align)1, fmt::v12::basic_appender<char>, fmt::v12::detail::write_char<char, fmt::v12::basic_appender<char> >(fmt::v12::basic_appender<char>, char, fmt::v12::format_specs const&)::{lambda(fmt::v12::basic_appender<char>)#1}&>"
const FMT_LAMBDA_MANGLED = "_ZN3fmt3v126detail12write_paddedIcLNS0_5alignE1ENS0_14basic_appenderIcEERZNS1_10write_charIcS5_EET0_S7_T_RKNS0_12format_specsEEUlS5_E_EET1_SE_SB_mmOT2_"
# The name the fixed derivation gives it. Pinned: consumers call this.
const FMT_LAMBDA_JL = "fmt_v12_detail_write_padded_char_fmt_v12_align1_fmt_v12_basic_appender_char_fmt_v12_detail_write_char_char_fmt_v12_basic_appender_char_fmt_v12_basic_appender_char_char_fmt_v12_format_specs_constref_lambda_fmt_v12_basic_appender_char_1_ref"
# Name "" and class "" in the metadata until 2026-09-24, when the build's
# derivation learned to skip `decltype ({parm#1}(0))` (test_local_entity_names.jl).
# The fallback it drives here stays, as the net for any other empty name.
const FMT_DECLTYPE_MANGLED = "_ZN3fmt3v129loc_value5visitINS0_6detail10loc_writerIcEEEEDTclfp_Li0EEEOT_"
const FMT_DECLTYPE_DEMANGLED = "decltype ({parm#1}(0)) fmt::v12::loc_value::visit<fmt::v12::detail::loc_writer<char> >(fmt::v12::detail::loc_writer<char>&&)"

# `#2`/`#3` siblings. fmt has these for real (write_float's three lambdas);
# derived here by renumbering so the fixture stays one vendored spelling.
_fmt_lambda(n) = replace(FMT_LAMBDA_NAME, "#1}" => "#$(n)}")

# Names that were already valid identifiers, and what they must stay. These are
# the old derivation's answers, byte-for-byte. Hub consumers call these names,
# so a "better" spelling here is an API break, not a fix.
const PINNED = [
    "Foo_~Foo"                 => "Foo_destroy_Foo",
    "Vec_operator+="           => "Vec_operatorplusassign",
    "Vec_operator=="           => "Vec_operatorassignassign",
    "Vec_operator<="           => "Vec_operator_assign",
    "Cls_operator!"            => "Cls_operator!",       # `!` is an identifier char
    # Known pre-existing collision, kept deliberately. `()` and `[]` both land
    # on `Vec_operator`; `_dedup_method_chunks` reports it when the signatures
    # also match. Renaming either would rename a shipped function.
    "Vec_operator()"           => "Vec_operator",
    "Vec_operator[]"           => "Vec_operator",
    "Box<double>_get"          => "Box_double_get",
    "ns::Foo_bar"              => "ns_Foo_bar",
    "(anonymous namespace)::helper" => "_anonymous_namespace_helper",
    "memcpy@@GLIBC_2"          => "memcpy_GLIBC_2",      # ELF symbol versioning
    "replibuild_shim_SYNTH_OK" => "SYNTH_OK",            # macro shim prefix
    "push_back_"               => "push_back",           # trailing `_` rstrip
    "lua_pushfstring"          => "lua_pushfstring",
    "café"                     => "café",                # Unicode letters survive
    # Contextual words are legal function names and appear in real C APIs.
    "in"                       => "in",
    "type"                     => "type",
    "where"                    => "where",
]

# ── Generator harness (same shape as test_enum_alias_collision) ──────────────
# Any loadable library satisfies the module's load-time check; nothing here is
# called. libjulia is in every Julia process on every OS.
const _LIBREF = abspath(first(filter(p -> occursin("libjulia", basename(p)), Libdl.dllist())))

function _name_config(dir::String, lang::String, extra::String = "")
    toml = joinpath(dir, "replibuild.toml")
    write(toml, """
    [project]
    name = "namesynth"
    root = "$(escape_string(dir))"

    [link]
    enable_lto = false

    [wrap]
    language = "$lang"
    $extra
    """)
    return RepliBuild.ConfigurationManager.load_config(toml)
end

_p(n, jt, ct) = Dict{String,Any}("name" => n, "julia_type" => jt, "c_type" => ct)
_fn(; name, mangled, demangled = name, class = "", is_method = false, is_vararg = false,
      params = Any[_p("n", "Cint", "int")]) =
    Dict{String,Any}("name" => name, "mangled" => mangled, "demangled" => demangled,
                     "class" => class, "is_method" => is_method, "is_vararg" => is_vararg,
                     "parameters" => params,
                     "return_type" => Dict{String,Any}("julia_type" => "Cint", "c_type" => "int"))

function _generate(lang::Symbol, functions::Vector{Any}; extra::String = "")
    dir = mktempdir()
    try
        cfg = _name_config(dir, lang === :cpp ? "cpp" : "c", extra)
        md = Dict{String,Any}("functions" => functions, "globals" => Dict{String,Any}(),
                              "struct_definitions" => Dict{String,Any}())
        gen = lang === :cpp ? W.generate_introspective_module_cpp :
                              W.generate_introspective_module_c
        # As `wrap_library` runs it: the config's `[wrap.rename]` active for the
        # generator, then checked against the text.
        (out, rt) = W.with_renames(cfg.wrap.rename) do
            gen(cfg, _LIBREF, md, "NameSynth", W.create_type_registry(cfg), false)
        end
        code = out isa Tuple ? String(first(out)) : String(out)
        W._check_renames(rt, code, "NameSynth")
        return code
    finally
        rm(dir; recursive = true, force = true)
    end
end

@testset "function name derivation" begin

    @testset "fmt's lambda instantiation names a valid function" begin
        jl = F("$(FMT_CLASS)_$(FMT_LAMBDA_NAME)", FMT_LAMBDA_MANGLED)
        @test _is_fn_name(jl)                    # the original failure
        @test !any(in(jl), ('{', '}', '#'))
        @test jl == FMT_LAMBDA_JL
    end

    @testset "distinct lambdas keep distinct names" begin
        # Deleting the `#N` instead of sanitizing it would parse just as well
        # and merge these. They share a dispatch signature, so dedup would keep
        # one and make the others unreachable. That is 10 real fmt collisions.
        names = [F("$(FMT_CLASS)_$(_fmt_lambda(n))", "") for n in 1:3]
        @test all(_is_fn_name, names)
        @test length(unique(names)) == 3
        @test endswith(names[2], "_lambda_fmt_v12_basic_appender_char_2_ref")
    end

    @testset "an empty name falls back to the mangled symbol" begin
        jl = F("", FMT_DECLTYPE_MANGLED)
        @test !isempty(jl)                       # was `function (this, vis)`
        @test _is_fn_name(jl)
        # Unique per symbol, like the names demangle failures already get.
        @test startswith(FMT_DECLTYPE_MANGLED, jl)
        @test jl != F("", FMT_LAMBDA_MANGLED)
        # Nothing to fall back to: the escape both type sanitizers use.
        @test F("") == "c__"
        @test F(")") == "c__"
        # All-underscore stays `c__` even with a mangled name available: it
        # reads back like the C name, and `_` itself is write-only in Julia.
        @test F("_", "_Z1_v") == "c__"
        @test F("__") == "c__"
    end

    @testset "names that already worked are byte-identical" begin
        checked = 0
        for (raw, want) in PINNED
            @test _is_fn_name(want)              # the pin itself is sound
            @test F(raw) == want
            checked += 1
        end
        @test checked == length(PINNED)
    end

    @testset "operators the list missed get words, not `_`" begin
        # As `_`, all three collapse onto operator()'s `Cls_operator`.
        ops = ["Cls_operator%", "Cls_operator^", "Cls_operator|", "Cls_operator()"]
        names = F.(ops)
        @test all(_is_fn_name, names)
        @test length(unique(names)) == length(ops)
        @test F("Cls_operator%=") == "Cls_operatormodassign"
        @test F("Cls_operator||") == "Cls_operatororor"
    end

    @testset "reserved words are escaped, contextual words are not" begin
        # Computed against the parser rather than stated, so a Julia release
        # that reserves a new word turns this red instead of shipping a module
        # that does not parse.
        candidates = union(W._JULIA_RESERVED_WORDS,
            ["type", "in", "isa", "where", "abstract", "mutable", "primitive",
             "public", "outer", "var", "and", "or", "not", "nothing", "missing"])
        checked = 0
        for w in candidates
            @test (w in W._JULIA_RESERVED_WORDS) == !_is_fn_name(w)
            @test _is_fn_name(F(w))
            checked += 1
        end
        @test checked == length(candidates)
        @test F("end") == "end_"
        @test F("local") == "local_"
    end

    @testset "every spelling yields a valid function name" begin
        corpus = vcat(first.(PINNED),
            ["$(FMT_CLASS)_$(_fmt_lambda(n))" for n in 1:3],
            ["foo()::{lambda(int, ...)#1}_operator()",   # a variadic lambda
             "Cls_{unnamed type#1}_get",
             "foo(int) [clone .constprop.0]",           # GCC clone suffix
             "'lambda'(int)",                           # LLVM demangler spelling
             "ns::\$_0_run",                            # clang unnamed type
             "operator\"\"_km",                         # user-defined literal
             "f′", "′f", "4ever", "a\\b", "x;y", "`q`"])
        checked = 0
        for raw in corpus
            jl = F(raw, "_Zfallback")
            @test _is_fn_name(jl)
            checked += 1
        end
        @test checked == length(corpus)
        @test F("′f") == "_′f"                   # `′` cannot START a name
    end

    @testset "both generators emit it" begin
        # Driven through the real emitters, main and varargs paths, because a
        # sixth private copy of the list is how this broke in the first place.
        cpp = _generate(:cpp, Any[
            _fn(name = FMT_LAMBDA_NAME, class = FMT_CLASS, is_method = true,
                mangled = FMT_LAMBDA_MANGLED),
            _fn(name = _fmt_lambda(2), class = FMT_CLASS, is_method = true,
                mangled = replace(FMT_LAMBDA_MANGLED, "EEUlS5_E_EE" => "EEUlS5_E0_EE")),
            _fn(name = "", mangled = FMT_DECLTYPE_MANGLED, demangled = FMT_DECLTYPE_DEMANGLED,
                params = Any[_p("vis", "Ptr{Cvoid}", "void*")]),
            _fn(name = "operator()", class = "foo()::{lambda(int, ...)#1}", is_method = true,
                is_vararg = true, mangled = "_ZZ3foovENKUliziE_clEiz"),
        ]; extra = "[wrap.varargs]\n\"operator()\" = [[\"Cint\"]]\n")
        c = _generate(:c, Any[
            _fn(name = "end", mangled = "end"),
            _fn(name = "local", mangled = "local", is_vararg = true),
        ]; extra = "[wrap.varargs]\nlocal = [[\"Cint\"]]\n")

        for (lang, code, want) in (
                (:cpp, cpp, [FMT_LAMBDA_JL, replace(FMT_LAMBDA_JL, r"_1_ref$" => "_2_ref"),
                             F("", FMT_DECLTYPE_MANGLED),
                             "foo_lambda_int_1_operator", "foo_lambda_int_1_operator_Cint"]),
                (:c,   c,   ["end_", "local_", "local__Cint"]))
            @testset "$lang" begin
                @test (W._assert_wrapper_parses(code, "NameSynth_$lang"); true)
                defined = W._defined_names(code)
                exported = W._exported_names(code)
                for n in want
                    @test n in defined
                    @test n in exported
                end
                # A definition that binds nothing parses fine, which is why
                # the refusal above cannot catch it.
                @test !occursin(r"^function \("m, code)
            end
        end
    end
end

# abi_fuzz M1: a project directory starting with a digit gave `module 3cCallF21`,
# which parses, as a module named `3`, and fails at load. The parse guard now
# refuses a module whose name is not an identifier, and the derived name never
# starts with a digit or comes out empty.
# ── [wrap.rename] (2026-11-13) ─────────────────────────────────────────────
# A template instantiation's derived name carries its full template arguments
# (Boost.Math's `pdf<double, policy<13 defaults>>` is 503 characters), and the
# curated derivation is frozen. `[wrap.rename]` is the one hand-chosen name,
# applied INSIDE `_julia_function_name` so every emitter sees it, with its
# checks: a key that matches nothing, and a name that collides, are errors.
const SCALED_L = "_ZN8stl_test6scaledIlEET_S1_i"
const SCALED_D = "_ZN8stl_test6scaledIdEET_S1_i"

@testset "[wrap.rename]" begin
    @testset "a key renames by mangled symbol or derived name, as a glob" begin
        (got, rt) = W.with_renames(Dict("_Z3fooi" => "foo_int", "ns_pdf_*" => "pdf")) do
            (F("foo", "_Z3fooi"),
             F("ns_pdf<double, policy<a> >", "_ZN2ns3pdfIdE"),
             F("ns_pdf<float>", "_ZN2ns3pdfIfE"),
             F("bar", "_Z3barv"))
        end
        @test got == ("foo_int", "pdf", "pdf", "bar")
        @test W._RENAMES[] === nothing              # scoped to the block
        @test rt.hit["_Z3fooi"] == 1 && rt.hit["ns_pdf_*"] == 2
        @test rt.derived["bar"] == Set(["_Z3barv"])
        @test Set(s for (_, s) in rt.renamed["pdf"]) == Set(["_ZN2ns3pdfIdE", "_ZN2ns3pdfIfE"])
        text = "function foo_int() end
function pdf() end
function bar() end
"
        @test W._check_renames(rt, text, "M") === nothing
        # The recursive mangled fallback does not consult the table twice.
        @test first(W.with_renames(Dict("_Z1_v" => "x")) do; F("", "_Z1_v") end) == "x"
    end

    @testset "no table, no change" begin
        (got, rt) = W.with_renames(Dict{String,String}()) do; F("foo", "_Z3fooi") end
        @test got == "foo" && rt === nothing
        @test W._check_renames(nothing, "", "M") === nothing
        @test F("ns_pdf<double>", "_ZN2ns3pdfIdE") == "ns_pdf_double"   # outside any block
    end

    @testset "a key that matches nothing is an error naming it" begin
        (_, rt) = W.with_renames(Dict("nomatch_*" => "x", "_Z3fooi" => "f")) do
            F("foo", "_Z3fooi")
        end
        err = try; W._check_renames(rt, "function f() end
", "M"); nothing; catch e; e; end
        @test err isa ErrorException
        msg = sprint(showerror, err)
        @test occursin("nomatch_*", msg) && occursin("matched no function", msg)
        @test !occursin("_Z3fooi", msg)
    end

    @testset "a chosen name that collides is an error naming both sides" begin
        # ...with a function the derivation names the same...
        (_, rt) = W.with_renames(Dict("_Z3fooi" => "bar")) do
            (F("foo", "_Z3fooi"), F("bar", "_Z3barv"))
        end
        err = try; W._check_renames(rt, "function bar() end
", "M"); nothing; catch e; e; end
        @test err isa ErrorException
        msg = sprint(showerror, err)
        @test occursin("`bar`", msg) && occursin("_Z3fooi", msg) && occursin("_Z3barv", msg)
        @test occursin("derived name", msg)
        # ...and with a type the wrapper binds.
        (_, rt) = W.with_renames(Dict("_Z3fooi" => "Shape")) do; F("foo", "_Z3fooi") end
        err = try; W._check_renames(rt, "struct Shape
    x::Cint
end
function Shape() end
", "M"); nothing; catch e; e; end
        @test err isa ErrorException
        @test occursin("bound as a struct", sprint(showerror, err))
        # Two renames onto one name are an overload set, not a collision.
        (_, rt) = W.with_renames(Dict("_Z3fooi" => "f", "_Z3bari" => "f")) do
            (F("foo", "_Z3fooi"), F("bar", "_Z3bari"))
        end
        @test W._check_renames(rt, "function f() end
", "M") === nothing
    end

    @testset "two keys giving one symbol two names is an error at the symbol" begin
        err = try
            W.with_renames(Dict("_Z3fooi" => "a", "foo" => "b")) do; F("foo", "_Z3fooi") end
            nothing
        catch e; e; end
        @test err isa ErrorException
        @test occursin("more than one name", sprint(showerror, err))
        @test W._RENAMES[] === nothing              # the finally ran
    end

    @testset "the overload prediction moves with the rename" begin
        # Two instantiations the derivation names apart (`take_int_`, `take_double_`)
        # are two one-member sets: no position is strict. Renamed onto one name
        # they are one set that disagrees in the pointer's target, so position
        # 1 is typed and Julia dispatches between them.
        G = W
        fns = Any[_fn(name = "take<int*>", mangled = "_Z4takeIPiEvT_",
                      params = Any[_p("p", "Ptr{Cint}", "int*")]),
                  _fn(name = "take<double*>", mangled = "_Z4takeIPdEvT_",
                      params = Any[_p("p", "Ptr{Cdouble}", "double*")])]
        ident(jt, ct) = String(jt)
        plain = G._integer_width_overloads(fns, Set{String}(), ident)
        @test isempty(plain)
        (strict, _) = W.with_renames(Dict("take_*" => "take")) do
            G._integer_width_overloads(fns, Set{String}(), ident)
        end
        @test strict == Dict("_Z4takeIPiEvT_" => Set([1]), "_Z4takeIPdEvT_" => Set([1]))
    end

    @testset "both generators bind the chosen name, and the old one nowhere" begin
        cpp = _generate(:cpp, Any[
            # A namespace function arrives with `class` = its namespace and
            # `is_method` set (no receiver: `has_object_pointer` is false).
            _fn(name = "scaled<long>", class = "stl_test", is_method = true, mangled = SCALED_L,
                params = Any[_p("x", "Clong", "long"), _p("k", "Cint", "int")]),
            _fn(name = "scaled<double>", class = "stl_test", is_method = true, mangled = SCALED_D,
                params = Any[_p("x", "Cdouble", "double"), _p("k", "Cint", "int")]),
            _fn(name = "untouched", mangled = "_Z9untouchedi"),
        ]; extra = "[wrap.rename]\n\"stl_test_scaled_*\" = \"scaled\"\n")
        c = _generate(:c, Any[
            _fn(name = "very_long_generated_name_v2", mangled = "very_long_generated_name_v2"),
            _fn(name = "local", mangled = "local", is_vararg = true),
        ]; extra = "[wrap.varargs]\nlocal = [[\"Cint\"]]\n" *
                   "[wrap.rename]\nvery_long_generated_name_v2 = \"short\"\nlocal = \"loc\"\n")
        for (lang, code, want, gone) in (
                (:cpp, cpp, ["scaled", "untouched"], ["stl_test_scaled_long", "stl_test_scaled_double"]),
                (:c,   c,   ["short", "loc", "loc_Cint"], ["very_long_generated_name_v2", "local_", "local__Cint"]))
            @testset "$lang" begin
                @test (W._assert_wrapper_parses(code, "NameSynth_$lang"); true)
                defined = W._defined_names(code)
                exported = W._exported_names(code)
                for n in want
                    @test n in defined
                    @test n in exported
                end
                for n in gone
                    @test !(n in defined)
                    @test !(n in exported)
                    # No alias and no second definition. (The C symbol itself
                    # still appears, as the ccall target and in the docstring.)
                    @test !occursin(Regex("^function " * n * "\\(", "m"), code)
                    @test !occursin(Regex("^(const|global) " * n * "\\b", "m"), code)
                    @test !occursin(Regex("^export .*\\b" * n * "\\b", "m"), code)
                end
            end
        end
        # The two renamed instantiations are one overload set with typed dispatch.
        @test count(==("scaled"), W._defined_names(cpp)) >= 1
        @test occursin(r"function scaled\(x::Integer, k::Integer\)", cpp) ||
              occursin(r"function scaled\(x::Clong, k::Integer\)", cpp)
        @test occursin(r"function scaled\(x::(Cdouble|Float64|Real), k::Integer\)", cpp)
    end

    @testset "the config refuses a value that is not a binding name, and round-trips" begin
        mktempdir() do dir
            toml = joinpath(dir, "replibuild.toml")
            write(toml, "[project]\nname = \"r\"\nroot = \"$(escape_string(dir))\"\n[wrap]\nlanguage = \"cpp\"\n[wrap.rename]\n\"_Z1\" = \"not an id\"\n")
            err = try; RepliBuild.ConfigurationManager.load_config(toml); nothing; catch e; e; end
            @test err isa ErrorException && occursin("not a Julia identifier", sprint(showerror, err))
            write(toml, "[project]\nname = \"r\"\nroot = \"$(escape_string(dir))\"\n[wrap]\nlanguage = \"cpp\"\n[wrap.rename]\n\"_Z1\" = \"end\"\n")
            err = try; RepliBuild.ConfigurationManager.load_config(toml); nothing; catch e; e; end
            @test err isa ErrorException && occursin("not a Julia identifier", sprint(showerror, err))
            write(toml, "[project]\nname = \"r\"\nroot = \"$(escape_string(dir))\"\n[wrap]\nlanguage = \"cpp\"\n[wrap.rename]\n\"boost_math_pdf_*\" = \"pdf\"\n_Z3fooi = \"foo!\"\n")
            cfg = RepliBuild.ConfigurationManager.load_config(toml)
            @test cfg.wrap.rename == Dict("boost_math_pdf_*" => "pdf", "_Z3fooi" => "foo!")
            RepliBuild.ConfigurationManager.save_config(cfg)
            @test RepliBuild.ConfigurationManager.load_config(toml).wrap.rename == cfg.wrap.rename
            @test ("wrap", "rename") in RepliBuild.Discovery.PRESERVED_TOML_KEYS
        end
    end
end

@testset "module names are identifiers (abi_fuzz M1)" begin
    modname(project) = mktempdir() do dir
        write(joinpath(dir, "replibuild.toml"), """
        [project]
        name = "$project"
        root = "$(escape_string(dir))"
        [wrap]
        language = "c"
        """)
        RepliBuild.ConfigurationManager.get_module_name(
            RepliBuild.ConfigurationManager.load_config(joinpath(dir, "replibuild.toml")))
    end
    @test modname("3c_call_f2") == "_3cCallF2"
    @test modname("zlib") == "Zlib"
    @test modname("my_lib") == "MyLib"
    @test modname("---") == "Wrapper"
    for p in ("3c_call_f2", "zlib", "my_lib", "---", "42")
        @test (p, _is_fn_name(modname(p))) == (p, true)
    end
    @test_throws ErrorException W._assert_wrapper_parses("module 3cCallF21\nf() = 1\nend\n", "Synth3c")
    @test (W._assert_wrapper_parses("module _3cCallF21\nf() = 1\nend\n", "Synth3c"); true)
end
