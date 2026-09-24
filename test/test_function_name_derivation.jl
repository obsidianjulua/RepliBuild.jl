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
        out = gen(cfg, _LIBREF, md, "NameSynth", W.create_type_registry(cfg), false)
        return out isa Tuple ? String(first(out)) : String(out)
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
