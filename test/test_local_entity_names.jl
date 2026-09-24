#!/usr/bin/env julia
# test/test_local_entity_names.jl — the build derives name and class from the
# function's OWN parameter list (2026-09-24, Hub fmt 12.1.0 and imgui).
#
# `_qualified_name_parts` cut the demangled string at the first `(` outside
# `<>`. Two GNU shapes put a paren group in front of the parameter list:
#
#   * A LOCAL entity's enclosing function.
#         fmt::v12::detail::write_fixed<…>(…)::{lambda(…)#3}::operator()(…) const
#     got name `write_fixed<…>`, class `fmt::v12::detail`, so every lambda in
#     write_fixed shared one Julia name and dispatch signature, and the wrap
#     reported "10 distinct C++ entry point(s) collapsed … UNREACHABLE". The
#     survivors were extra methods on the REAL `write_fixed`'s Julia function.
#     imgui's static lambda invokers (`…::{lambda(…)#1}::__invoke`) were given
#     class `ExampleDualListBox`, a real struct, so both receiver gates added a
#     `this` the function does not take and every argument shifted one slot.
#   * A `decltype (…)` return type.
#         decltype ({parm#1}(0)) fmt::v12::loc_value::visit<…>(…)
#     cut to `"decltype "`, and the return-type strip left name and class "".
#
# `_param_list_paren` skips both. Everything else must get the historical
# answer back, because Hub names depend on it. The old-vs-new sweep over every
# Hub package moved exactly these 47 functions (fmt 45, imgui 2).
#
# Library-free. The fmt and imgui spellings are copied verbatim from those
# packages' compilation_metadata.json. The rest are GNU c++filt output for the
# mangled name beside each. Nothing reads the Hub.

using Test
using RepliBuild
using Libdl

const C  = RepliBuild.Compiler
const W  = RepliBuild.Wrapper
const FG = RepliBuild.JLCSIRGenerator.FunctionGen

# ── Vendored from Hub fmt 12.1.0 ─────────────────────────────────────────────
const WF_ENCLOSING = "fmt::v12::detail::write_fixed<char, fmt::v12::detail::fallback_digit_grouping<char>, fmt::v12::basic_appender<char>, fmt::v12::detail::dragonbox::decimal_fp<float> >(fmt::v12::basic_appender<char>, fmt::v12::detail::dragonbox::decimal_fp<float> const&, int, char, fmt::v12::format_specs const&, fmt::v12::sign, fmt::v12::locale_ref)"
const WF_LAMBDA = Dict(
    1 => ("_ZZN3fmt3v126detail11write_fixedIcNS1_23fallback_digit_groupingIcEENS0_14basic_appenderIcEENS1_9dragonbox10decimal_fpIfEEEET1_SA_RKT2_iT_RKNS0_12format_specsENS0_4signENS0_10locale_refEENKUlS6_E_clES6_",
          WF_ENCLOSING * "::{lambda(fmt::v12::basic_appender<char>)#1}::operator()(fmt::v12::basic_appender<char>) const"),
    3 => ("_ZZN3fmt3v126detail11write_fixedIcNS1_23fallback_digit_groupingIcEENS0_14basic_appenderIcEENS1_9dragonbox10decimal_fpIfEEEET1_SA_RKT2_iT_RKNS0_12format_specsENS0_4signENS0_10locale_refEENKUlS6_E1_clES6_",
          WF_ENCLOSING * "::{lambda(fmt::v12::basic_appender<char>)#3}::operator()(fmt::v12::basic_appender<char>) const"),
)
const LOC_VISIT = ("_ZN3fmt3v129loc_value5visitINS0_6detail10loc_writerIcEEEEDTclfp_Li0EEEOT_",
                   "decltype ({parm#1}(0)) fmt::v12::loc_value::visit<fmt::v12::detail::loc_writer<char> >(fmt::v12::detail::loc_writer<char>&&)")
const IMGUI_INVOKE = ("_ZZN18ExampleDualListBox22ApplySelectionRequestsEP18ImGuiMultiSelectIOiENUlP26ImGuiSelectionBasicStorageiE_8__invokeES3_i",
                      "ExampleDualListBox::ApplySelectionRequests(ImGuiMultiSelectIO*, int)::{lambda(ImGuiSelectionBasicStorage*, int)#1}::__invoke(ImGuiSelectionBasicStorage*, int)")

# (demangled, class, name). Class is the demangler's prefix: everything before
# the last `::` of the qualified name, enclosing signature and all.
const CASES = [
    # ── Hub, verbatim ──
    (WF_LAMBDA[3][2], WF_ENCLOSING * "::{lambda(fmt::v12::basic_appender<char>)#3}", "operator"),
    (WF_LAMBDA[1][2], WF_ENCLOSING * "::{lambda(fmt::v12::basic_appender<char>)#1}", "operator"),
    (LOC_VISIT[2], "fmt::v12::loc_value", "visit<fmt::v12::detail::loc_writer<char> >"),
    ("decltype ({parm#1}(0)) fmt::v12::basic_format_arg<fmt::v12::context>::visit<fmt::v12::detail::dynamic_spec_getter>(fmt::v12::detail::dynamic_spec_getter&&) const",
     "fmt::v12::basic_format_arg<fmt::v12::context>", "visit<fmt::v12::detail::dynamic_spec_getter>"),
    ("decltype ({parm#1}(0)) fmt::v12::basic_format_arg@fmt<fmt::v12::context@fmt>::visit<fmt::v12::detail::dynamic_spec_getter@fmt>(fmt::v12::detail::dynamic_spec_getter@fmt&&) const",
     "fmt::v12::basic_format_arg@fmt<fmt::v12::context@fmt>", "visit<fmt::v12::detail::dynamic_spec_getter@fmt>"),
    # A lambda type inside the enclosing function's template and parameter lists.
    ("fmt::v12::detail::for_each_codepoint<fmt::v12::detail::find_escape(char const*, char const*)::{lambda(unsigned int, fmt::v12::basic_string_view<char>)#1}>(fmt::v12::basic_string_view<char>, fmt::v12::detail::find_escape(char const*, char const*)::{lambda(unsigned int, fmt::v12::basic_string_view<char>)#1})::{lambda(char const*, char const*)#1}::operator()(char const*, char const*) const",
     "fmt::v12::detail::for_each_codepoint<fmt::v12::detail::find_escape(char const*, char const*)::{lambda(unsigned int, fmt::v12::basic_string_view<char>)#1}>(fmt::v12::basic_string_view<char>, fmt::v12::detail::find_escape(char const*, char const*)::{lambda(unsigned int, fmt::v12::basic_string_view<char>)#1})::{lambda(char const*, char const*)#1}",
     "operator"),
    (IMGUI_INVOKE[2],
     "ExampleDualListBox::ApplySelectionRequests(ImGuiMultiSelectIO*, int)::{lambda(ImGuiSelectionBasicStorage*, int)#1}", "__invoke"),
    # ── c++filt ──
    # _ZZN3fmt3v126detail13utf8_to_utf16C1ENS0_17basic_string_viewIcEEENKUljS4_E_clEjS4_
    # (the old cut made this look like utf8_to_utf16's constructor)
    ("fmt::v12::detail::utf8_to_utf16::utf8_to_utf16(fmt::v12::basic_string_view<char>)::{lambda(unsigned int, fmt::v12::basic_string_view<char>)#1}::operator()(unsigned int, fmt::v12::basic_string_view<char>) const",
     "fmt::v12::detail::utf8_to_utf16::utf8_to_utf16(fmt::v12::basic_string_view<char>)::{lambda(unsigned int, fmt::v12::basic_string_view<char>)#1}", "operator"),
    # _ZZZ3foovENKUlvE_clEvENKUlvE_clEv — a lambda inside a lambda
    ("foo()::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const",
     "foo()::{lambda()#1}::operator()() const::{lambda()#1}", "operator"),
    # _ZZNK1A1fEvENKUlvE_clEv, _ZZNR1A…, _ZZNO1A…, _ZZNVK1A…, _ZZNKR1A… — qualifiers
    ("A::f() const::{lambda()#1}::operator()() const",          "A::f() const::{lambda()#1}", "operator"),
    ("A::f() &::{lambda()#1}::operator()() const",              "A::f() &::{lambda()#1}", "operator"),
    ("A::f() &&::{lambda()#1}::operator()() const",             "A::f() &&::{lambda()#1}", "operator"),
    ("A::f() const volatile::{lambda()#1}::operator()() const", "A::f() const volatile::{lambda()#1}", "operator"),
    ("A::f() const &::{lambda()#1}::operator()() const",        "A::f() const &::{lambda()#1}", "operator"),
    # _ZZNK1A1fEvEN5Local1gEv, _ZZNK1A1fEvENUt_1gEv — the qualifier's space is not a return type
    ("A::f() const::Local::g()",                 "A::f() const::Local", "g"),
    ("A::f() const::{unnamed type#1}::g()",      "A::f() const::{unnamed type#1}", "g"),
    # _ZZ3foovEN3Bar3bazIiEEvv — only the outermost name carries a return type
    ("void foo()::Bar::baz<int>()",              "foo()::Bar", "baz<int>"),
    # _ZZ3foovENK3BarclEv, _ZZ3foovEN3Bar3bazEv — local classes
    ("foo()::Bar::operator()() const",           "foo()::Bar", "operator"),
    ("foo()::Bar::baz()",                        "foo()::Bar", "baz"),
    # _ZZN1FclEiENKUlvE_clEv — the enclosing function is itself an operator()
    ("F::operator()(int)::{lambda()#1}::operator()() const", "F::operator()(int)::{lambda()#1}", "operator"),
    # _ZZ3foovENUlvE_8__invokeEv, _ZZ3foovENKUlT_E_clIiEEDaS_, _ZZ3fooiEd_NKUlvE_clEv
    ("foo()::{lambda()#1}::__invoke()",          "foo()::{lambda()#1}", "__invoke"),
    ("auto foo()::{lambda(auto:1)#1}::operator()<int>(int) const", "foo()::{lambda(auto:1)#1}", "operator"),
    ("foo(int)::{default arg#1}::{lambda()#1}::operator()() const", "foo(int)::{default arg#1}::{lambda()#1}", "operator"),
    # _ZZN3fooB5cxx11EvENKUlvE_clEv, _ZZNK1AcvbEvENKUlvE_clEv
    ("foo[abi:cxx11]()::{lambda()#1}::operator()() const", "foo[abi:cxx11]()::{lambda()#1}", "operator"),
    ("A::operator bool() const::{lambda()#1}::operator()() const", "A::operator bool() const::{lambda()#1}", "operator"),
    # _ZN12_GLOBAL__N_13fooEv, _ZZN12_GLOBAL__N_13fooEvENKUlvE_clEv — the other parenthesized scope
    ("(anonymous namespace)::foo()",             "(anonymous namespace)", "foo"),
    ("(anonymous namespace)::foo()::{lambda()#1}::operator()() const",
     "(anonymous namespace)::foo()::{lambda()#1}", "operator"),
    # _ZNK1fMUlvE_clEv — a namespace-scope lambda: the first `(` is inside `{}`
    ("f::{lambda()#1}::operator()() const",      "f::{lambda()#1}", "operator"),
    # _ZN1A3getIiEEDTcl1gIT_EEEv, _Z1fIiEDTcl1gIT_EEEv — decltype, member and free
    ("decltype ((g<int>)()) A::get<int>()",      "A", "get<int>"),
    ("decltype ((g<int>)()) f<int>()",           "", "f<int>"),
]

# Shapes whose answer is the historical one, pinned in test_symbol_hygiene.jl
# too. `_param_list_paren` must hand these back unchanged, ragged names included.
const HISTORICAL = [
    ("Foo::operator()(int) const",                                    "Foo", "operator"),
    ("pugi::xml_node::operator void (*)(pugi::xml_node***)() const",  "pugi::xml_node", "operator void"),
    ("pugi::xml_attribute::operator<(pugi::xml_attribute const&) const",
     "pugi::xml_attribute", "operator<(pugi::xml_attribute const&) const"),
    ("bool gguf_reader::read<int>(std::vector<int>&, unsigned long) const", "gguf_reader", "read<int>"),
    # _Z11signal_likeIiEPFvT_Ei. A function-pointer return wraps the name in
    # the declarator. Not a local entity or a decltype, so still unparsed.
    ("void (*signal_like<int>(int))(int)",                            "", ""),
    ("sum_vector(std::vector<int, std::allocator<int> > const&)",     "", "sum_vector"),
]

# ── Generator harness (same shape as test_function_name_derivation) ──────────
const _LIBREF = abspath(first(filter(p -> occursin("libjulia", basename(p)), Libdl.dllist())))

function _generate_cpp(functions::Vector{Dict{String,Any}})
    dir = mktempdir()
    try
        toml = joinpath(dir, "replibuild.toml")
        write(toml, """
        [project]
        name = "localnames"
        root = "$(escape_string(dir))"

        [link]
        enable_lto = false

        [wrap]
        language = "cpp"
        """)
        cfg = RepliBuild.ConfigurationManager.load_config(toml)
        md = Dict{String,Any}("functions" => functions, "globals" => Dict{String,Any}(),
                              "struct_definitions" => Dict{String,Any}())
        out = W.generate_introspective_module_cpp(cfg, _LIBREF, md, "LocalNames",
                                                  W.create_type_registry(cfg), false)
        return out isa Tuple ? String(first(out)) : String(out)
    finally
        rm(dir; recursive = true, force = true)
    end
end

_param(n, jt, ct) = Dict{String,Any}("name" => n, "julia_type" => jt, "c_type" => ct)

# A metadata row the way the build assembles one: name, class and is_method from
# `parse_function_signatures`, then parameters and return type merged from DWARF.
function _built_row(mangled, demangled, params)
    row = only(C.parse_function_signatures(
        [Dict{String,Any}("mangled" => mangled, "demangled" => demangled)]))
    row["parameters"] = params
    row["return_type"] = Dict{String,Any}("julia_type" => "Cvoid", "c_type" => "void", "size" => 0)
    row["is_vararg"] = false
    return row
end

# The Julia name GeneratorCpp gives a row.
_jl_name(row) = W._julia_function_name(
    (row["is_method"] && !isempty(row["class"])) ? "$(row["class"])_$(row["name"])" : row["name"],
    row["mangled"])

@testset "local entities and decltype returns: name and class" begin

    @testset "each shape splits at its own parameter list" begin
        checked = 0
        for (dem, cls, nm) in CASES
            @test C.extract_class_name(dem) == cls
            @test C.extract_function_name(dem) == nm
            # Class is the demangler's prefix: the qualified name is in the text.
            isempty(cls) || @test occursin("$(cls)::$(nm)", dem)
            checked += 1
        end
        @test checked == length(CASES)
    end

    @testset "historical answers are unchanged" begin
        checked = 0
        for (dem, cls, nm) in HISTORICAL
            @test C.extract_class_name(dem) == cls
            @test C.extract_function_name(dem) == nm
            checked += 1
        end
        @test checked == length(HISTORICAL)
    end

    @testset "is_method follows the same prefix" begin
        # `is_method` is `contains(_name_prefix(d), "::")`. Cut at the decltype,
        # the prefix was "decltype " and a member function read as free.
        @test contains(C._name_prefix(LOC_VISIT[2]), "::")
        @test C._name_prefix(LOC_VISIT[2]) ==
              "decltype ({parm#1}(0)) fmt::v12::loc_value::visit<fmt::v12::detail::loc_writer<char> >"
        @test !contains(C._name_prefix("decltype ((g<int>)()) f<int>()"), "::")
        @test C._name_prefix(WF_LAMBDA[3][2]) ==
              WF_ENCLOSING * "::{lambda(fmt::v12::basic_appender<char>)#3}::operator"
    end

    @testset "lambdas of one function stay distinct" begin
        a = C.extract_class_name(WF_LAMBDA[1][2])
        b = C.extract_class_name(WF_LAMBDA[3][2])
        @test a != b
        # Neither may land on the enclosing function's own scope and name,
        # which is where they used to become extra methods of write_fixed.
        @test a != "fmt::v12::detail" && b != "fmt::v12::detail"
        @test startswith(a, WF_ENCLOSING) && startswith(b, WF_ENCLOSING)
    end

    @testset "the C++ generator binds every entry point" begin
        this = _param("this", "Ptr{Cvoid}", "const unknown_class*")
        it   = _param("it", "Cint", "int")
        rows = [_built_row(WF_LAMBDA[1]..., Any[this, it]),
                _built_row(WF_LAMBDA[3]..., Any[this, it]),
                _built_row(LOC_VISIT..., Any[_param("this", "Ptr{Cvoid}", "loc_value*"),
                                             _param("vis", "Ptr{Cvoid}", "void*")])]
        want = _jl_name.(rows)
        @test length(unique(want)) == 3
        # The decltype member is named like any other method, not by its symbol.
        @test want[3] == "fmt_v12_loc_value_visit_fmt_v12_detail_loc_writer_char"
        @test all(w -> !startswith(w, "_Z"), want)

        code = _generate_cpp(rows)
        @test (W._assert_wrapper_parses(code, "LocalNames"); true)
        defined = W._defined_names(code)
        for w in want
            @test w in defined
        end
        # One definition each. Same name + signature is what dedup drops.
        for w in want
            @test count(Regex("^function " * w * "\\(", "m"), code) == 1
        end
    end

    @testset "receiver gates on the new class spellings" begin
        names = ["ExampleDualListBox", "A", "F", "Bar", "Local", "loc_value",
                 "basic_appender<char>", "basic_format_arg<fmt::v12::context>",
                 "utf8_to_utf16"]
        structs = Dict{String,Any}(n => nothing for n in names)
        stypes  = Set{String}(names)
        verdict(cls, nm) = (FG._has_receiver(Dict("class" => cls, "name" => nm), structs),
                            W._cpp_this_param(cls, nm, stypes) !== nothing)
        # What the build derives, not the table's expectation, so this checks
        # the spellings that actually reach the gates.
        derived(dem) = (C.extract_class_name(dem), C.extract_function_name(dem))
        checked = 0
        for (dem, _, _) in CASES
            cls, nm = derived(dem)
            isempty(cls) && continue
            fg, cpp = verdict(cls, nm)
            @test fg == cpp
            checked += 1
        end
        @test checked == count(c -> !isempty(c[2]), CASES)

        # imgui's lambda invoker is static: no `this`. Under its old class, a
        # real struct, both gates synthesized one.
        @test verdict("ExampleDualListBox", "ApplySelectionRequests") == (true, true)
        @test verdict(derived(IMGUI_INVOKE[2])...) == (false, false)

        # A lambda inside a constructor is not the constructor. The old cut gave
        # it name `utf8_to_utf16` in class `…::utf8_to_utf16`.
        ctor_dem = CASES[findfirst(c -> startswith(c[1], "fmt::v12::detail::utf8_to_utf16::"), CASES)][1]
        cls, nm = derived(ctor_dem)
        @test !FG._is_ctor_or_dtor(Dict("class" => cls, "name" => nm))
        @test !W._is_ctor_or_dtor_cpp(cls, nm)
    end
end
