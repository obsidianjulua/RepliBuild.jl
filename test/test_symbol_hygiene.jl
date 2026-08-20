# =============================================================================
# Symbol hygiene — ABI-artifact symbols must never reach the wrappable API
# =============================================================================
#
# `extract_symbols_from_binary` feeds `parse_function_signatures`, which infers
# a signature from the DEMANGLED STRING for anything DWARF does not describe.
# That inference is only meaningful for symbols that are actually functions the
# user could call. Two classes are not:
#
#   • `__rb_*` promoted statics (static-promotion pass) — slice-binding surface.
#   • Itanium thunks (`_ZTh`/`_ZTv`/`_ZTc`) — vtable-slot entry points that
#     adjust `this` and tail-jump to the real method.
#
# Both fail the same way: no DWARF subprogram, so the "class" comes out of the
# demangled prefix. For a thunk that prefix is the demangler's PHRASE — class
# `"non-virtual thunk to Derived"` — and no aggregate is named that, so neither
# receiver gate (`FunctionGen._has_receiver`, `GeneratorCpp`'s `struct_types`
# check) synthesizes `this`. The wrapper then emits a zero-argument function
# calling a method that needs `this` in rdi.
#
# Not a theoretical hazard: before the filter, `ViTest.virtual_thunk_to_Diamond_tag()`
# was exported, documented, zero-argument, and SIGSEGV'd on call — the virtual
# thunk dereferences the garbage `this` twice (`mov (%rdi),%rax` →
# `mov -0x20(%rax),%rax`) to read the vcall offset out of the vptr. Found
# 2026-08-13.
#
# No toolchain: the predicate is a pure function of two strings, and the
# fixture-level end-to-end assertions live in mi_test/vi_test verify.jl.

using Test
import JSON   # runtests.jl loads only Test + RepliBuild; this file needs JSON itself

@testset "Symbol hygiene" begin
    C = RepliBuild.Compiler

    @testset "Itanium thunk classification" begin
        # Real thunks, from the MI and VI fixtures' own nm output.
        for (m, d) in [
            ("_ZThn16_N7DerivedD0Ev",      "non-virtual thunk to Derived::~Derived()"),
            ("_ZThn16_N7DerivedD1Ev",      "non-virtual thunk to Derived::~Derived()"),
            ("_ZThn16_NK7Derived5get_bEv", "non-virtual thunk to Derived::get_b() const"),
            ("_ZThn16_N7DiamondD0Ev",      "non-virtual thunk to Diamond::~Diamond()"),
            ("_ZTv0_n24_N4LeftD1Ev",       "virtual thunk to Left::~Left()"),
            ("_ZTv0_n24_N5RightD0Ev",      "virtual thunk to Right::~Right()"),
            ("_ZTv0_n32_NK7Diamond3tagEv", "virtual thunk to Diamond::tag() const"),
        ]
            @test C._is_itanium_thunk(m, d)
        end

        # Covariant-return thunks: no fixture produces one, so this is the
        # synthetic arm. Kept because the grammar admits it and the cost of
        # missing it is the same SIGSEGV.
        @test C._is_itanium_thunk("_ZTch0_h0_N1D5cloneEv",
                                  "covariant return thunk to D::clone()")

        # The demangled phrase alone must classify: `mangled_name` falls back to
        # the demangled string when the address→mangled lookup misses, and an
        # ingested binary may carry no mangled form at all.
        @test C._is_itanium_thunk("virtual thunk to Diamond::tag() const",
                                  "virtual thunk to Diamond::tag() const")

        # Ordinary methods and free functions must survive. `_ZTV`/`_ZTI`/`_ZTS`
        # are the OTHER `_ZT*` special names — they are data symbols and never
        # reach this filter, but a prefix test written as `_ZT` would eat them
        # AND, more importantly, would be the wrong shape.
        for (m, d) in [
            ("_ZNK7Derived5get_bEv", "Derived::get_b() const"),
            ("_ZN7DerivedD2Ev",      "Derived::~Derived()"),
            ("_ZTV7Derived",         "vtable for Derived"),
            ("_ZTI7Derived",         "typeinfo for Derived"),
            ("_ZTS7Derived",         "typeinfo name for Derived"),
            ("_ZTT7Diamond",         "VTT for Diamond"),
            ("b2World_Step",         "b2World_Step"),
        ]
            @test !C._is_itanium_thunk(m, d)
        end

        # A class whose name merely STARTS with a thunk-ish letter sequence is a
        # normal nested-name encoding (`_ZN` + length + name), not a special
        # name. This is what makes the `_ZT` prefix safe rather than lucky.
        @test !C._is_itanium_thunk("_ZN3Thx3fooEv", "Thx::foo()")
        @test !C._is_itanium_thunk("_ZN3Tvx3barEv", "Tvx::bar()")
        @test !C._is_itanium_thunk("_ZN9thunk_lib3runEv", "thunk_lib::run()")
    end

    @testset "the fixtures' own thunk symbols are filtered" begin
        # Drives the real predicate over the exact symbol set `nm` reports for
        # the MI/VI fixtures, so the test pins BEHAVIOUR on real input rather
        # than on hand-written strings. Skips when the fixtures aren't built —
        # this file runs in the no-toolchain suite.
        for fixture in ("mi_test", "vi_test")
            so = joinpath(@__DIR__, fixture, "julia", "lib$(fixture).so")
            isfile(so) || continue
            out = try
                read(`nm -g --defined-only $so`, String)
            catch
                continue
            end
            code_syms = String[]
            for line in split(out, '\n')
                parts = split(strip(line))
                length(parts) >= 3 && parts[2] in ("T", "W", "t", "w") &&
                    push!(code_syms, String(parts[3]))
            end
            @test !isempty(code_syms)

            thunks = filter(s -> startswith(s, "_ZTh") || startswith(s, "_ZTv"), code_syms)
            @test !isempty(thunks)                       # fixture still exercises the class
            @test all(s -> C._is_itanium_thunk(s, s), thunks)

            # And nothing else in the fixture is caught by it.
            others = filter(s -> !(s in thunks), code_syms)
            @test !any(s -> C._is_itanium_thunk(s, s), others)
        end
    end

    # ── `class` must be a SCOPE, not the demangler's prefix ─────────────────
    # Itanium mangles a function TEMPLATE's return type into the symbol, so the
    # demangler prints `bool gguf_reader::read<int>(...)`. `gguf_reader` is a
    # real class, so the old flat `split(prefix, "::")` produced the scope
    # `"bool gguf_reader"` — no aggregate is named that, both receiver gates
    # declined `this`, and every argument shifted one register. That is the
    # SILENT direction `_has_receiver`'s docstring warns about, and it was live
    # in the Hub: 62 methods across box2d/llamacpp/pugixml/tinyxml2.
    #
    # Every case below is a real demangled name taken from a Hub package or
    # fixture, not an invented one.
    @testset "extract_class_name / extract_function_name" begin
        for (dem, cls, nm) in [
            # Plain method and free function — the shapes that always worked.
            ("Calculator::compute(int, int, char)",                  "Calculator",       "compute"),
            ("sum_vector(std::vector<int, std::allocator<int> > const&)", "",             "sum_vector"),
            ("Derived::get_b() const",                               "Derived",          "get_b"),
            ("b2Distance(b2DistanceOutput*, b2SimplexCache*, b2DistanceInput const*)", "", "b2Distance"),

            # Function template: return type is mangled in, and must not become
            # part of the scope.
            ("bool gguf_reader::read<signed char>(std::vector<signed char>&, unsigned long) const",
             "gguf_reader", "read<signed char>"),
            ("void b2DynamicTree::Query<b2BroadPhase>(b2BroadPhase*, b2AABB const&) const",
             "b2DynamicTree", "Query<b2BroadPhase>"),

            # Multi-word return type — the cut is the LAST top-level space.
            ("unsigned long long ImGui::RoundScalarWithFormatT<unsigned long long>(char const*, int, unsigned long long)",
             "ImGui", "RoundScalarWithFormatT<unsigned long long>"),

            # Return type that is itself `::`-qualified AND template-bearing.
            ("std::enable_if<std::is_integral<unsigned int>::value, bool>::type llama_model_loader::get_arr_n<unsigned int>(llm_kv, unsigned int&, bool)",
             "llama_model_loader", "get_arr_n<unsigned int>"),

            # Template ARGUMENTS containing `::` — what broke the flat split.
            ("bool llama_model_loader::get_arr<std::vector<int, std::allocator<int> > >(llm_kv, std::vector<int, std::allocator<int> >&, bool)",
             "llama_model_loader", "get_arr<std::vector<int, std::allocator<int> > >"),
            ("tinyxml2::XMLElement* tinyxml2::XMLDocument::CreateUnlinkedNode<tinyxml2::XMLElement, 120>(tinyxml2::MemPoolT<120>&)",
             "tinyxml2::XMLDocument", "CreateUnlinkedNode<tinyxml2::XMLElement, 120>"),

            # A cast inside template arguments — paren depth, not just angle.
            ("void ggml::cpu::repack::gemm<block_q2_K, 8l, 8l, (ggml_type)15>(int, float*, unsigned long, void const*)",
             "ggml::cpu::repack", "gemm<block_q2_K, 8l, 8l, (ggml_type)15>"),

            # Free FUNCTION template — scope is empty, not a truncated string.
            ("bool gguf_read_emplace_helper<short>(gguf_reader const&, std::vector<gguf_kv>&)",
             "", "gguf_read_emplace_helper<short>"),

            # Namespaced classes, nested classes, template classes.
            ("pugi::xml_document::load_string(char const*, unsigned int)", "pugi::xml_document", "load_string"),
            ("llama_kv_cache_dsv4_context::comp_plan::operator=(llama_kv_cache_dsv4_context::comp_plan&&)",
             "llama_kv_cache_dsv4_context::comp_plan", "operator="),
            ("std::vector<int, std::allocator<int> >::push_back(int const&)",
             "std::vector<int, std::allocator<int> >", "push_back"),

            # Conversion operators — the ONLY names with a legitimate top-level
            # space. Cutting at it would eat the name, so they bail unchanged.
            ("pugi::xml_parse_result::operator bool() const", "pugi::xml_parse_result", "operator bool"),
        ]
            @test C.extract_class_name(dem)    == cls
            @test C.extract_function_name(dem) == nm
        end

        # ── Relational/arrow operators: SCOPE exact, NAME knowingly ragged ──
        # `_find_toplevel_paren` tracks `<`/`>` without clamping, so `operator>`
        # drives it negative and `operator<` drives it positive — either way the
        # argument list's `(` is never seen at depth 0 and the prefix is never
        # cut. `_qualified_name_parts` has its own CLAMPED walker, so the scope
        # is right (these were `"pugi::xml_attribute::operator>=(pugi"` before)
        # and `this` is synthesized correctly. Only the emitted Julia function
        # name carries the parameter text, which the generator then sanitizes
        # the same way it already does for `operator=` → `operatorassign`.
        #
        # Deliberately NOT fixed by clamping `_find_toplevel_paren`: measured
        # over 10,618 Hub names it changes exactly 7 (`operator>`/`>=`/`->`) and
        # leaves the `operator<`/`<=` siblings ragged, i.e. it would split one
        # family into two spellings. Uniform beats half-fixed; a real fix skips
        # the `operator` token outright.
        for (dem, cls) in [
            ("pugi::xml_attribute::operator>=(pugi::xml_attribute const&) const", "pugi::xml_attribute"),
            ("pugi::xml_attribute::operator<(pugi::xml_attribute const&) const",  "pugi::xml_attribute"),
            ("pugi::xml_node::operator>(pugi::xml_node const&) const",            "pugi::xml_node"),
            ("pugi::xml_node_iterator::operator->() const",                       "pugi::xml_node_iterator"),
            # Conversion-to-function-pointer: prefix is cut at the `(` of `(*)`.
            ("pugi::xml_node::operator void (*)(pugi::xml_node***)() const",      "pugi::xml_node"),
        ]
            @test C.extract_class_name(dem) == cls
            @test startswith(C.extract_function_name(dem), "operator")
        end

        # Direction invariant: the scope may only ever get MORE precise. A
        # correct scope is a suffix-preserving refinement of the old one — it
        # never introduces a name the old answer did not already contain.
        for dem in ["bool gguf_reader::read<int>(std::vector<int>&, unsigned long) const",
                    "void b2BroadPhase::UpdatePairs<b2ContactManager>(b2ContactManager*)",
                    "unsigned int ImGui::RoundScalarWithFormatT<unsigned int>(char const*, int, unsigned int)"]
            @test occursin(C.extract_class_name(dem), dem)
        end
    end

    # ── The two receiver gates must agree ───────────────────────────────────
    # `FunctionGen._has_receiver` (MLIR thunk) and `GeneratorCpp`'s
    # `struct_types` check (Julia wrapper) decide the SAME argument array from
    # opposite ends of the pipeline. When they disagree the thunk reads a slot
    # the wrapper never wrote — that is the Dear ImGui incident (788 thunked
    # functions, 174 crashing immediately, 614 with every argument shifted) and
    # it is the same shape as `ffe_call`/`try_call` each carrying their own SysV
    # coercion. The code is duplicated on purpose (generator isolation), so
    # AGREEMENT is what has to be pinned, not the implementation.
    @testset "Both receiver gates agree" begin
        FG = RepliBuild.JLCSIRGenerator.FunctionGen
        W  = RepliBuild.Wrapper

        # ctor/dtor predicate: same verdict from both copies, on both the
        # obvious shapes and the ones that broke real packages.
        for (cls, nm, want) in [
            ("b2CircleContact",                        "~b2CircleContact", true),   # box2d: no type DIE
            ("Derived",                                "~Derived",         true),
            ("pugi::xml_document",                     "xml_document",     true),   # namespaced ctor
            ("Box<double>",                            "Box",              true),   # template class ctor
            ("std::vector<int, std::allocator<int> >", "vector",           true),
            ("b2DynamicTree",                          "Query<b2BroadPhase>", false),
            ("gguf_reader",                            "read<int>",        false),
            ("ImGui",                                  "GetVersion",       false),  # namespace, free fn
            ("pugi::xml_node",                         "operator>=",       false),
            ("",                                       "~Orphan",          false),  # no class ⇒ no verdict
        ]
            got_fg = FG._is_ctor_or_dtor(Dict("class" => cls, "name" => nm))
            got_cpp = W._is_ctor_or_dtor_cpp(cls, nm)
            @test got_fg == want
            @test got_cpp == want
        end

        # Drive BOTH full gates over real C++ metadata and fail on any function
        # where they disagree. This is the assertion that would have caught the
        # ImGui incident, the adjustor thunks, and the box2d destructors — all
        # three were gate disagreements or shared blind spots.
        #
        # The corpus is VENDORED, not read from RepliBuild-Hub. This sweep used
        # to walk `~/Desktop/Projects/RepliBuild-Hub` live, which made a
        # no-toolchain CI test depend on a working tree in another repo. It was
        # `isdir`-guarded so it did not fail elsewhere — it went **vacuously
        # green**, which is the worse failure: the strongest assertion in this
        # file only ever executed on one machine, against whatever happened to
        # be built there that day.
        #
        # Both gates are pure functions of (class, name) plus the aggregate-NAME
        # set — `_fuzzy_struct_lookup` reaches `structs` only through `haskey`
        # and `keys`, never a value — so the fixture stores exactly that and
        # nothing else. `test/gen_receiver_corpus.jl` regenerates it from the
        # Hub and refuses to write unless both gates give identical verdicts on
        # the real metadata and on the reduction.
        corpus_path = joinpath(@__DIR__, "fixtures", "receiver_gate_corpus.json")
        @test isfile(corpus_path)
        # Deliberately NOT wrapped in try/catch. The first draft was, and it
        # swallowed an `UndefVarError: JSON` — every package was skipped, the
        # sweep proved nothing, and only the `checked` counter below revealed
        # it. A bare catch around the one call that loads the corpus can
        # silently turn this whole testset into a no-op; that is the failure
        # this guard exists to prevent.
        corpus = JSON.parsefile(corpus_path)
        pkgs = get(corpus, "packages", [])

        checked = 0
        expected = 0
        disagreements = String[]
        for p in pkgs
            pkg = String(get(p, "name", "?"))
            names = String[String(n) for n in get(p, "structs", [])]
            # Reconstruct both shapes the gates want. Values are never read, so
            # `nothing` is faithful — proven by the generator, not assumed.
            structs = Dict{String,Any}(n => nothing for n in names)
            stypes  = Set{String}(names)
            rows = get(p, "methods", [])
            expected += length(rows)
            for row in rows
                cls, nm, mangled = String(row[1]), String(row[2]), String(row[3])
                checked += 1
                fg  = FG._has_receiver(Dict("class" => cls, "name" => nm), structs)
                # The REAL GeneratorCpp gate, not a paraphrase of it.
                cpp = W._cpp_this_param(cls, nm, stypes) !== nothing
                fg == cpp || push!(disagreements,
                    "$pkg :: $mangled class=$(repr(cls)) FunctionGen=$fg GeneratorCpp=$cpp")
            end
        end
        isempty(disagreements) || @warn "Receiver gates disagree" n=length(disagreements) first=first(disagreements, 5)
        @test isempty(disagreements)
        # The sweep must actually have run, or it proves nothing — and it must
        # have consumed every row it loaded, not stopped early.
        @test checked == expected
        @test checked > 3000
        @test length(pkgs) >= 5
    end
end
