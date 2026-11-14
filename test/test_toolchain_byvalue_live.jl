# =============================================================================
# Toolchain records by value, live (2026-10-29)
#
# A library function that returns or takes `std::pair`, `std::array` or
# `std::string_view` by value, or a library record that holds one, needs that
# record's layout. parse_dwarf_dump's provenance gate left every toolchain record
# out of struct_definitions, so all of them were ABI traps, and the library's
# own `Span { std::pair<int, int> range; double weight; }` had a byte-blob member
# and two trapped functions. `Compiler._promote_toolchain_records!` keeps exactly
# the toolchain records the library reaches by value; an empty base is marked so
# the SysV classifier skips it (std::pair's `__pair_base`, and a library `Tag`);
# GeneratorCpp's STL screen lets a promoted record through. tuple and optional
# are laid out from their bases and complex from its sized member (2026-11-13).
#
# This builds toolchain_byvalue_test, checks what the metadata and the wrapper
# say, and calls every function in a child process (a wrong crossing returns
# garbage or crashes). Needs clang++ and libJLCS (every call is Tier 2).
# =============================================================================

using Test
using RepliBuild

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

const TBV_DIR = joinpath(@__DIR__, "toolchain_byvalue_test")
const TBV_PROBE = joinpath(TBV_DIR, "probe_tbv.jl")

if requires("Toolchain records by value, live", :libJLCS, :clangxx)
@testset "Toolchain records by value, live" begin
    toml_path = joinpath(TBV_DIR, "replibuild.toml")
    # Regenerated each run: it holds absolute, machine-local paths.
    write(toml_path, """
    [project]
    name = "tbv"
    root = "$(escape_string(TBV_DIR))"

    [compile]
    flags = ["-std=c++17", "-fPIC"]
    source_files = ["$(escape_string(joinpath(TBV_DIR, "src", "tbv.cpp")))"]
    include_dirs = ["$(escape_string(joinpath(TBV_DIR, "include")))"]

    [link]
    enable_lto = false
    optimization_level = "0"

    [binary]
    type = "shared"

    [wrap]
    language = "cpp"
    exclude_symbols = ["tbv_ledger"]

    [types]
    strictness = "warn"
    allow_unknown_structs = true

    [cache]
    enabled = false
    """)
    RepliBuild.build(toml_path)
    wrapper = RepliBuild.wrap(toml_path)
    @test isfile(wrapper)
    code = read(wrapper, String)
    md = RepliBuild._read_json(joinpath(dirname(wrapper), "compilation_metadata.json"))
    sd = md["struct_definitions"]

    @testset "the metadata keeps what the library reaches by value" begin
        for k in ("pair<double, double>", "pair<int, int>", "pair<int, double>", "pair<float, float>",
                  "pair<double, std::pair<double, double> >", "pair<std::pair<int, int>, double>",
                  "pair<int *, bool>", "array<double, 2UL>")
            @test get(get(sd, k, Dict()), "toolchain", false) == true
        end
        @test any(k -> startswith(k, "basic_string_view<char") && get(sd[k], "toolchain", false) == true, keys(sd))
        # Laid out from their bases (2026-11-13): members at the bases' offsets,
        # the bases gone but the empty ones, a shared name numbered.
        layout(k) = [(m["name"], m["c_type"], m["offset"]) for m in sd[k]["members"]]
        for k in ("tuple<int, double>", "optional<double>", "optional<int>", "complex<double>",
                  "complex<float>", "complex<long double>")
            @test (k, get(get(sd, k, Dict()), "toolchain", false)) == (k, true)
        end
        @test layout("tuple<int, double>") ==
              [("_M_head_impl", "double", "0x0"), ("_M_head_impl_1", "int", "0x8")]
        @test !haskey(sd["tuple<int, double>"], "base_classes")
        @test layout("optional<double>") == [("_M_payload", "_Optional_payload<double, true, true, true>", "0x0")]
        @test all(b -> get(b, "empty", false) == true, sd["optional<double>"]["base_classes"])
        @test layout("_Optional_payload<double, true, true, true>") ==
              [("_M_payload", "_Storage<double, true>", "0x0"), ("_M_engaged", "bool", "0x8")]
        # The union drops the empty `_Empty_byte` beside the value.
        @test layout("_Storage<double, true>") == [("_M_value", "double", "0x0")]
        # The complex width, read off the base type's byte_size.
        @test layout("complex<double>") == [("_M_value", "complex double", "0x0")]
        @test layout("complex<float>") == [("_M_value", "complex float", "0x0")]
        # The intermediate bases are flattened, not promoted.
        @test !any(k -> startswith(k, "_Tuple_impl") || startswith(k, "_Head_base"), keys(sd))
        # Only what an exported, non-std function crosses by value: a hidden
        # function's pair stays out, and so do the iterators libstdc++'s exported
        # std::map instantiations return.
        @test !haskey(sd, "pair<short, short>")
        # libstdc++'s exported _M_get_insert_unique_pos returns this by value.
        # (System ENUMS, `_Rb_tree_color` among them, were always kept.)
        @test !haskey(sd, "pair<std::_Rb_tree_node_base *, std::_Rb_tree_node_base *>")
        @test !any(k -> occursin("_Rb_tree", k) && !startswith(k, "__enum__"), keys(sd))
        # Nothing the library does not reach: pair's own empty base, allocators.
        @test !any(k -> startswith(k, "__pair_base") || startswith(k, "allocator<"), keys(sd))
        # The empty base is marked on the derived record, toolchain or not.
        @test all(b -> get(b, "empty", false) == true, sd["pair<double, double>"]["base_classes"])
        @test all(b -> get(b, "empty", false) == true, sd["LP"]["base_classes"])
        @test get(sd["Span"], "toolchain", false) == false
        # An exported global holding `pair<long, long>` by value, dropped by
        # `exclude_symbols`: wrap will not bind it, so it seeds nothing. The
        # record itself is the library's and keeps its entry.
        @test haskey(sd, "Ledger")
        @test !haskey(sd, "pair<long, long>")
    end

    @testset "the wrapper emits them as structs and calls through them" begin
        @test occursin(r"struct pair_double_double\n\s+first::Cdouble\n\s+second::Cdouble\nend", code)
        @test occursin(r"struct Span\n\s+range::pair_int_int\n", code)
        for f in ("minmax_pair", "tagged", "divmod", "fpair", "triple", "nested", "find_first",
                  "pair_sum", "pair_imix", "make_span", "span_weight", "arr2", "tail_view",
                  "make_lp", "lp_sum", "make_outer3",
                  "tup", "maybe", "maybe_int", "cplx", "cplxf", "tup_sum", "maybe_or",
                  "maybe_int_or", "cplx_norm", "cplxf_norm", "cplxl", "cplxl_re", "cplxl_im")
            @test !occursin("ABI Safety Trap: cannot call '$f'", code)
        end
        @test !occursin("tbv_ledger", code)
    end

    @testset "every call returns what the C++ computed" begin
        names = ["pair_sse_sse", "pair_int_sse", "pair_one_integer", "pair_two_floats",
                 "pair_memory_return", "pair_inside_pair", "pair_pointer_in_arguments",
                 "pair_parameters", "library_record_holds_pair", "std_array", "string_view",
                 "map_lookup_is_a_plain_call", "tuple_from_bases", "optional_from_bases",
                 "complex_width", "complex_long_double", "control_empty_base",
                 "control_template_in_memory_return"]
        r = run_isolated(`$TBV_PROBE $wrapper`)
        v = probe_verdicts(r.output)
        @test r.ok
        @test haskey(v, "PROBE_DONE")
        for n in names
            @test (n, get(v, n, "no PROBE line")) == (n, "PASS")
        end
        @test count(n -> haskey(v, n), names) == length(names)
    end
end
end # requires
