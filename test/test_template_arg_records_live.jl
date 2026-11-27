# =============================================================================
# Records named with a pointer, array or function type in their template
# arguments, and std::vector<bool>, live (2026-11-23)
#
# The type matrix found two ways a by-value return killed the process:
#
#   * `std::unique_ptr<int[]>` came back as eight bytes in a register. C++
#     returns it through a hidden pointer; the fact that says so
#     (`record_abi`'s `pass`) was missing, because the name was read as an
#     array spelling (`_by_value_record_name`).
#   * `std::vector<bool>` got a 24-byte return buffer and is 40 bytes.
#
# The first is every record whose template arguments hold a `*`, `&`, `(` or
# `[`, so the fixture crosses each kind: a smart pointer to an array, a
# container of pointers, an empty class (C++ passes nothing for it), a class
# with a destructor (by reference both ways), and `std::function` by value
# (the address of the caller's object, and nothing else).
#
# tmplarg_records_test's functions are called through a wrapper in a child: a
# wrong convention here is a crash or a wrong register. Needs clang++ and
# libJLCS.
# =============================================================================

using Test
using RepliBuild

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

const TMPLARG_DIR = joinpath(@__DIR__, "tmplarg_records_test")

if requires("Records named with a declarator in their template arguments, live", :libJLCS, :clangxx)
@testset "Records named with a declarator in their template arguments, live" begin
    toml_path = joinpath(TMPLARG_DIR, "replibuild.toml")
    # Regenerated each run: it holds absolute, machine-local paths.
    write(toml_path, """
    [project]
    name = "tmplarg_records"
    root = "$(escape_string(TMPLARG_DIR))"

    [compile]
    flags = ["-std=c++17", "-fPIC", "-fstandalone-debug"]
    source_files = ["$(escape_string(joinpath(TMPLARG_DIR, "src", "records.cpp")))"]

    [link]
    enable_lto = false
    optimization_level = "0"

    [binary]
    type = "shared"

    [wrap]
    language = "cpp"

    [types]
    strictness = "warn"
    allow_unknown_structs = true

    [cache]
    enabled = false
    """)
    RepliBuild.build(toml_path)
    wrapper = RepliBuild.wrap(toml_path)
    @test isfile(wrapper)

    # The build kept the facts of every record crossed by value, whatever its
    # template arguments hold. Keys are the compiler's spellings: ask by prefix.
    md = RepliBuild._read_json(joinpath(dirname(wrapper), "compilation_metadata.json"))
    ra = md["record_abi"]
    fact(prefix) = (k = findfirst(startswith(prefix), sort!(collect(keys(ra)))); k === nothing ? nothing : ra[sort!(collect(keys(ra)))[k]])
    found = 0
    for prefix in ("unique_ptr<int[]", "shared_ptr<int[]", "vector<int *", "vector<bool", "Holder<int *", "function<int (")
        f = fact(prefix)
        @test (prefix, f === nothing) == (prefix, false)
        f === nothing && continue
        @test (prefix, get(f, "pass", nothing)) == (prefix, "reference")
        found += 1
    end
    @test found == 6
    tag = fact("Tag<int *")
    @test tag !== nothing && get(tag, "empty", false) == true

    # The wrapper's buffer for a container is the size the DWARF gave it, and
    # the library says the same of itself (probe `vector_bool_size`).
    code = read(wrapper, String)
    vb = fact("vector<bool")
    if vb !== nothing
        n = RepliBuild.JLCSIRGenerator.FunctionGen._parse_num(vb["byte_size"])
        @test occursin("function vb_make(n::Integer)\n", code)
        # (The build asked its compiler about the type, so the storage is held
        # as the object and a by-value pass gets a copy: test_cxx_objects.jl.)
        @test occursin("CxxObjects.slot(NTuple{$n, UInt8}, ", match(r"function vb_make\(.*?\nend"s, code).match)
        @test occursin("CxxObjects.byvalue(NTuple{$n, UInt8}, ", match(r"function vb_count_val\(.*?\nend"s, code).match)
    end
    # std::function by value: converted like any pointer (inside `byvalue`,
    # which asks `pointer_arg`), never passed as it came.
    @test occursin(r"RepliBuild\.CxxObjects\.byvalue\(NTuple\{\d+, UInt8\}, _RB_OWN_\d+, LIBRARY_PATH, f\)",
                   match(r"function fn_apply\(.*?\nend"s, code).match)
    @test !occursin(r"invoke\([^\n]*, f, ", match(r"function fn_apply\(.*?\nend"s, code).match)

    r = run_isolated(`$(joinpath(TMPLARG_DIR, "probe_records.jl")) $wrapper`)
    v = probe_verdicts(r.output)
    @test r.ok
    @test haskey(v, "PROBE_DONE")
    names = ["unique_ptr_array_return", "shared_ptr_array_return", "vector_of_pointers_return",
             "vector_bool_size", "vector_bool_return", "vector_bool_by_value",
             "empty_record_is_dropped", "class_with_destructor_return", "class_with_destructor_by_value",
             "std_function_by_value_takes_a_pointer", "std_function_by_value_refuses_a_julia_function"]
    for n in names
        @test (n, get(v, n, "no PROBE line")) == (n, "PASS")
    end
    @test count(n -> haskey(v, n), names) == length(names)
end
end # requires
