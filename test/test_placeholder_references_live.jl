# =============================================================================
# A reference to a smart pointer takes the bytes one is returned as, live
# (2026-11-24)
#
# A smart pointer and a `std::function` are `struct T end` in a wrapper: no
# value of one exists, and a reference parameter typed `Ref{T}` took only a
# pointer already typed that way. So what the library hands out did not pass
# (the type matrix, 2026-11-23): the bytes a `std::shared_ptr` was returned as,
# in a `Ref` (the manual's way for a returned container), and a
# `std::function*` the library returned, which came back untyped.
#
# The reference now also takes a `Ref` of exactly those bytes, and the
# `std::function*` return is typed. Nothing else passes: an untyped pointer
# and a `@cfunction` pointer are a `MethodError`, as they were.
# placeholder_refs_test's functions are called through a wrapper in a child.
# Needs clang++ and libJLCS.
# =============================================================================

using Test
using RepliBuild

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

const PLACEHOLDER_DIR = joinpath(@__DIR__, "placeholder_refs_test")

if requires("A reference to a smart pointer takes the bytes one is returned as, live", :libJLCS, :clangxx)
@testset "A reference to a smart pointer takes the bytes one is returned as, live" begin
    toml_path = joinpath(PLACEHOLDER_DIR, "replibuild.toml")
    # Regenerated each run: it holds absolute, machine-local paths.
    write(toml_path, """
    [project]
    name = "placeholder_refs"
    root = "$(escape_string(PLACEHOLDER_DIR))"

    [compile]
    flags = ["-std=c++17", "-fPIC", "-fstandalone-debug"]
    source_files = ["$(escape_string(joinpath(PLACEHOLDER_DIR, "src", "handles.cpp")))"]

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

    code = read(wrapper, String)
    body(f) = [m.match for m in eachmatch(Regex("\\nfunction $f\\(.*?\\nend", "s"), code)]
    # The placeholders.
    for name in ("shared_ptr_Node", "function_int_int")
        @test occursin("\nstruct $name end\n", code)
    end
    # A reference to a smart pointer: its typed pointer, or a Ref of the bytes
    # a by-value return gives, whose size is read off that return.
    n = match(r"return_slot\(NTuple\{(\d+), UInt8\}\)", only(body("sp_make")))
    @test n !== nothing
    if n !== nothing
        for f in ("sp_read", "sp_count", "sp_set")
            b = only(body(f))
            @test occursin("$f(p::Union{Ref{shared_ptr_Node}, Base.RefValue{NTuple{$(n.captures[1]), UInt8}}}", b)
            @test occursin("pointer_arg(Ptr{shared_ptr_Node}, p)", b)
        end
    end
    # A reference to a std::function is what it was, and the pointer the
    # library returns is typed for it.
    @test occursin("function fn_empty(f::Ref{function_int_int})", only(body("fn_empty")))
    @test occursin("Ptr{function_int_int})", only(body("fn_twice")))
    # The overload set: two methods.
    @test length(body("which")) == 2

    r = run_isolated(`$(joinpath(PLACEHOLDER_DIR, "probe_handles.jl")) $wrapper`)
    v = probe_verdicts(r.output)
    @test r.ok
    @test haskey(v, "PROBE_DONE")
    names = ["smart_pointer_bytes_by_reference", "reference_is_the_callers_object",
             "std_function_pointer_as_returned", "typed_pointer_still_passes",
             "anything_else_is_refused", "function_pointer_is_no_std_function",
             "overload_set_keeps_two_methods"]
    for n in names
        @test (n, get(v, n, "no PROBE line")) == (n, "PASS")
    end
    @test count(n -> haskey(v, n), names) == length(names)
end
end # requires
