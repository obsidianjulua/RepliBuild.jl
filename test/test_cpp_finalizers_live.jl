# =============================================================================
# What a C++ finalizer frees, live (2026-10-26)
#
# test_cpp_finalizers.jl (runtests) checks what the generator emits from
# synthetic metadata. This builds the same shapes from C++ (finalizer_test) and
# counts what the finalizers free: the classes with a deleting destructor give
# back their storage through a class-specific operator delete that counts, and
# `destroy_widget` and the static `Pooled::DeletePooled` count their own calls.
# Before the fix, the struct `make_rect` makes ran `Rect`'s non-deleting
# destructor: 20 destructor bodies ran and nothing was freed. Needs clang++ and
# libJLCS (the method proxy is a Tier 2 thunk).
# =============================================================================

using Test
using RepliBuild

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

const FINALIZER_DIR = joinpath(@__DIR__, "finalizer_test")
const FINALIZER_PROBE = joinpath(FINALIZER_DIR, "probe_finalizers.jl")

if requires("C++ finalizers, live", :libJLCS, :clangxx)
@testset "C++ finalizers free what they own, live" begin
    toml_path = joinpath(FINALIZER_DIR, "replibuild.toml")
    # Regenerated each run: it holds absolute, machine-local paths.
    write(toml_path, """
    [project]
    name = "finalizers"
    root = "$(escape_string(FINALIZER_DIR))"

    [compile]
    flags = ["-std=c++17", "-fPIC"]
    source_files = ["$(escape_string(joinpath(FINALIZER_DIR, "src", "finalizers.cpp")))"]
    include_dirs = ["$(escape_string(joinpath(FINALIZER_DIR, "include")))"]

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

    @testset "what each finalizer calls" begin
        rect = (m = match(r"mutable struct Rect\n(.*?)\n# Allow passing a Rect"s, code);
                m === nothing ? "" : String(m.captures[1]))
        @test occursin("ccall((:_ZN4RectD0Ev, LIBRARY_PATH), Cvoid, (Ptr{Cvoid},), o.handle)", rect)
        @test occursin("ccall((:_Z14destroy_widgetP6Widget, LIBRARY_PATH), Cvoid, (Ptr{Cvoid},), o.handle)", code)
        @test occursin("ccall((:_ZN4NodeD0Ev, LIBRARY_PATH), Cvoid, (Ptr{Node},), x.handle)", code)
        @test occursin("ccall((:_ZN4LeafD0Ev, LIBRARY_PATH), Cvoid, (Ptr{Leaf},), x.handle)", code)
        @test occursin("ccall((:_ZN6Pooled12DeletePooledEPS_, LIBRARY_PATH), Cvoid, (Ptr{Pooled},), x.handle)", code)
        @test !occursin("ccall((:_ZN4Node10FreeMemoryEv, LIBRARY_PATH), Cvoid, (Ptr{Node},), x.handle)", code)
    end

    @testset "collected objects are freed" begin
        names = ["rect_finalizer_deletes", "widget_finalizer_calls_destroy_widget", "managed_leaf_deletes",
                 "managed_node_deletes", "managed_pooled_calls_its_static_deleter", "nothing_owns_a_plain",
                 "no_invented_class"]
        r = run_isolated(`$FINALIZER_PROBE $wrapper`)
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
