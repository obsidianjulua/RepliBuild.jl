# =============================================================================
# Thread-local globals, live (2026-11-23; the 2026-11-17 audit)
#
# A thread-local global's accessor was a `cglobal`, resolved once in the
# thread that called first: on any other thread `tls_counter_ptr()` was the
# first thread's instance, a write through it changed that thread's value,
# and a C++ variable with a dynamic initializer read 0 in a thread whose
# first use was the accessor. This builds test/tls_global_test as C and as
# C++, wraps it, and runs test/tls_global_test/probe.jl in a child with two
# threads, comparing each accessor with the library's own `&variable` on a
# thread other than the first caller's. The static data member `Slot::n`
# must refuse by name. test_tls_globals.jl (unit) holds the plan, the unit
# and the emitted text.
#
# The C half needs the C bucket; the C++ half needs clang++. The wrappers
# call only `ccall`s (the C++ functions are `noexcept`). The child is started
# with `--threads=2`: the defect needs a second thread to show.
# =============================================================================

using Test
using RepliBuild

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

haskey(ENV, "REPLIBUILD_HOME") || (ENV["REPLIBUILD_HOME"] = mktempdir())

const TLS_DIR = joinpath(@__DIR__, "tls_global_test")

function tls_build(name, language, flags, src)
    for d in ("build", "julia", ".replibuild_cache", ".debug")
        p = joinpath(TLS_DIR, d)
        ispath(p) && RepliBuild._rm_tree(p)
    end
    toml_path = joinpath(TLS_DIR, "replibuild.toml")
    # Regenerated each run: it holds absolute, machine-local paths.
    write(toml_path, """
    [project]
    name = "$name"
    # escape_string on every interpolated path: TOML basic strings take escapes.
    root = "$(escape_string(TLS_DIR))"

    [compile]
    flags = [$(join(repr.(flags), ", "))]
    source_files = ["$(escape_string(joinpath(TLS_DIR, src)))"]
    include_dirs = ["$(escape_string(TLS_DIR))"]

    [link]
    enable_lto = false
    optimization_level = "0"

    [binary]
    type = "shared"

    [wrap]
    language = "$language"

    [types]
    strictness = "warn"
    allow_unknown_structs = true

    [cache]
    enabled = false
    """)
    build_out = mktemp() do path, io
        redirect_stdout(io) do
            RepliBuild.build(toml_path)
        end
        close(io)
        read(path, String)
    end
    wrap_out = mktemp() do path, io
        redirect_stdout(io) do
            RepliBuild.wrap(toml_path)
        end
        close(io)
        read(path, String)
    end
    return (build_out, wrap_out)
end

tls_metadata() = RepliBuild._read_json(joinpath(TLS_DIR, "julia", "compilation_metadata.json"))

function tls_probe(wrapper, lang, cases)
    r = run_isolated(`--threads=2 $(joinpath(TLS_DIR, "probe.jl")) $wrapper $lang`)
    v = probe_verdicts(r.output)
    @test r.ok
    @test haskey(v, "PROBE_DONE")
    for n in cases
        @test (n, get(v, n, "no PROBE line")) == (n, "PASS")
    end
    @test count(n -> haskey(v, n), cases) == length(cases)
end

# One top-level testset, so a red C half does not hide the C++ half.
@testset "Thread-local globals, live" begin
@testset "Thread-local globals, live: C" begin
    (bout, wout) = tls_build("tls_c", "c", ["-O0", "-fPIC"], "tls.c")
    wrapper = joinpath(TLS_DIR, "julia", "TlsC.jl")
    @test isfile(wrapper)
    @test occursin("tls: 2 thread-local global(s): compiling an accessor for each", bout)
    @test occursin("2 thread-local global(s), read in the calling thread through an accessor", wout)

    @testset "the metadata marks each thread-local global with its accessor" begin
        g = tls_metadata()["globals"]
        @test g["tls_counter"]["thread_local"] == true
        @test g["tls_counter"]["tls_accessor"] == "replibuild_tls_tls_counter"
        @test g["tls_scale"]["tls_accessor"] == "replibuild_tls_tls_scale"
        @test !haskey(g["plain_count"], "thread_local")
        # The accessors are not functions of the API.
        names = [f["name"] for f in tls_metadata()["functions"]]
        @test !any(n -> startswith(n, "replibuild_tls_"), names)
    end
    @testset "what was emitted" begin
        wsrc = isfile(wrapper) ? read(wrapper, String) : ""
        @test occursin("ccall((:replibuild_tls_tls_counter, LIBRARY_PATH), Ptr{Cint}, ())", wsrc)
        @test !occursin("cglobal((:tls_counter", wsrc)
        @test occursin("cglobal((:plain_count, LIBRARY_PATH), Cint)", wsrc)
    end
    @testset "used from two threads" begin
        tls_probe(wrapper, "c", ["main_thread", "other_thread_pointer", "other_thread_value",
                                 "other_thread_write", "double_other_thread", "plain_global_is_shared"])
    end
end

if requires("Thread-local globals, live: C++", :clangxx)
@testset "Thread-local globals, live: C++" begin
    (bout, wout) = tls_build("tls_cpp", "cpp", ["-O0", "-fPIC", "-std=c++17"], "tls.cpp")
    wrapper = joinpath(TLS_DIR, "julia", "TlsCpp.jl")
    @test isfile(wrapper)
    @test occursin("2 thread-local global(s), read in the calling thread through an accessor", wout)
    @test occursin("1 thread-local global(s) have no accessor and refuse: n (it is a static data member", wout)

    @testset "the metadata marks each thread-local global with its accessor" begin
        md = tls_metadata()
        g = md["globals"]
        @test g["_ZN3cfg5depthE"]["tls_accessor"] == "replibuild_tls__ZN3cfg5depthE"
        @test g["dyn"]["tls_accessor"] == "replibuild_tls_dyn"
        @test occursin("static data member", g["_ZN4Slot1nE"]["tls_refused"])
        @test get(g["_ZN4Slot1nE"], "record_member", false) == true
        # `_ZTH3dyn` (C++'s initializer for `dyn`) is the compiler's, not API:
        # it was wrapped as a stub that refused for want of debug info.
        mangled = [f["mangled"] for f in md["functions"]]
        @test !any(m -> startswith(m, "_ZTH") || startswith(m, "_ZTW") || startswith(m, "replibuild_tls_"), mangled)
    end
    @testset "used from two threads" begin
        tls_probe(wrapper, "cpp", ["namespace_member", "dynamic_initialization", "static_member_refused"])
    end
end
end # requires
end
