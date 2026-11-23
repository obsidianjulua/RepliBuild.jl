# =============================================================================
# Thread-local globals (2026-11-23; the 2026-11-17 audit)
#
# A thread-local variable has one instance per thread. Its accessor was a
# `cglobal`, which resolves once, in whichever thread calls first, so every
# other thread read and wrote that thread's instance; and a C++ variable with
# a dynamic initializer read zero in a thread whose first use was the
# accessor. The build now reads which variables are thread-local off the
# linked IR (`Compiler._thread_local_definitions`), compiles one function per
# such variable into the library that returns the calling thread's address
# (`_tls_accessor_plan`, `_tls_unit_text`), and the wrapper calls it on every
# use (`Wrapper._thread_local_global_chunk`). A variable no unit can declare
# (a static data member) gets accessors that refuse, with the reason.
#
# Library-free: IR lines as clang 23 prints them (x86-64 Linux, with
# `-fvisibility=hidden`, x86_64-w64-windows-gnu), synthetic metadata, and the
# generators' text. test_tls_globals_live.jl builds, wraps and calls a library
# from two threads.
# =============================================================================

using Test
using RepliBuild
using Libdl

const _TL_C = RepliBuild.Compiler
const _TL_W = RepliBuild.Wrapper
const _TL_LIBREF = abspath(first(filter(p -> occursin("libjulia", basename(p)), Libdl.dllist())))

# `clang++ -std=c++17 -S -emit-llvm` of
#     thread_local int a = 1;  static thread_local int b = 2;  inline thread_local int c = 3;
#     __attribute__((tls_model("initial-exec"))) thread_local int d = 4;
#     extern thread_local int e;  namespace n { thread_local struct S { int x; S(); } s; }
# (and functions using each), its `^@` lines, per target.
const TL_IR_LINUX = """
@a = thread_local global i32 1, align 4
@d = thread_local(initialexec) global i32 4, align 4
@e = external thread_local global i32, align 4
@_ZL1b = internal thread_local global i32 2, align 4
@c = linkonce_odr thread_local global i32 3, comdat, align 4
@_ZN1n1sE = thread_local global %"struct.n::S" zeroinitializer, align 4
@__tls_guard = internal thread_local global i8 0, align 1
@_ZTHN1n1sE = alias void (), ptr @__tls_init
@plain = global i32 3, align 4
@"quoted.name" = hidden thread_local global i32 0, align 4
"""
const TL_IR_HIDDEN = """
@a = hidden thread_local global i32 1, align 4
@d = hidden thread_local(initialexec) global i32 4, align 4
@e = external thread_local global i32, align 4
@_ZL1b = internal thread_local global i32 2, align 4
@c = linkonce_odr hidden thread_local global i32 3, comdat, align 4
@_ZN1n1sE = hidden thread_local global %"struct.n::S" zeroinitializer, align 4
@__tls_guard = internal thread_local global i8 0, align 1
@_ZTHN1n1sE = hidden alias void (), ptr @__tls_init
"""
const TL_IR_MINGW = """
@a = dso_local thread_local global i32 1, align 4
@d = dso_local thread_local(initialexec) global i32 4, align 4
@e = external dso_local thread_local global i32, align 4
@_ZL1b = internal thread_local global i32 2, align 4
@c = linkonce_odr dso_local thread_local global i32 3, comdat, align 4
@_ZN1n1sE = dso_local thread_local global %"struct.n::S" zeroinitializer, align 4
@__tls_guard = internal thread_local global i8 0, align 1
@_ZTHN1n1sE = dso_local alias void (), ptr @__tls_init
"""

function _tl_ir(text)
    path = tempname() * ".ll"
    write(path, text)
    return path
end

@testset "which variables the linked IR defines as thread-local" begin
    for (label, text) in (("Linux", TL_IR_LINUX), ("hidden", TL_IR_HIDDEN), ("MinGW", TL_IR_MINGW))
        @testset "$label" begin
            got = _TL_C._thread_local_definitions([_tl_ir(text)])
            # Defined, with a linkage another module names: never `e` (a
            # declaration), `_ZL1b` or `__tls_guard` (internal), the alias or
            # the plain global.
            expected = Set(["a", "c", "d", "_ZN1n1sE"])
            label == "Linux" && push!(expected, "quoted.name")
            @test got == expected
        end
    end
    # Several files: the union; a missing file is skipped.
    two = _TL_C._thread_local_definitions([_tl_ir("@x = thread_local global i32 0\n"),
                                           _tl_ir("@y = dso_local thread_local global i64 0\n"),
                                           "/nonexistent.ll"])
    @test two == Set(["x", "y"])
end

# The globals of the fixture library, as the parser writes them.
function _tl_globals()
    g(name, scope, symbol, extra::Pair...) = Dict{String,Any}("name" => name, "scope" => scope, "symbol" => symbol,
                                                              "c_type" => "int", "julia_type" => "Cint", extra...)
    return Dict{String,Any}(
        "tls_counter" => g("tls_counter", "", "tls_counter", "thread_local" => true),
        "plain_count" => g("plain_count", "", "plain_count"),
        "_ZN3cfg5depthE" => g("depth", "cfg", "_ZN3cfg5depthE", "thread_local" => true),
        "dyn" => g("dyn", "", "dyn", "thread_local" => true),
        "_ZN4Slot1nE" => g("n", "Slot", "_ZN4Slot1nE", "thread_local" => true, "record_member" => true),
        "_ZN4Slot2pnE" => g("pn", "Slot", "_ZN4Slot2pnE", "thread_local" => true, "record_member" => true,
                            "accessibility" => "private"),
        "_ZN1a1b5innerE" => g("inner", "a::b", "_ZN1a1b5innerE", "thread_local" => true),
        "_ZN2tvIiE1vE" => g("v", "tv<int>", "_ZN2tvIiE1vE", "thread_local" => true),
        "hidden_tls" => g("hidden_tls", "", "hidden_tls", "thread_local" => true))
end

@testset "the plan: one accessor per thread-local global a unit can declare" begin
    gl = _tl_globals()
    exported = Set(keys(gl))
    delete!(exported, "hidden_tls")
    rows = _TL_C._tls_accessor_plan(gl, :cpp; exported)
    by = Dict(r["symbol"] => r for r in rows)
    # Sorted by symbol; no row for a plain global or a private member.
    @test [r["symbol"] for r in rows] == sort!(["tls_counter", "_ZN3cfg5depthE", "dyn", "_ZN4Slot1nE",
                                                "_ZN1a1b5innerE", "_ZN2tvIiE1vE", "hidden_tls"])
    @test by["_ZN3cfg5depthE"]["accessor"] == "replibuild_tls__ZN3cfg5depthE"
    @test (by["_ZN3cfg5depthE"]["name"], by["_ZN3cfg5depthE"]["scope"]) == ("depth", "cfg")
    @test by["dyn"]["accessor"] == "replibuild_tls_dyn"
    @test by["_ZN1a1b5innerE"]["scope"] == "a::b"
    @test occursin("static data member", by["_ZN4Slot1nE"]["reason"]) && !haskey(by["_ZN4Slot1nE"], "accessor")
    @test occursin("not a namespace", by["_ZN2tvIiE1vE"]["reason"])
    @test by["hidden_tls"]["reason"] == "the library does not export it"
    # `exported === nothing` is "could not look": nothing refused for it.
    @test haskey(Dict(r["symbol"] => r for r in _TL_C._tls_accessor_plan(gl, :cpp))["hidden_tls"], "accessor")

    # C: the symbol is the name, and there is no scope.
    crows = _TL_C._tls_accessor_plan(Dict{String,Any}(
        "tls_counter" => gl["tls_counter"],
        "renamed" => Dict{String,Any}("name" => "x", "scope" => "", "symbol" => "renamed", "thread_local" => true)), :c)
    cby = Dict(r["symbol"] => r for r in crows)
    @test cby["tls_counter"]["accessor"] == "replibuild_tls_tls_counter"
    @test cby["renamed"]["reason"] == "its symbol is not its name"
    # Metadata without the mark plans nothing.
    @test isempty(_TL_C._tls_accessor_plan(Dict{String,Any}("plain_count" => gl["plain_count"]), :c))
end

@testset "the unit declares each variable in its scope and returns its address" begin
    gl = _tl_globals()
    rows = _TL_C._tls_accessor_plan(gl, :cpp; exported = Set(keys(gl)))
    cpp = _TL_C._tls_unit_text(rows, :cpp)
    lines = split(cpp, '\n')
    @test "namespace cfg { extern thread_local ::replibuild_tls_opaque depth; }" in lines
    @test "extern \"C\" RB_TLS_EXPORT void *replibuild_tls__ZN3cfg5depthE(void) { return (void *)__builtin_addressof(::cfg::depth); }" in lines
    @test "namespace a { namespace b { extern thread_local ::replibuild_tls_opaque inner; } }" in lines
    @test "extern thread_local ::replibuild_tls_opaque dyn;" in lines
    @test "extern \"C\" RB_TLS_EXPORT void *replibuild_tls_dyn(void) { return (void *)__builtin_addressof(::dyn); }" in lines
    # A refused row writes nothing.
    @test !occursin("Slot", cpp) && !occursin("tv<", cpp)
    # The export macro, per object format, as the macro shims have it.
    @test occursin("#  define RB_TLS_EXPORT __attribute__((used)) __declspec(dllexport)", cpp)
    @test occursin("#  define RB_TLS_EXPORT __attribute__((used, visibility(\"default\")))", cpp)

    c = _TL_C._tls_unit_text(_TL_C._tls_accessor_plan(Dict{String,Any}("tls_counter" => gl["tls_counter"]), :c), :c)
    @test occursin("\nextern _Thread_local struct replibuild_tls_opaque tls_counter;\n", c)
    @test occursin("\nRB_TLS_EXPORT void *replibuild_tls_tls_counter(void) { return (void *)&tls_counter; }\n", c)
end

@testset "a row is kept only when the compiled unit names its symbol" begin
    # The unit's IR as clang 23 prints it for the C++ fixture (its `^@` and
    # `define` lines).
    ir = _tl_ir("""
    @_ZN3cfg5depthE = external thread_local global %struct.replibuild_tls_opaque, align 1
    @dyn = external dso_local thread_local global %struct.replibuild_tls_opaque, align 1
    @llvm.compiler.used = appending global [2 x ptr] [ptr @replibuild_tls__ZN3cfg5depthE, ptr @replibuild_tls_dyn], section "llvm.metadata"
    define ptr @replibuild_tls__ZN3cfg5depthE() #0 !dbg !10 {
    define linkonce_odr hidden noundef ptr @_ZTWN3cfg5depthE() #1 comdat {
    define ptr @replibuild_tls_dyn() #0 !dbg !15 {
    """)
    (defined, tls) = _TL_C._tls_unit_names(ir)
    @test tls == Set(["_ZN3cfg5depthE", "dyn"])
    @test "replibuild_tls_dyn" in defined
    rows = [Dict{String,Any}("symbol" => "_ZN3cfg5depthE", "accessor" => "replibuild_tls__ZN3cfg5depthE"),
            Dict{String,Any}("symbol" => "dyn", "accessor" => "replibuild_tls_dyn"),
            Dict{String,Any}("symbol" => "_ZN3cfg5otherE", "accessor" => "replibuild_tls__ZN3cfg5otherE"),
            Dict{String,Any}("symbol" => "_ZN4Slot1nE", "reason" => "static")]
    _TL_C._tls_rows_checked(rows, ir)
    @test [haskey(r, "accessor") for r in rows] == [true, true, false, false]
    @test occursin("did not name the symbol _ZN3cfg5otherE", rows[3]["reason"])
    @test rows[4]["reason"] == "static"
    # A unit the library did not build with refuses every row it held.
    _TL_C._tls_unit_refused(rows, "multiple definition of `x'")
    @test !any(r -> haskey(r, "accessor"), rows)
    @test occursin("did not build with its thread-local accessors: multiple definition", rows[1]["reason"])
end

@testset "the metadata marks a thread-local global and says how it is read" begin
    gl = _tl_globals()
    for v in values(gl)
        delete!(v, "thread_local")
    end
    rows = _TL_C._tls_accessor_plan(merge(gl, Dict{String,Any}(
        k => merge(gl[k], Dict{String,Any}("thread_local" => true)) for k in ("tls_counter", "dyn", "_ZN4Slot1nE"))), :cpp)
    tls = Set(["tls_counter", "dyn", "_ZN4Slot1nE", "_ZN3cfg5depthE"])
    runtime = Set(["tls_counter", "dyn", "_ZN4Slot1nE", "_ZN3cfg5depthE", "plain_count", "replibuild_tls_tls_counter"])
    _TL_C._mark_thread_locals!(gl, tls, rows, runtime; language = :cpp)
    @test gl["tls_counter"]["tls_accessor"] == "replibuild_tls_tls_counter"
    # The row's accessor is not in the library: refused, never bound.
    @test occursin("not in the library's dynamic symbol table", gl["dyn"]["tls_refused"])
    @test occursin("static data member", gl["_ZN4Slot1nE"]["tls_refused"])
    # No row: a build that compiled none (graph mode).
    @test gl["_ZN3cfg5depthE"]["tls_refused"] == "no accessor was compiled for it"
    @test all(k -> gl[k]["thread_local"] == true, collect(tls))
    @test !haskey(gl["plain_count"], "thread_local") && !haskey(gl["plain_count"], "tls_refused")

    # The extraction before step 4c (and graph mode's) plans here: a refused
    # row has its reason already, one a unit could serve has none compiled.
    gl2 = _tl_globals()
    for v in values(gl2)
        delete!(v, "thread_local")
    end
    _TL_C._mark_thread_locals!(gl2, Set(["tls_counter", "_ZN4Slot1nE", "hidden_tls"]), nothing,
                               Set(["tls_counter", "_ZN4Slot1nE"]); language = :cpp)
    @test gl2["tls_counter"]["tls_refused"] == "no accessor was compiled for it"
    @test occursin("static data member", gl2["_ZN4Slot1nE"]["tls_refused"])
    @test gl2["hidden_tls"]["tls_refused"] == "the library does not export it"
    @test !any(g -> haskey(g, "tls_accessor"), values(gl2))
end

@testset "C++'s thread-local helpers are not API" begin
    @test _TL_C._is_tls_helper("_ZTH3dyn", "TLS init function for dyn")
    @test _TL_C._is_tls_helper("_ZTWN3cfg5depthE", "TLS wrapper function for cfg::depth")
    @test _TL_C._is_tls_helper("", "TLS init function for dyn")
    @test !_TL_C._is_tls_helper("_ZTV4Base", "vtable for Base")
    @test !_TL_C._is_tls_helper("tls_counter_get", "tls_counter_get")
end

# `record_member` (a static data member, which no unit can declare outside
# its class) is asserted on real DWARF 4 and 5 dumps in test_cpp_globals.jl.

# ── The generators ──────────────────────────────────────────────────────────

function _tl_config(dir::String, lang::String)
    toml = joinpath(dir, "replibuild.toml")
    write(toml, """
    [project]
    name = "tlsynth"
    root = "$(escape_string(dir))"

    [link]
    enable_lto = false

    [wrap]
    language = "$lang"
    """)
    return RepliBuild.ConfigurationManager.load_config(toml)
end

function _tl_generate(lang::Symbol, globals)
    md = Dict{String,Any}("functions" => Any[], "struct_definitions" => Dict{String,Any}(), "globals" => globals)
    dir = mktempdir()
    try
        cfg = _tl_config(dir, lang === :cpp ? "cpp" : "c")
        gen = lang === :cpp ? _TL_W.generate_introspective_module_cpp : _TL_W.generate_introspective_module_c
        out = redirect_stdout(devnull) do
            gen(cfg, _TL_LIBREF, md, "TlSynth", _TL_W.create_type_registry(cfg), false)
        end
        return out isa Tuple ? String(first(out)) : String(out)
    finally
        rm(dir; recursive = true, force = true)
    end
end

function _tl_body(code, name)
    m = match(Regex("^function $(name)\\(.*?^end\$", "ms"), code)
    return m === nothing ? "" : String(m.match)
end

@testset "both generators read a thread-local global through its accessor" begin
    for lang in (:c, :cpp)
        @testset "$lang" begin
            gl = Dict{String,Any}(
                "tls_counter" => Dict{String,Any}("name" => "tls_counter", "scope" => "", "symbol" => "tls_counter",
                                                  "c_type" => "int", "julia_type" => "Cint", "thread_local" => true,
                                                  "tls_accessor" => "replibuild_tls_tls_counter"),
                "tls_blob" => Dict{String,Any}("name" => "tls_blob", "scope" => "", "symbol" => "tls_blob",
                                               "c_type" => "struct opaque", "julia_type" => "Any", "thread_local" => true,
                                               "tls_accessor" => "replibuild_tls_tls_blob"),
                "tls_member" => Dict{String,Any}("name" => "tls_member", "scope" => "", "symbol" => "tls_member",
                                                 "c_type" => "int", "julia_type" => "Cint", "thread_local" => true,
                                                 "tls_refused" => "it is a static data member, which can be declared only inside its class"),
                "plain_count" => Dict{String,Any}("name" => "plain_count", "scope" => "", "symbol" => "plain_count",
                                                  "c_type" => "int", "julia_type" => "Cint"))
            code = _tl_generate(lang, gl)
            v = _tl_body(code, "tls_counter")
            p = _tl_body(code, "tls_counter_ptr")
            @test occursin("unsafe_load(ccall((:replibuild_tls_tls_counter, LIBRARY_PATH), Ptr{Cint}, ()))", v)
            @test occursin("return ccall((:replibuild_tls_tls_counter, LIBRARY_PATH), Ptr{Cint}, ())", p)
            @test occursin("Valid only on the thread that called", code)
            # No `cglobal` for a thread-local global, of either kind.
            @test !occursin("cglobal((:tls_", code)
            # A type it cannot hold: the pointer alone, untyped.
            @test occursin("return ccall((:replibuild_tls_tls_blob, LIBRARY_PATH), Ptr{Cvoid}, ())", _tl_body(code, "tls_blob_ptr"))
            @test _tl_body(code, "tls_blob") == ""
            # Refused: both names say why.
            @test occursin("Base.error(", _tl_body(code, "tls_member")) && occursin("Base.error(", _tl_body(code, "tls_member_ptr"))
            @test occursin("static data member", _tl_body(code, "tls_member_ptr"))
            # A plain global is what it was.
            @test occursin("cglobal((:plain_count, LIBRARY_PATH), Cint)", code)

            m = Module(:TlSynthProbe)
            Base.include_string(m, code)
            mod = Base.invokelatest(getfield, m, :TlSynth)
            has(s) = Base.invokelatest(isdefined, mod, s)
            @test all(has, (:tls_counter, :tls_counter_ptr, :tls_blob_ptr, :tls_member, :tls_member_ptr, :plain_count))
            err = try
                Base.invokelatest(getfield(mod, :tls_member_ptr)); ""
            catch e
                sprint(showerror, e)
            end
            @test occursin("`tls_member` is a thread-local global", err)
        end
    end
end

@testset "a thread-local global is resolved through its accessor" begin
    gl() = Dict{String,Any}(
        "tls_counter" => Dict{String,Any}("symbol" => "tls_counter", "thread_local" => true,
                                          "tls_accessor" => "replibuild_tls_tls_counter"),
        "tls_refused" => Dict{String,Any}("symbol" => "tls_refused", "thread_local" => true, "tls_refused" => "x"))
    md = Dict{String,Any}("globals" => gl())
    # On PE the variable may not be exported; its accessor is what binds.
    @test _TL_W._drop_unresolvable_globals!(md, Set(["replibuild_tls_tls_counter", "tls_refused"])) == 0
    md = Dict{String,Any}("globals" => gl())
    @test _TL_W._drop_unresolvable_globals!(md, Set(["tls_counter"])) == 2
    @test _TL_W._global_runtime_symbol("k", Dict{String,Any}("symbol" => "s")) == "s"
    @test _TL_W._global_runtime_symbol("k", Dict{String,Any}()) == "k"
end
