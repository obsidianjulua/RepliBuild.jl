#!/usr/bin/env julia
# C linkage in a C++ library: which functions may be called through a plain ccall.
#
# `extern "C"` changes a function's name, not the language of its body:
#
#     extern "C" int c_linkage_throws(int x) {
#         if (x < 0) throw std::runtime_error("from extern C");
#         return x;
#     }
#
# The router sent every unmangled name to Tier 3 ("plain C functions never
# throw"), and a throw through a `ccall` has no landing pad: `terminate called
# after throwing an instance of 'std::runtime_error'`, SIGABRT (2026-11-17).
#
# No source spelling says which C-linkage functions can throw, so the answer is
# the compiler's: LLVM's `nounwind` on the function's own definition in the IR
# the library is linked from (`_nounwind_definitions`), which `_mark_noexcept!`
# records as `nounwind`. In a C++ library a C-linkage function reaches ccall
# only with that proof; any other takes a Tier 2 thunk, whose `try_call`
# catches. A mangled function still needs the proof AND a declared `noexcept`,
# and a C library never asks. Toolchain-free: the IR here is text. The live
# half is test_c_linkage_unwind_live.jl.
#
# Negative check: with `is_ccall_safe`'s exception gate back to
# `_is_cpp_function(f) && !_is_noexcept(f)`, the routing testsets fail.

using Test
using RepliBuild
using Libdl

const CLU_C = RepliBuild.Compiler
const CLU_W = RepliBuild.Wrapper
const CLU_F = RepliBuild.JLCSIRGenerator.FunctionGen

# The define lines clang 23 wrote for the audit's probe at -O0: the throwing
# function carries a personality and group #0, the one that cannot throw #3.
const CLU_IR = raw"""
; ModuleID = 'unwind.cpp'
@cl_twin = alias i32 (i32, i32), ptr @cl_add
define i32 @c_linkage_throws(i32 noundef %0) #0 personality ptr @__gxx_personality_v0 !dbg !2192 {
  ret i32 0
}
define i32 @cl_add(i32 noundef %0, i32 noundef %1) #3 !dbg !2001 {
  ret i32 0
}
define noundef i32 @_Z12never_throwsi(i32 noundef %0) #3 !dbg !2002 {
  ret i32 0
}
define noundef i32 @_Z6provenv() #3 !dbg !2003 {
  ret i32 0
}
declare i32 @cl_elsewhere(i32) #3
attributes #0 = { mustprogress noinline optnone sspstrong uwtable "frame-pointer"="all" }
attributes #3 = { mustprogress noinline nounwind optnone sspstrong uwtable "frame-pointer"="all" }
"""

_clu_fn(name, mangled; extra...) = merge(Dict{String,Any}(
    "name" => name, "mangled" => mangled, "demangled" => startswith(mangled, "_Z") ? "$name(int)" : name,
    "class" => "", "is_method" => false, "is_vararg" => false, "exported" => true,
    "parameters" => Any[Dict{String,Any}("name" => "x", "c_type" => "int", "julia_type" => "Cint")],
    "return_type" => Dict{String,Any}("c_type" => "int", "julia_type" => "Cint", "size" => 4),
    "return_type_source" => "dwarf", "parameters_source" => "dwarf"),
    Dict{String,Any}(String(k) => v for (k, v) in extra))

@testset "C linkage in a C++ library: routed by the nounwind proof" begin

    nounwind = mktempdir() do dir
        f = joinpath(dir, "m.ll")
        write(f, CLU_IR)
        CLU_C._nounwind_definitions([f])
    end
    @test nounwind == Set(["cl_add", "_Z12never_throwsi", "_Z6provenv"])

    @testset "_mark_noexcept! records the proof beside is_noexcept" begin
        funcs = [_clu_fn("c_linkage_throws", "c_linkage_throws"),
                 _clu_fn("cl_add", "cl_add"),
                 _clu_fn("cl_elsewhere", "cl_elsewhere"),          # only declared in the IR
                 _clu_fn("never_throws", "_Z12never_throwsi"),     # declared noexcept
                 _clu_fn("proven", "_Z6provenv")]                  # only proven
        CLU_C._mark_noexcept!(funcs, Set(["never_throws"]), nounwind)
        @test [f["nounwind"] for f in funcs] == [false, true, false, true, true]
        # `is_noexcept` is what it was: the proof AND a declared `noexcept`.
        @test [f["is_noexcept"] for f in funcs] == [false, false, false, true, false]

        # An alias has no definition: its code is its aliasee's.
        al = [_clu_fn("cl_twin", "cl_twin"; alias_of = "cl_add"),
              _clu_fn("cl_other", "cl_other"; alias_of = "c_linkage_throws")]
        CLU_C._mark_noexcept!(al, Set{String}(), nounwind)
        @test [f["nounwind"] for f in al] == [true, false]

        # No IR (ingest): no key at all, which a reader takes as "not proven".
        none = [_clu_fn("cl_add", "cl_add")]
        CLU_C._mark_noexcept!(none, Set{String}(), nothing)
        @test !haskey(none[1], "nounwind")
        @test none[1]["is_noexcept"] == false
    end

    @testset "is_ccall_safe: C linkage needs the proof" begin
        sd = Dict{String,Any}()
        safe(f) = CLU_W.is_ccall_safe(f, sd)
        @test safe(_clu_fn("cl_add", "cl_add"; nounwind = true))
        @test !safe(_clu_fn("c_linkage_throws", "c_linkage_throws"; nounwind = false))
        # Metadata from before the key, or built with no IR: not proven.
        @test !safe(_clu_fn("cl_add", "cl_add"))
        @test !safe(_clu_fn("cl_add", "cl_add"; is_noexcept = false))
        # Older metadata did prove a C-linkage function declared `noexcept`.
        @test safe(_clu_fn("cl_quiet", "cl_quiet"; is_noexcept = true))
        # Only `true` is a proof.
        @test !safe(_clu_fn("cl_add", "cl_add"; nounwind = nothing))
        @test !safe(_clu_fn("cl_add", "cl_add"; nounwind = "true"))
        # A mangled function is unchanged: the proof alone does not widen Tier 3.
        @test !safe(_clu_fn("proven", "_Z6provenv"; nounwind = true, is_noexcept = false))
        @test safe(_clu_fn("never_throws", "_Z12never_throwsi"; nounwind = true, is_noexcept = true))
        @test CLU_W._is_nounwind(_clu_fn("a", "a"; nounwind = true))
        @test CLU_W._is_nounwind(_clu_fn("a", "a"; is_noexcept = true))
        @test !CLU_W._is_nounwind(_clu_fn("a", "a"))
    end

    dir = mktempdir()
    config(language) = begin
        toml = joinpath(dir, "replibuild_$language.toml")
        write(toml, """
        [project]
        name = "unwindsynth"
        root = "$(escape_string(dir))"
        [link]
        enable_lto = false
        [wrap]
        language = "$language"
        [types]
        strictness = "warn"
        allow_unknown_structs = true
        [cache]
        enabled = false
        """)
        RepliBuild.ConfigurationManager.load_config(toml)
    end
    libref = abspath(first(filter(p -> occursin("libjulia", basename(p)), Libdl.dllist())))
    functions() = Any[_clu_fn("cl_add", "cl_add"; nounwind = true, is_noexcept = false),
                      _clu_fn("c_linkage_throws", "c_linkage_throws"; nounwind = false, is_noexcept = false),
                      _clu_fn("cl_old", "cl_old"; is_noexcept = false)]     # metadata without the key
    meta(language) = Dict{String,Any}("functions" => functions(), "globals" => Dict{String,Any}(),
                                      "struct_definitions" => Dict{String,Any}(), "record_abi" => Dict{String,Any}(),
                                      "language" => language)
    body(out, name) = (m = match(Regex("^function $(name)\\(.*?^end\$", "ms"), out); m === nothing ? "" : String(m.match))

    @testset "the C++ generator: ccall with the proof, a thunk without" begin
        cfg = config("cpp")
        (out, thunks) = CLU_W.generate_introspective_module_cpp(cfg, libref, meta("c++"), "UnwindSynth",
                                                                CLU_W.create_type_registry(cfg), true)
        @test occursin("ccall((:cl_add, LIBRARY_PATH), Cint, (Cint,), x_c)", body(out, "cl_add"))
        @test !occursin("JITManager", body(out, "cl_add"))
        for name in ("c_linkage_throws", "cl_old")
            b = body(out, name)
            @test occursin("RepliBuild.JITManager.invoke(\"_mlir_ciface_$(name)_thunk\", Cint, Cint(x))", b)
            @test !occursin("ccall((:", b)
        end
        # The thunks the wrapper asks for are exactly those two.
        @test sort!(collect(String, thunks)) == ["c_linkage_throws", "cl_old"]
    end

    @testset "a C-linkage function keeps its `_safe` wrapper on the thunk path" begin
        # `<f>_safe` (the result owned by a `Managed<X>`) was emitted on the
        # ccall path only, where every unmangled name used to be. An `extern
        # "C"` function returning a new object calls `new`, so it is never
        # `nounwind`, and moving it to the thunk must not take the wrapper away.
        cfg = config("cpp")
        widget(name, mangled; extra...) = merge(
            _clu_fn(name, mangled; extra...),
            Dict{String,Any}("return_type" => Dict{String,Any}("c_type" => "Widget*", "julia_type" => "Ptr{Cvoid}", "size" => 8)))
        m = meta("c++")
        m["functions"] = Any[
            merge(_clu_fn("widget_destroy", "widget_destroy"; nounwind = true), Dict{String,Any}(
                "parameters" => Any[Dict{String,Any}("name" => "w", "c_type" => "Widget*", "julia_type" => "Ptr{Widget}")],
                "return_type" => Dict{String,Any}("c_type" => "void", "julia_type" => "Cvoid", "size" => 0))),
            widget("widget_clone", "widget_clone"; nounwind = false),      # C linkage, may throw: thunk
            widget("widget_peek", "widget_peek"; nounwind = true),         # C linkage, proven: ccall
            widget("widget_spawn", "_Z12widget_spawni"; is_noexcept = false)]   # mangled, may throw: thunk
        m["struct_definitions"] = Dict{String,Any}("Widget" => Dict{String,Any}(
            "kind" => "struct", "byte_size" => "0x4",
            "members" => Any[Dict{String,Any}("name" => "id", "c_type" => "int", "julia_type" => "Cint",
                                              "offset" => "0x0", "size" => 4)]))
        (out, thunks) = CLU_W.generate_introspective_module_cpp(cfg, libref, m, "UnwindSafe",
                                                                CLU_W.create_type_registry(cfg), true)
        @test occursin("mutable struct ManagedWidget\n", out)
        @test occursin("JITManager.invoke(\"_mlir_ciface_widget_clone_thunk\", Ptr{Widget}, Cint(x))", body(out, "widget_clone"))
        s = body(out, "widget_clone_safe")
        @test startswith(s, "function widget_clone_safe(x::Integer)::ManagedWidget\n")
        @test occursin("ptr = widget_clone(x)\n    return ManagedWidget(ptr)", s)
        @test occursin(r"^export [^\n]*\bwidget_clone_safe\b"m, out)
        # On the ccall path it is what it was.
        @test occursin("ccall((:widget_peek, LIBRARY_PATH), Ptr{Widget}", body(out, "widget_peek"))
        @test startswith(body(out, "widget_peek_safe"), "function widget_peek_safe(x::Integer)::ManagedWidget\n")
        # A mangled function on the thunk path never had one.
        @test occursin("_mlir_ciface__Z12widget_spawni_thunk", body(out, "widget_spawn"))
        @test body(out, "widget_spawn_safe") == ""
        @test sort!(collect(String, thunks)) == ["_Z12widget_spawni", "widget_clone"]
    end

    @testset "the thunk catches: try_call for a C-linkage function" begin
        ir = CLU_F.generate_function_thunks(functions()[2:2], Dict{String,Any}(); may_throw = true)
        @test occursin("func.func @c_linkage_throws_thunk(", ir)
        @test occursin("jlcs.try_call", ir)
    end

    @testset "a C library is unchanged" begin
        cfg = config("c")
        out = CLU_W.generate_introspective_module_c(cfg, libref, meta("c"), "UnwindSynthC",
                                                    CLU_W.create_type_registry(cfg), true)
        out isa Tuple && (out = out[1])
        for name in ("cl_add", "c_linkage_throws", "cl_old")
            @test occursin("ccall((:$name, LIBRARY_PATH), Cint, (Cint,), x_c)", body(out, name))
        end
        @test !occursin("JITManager.invoke", out)
    end
end
