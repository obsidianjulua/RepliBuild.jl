#!/usr/bin/env julia
# Wrapper type-binding guard — library-free traces.
#
# The class: a type name reaches a foreign-call signature that the generated
# module never declares. Julia raises `UndefVarError` at INCLUDE time, so the
# whole wrapper is dead — every function, not just the offending one — and the
# error names a type the user never asked about.
#
# Live instance (2026-08-02, miniaudio): `ma_fopen(FILE** ppFile, …)`. DWARF
# resolves FILE to `struct _IO_FILE`, which sits on _INTERNAL_TYPE_BLOCKLIST
# because libc internals shouldn't be exported. The blocklist suppressed the
# DECLARATION but nothing suppressed the USES, so `Ptr{Ptr{_IO_FILE}}` went
# into the ccall tuple and all 1178 functions died on one missing line.
#
# Two things are pinned here:
#   1. `_undefined_ccall_types` finds that shape and ignores the shapes that
#      only look like it (docstrings, defined types, Julia builtins).
#   2. `_assert_wrapper_loadable` REFUSES to write such a wrapper, naming the
#      type and the function — the failure is caught at generation, not at the
#      user's include.
#
# No toolchain: these drive the checker with synthetic module text. The
# end-to-end assertions (a real FILE** in the API surface) live in
# test/c_abomination_test/verify.jl §8.
#
# Usage: julia --project=. test/test_wrapper_type_bindings.jl

using Test
using RepliBuild
using Libdl

const WU = RepliBuild.Wrapper

@testset "Wrapper type bindings" begin

@testset "_defined_type_names" begin
    src = """
    module Demo
    struct Plain end
    mutable struct Mut
        x::Cint
    end
    @enum Color::Cuint begin
        red = 0
    end
    abstract type Base_ end
    primitive type Prim 8 end
    const Alias = Ptr{Cvoid}
    end
    """
    names = WU._defined_type_names(src)
    for n in ("Plain", "Mut", "Color", "Base_", "Prim", "Alias")
        @test n in names
    end
    # Not a definition: a use.
    @test !("Cvoid" in names)
end

@testset "Undefined type in a ccall signature is found" begin
    # The exact miniaudio shape, minimized.
    bad = """
    module Bad
    const LIBRARY_PATH = "libbad.so"
    function ma_fopen(ppFile, path)
        ccall((:ma_fopen, LIBRARY_PATH), Cint, (Ptr{Ptr{_IO_FILE}}, Ptr{UInt8},), ppFile, path)
    end
    end
    """
    found = WU._undefined_ccall_types(bad)
    @test length(found) == 1
    @test found[1][1] == "_IO_FILE"
    @test found[1][2] == "ma_fopen"          # blames the right function

    # ...and generation refuses to write it, naming both.
    err = try
        WU._assert_wrapper_loadable(bad, "Bad"); nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("_IO_FILE", err.msg)
    @test occursin("ma_fopen", err.msg)
    @test occursin("Refusing to write", err.msg)
end

@testset "Clean wrappers are not flagged" begin
    # Every referenced type is declared, a builtin, or a type constructor.
    good = """
    module Good
    const LIBRARY_PATH = "libgood.so"
    struct Handle end
    @enum Mode::Cuint begin
        fast = 0
    end
    function open_it(h, m)
        ccall((:open_it, LIBRARY_PATH), Ptr{Handle}, (Ptr{Ptr{Handle}}, Mode, Csize_t,), h, m, 0)
    end
    function raw(p)
        ccall((:raw, LIBRARY_PATH), Cvoid, (Ptr{Cvoid}, Cstring,), p, "x")
    end
    end
    """
    @test isempty(WU._undefined_ccall_types(good))
    @test WU._assert_wrapper_loadable(good, "Good") === nothing
end

@testset "Docstrings are not code" begin
    # The generator documents the C-level type even where the emitted ccall
    # degrades it to Ptr{Cvoid}. That mention is inside a string and cannot
    # raise, so flagging it would be a false positive that blocks a wrapper
    # which loads perfectly well — the exact shape c_abomination_test emits.
    doc_only = """
    module Doc
    const LIBRARY_PATH = "libdoc.so"
    \"\"\"
        stream_open(out, path)

    # Arguments
    - `out::Ptr{Ptr{_IO_FILE}}`
    \"\"\"
    function stream_open(out, path)
        ccall((:stream_open, LIBRARY_PATH), Clong, (Ptr{Ptr{Cvoid}}, Ptr{UInt8},), out, path)
    end
    end
    """
    @test isempty(WU._undefined_ccall_types(doc_only))
end

@testset "By-value struct params are covered too" begin
    # Not every undeclared type arrives behind a pointer: a struct passed by
    # value appears as a bare name in the tuple.
    byval = """
    module ByVal
    const LIBRARY_PATH = "libbyval.so"
    function takes_it(v)
        ccall((:takes_it, LIBRARY_PATH), Cvoid, (SomeStruct,), v)
    end
    end
    """
    found = WU._undefined_ccall_types(byval)
    @test length(found) == 1
    @test found[1][1] == "SomeStruct"
    @test found[1][2] == "takes_it"
end

@testset "Ptr{Cvoid} degradation is ABI-identical" begin
    # Why the fix degrades rather than declares: pointer width is the whole
    # ABI contract for an opaque handle, so Ptr{Cvoid} loses nothing a caller
    # could have used — _IO_FILE has no accessible fields in Julia either way.
    @test sizeof(Ptr{Cvoid}) == sizeof(Ptr{Int})

    # Indirection DEPTH is preserved — only the undeclared leaf is swapped.
    # FILE** stays a pointer-to-pointer, matching what the generator emits for
    # c_abomination_test's stream_open: Ptr{Ptr{Cvoid}}.
    @test WU._resolve_forward_ptr("Ptr{Ptr{_IO_FILE}}", Set{String}()) == "Ptr{Ptr{Cvoid}}"
    @test WU._resolve_forward_ptr("Ptr{_IO_FILE}", Set{String}()) == "Ptr{Cvoid}"
    @test WU._resolve_forward_ptr("Ptr{Known}", Set(["Known"])) == "Ptr{Known}"
end

# ── One prior generation kept beside a regenerated wrapper ───────────────────
#
# `julia/` output dirs are gitignored, so regenerating destroyed the only copy
# of what the generator produced last time — and regenerating is exactly when
# that copy becomes interesting. `.prev` answers "what did this change?".

@testset "wrapper rotation keeps exactly one prior generation" begin
    W = RepliBuild.Wrapper

    mktempdir() do dir
        out = joinpath(dir, "Rot.jl")
        v1 = "# Generated: 2026-01-01 00:00:00\nmodule Rot\nexport a\na() = 1\nend\n"
        v2 = "# Generated: 2026-01-02 00:00:00\nmodule Rot\nexport a, b\na() = 1\nb() = 2\nend\n"

        # First write: nothing to rotate.
        W._write_wrapper(out, v1, "Rot")
        @test read(out, String) == v1
        @test !isfile(out * ".prev")

        # Same content, different wrap-time header. The timestamp alone must NOT
        # trigger rotation, or `.prev` becomes "the wrap from moments ago" and
        # the last real diff is gone — the failure you only notice when you need it.
        v1b = replace(v1, "2026-01-01 00:00:00" => "2026-06-06 12:34:56")
        @test v1b != v1                      # the bytes really do differ
        W._write_wrapper(out, v1b, "Rot")
        @test !isfile(out * ".prev")

        # Real change: rotate, and `.prev` holds what was there before.
        W._write_wrapper(out, v2, "Rot")
        @test isfile(out * ".prev")
        @test read(out, String) == v2
        @test read(out * ".prev", String) == v1b
        @test occursin("export a, b", read(out, String))
        @test !occursin("export a, b", read(out * ".prev", String))

        # Exactly one generation is kept — a third write overwrites `.prev`
        # rather than accumulating .prev.prev.
        v3 = "# Generated: 2026-01-03 00:00:00\nmodule Rot\nexport a\na() = 3\nend\n"
        W._write_wrapper(out, v3, "Rot")
        @test read(out * ".prev", String) == v2
        @test !isfile(out * ".prev.prev")
        @test count(f -> endswith(f, ".prev"), readdir(dir)) == 1

        # `.prev` must not look like a Julia source file to anything scanning
        # the output dir for wrappers.
        @test !endswith(out * ".prev", ".jl")

        # The guard still runs on the write path: a wrapper exporting a name it
        # never binds is refused, and nothing is written or rotated.
        before = read(out, String)
        @test_throws ErrorException W._write_wrapper(
            out, "module Rot\nexport a, ghost\na() = 1\nend\n", "Rot")
        @test read(out, String) == before
        @test read(out * ".prev", String) == v2
    end

    @test W._wrapper_differs("# Generated: A\nx = 1\n", "# Generated: B\nx = 1\n") == false
    @test W._wrapper_differs("# Generated: A\nx = 1\n", "# Generated: A\nx = 2\n") == true
end

# ── A type CONSTRUCTOR is not a type name ────────────────────────────────────
#
# The screens above all work by scraping names out of emitted source and asking
# "does the module bind this?". `Ptr` answers no — nothing declares it — but the
# question is meaningless for a constructor, and the union-accessor filter asked
# it anyway: its `::(\w+)` regex captures `Ptr` from `::Ptr{Cvoid}`, so EVERY
# pointer-typed union accessor was dropped no matter how ordinary its pointee.
# Cost across the Hub was 76 accessors — sqlite's `p4union` kept 1 member of 16
# (`i::Cint`), lua's `Value` 3 of 6 — every one of them a fully-resolved type the
# module did bind.
#
# It is invisible three times over: the wrapper loads, the build is green, and a
# missing accessor is indistinguishable from a member the generator was right to
# skip. So this pins the behaviour AND the guard, since only the guard
# generalizes to the next constructor someone scrapes.

@testset "union accessors survive a Ptr return type" begin
    _member(name, jt, ct) = Dict{String,Any}(
        "name" => name, "julia_type" => jt, "c_type" => ct, "offset" => "0x0", "size" => 8)

    metadata = Dict{String,Any}(
        "functions" => Any[],
        "struct_definitions" => Dict{String,Any}(
            # A struct the union can legitimately point at.
            "Pointee" => Dict{String,Any}(
                "kind" => "struct", "byte_size" => "0x4",
                "members" => Any[Dict{String,Any}(
                    "name" => "v", "julia_type" => "Cint", "c_type" => "int",
                    "offset" => "0x0", "size" => 4)]),
            # On `_INTERNAL_TYPE_BLOCKLIST`, so it is a known DWARF struct that
            # deliberately gets no declaration — the only way to reach the
            # filter's reject path, since the emitter's own gate gets anything
            # DWARF never described. This is the miniaudio shape exactly.
            "_IO_FILE" => Dict{String,Any}(
                "kind" => "struct", "byte_size" => "0x8",
                "members" => Any[Dict{String,Any}(
                    "name" => "flags", "julia_type" => "Cint", "c_type" => "int",
                    "offset" => "0x0", "size" => 4)]),
            # sqlite's p4union in miniature: one scalar arm, the rest pointers.
            "Arms" => Dict{String,Any}(
                "kind" => "union", "byte_size" => "0x8",
                "members" => Any[
                    _member("i",       "Cint",            "int"),
                    _member("p",       "Ptr{Cvoid}",      "void*"),
                    _member("z",       "Ptr{UInt8}",      "char*"),
                    _member("deep",    "Ptr{Ptr{Cvoid}}", "void**"),
                    _member("known",   "Ptr{Pointee}",    "Pointee*"),
                    # The two shapes the filter exists for, one per regex:
                    # by value (first regex reads the annotation) and behind a
                    # pointer (second regex reads the pointee).
                    _member("blocked", "_IO_FILE",        "struct _IO_FILE"),
                    _member("fp",      "Ptr{_IO_FILE}",   "struct _IO_FILE*"),
                ]),
        ),
        "globals" => Dict{String,Any}(),
        "function_pointer_typedefs" => Dict{String,Any}(),
    )

    dir = mktempdir()
    toml = joinpath(dir, "replibuild.toml")
    write(toml, """
    [project]
    name = "unionsynth"
    root = "$(dir)"

    [link]
    enable_lto = false

    [wrap]
    language = "c"

    [types]
    strictness = "warn"
    allow_unknown_structs = true

    [cache]
    enabled = false
    """)
    cfg = RepliBuild.ConfigurationManager.load_config(toml)
    libref = abspath(first(filter(p -> occursin("libjulia", basename(p)), Libdl.dllist())))
    code = WU.generate_introspective_module_c(
        cfg, libref, metadata, "UnionSynth", WU.create_type_registry(cfg), true)

    # The scalar arm always worked — it is the control that proves the union
    # itself was emitted, so a failure below is about the member's TYPE.
    @test occursin("function get_Arms_i(u::Arms)::Cint", code)

    # Every pointer arm: this is the regression.
    @test occursin("function get_Arms_p(u::Arms)::Ptr{Cvoid}", code)
    @test occursin("function set_Arms_p!(u::Arms, v::Ptr{Cvoid})", code)
    @test occursin("function get_Arms_z(u::Arms)::Ptr{UInt8}", code)
    @test occursin("function get_Arms_deep(u::Arms)::Ptr{Ptr{Cvoid}}", code)
    @test occursin("function get_Arms_known(u::Arms)::Ptr{Pointee}", code)

    # …and the filter still does its job. `_IO_FILE` is a struct DWARF fully
    # described, so both members clear the emitter's gate and reach the filter;
    # the blocklist then means no chunk declares it. Widening the accepted set
    # must not cost these — the pointer one especially, since it is the pointee
    # the second regex has to keep reading.
    @test !occursin("get_Arms_blocked", code)
    @test !occursin("get_Arms_fp", code)
    @test !occursin("struct _IO_FILE\n", code)

    # Restored accessors are exported, not merely defined.
    export_line = match(r"^export .*$"m, code)
    @test export_line !== nothing
    for n in ("get_Arms_p", "set_Arms_p!", "get_Arms_z", "get_Arms_known")
        @test occursin(n, export_line.match)
    end

    # The whole point is that they RUN. Emitted code, executed.
    m = Module(:UnionSynthProbe)
    Core.eval(m, Meta.parseall(code))
    Arms = Core.eval(m, :(UnionSynth.Arms))
    u = Base.invokelatest(Arms)
    probe = Ptr{Cvoid}(UInt(0xdeadbeef))
    Base.invokelatest(Core.eval(m, :(UnionSynth.set_Arms_p!)), u, probe)
    @test Base.invokelatest(Core.eval(m, :(UnionSynth.get_Arms_p)), u) == probe
    # Same eight bytes, read through a different arm — that a union CAN do this
    # is why the missing accessors mattered.
    @test Base.invokelatest(Core.eval(m, :(UnionSynth.get_Arms_i)), u) ==
          reinterpret(Int32, UInt32(0xdeadbeef))
end

@testset "rejecting on a bound name is refused" begin
    # The guard is the durable half: `_JULIA_TYPE_CTORS` is a list someone must
    # remember to extend, and this fires when they didn't.
    @test_throws ErrorException WU._assert_no_bound_name_rejected(Set(["Ptr"]), "probe")
    for ctor in WU._JULIA_TYPE_CTORS
        @test_throws ErrorException WU._assert_no_bound_name_rejected(Set([ctor]), "probe")
    end

    # A genuinely undeclared DWARF leak is exactly what these filters SHOULD
    # drop — the guard must stay out of the way of it.
    @test WU._assert_no_bound_name_rejected(
        Set(["ma_dr_flac__memory_stream", "_IO_FILE", "Ghost"]), "probe") === nothing
    @test WU._assert_no_bound_name_rejected(Set{String}(), "probe") === nothing

    # The message has to name the offender — the whole failure mode was not
    # knowing which name did it.
    err = try
        WU._assert_no_bound_name_rejected(Set(["Ptr", "_IO_FILE"]), "Union accessor filter")
        nothing
    catch e
        sprint(showerror, e)
    end
    @test err !== nothing
    @test occursin("Union accessor filter", err)
    @test occursin("Ptr", err)
    @test !occursin("_IO_FILE", err)   # not a false positive; not reported as one
end

# The mirror image of the undefined-type class above: there the ccall names a
# type nothing defines, here it names one a PARAMETER has shadowed. Both kill
# the module at include, because ccall resolves its argument tuple eagerly at
# method definition. Found on libcurl (26 functions, 5 types), reproduced with
# no library in play — `struct bufq *bufq` is ordinary C.
@testset "Parameter must not shadow a ccall argument type" begin
    W = RepliBuild.Wrapper

    shadowed = """
    struct bufq end
    function sh_take_bufq(bufq::Any, n::Integer)::Cint
        n_c = Cint(n)
        return ccall((:sh_take_bufq, LIBRARY_PATH), Cint, (Ptr{bufq}, Cint,), bufq, n_c)
    end
    """
    err = try
        W._assert_no_shadowed_ccall_types(shadowed, "M"); ""
    catch e; sprint(showerror, e) end
    @test !isempty(err)
    @test occursin("bufq", err)
    @test occursin("sh_take_bufq", err)
    # The message has to say which way to fix it — renaming the TYPE would be
    # the wrong move, since the ccall tuple must still reach it.
    @test occursin("renaming the PARAMETER", err) || occursin("rename the PARAMETER", err) ||
          occursin("Fix by renaming the PARAMETER", err)

    # Renaming the parameter is the fix, and must pass.
    fixed = replace(shadowed,
        "sh_take_bufq(bufq::Any" => "sh_take_bufq(bufq_1::Any",
        "Cint,), bufq, n_c"      => "Cint,), bufq_1, n_c")
    @test W._assert_no_shadowed_ccall_types(fixed, "M") === nothing

    # Must-not-flag shapes ------------------------------------------------
    # A parameter sharing a name with a type NOT in this function's tuple.
    @test W._assert_no_shadowed_ccall_types("""
    function f(other::Any, n::Integer)::Cint
        return ccall((:f, LIBRARY_PATH), Cint, (Ptr{bufq}, Cint,), other, n)
    end
    """, "M") === nothing

    # A comma inside a parameter's own type must not be read as an extra
    # parameter — the split has to be depth-aware, same as _method_sig_keys.
    @test W._assert_no_shadowed_ccall_types("""
    function f(s::Union{AbstractString,Cstring}, n::Integer)::Cint
        return ccall((:f, LIBRARY_PATH), Cint, (Cstring, Cint,), s, n)
    end
    """, "M") === nothing

    # And it must be reachable from the real write path, or it prevents nothing.
    @test occursin("_assert_no_shadowed_ccall_types",
                   read(joinpath(@__DIR__, "..", "src", "Wrapper", "Generator.jl"), String))
end

# The third member of the family, and the one that is NOT about resolution:
# `Any` resolves fine, it just means the wrong thing. In a foreign call it
# declares that the callee returns a `jl_value_t*`, so the returned integer is
# dereferenced as a Julia object — a segfault inside dispatch on some LATER
# call, with a stack naming neither the wrapper nor the library. libcurl shipped
# 18: every curl_*_setopt and curl_easy_getinfo overload.
@testset "Foreign call must not return Any" begin
    W = RepliBuild.Wrapper

    # The variadic shape that actually shipped: @ccall, return spliced as `Any`.
    err = try
        W._assert_no_any_ccall_return("""
        function curl_easy_setopt(handle::Any, option::Any, va_1::Integer)::Any
            return @ccall LIBRARY_PATH.var"curl_easy_setopt"(handle::Ptr{Cvoid}, option::Cint; va_1::Cint)::Any
        end
        """, "M"); ""
    catch e; sprint(showerror, e) end
    @test !isempty(err)
    @test occursin("curl_easy_setopt", err)
    @test occursin("Any", err)
    # The message must say what to do, and `Cvoid` is the safe degradation —
    # discarding a value is recoverable, corrupting one is not.
    @test occursin("Cvoid", err)

    # The classic form is equally eager and must be caught too.
    err2 = try
        W._assert_no_any_ccall_return("""
        function g(x::Integer)::Any
            return ccall((:g, LIBRARY_PATH), Any, (Cint,), x)
        end
        """, "M"); ""
    catch e; sprint(showerror, e) end
    @test !isempty(err2)
    @test occursin("g", err2)

    # Must-not-flag shapes ------------------------------------------------
    # `::Any` in a Julia SIGNATURE is ordinary and correct — the C generator
    # emits it for every unmodellable parameter. Flagging it would refuse
    # essentially every wrapper, so this is the assertion that keeps the guard
    # narrow enough to be reachable.
    @test W._assert_no_any_ccall_return("""
    function f(x::Any, y::Any)::Cint
        return ccall((:f, LIBRARY_PATH), Cint, (Ptr{Cvoid}, Ptr{Cvoid}), x, y)
    end
    """, "M") === nothing

    # `Any` in the ARGUMENT tuple is also legitimate — it is the return
    # position alone that is wrong. The classic-form pattern anchors on the
    # element right after the (name, lib) pair for exactly this reason.
    @test W._assert_no_any_ccall_return(
        "function h(x)::Cint\n    return ccall((:h, LIBRARY_PATH), Cint, (Any,), x)\nend\n",
        "M") === nothing

    # A Julia function annotated `::Any` with no foreign call in it at all.
    @test W._assert_no_any_ccall_return(
        "function k(x)::Any\n    return x\nend\n", "M") === nothing

    # And an @ccall with a real return type must pass.
    @test W._assert_no_any_ccall_return(
        "function m(x::Integer)::Cint\n    return @ccall LIBRARY_PATH.var\"m\"(x::Cint;)::Cint\nend\n",
        "M") === nothing

    # Reachable from the real write path, or it prevents nothing.
    @test occursin("_assert_no_any_ccall_return",
                   read(joinpath(@__DIR__, "..", "src", "Wrapper", "Generator.jl"), String))
end

end  # testset
