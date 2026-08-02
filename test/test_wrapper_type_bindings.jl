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

end  # testset
