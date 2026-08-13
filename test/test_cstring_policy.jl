#!/usr/bin/env julia
# test/test_cstring_policy.jl — the char* return policy, and the guard that
# keeps a dispatch tier from deciding it (2026-08-12).
#
# A `char*` return must reach Julia one of exactly two ways: the policy wrapper
# (`::Union{String,Nothing}` — NULL is a value, the buffer is copied, and a
# `[wrap.cstring_owned]` deallocator runs), or a raw `<name>_ptr` sibling that
# is NAMED for handing back an unmanaged pointer.
#
# That held on the ccall path and nowhere else. The C++ generator's MLIR
# dispatch branch `continue`s past the ccall path, and the policy lived only
# there — so 75 functions across five Hub packages returned a bare `Cstring`
# with no `_ptr` sibling, and any `cstring_owned` declaration on one of them was
# silently discarded. Surfaced by a user calling `hello_message()` and getting
# `Cstring(0x7fb608c37000)`.
#
# The fix is one derivation (`_cstring_wrapper_pair`) consumed by every tier and
# both generators, plus `_assert_cstring_policy` on the single write path so a
# future emission site cannot quietly opt out again.
#
# Library-free: drives the emitter and the guard directly, no toolchain, no
# built fixture.

using Test
using RepliBuild

const W = RepliBuild.Wrapper

@testset "Cstring return policy" begin

    @testset "The pair is emitted from one derivation" begin
        chunk = W._cstring_wrapper_pair("get_name", "obj::Any",
                    "    ptr = ccall((:get_name, LIBRARY_PATH), Cstring, (Ptr{Cvoid},), obj)",
                    "    return ccall((:get_name, LIBRARY_PATH), Cstring, (Ptr{Cvoid},), obj)",
                    "")

        @test occursin("function get_name(obj::Any)::Union{String,Nothing}", chunk)
        @test occursin("function get_name_ptr(obj::Any)::Cstring", chunk)
        # The policy, verbatim from _cstring_policy_lines
        @test occursin("ptr == C_NULL && return nothing", chunk)
        @test occursin("s = unsafe_string(ptr)", chunk)
        # No deallocator declared → nothing is freed, and the _ptr doc says so
        @test !occursin("Cvoid, (Cstring,), ptr", chunk)
        @test occursin("(no copy, no NULL check).", chunk)
        # Both halves parse
        @test Meta.parse("begin\n" * chunk * "\nend") isa Expr
    end

    @testset "An owned buffer is freed, and only in the policy wrapper" begin
        chunk = W._cstring_wrapper_pair("dup", "x::Any",
                    "    ptr = ccall((:dup, LIBRARY_PATH), Cstring, (Ptr{Cvoid},), x)",
                    "    return ccall((:dup, LIBRARY_PATH), Cstring, (Ptr{Cvoid},), x)",
                    "lib_free")

        @test occursin("ccall((:lib_free, LIBRARY_PATH), Cvoid, (Cstring,), ptr)", chunk)
        # exactly once — the raw variant must NOT free what the caller now owns
        @test count("lib_free", chunk) == 1
        @test occursin("NOT freed — caller owns the buffer", chunk)

        policy_half, ptr_half = split(chunk, "function dup_ptr(")
        @test occursin("lib_free", policy_half)
        @test !occursin("lib_free", ptr_half)
    end

    @testset "Every tier's call body flows through the same policy" begin
        # The tier supplies only the two call bodies. Whatever it supplies, the
        # presentation is identical — that is the invariant the bug violated.
        bodies = [
            ("ccall",  "    ptr = ccall((:f, LIBRARY_PATH), Cstring, (Cint,), n)"),
            ("jit",    "    ptr = RepliBuild.JITManager.invoke(\"_mlir_ciface__Z1fi_thunk\", Cstring, n)"),
            ("aot",    "    ptr = ccall((:_mlir_ciface__Z1fi_thunk, THUNKS_LIBRARY_PATH), Cstring, (Ptr{Ptr{Cvoid}},), inner_ptrs)"),
        ]
        for (tier, bind) in bodies
            chunk = W._cstring_wrapper_pair("f", "n::Cint", bind,
                        replace(bind, "    ptr = " => "    return "), "")
            @test occursin("::Union{String,Nothing}", chunk)
            @test occursin("function f_ptr(n::Cint)::Cstring", chunk)
            @test occursin("ptr == C_NULL && return nothing", chunk)
            # and what one derivation produced must satisfy the guard
            @test W._assert_cstring_policy(chunk, "tier-$tier") === nothing
        end
    end

    @testset "Guard refuses a bare Cstring return, per emission shape" begin
        shapes = Dict(
            "tier-2 JIT" => "    return RepliBuild.JITManager.invoke(\"_mlir_ciface__Z1fv_thunk\", Cstring)",
            "tier-2 AOT" => "    return ccall((:_mlir_ciface__Z1fv_thunk, THUNKS_LIBRARY_PATH), Cstring, (Ptr{Ptr{Cvoid}},), inner_ptrs)",
            "plain ccall" => "    return ccall((:f, LIBRARY_PATH), Cstring, (), )",
            "llvmcall"    => "    return Base.llvmcall((_SLICE_f, \"f\"), Cstring, Tuple{}, )",
            "varargs"     => "    return @ccall LIBRARY_PATH.var\"f\"(;)::Cstring",
        )
        for (shape, body) in shapes
            txt = "function leaky()\n$body\nend\n"
            err = try
                W._assert_cstring_policy(txt, "M"); nothing
            catch e
                sprint(showerror, e)
            end
            @test err !== nothing            # $shape must be refused
            @test occursin("leaky", err)
            @test occursin("_cstring_wrapper_pair", err)
            @test occursin("cstring_owned", err)   # names the silent-discard consequence
        end
    end

    @testset "Guard accepts what it must not flag" begin
        # A _ptr variant is exempt BY NAME — that is the contract.
        @test W._assert_cstring_policy(
            "function f_ptr(x::Any)::Cstring\n    return ccall((:f, LIBRARY_PATH), Cstring, (Cint,), x)\nend\n",
            "M") === nothing

        # The deallocator names Cstring in an ARGUMENT tuple, not a return.
        @test W._assert_cstring_policy(
            """
            function f(x::Any)::Union{String,Nothing}
                ptr = ccall((:f, LIBRARY_PATH), Cstring, (Cint,), x)
                ptr == C_NULL && return nothing
                s = unsafe_string(ptr)
                ccall((:lib_free, LIBRARY_PATH), Cvoid, (Cstring,), ptr)
                return s
            end
            """, "M") === nothing

        # A variadic overload that TAKES strings and returns void. An
        # unanchored `::Cstring` match flagged all four of box2d's
        # b2Log/b2Dump overloads — the guard's own first draft did.
        @test W._assert_cstring_policy(
            """
            function b2Log_Cstring(fmt::Any, va_1::Cstring)::Cvoid
                return @ccall LIBRARY_PATH.var"b2Log"(fmt::Cstring; va_1::Cstring)::Cvoid
            end
            """, "M") === nothing

        # A Cstring-typed ARGUMENT on an unrelated return type.
        @test W._assert_cstring_policy(
            "function g(s::Any)::Cint\n    return ccall((:g, LIBRARY_PATH), Cint, (Cstring,), s)\nend\n",
            "M") === nothing
    end

    @testset "Guard reports every offender, not the first" begin
        txt = join(("function f$(i)()\n    return RepliBuild.JITManager.invoke(\"t$(i)\", Cstring)\nend\n"
                    for i in 1:3), "\n")
        err = try
            W._assert_cstring_policy(txt, "M"); ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("3 function(s)", err)
        for i in 1:3
            @test occursin("f$i", err)
        end
    end

    @testset "The guard runs on the single write path" begin
        # _assert_wrapper_loadable is what _write_wrapper calls; the policy check
        # must be part of it, or the guard exists and never fires.
        bad = """
        module M
        function leaky()
            return RepliBuild.JITManager.invoke("t", Cstring)
        end
        end
        """
        @test_throws ErrorException W._assert_wrapper_loadable(bad, "M")
    end
end
