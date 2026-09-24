#!/usr/bin/env julia
# test/test_enum_underscore_name.jl — an enumerator whose C++ name is `_`
# (2026-09-20, onednn 3.11.3 GENERATOR-enum-underscore-name.md).
#
# oneDNN spells the zero enumerator of two enums literally `_`:
#
#     enum class block_dim_t { _, _A, _B, _C, _D, _E, _AB, _BC, _CD, _CE };
#     enum class inner_blk_t { _, _4a, _4b, … };            // tag_traits.hpp
#
# `_` is a legal C++ identifier, DWARF records it faithfully, and the TOML did
# not invent it. The emitter is what broke: `_sanitize_cpp_type_name` collapsed
# it to "_" and then `rstrip(s, '_')` emptied it, so the module carried
#
#     @enum inner_blk_t::Cint begin
#          = 0
#
# and `_assert_wrapper_parses` refused a 271 923-line wrapper over two
# enumerators. That guard did its job; this test is so it never has to.
#
# The repair is NOT `_`. Julia parses `@enum T _ = 0` and evaluates it, but
# all-underscore identifiers are WRITE-ONLY — their values cannot be used in
# expressions — so the enumerator would be unreferenceable. For inner_blk_t
# that is value 0, the C++ default (`inner_blk_t{}`), and a member nobody can
# name is barely better than the dropped member the report warned against.
#
# Library-free: drives the sanitizers and EXECUTES what they produce.

using Test
using RepliBuild

const W = RepliBuild.Wrapper

@testset "enumerator named `_`" begin

    @testset "all-underscore names survive sanitizing in BOTH generators" begin
        # The C++ side had no empty guard at all; the C side had one but
        # answered "_UnknownType", which parses yet silently renames a member
        # the user is reading out of a header. Same two-generators-one-guard
        # split as the namespace phantom-`this` bug — they must agree.
        for f in (W._sanitize_cpp_type_name, W._sanitize_c_type_name)
            for raw in ("_", "__", "___")
                s = f(raw)
                @test !isempty(s)
                @test s == "c__"
                # The actual defect: must not be all-underscore, or it is
                # write-only and the enumerator cannot be referenced.
                @test !all(==('_'), s)
                @test Base.isidentifier(s)
            end
        end
    end

    @testset "neighbouring enumerators are untouched" begin
        # rstrip only ever struck TRAILING underscores, so `_4a` and `_A` were
        # never at risk. Pinned so a broader "strip underscores" repair cannot
        # quietly rename the rest of the enum.
        for f in (W._sanitize_cpp_type_name, W._sanitize_c_type_name)
            @test f("_4a") == "_4a"
            @test f("_4b") == "_4b"
            @test f("_A")  == "_A"
            @test f("_AB") == "_AB"
        end
    end

    @testset "pre-existing collapse/trim behaviour is unchanged" begin
        for f in (W._sanitize_cpp_type_name, W._sanitize_c_type_name)
            @test f("foo_")     == "foo"
            @test f("foo__bar") == "foo_bar"
            @test f("4x")       == "_4x"
            @test f("for")      == "c_for"
        end
    end

    @testset "empty stays a SENTINEL on the C++ side" begin
        # Deliberate asymmetry. C returns "_UnknownType"; C++ must keep
        # returning "" because callers read it as a signal, not as a name:
        # GeneratorCpp.jl:2390 does
        #   isempty(inner_safe) ? "Ptr{Cvoid}" : "$wrapper_kw{$inner_safe}"
        # so a placeholder here would emit Ptr{_UnknownType} where Ptr{Cvoid}
        # is meant. GeneratorCpp.jl:1553 gates on it too. All-underscore is a
        # DIFFERENT case from empty, which is why only the former was repaired.
        @test W._sanitize_cpp_type_name("") == ""
        @test W._sanitize_c_type_name("")   == "_UnknownType"
    end

    @testset "the emitted @enum parses, evaluates, and reads back" begin
        # The end-to-end property the wrapper needs: not just parseable, but a
        # value-0 member that can actually be NAMED and round-tripped.
        zero_name = W._sanitize_cpp_type_name("_")
        src = "@enum inner_blk_t::Cint $zero_name = 0 $(W._sanitize_cpp_type_name("_4a")) = 1"

        @test Meta.parse("begin\n$src\nend") isa Expr    # the original failure

        m = Module(:EnumUnderscoreProbe)
        Base.include_string(m, "using Base: @enum\n" * src)

        T = getfield(m, :inner_blk_t)
        zero_member = getfield(m, Symbol(zero_name))

        @test Int(zero_member) == 0                      # readable, not write-only
        @test T(0) === zero_member                       # round-trips
        @test length(instances(T)) == 2
        @test Symbol(zero_name) in Symbol.(instances(T))
    end

    @testset "the enumerator is not dropped" begin
        # Skipping value 0 would compile a wrapper and silently change the C++
        # default (`inner_blk_t{}`), which is worse than the syntax error.
        names = [W._sanitize_cpp_type_name(n) for n in ("_", "_4a", "_4b")]
        @test length(unique(names)) == 3                 # no collision/merge
        @test all(!isempty, names)
    end
end
