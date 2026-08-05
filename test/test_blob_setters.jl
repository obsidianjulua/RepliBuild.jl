#!/usr/bin/env julia
# test/test_blob_setters.jl — writes for byte-blob structs, and the namespace
# guard that makes their error paths work (2026-08-05).
#
# A struct whose layout could not be modelled field-by-field is emitted as an
# immutable `_data::NTuple{N,UInt8}` blob with `getproperty` accessors. Nothing
# was emitted in the other direction, so a param struct built BY the library
# was read-only — and on llama.cpp that is the only path in (an embedding model
# returns NULL unless `ctx_params.embeddings` is set). Callers were left
# patching bytes through hand-rolled offset tables copied out of
# `compilation_metadata.json`.
#
# Everything here is library-free: it drives the emitters directly and executes
# what they produce, so it needs no toolchain and no built fixture.

using Test
using RepliBuild

const W = RepliBuild.Wrapper

@testset "Byte-blob struct setters" begin

    @testset "one store per getter shape" begin
        @test occursin("unsafe_store!(Ptr{Ptr{Cvoid}}(p + 8), convert(Ptr{Cvoid}, v))",
                       W._blob_store_expr("p", 8, :pointer))
        @test occursin("unsafe_store!(Ptr{Int32}(p + 4), convert(Int32, v))",
                       W._blob_store_expr("a", 4, :primitive; julia_type="Int32"))
        @test occursin("unsafe_store!(Ptr{Inner}(p + 0), convert(Inner, v))",
                       W._blob_store_expr("i", 0, :typed_struct; struct_name="Inner"))
        @test occursin("getfield(convert(Blob, v), :_data)",
                       W._blob_store_expr("b", 16, :blob_struct_full;
                                          struct_name="Blob", nested_size=8))
        # A member the struct has no room for is zero-padded on read, so the
        # write must stop at the bytes that FIT or it corrupts the next member.
        partial = W._blob_store_expr("b", 16, :blob_struct_partial;
                                     struct_name="Blob", nested_size=8, actual_size=4)
        @test occursin("for _i in 1:4", partial)
        @test !occursin("1:8", partial)
        # An unknown shape yields nothing rather than a guess — the getter skips
        # those members too, and a field settable but unreadable is worse than
        # neither.
        @test isempty(W._blob_store_expr("x", 0, :something_else))
    end

    @testset "chunk shape: elseif chain, single exit" begin
        chunk = W._blob_setter_chunk("P", [W._blob_store_expr("a", 0, :primitive; julia_type="Int32"),
                                           W._blob_store_expr("b", 4, :primitive; julia_type="Int32")])
        @test occursin("function setproperty(x::P, s::Symbol, v)", chunk)
        @test count("if s === :", chunk) == 2
        @test occursin("elseif s === :b", chunk)
        # One `return` outside the GC.@preserve region: returning from inside it
        # would skip the preserve-end.
        @test count("return P(buf[])", chunk) == 1
        @test occursin("Base.setproperty!(x::P", chunk)
        # `error` is rebindable by the library (see the namespace testset below).
        @test !occursin(r"(?<![.\w])error\(", chunk)
        @test isempty(W._blob_setter_chunk("P", String[]))
    end

    @testset "generated setters execute and round-trip" begin
        setter = W._blob_setter_chunk("P",
            [W._blob_store_expr("a", 0,  :primitive; julia_type="Int32"),
             W._blob_store_expr("p", 8,  :pointer),
             W._blob_store_expr("f", 16, :primitive; julia_type="Bool")])
        src = """
        module _BlobSetterProbe
        struct P
            _data::NTuple{24, UInt8}
        end
        P() = P(ntuple(i -> 0x00, 24))
        function Base.getproperty(x::P, s::Symbol)
            s === :_data && return getfield(x, :_data)
            if s === :a
                return GC.@preserve x unsafe_load(Ptr{Int32}(pointer_from_objref(Ref(x._data)) + 0))
            end
            if s === :p
                return GC.@preserve x unsafe_load(Ptr{Ptr{Cvoid}}(pointer_from_objref(Ref(x._data)) + 8))
            end
            if s === :f
                return GC.@preserve x unsafe_load(Ptr{Bool}(pointer_from_objref(Ref(x._data)) + 16))
            end
            Base.error("type P has no field \$s")
        end
        $setter
        $(W._blob_setproperties_chunk())
        end
        """
        m = include_string(Main, src)

        z = m.P()
        @test z.a == 0 && z.f == false

        one = m.setproperty(z, :a, 7)
        @test one.a == 7
        @test z.a == 0                       # immutable: the original is untouched

        many = m.setproperties(z; a = 11, f = true, p = Ptr{Cvoid}(UInt(0x1234)))
        @test many.a == 11
        @test many.f == true
        @test UInt(many.p) == 0x1234
        @test z.a == 0 && z.f == false       # still untouched

        # Writing one field must not disturb its neighbours.
        @test m.setproperty(many, :a, 99).f == true
        @test UInt(m.setproperty(many, :f, false).p) == 0x1234

        # Values are converted, so a wrong-width write is refused rather than
        # silently truncated into the next field.
        @test_throws InexactError m.setproperty(z, :a, 2^40)

        @test_throws ErrorException m.setproperty(z, :nope, 1)
        # `x.field = v` is what someone types first; it must name the way out.
        err = try
            (x = m.P(); x.a = 1; "")
        catch e
            sprint(showerror, e)
        end
        @test occursin("setproperties", err)
        @test occursin("immutable byte blob", err)
    end

    # ── the namespace guard ──────────────────────────────────────────────────
    #
    # A generated module is a namespace the LIBRARY populates. llama.cpp pulls in
    # libstdc++'s `std::codecvt_base::result`, whose members include one named
    # `error` — so `@enum result::Cuint begin … error = 2 … end` rebound `error`
    # for the whole module, and every failure path in the wrapper (including the
    # long-standing `getproperty` "no field" branch) raised
    # `MethodError: objects of type result are not callable` instead of its
    # message. Refusing to emit the library's own `error = 2` is not an option;
    # qualifying our own calls is.
    @testset "generated code may not call a rebindable Base name bare" begin
        ok = """
        module M
        @enum result::Cuint begin
            ok = 0
            error = 2
        end
        f(x) = Base.error("nope \$x")
        end
        """
        @test W._assert_base_calls_qualified(ok, "M") === nothing

        bad = replace(ok, "Base.error(" => "error(")
        err = try
            W._assert_base_calls_qualified(bad, "M"); ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("unqualified call", err)
        @test occursin("`error`", err)

        # Names that merely CONTAIN the word are not calls to it.
        for benign in ("_error(x)", "error_code(x)", "@error \"x\"", "obj.error(x)")
            @test W._assert_base_calls_qualified("module M\nf() = $benign\nend", "M") === nothing
        end

        # A library may OWN the name. cJSON has a `struct error`, so its wrapper
        # emits a `struct error`, a zero-arg `error()` constructor, and a call to
        # that constructor — the same wrapper that most needs `Base.error`
        # everywhere else. The check keys on a string-literal first argument,
        # which every generator emission has and none of these do.
        owns = """
        module M
        # C struct: error (2 members)
        struct error
            json::Ptr{UInt8}
        end
        function error()
            return error(Ptr{UInt8}())
        end
        g() = Base.error("real failure")
        end
        """
        @test W._assert_base_calls_qualified(owns, "M") === nothing
        @test occursin("unqualified call",
                       try W._assert_base_calls_qualified(
                               replace(owns, "Base.error(\"real" => "error(\"real"), "M"); ""
                           catch e; sprint(showerror, e) end)
    end
end
