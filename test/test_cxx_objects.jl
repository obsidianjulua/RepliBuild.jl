# =============================================================================
# A C++ object Julia holds where C++ built it (2026-11-27)
#
# A function that returns a `std::vector`, a `std::map`, a `std::unique_ptr` or
# a class with a destructor builds the object in storage its caller supplies.
# A wrapper used to copy those bytes into a Julia value: invalid wherever the
# object points into itself (`std::list`, `std::map`), and never ended. The
# value was then passed by value by lending its bytes, where C++ copies or
# moves.
#
# John's decision (2026-11-27, "by value and whatever best supports that"):
#
#   * a by-value return that has to be ended is held where C++ built it
#     (`CxxObjects.Owned`) and ended by the type's own destructor, once: when
#     Julia collects it, or at `release!`;
#   * a type with nothing to end stays a plain value;
#   * a by-value parameter gets a COPY where C++ can copy, and the caller keeps
#     its object. A type C++ cannot copy is moved, and `move(x)` asks for that:
#     the object is spent, and using it again is an error.
#
# No compiler: the rules on storage this file ends, copies and counts itself,
# and the wrapper text for made-up metadata. test_cxx_objects_live.jl calls a
# library.
# =============================================================================

using Test
using RepliBuild

const _CXO = RepliBuild.CxxObjects
const _CXO_W = RepliBuild.Wrapper
const _CXO_J = RepliBuild.JITManager

# What the test's stand-ins for a type's three functions did. An "object" is
# 16 bytes whose first 8 are its value; ending one counts, and writes a mark
# over it so a use after its end is seen.
const _CXO_ENDED = Ref(0)
const _CXO_COPIED = Ref(0)
const _CXO_SEEN = UInt64[]          # the value each ended object had
_cxo_end(p::Ptr{Cvoid}) = (push!(_CXO_SEEN, unsafe_load(Ptr{UInt64}(p))); _CXO_ENDED[] += 1;
                           unsafe_store!(Ptr{UInt64}(p), 0xdeaddeaddeaddead); nothing)
_cxo_copy(d::Ptr{Cvoid}, s::Ptr{Cvoid}) = (unsafe_store!(Ptr{UInt64}(d), unsafe_load(Ptr{UInt64}(s)));
                                           unsafe_store!(Ptr{UInt64}(d), 0, 2); _CXO_COPIED[] += 1; Cint(0))
_cxo_throwing_copy(d::Ptr{Cvoid}, s::Ptr{Cvoid}) = Cint(1)

const _CXO_T = NTuple{16, UInt8}

# A kind whose functions are this file's, marked found for this process.
function _cxo_kind(name; ends = true, copies = true, moves = true, copier = _cxo_copy)
    k = _CXO.Kind(name, 16, 8, ends, copies, moves, "end_x", "copy_x", "move_x")
    k.ender = @cfunction(_cxo_end, Cvoid, (Ptr{Cvoid},))
    k.copier = copier === _cxo_copy ? @cfunction(_cxo_copy, Cint, (Ptr{Cvoid}, Ptr{Cvoid})) :
                                      @cfunction(_cxo_throwing_copy, Cint, (Ptr{Cvoid}, Ptr{Cvoid}))
    k.mover = C_NULL
    k.process = _CXO._PROCESS[]
    return k
end

# An object "C++ built": the storage, a value written where it is, born.
function _cxo_make(k, value; T = _CXO_T)
    o = _CXO.slot(T, k, "unused: the kind is already resolved")
    GC.@preserve o unsafe_store!(Ptr{UInt64}(pointer_from_objref(o)), UInt64(value))
    return _CXO.born!(o)
end
_cxo_value(p::Ptr) = unsafe_load(Ptr{UInt64}(p))

@testset "A C++ object Julia holds where C++ built it" begin

    @testset "the object is the Julia object's first bytes, and stays there" begin
        @test fieldoffset(_CXO.Owned{_CXO_T}, 1) == 0
        @test fieldname(_CXO.Owned{_CXO_T}, 1) === :bytes
        @test ismutabletype(_CXO.Owned{_CXO_T})
        k = _cxo_kind("box")
        o = _cxo_make(k, 7)
        p = _CXO.address(o)
        @test p == pointer_from_objref(o)
        @test UInt(p) % 16 == 0                      # Julia's heap alignment: what `align <= 16` relies on
        GC.gc()
        @test _CXO.address(o) == p                   # the collector does not move it
        @test _cxo_value(p) == 7
        @test o[] isa _CXO_T && reinterpret(UInt64, collect(o[]))[1] == 7
        @test occursin("Owned C++ box (16 bytes, live)", sprint(show, o))
        _CXO.release!(o)
    end

    @testset "storage C++ has not built into is not an object" begin
        k = _cxo_kind("box")
        before = _CXO_ENDED[]
        o = _CXO.slot(_CXO_T, k, "unused")
        @test !_CXO.isalive(o)
        @test_throws ArgumentError _CXO.address(o)
        # A call that threw leaves it so: nothing ends it, at release or at collection.
        _CXO.release!(o)
        finalize(o)
        @test _CXO_ENDED[] == before
        # The wrapper's size and the library's must agree, and the holder must be plain bits.
        @test_throws ErrorException _CXO.slot(NTuple{8, UInt8}, k, "unused")
        @test_throws ErrorException _CXO.slot(Base.RefValue{Int}, _CXO.Kind("r", 8, 8, true, true, true, "e", "c", "m"), "unused")
    end

    @testset "ended once: at release, or when Julia collects it" begin
        k = _cxo_kind("box")
        before = _CXO_ENDED[]
        o = _cxo_make(k, 11)
        @test _CXO.isalive(o)
        _CXO.release!(o)
        @test _CXO_ENDED[] == before + 1 && last(_CXO_SEEN) == 11
        @test !_CXO.isalive(o)
        _CXO.release!(o)                             # twice is not an error, and not a second ending
        finalize(o)                                  # nor is the collector's turn
        @test _CXO_ENDED[] == before + 1
        # Using it afterwards is an error that says what happened.
        err = try _CXO.address(o); nothing catch e; e end
        @test err isa ArgumentError && occursin("was released", err.msg)
        @test_throws ArgumentError o[]
        @test_throws ArgumentError Base.unsafe_convert(Ptr{Cvoid}, o)
        @test occursin("released", sprint(show, o))

        # Left to the collector: its finalizer ends it, once.
        o2 = _cxo_make(k, 12)
        finalize(o2)
        @test _CXO_ENDED[] == before + 2 && last(_CXO_SEEN) == 12
        finalize(o2)
        @test _CXO_ENDED[] == before + 2

        # ...and really by the collector, with no call from here.
        made = 200
        (() -> (for i in 1:made; _cxo_make(k, 1000 + i); end; nothing))()
        for _ in 1:4; GC.gc(true); end
        @test _CXO_ENDED[] >= before + 2 + made - 8   # a conservative root may hold the last few
    end

    @testset "a type with nothing to end is never ended" begin
        k = _cxo_kind("plain"; ends = false)
        before = _CXO_ENDED[]
        o = _cxo_make(k, 5)
        _CXO.release!(o)
        finalize(o)
        @test _CXO_ENDED[] == before
        @test !_CXO.isalive(o)                       # released is still released
    end

    @testset "copy is the type's own copy constructor" begin
        k = _cxo_kind("box")
        o = _cxo_make(k, 21)
        copies = _CXO_COPIED[]
        c = copy(o)
        @test _CXO_COPIED[] == copies + 1
        @test c !== o && _CXO.address(c) != _CXO.address(o)
        @test _cxo_value(_CXO.address(c)) == 21 && _CXO.isalive(c) && _CXO.isalive(o)
        _CXO.release!(c)
        @test _CXO.isalive(o) && _cxo_value(_CXO.address(o)) == 21   # ending the copy leaves the original
        # A type C++ cannot copy: an error that names the way that works.
        u = _cxo_make(_cxo_kind("unique_ptr<T>"; copies = false), 3)
        err = try copy(u); nothing catch e; e end
        @test err isa ArgumentError && occursin("cannot be copied", err.msg) && occursin("RepliBuild.move", err.msg)
        @test _CXO.isalive(u)
        # A copy constructor that throws: no half-built object is ever ended.
        t = _cxo_make(_cxo_kind("thrower"; copier = _cxo_throwing_copy), 4)
        before = _CXO_ENDED[]
        @test_throws ErrorException copy(t)
        GC.gc()
        @test _CXO_ENDED[] == before
        foreach(_CXO.release!, (o, u, t))
    end

    @testset "by value: a copy where C++ can copy, and the caller keeps its object" begin
        k = _cxo_kind("vector<int>")
        o = _cxo_make(k, 31)
        (ended, copies) = (_CXO_ENDED[], _CXO_COPIED[])
        a = _CXO.byvalue(_CXO_T, k, "unused", o)
        @test a isa _CXO.Passed{_CXO_T}
        @test _CXO_COPIED[] == copies + 1
        @test a.ptr != _CXO.address(o) && _cxo_value(a.ptr) == 31     # the call gets a temporary
        @test a.temp !== nothing && a.spend === nothing
        _CXO.settle!(a)                                               # ...ended when the call returns
        @test _CXO_ENDED[] == ended + 1
        @test _CXO.isalive(o) && _cxo_value(_CXO.address(o)) == 31    # the caller's object is untouched
        _CXO.settle!(a)
        @test _CXO_ENDED[] == ended + 1                               # settled twice: ended once
        _CXO.release!(o)
    end

    @testset "by value: what C++ cannot copy is moved, and is spent" begin
        k = _cxo_kind("unique_ptr<T>"; copies = false)
        o = _cxo_make(k, 41)
        (ended, copies) = (_CXO_ENDED[], _CXO_COPIED[])
        a = _CXO.byvalue(_CXO_T, k, "unused", o)
        @test _CXO_COPIED[] == copies
        @test a.ptr == pointer_from_objref(o) && a.temp === nothing && a.spend === o   # the object itself
        @test _CXO.isalive(o)                        # until the call returns
        _CXO.settle!(a)
        @test _CXO_ENDED[] == ended + 1              # what the call left of it is ended, once
        @test !_CXO.isalive(o)
        err = try _CXO.address(o); nothing catch e; e end
        @test err isa ArgumentError && occursin("was moved into a C++ call", err.msg)
        @test_throws ArgumentError _CXO.byvalue(_CXO_T, k, "unused", o)   # passing it again is that error
        @test_throws ArgumentError RepliBuild.move(o)
        @test occursin("moved into a call", sprint(show, o))
        finalize(o)
        @test _CXO_ENDED[] == ended + 1
    end

    @testset "move(x) gives the object where a copy would have been made" begin
        k = _cxo_kind("vector<int>")
        o = _cxo_make(k, 51)
        (ended, copies) = (_CXO_ENDED[], _CXO_COPIED[])
        m = RepliBuild.move(o)
        @test m isa _CXO.Moved{_CXO_T}
        a = _CXO.byvalue(_CXO_T, k, "unused", m)
        @test _CXO_COPIED[] == copies && a.ptr == pointer_from_objref(o) && a.spend === o
        _CXO.settle!(a)
        @test _CXO_ENDED[] == ended + 1 && !_CXO.isalive(o)
        @test_throws ArgumentError _CXO.byvalue(_CXO_T, k, "unused", m)   # a move is one move
    end

    @testset "by value: an address, or the bytes one used to be returned as" begin
        k = _cxo_kind("vector<int>")
        o = _cxo_make(k, 61)
        p = _CXO.address(o)
        (ended, copies) = (_CXO_ENDED[], _CXO_COPIED[])
        # A pointer to an existing object: a temporary copied from it.
        a = _CXO.byvalue(_CXO_T, k, "unused", p)
        @test _CXO_COPIED[] == copies + 1 && a.ptr != p && _cxo_value(a.ptr) == 61 && a.spend === nothing
        _CXO.settle!(a)
        @test _CXO_ENDED[] == ended + 1 && _CXO.isalive(o)
        # The bytes (what `x[]` gives): the same, from a copy of those bytes.
        b = _CXO.byvalue(_CXO_T, k, "unused", o[])
        @test _CXO_COPIED[] == copies + 2 && _cxo_value(b.ptr) == 61 && b.temp !== nothing
        _CXO.settle!(b)
        # A null pointer is passed as it came: there is nothing to copy.
        n = _CXO.byvalue(_CXO_T, k, "unused", C_NULL)
        @test n.ptr == C_NULL && n.temp === nothing && n.spend === nothing
        # A type C++ cannot copy, by its address: the address itself, as before.
        ku = _cxo_kind("unique_ptr<T>"; copies = false)
        u = _CXO.byvalue(_CXO_T, ku, "unused", p)
        @test u.ptr == p && u.temp === nothing && u.spend === nothing
        _CXO.settle!(u)
        @test _CXO.isalive(o)
        # Not an address and not an object: refused, as a pointer position refuses it.
        @test_throws MethodError _CXO.byvalue(_CXO_T, k, "unused", 3)
        @test_throws MethodError _CXO.byvalue(_CXO_T, k, "unused", "text")
        _CXO.release!(o)
    end

    @testset "an object of another type is refused" begin
        k = _cxo_kind("vector<int>")
        other = _cxo_make(_cxo_kind("vector<double>"), 1)
        err = try _CXO.byvalue(_CXO_T, k, "unused", other); nothing catch e; e end
        @test err isa ArgumentError && occursin("vector<double>", err.msg) && occursin("vector<int>", err.msg)
        @test_throws ArgumentError _CXO.byvalue(_CXO_T, k, "unused", RepliBuild.move(other))
        # The same type from another wrapper (another `Kind` object of that name) passes.
        twin = _cxo_make(_cxo_kind("vector<int>"), 2)
        a = _CXO.byvalue(_CXO_T, k, "unused", twin)
        @test _cxo_value(a.ptr) == 2
        _CXO.settle!(a)
        # Held as another Julia type: refused, never reinterpreted.
        @test_throws ArgumentError _CXO.byvalue(NTuple{24, UInt8}, _CXO.Kind("v", 24, 8, true, true, true, "e", "c", "m") |>
                                                 (kk -> (kk.process = _CXO._PROCESS[]; kk)), "unused", twin)
        foreach(_CXO.release!, (other, twin))
    end

    @testset "a library class by value: the thunk copies from the object where it is" begin
        k = _cxo_kind("Tracked")
        o = _cxo_make(k, 71)
        (ended, copies) = (_CXO_ENDED[], _CXO_COPIED[])
        # The plain value passes as it is.
        plain = o[]
        l = _CXO.lend(_CXO_T, k, "unused", plain)
        @test l.arg === plain && l.spend === nothing
        # The object: the call is given the object itself, and nothing is copied here.
        l = _CXO.lend(_CXO_T, k, "unused", o)
        @test l.arg === o && l.spend === nothing && _CXO_COPIED[] == copies
        _CXO.settle!(l)
        @test _CXO.isalive(o) && _CXO_ENDED[] == ended
        # What the thunk's slot holds for it is the object's address.
        r = Ref(l.arg)
        @test Base.unsafe_convert(Ptr{Cvoid}, r) == pointer_from_objref(o)
        # move(x), or a class C++ can only move: spent after the call.
        l = _CXO.lend(_CXO_T, k, "unused", RepliBuild.move(o))
        @test l.arg === o && l.spend === o
        _CXO.settle!(l)
        @test !_CXO.isalive(o) && _CXO_ENDED[] == ended + 1
        ku = _cxo_kind("MoveOnly"; copies = false)
        u = _cxo_make(ku, 72)
        l = _CXO.lend(_CXO_T, ku, "unused", u)
        @test l.spend === u
        _CXO.settle!(l)
        @test !_CXO.isalive(u)
        @test_throws ArgumentError _CXO.lend(_CXO_T, ku, "unused", u)
        @test_throws MethodError _CXO.lend(_CXO_T, k, "unused", 3)
    end

    @testset "where C++ takes a pointer or a reference, the object's address passes" begin
        k = _cxo_kind("vector<int>")
        o = _cxo_make(k, 81)
        p = pointer_from_objref(o)
        @test Base.unsafe_convert(Ptr{Cvoid}, o) == p
        @test Base.unsafe_convert(Ptr{_CXO_T}, o) == Ptr{_CXO_T}(p)
        @test Base.cconvert(Ref{_CXO_T}, o) === o && Base.unsafe_convert(Ref{_CXO_T}, o) == Ptr{_CXO_T}(p)
        # A thunk's pointer position, whatever it is declared as.
        for P in (Ptr{Cvoid}, Ptr{_CXO_J.Opaque}, Ptr{_CXO_T})
            a = _CXO_J.pointer_arg(P, o)
            @test a.ptr == p && a.keep === o
            # `Ref(x)`, the manual's way while a container was its bytes: still the object.
            r = Ref(o)
            a = _CXO_J.pointer_arg(P, r)
            @test a.ptr == p && a.keep === r
        end
        # Never a C string, and never code.
        @test_throws MethodError _CXO_J.pointer_arg(Cstring, o)
        @test_throws MethodError _CXO_J.pointer_arg(Ptr{_CXO_J.Code}, o)
        # A `ccall` position typed `Ptr{Cvoid}` or `Ref{T}` gets the same address.
        @test ccall(:jl_value_ptr, Ptr{Cvoid}, (Ptr{Cvoid},), o) == p
        # An object that is gone passes nowhere.
        _CXO.release!(o)
        @test_throws ArgumentError _CXO_J.pointer_arg(Ptr{Cvoid}, o)
        @test_throws ArgumentError _CXO_J.pointer_arg(Ptr{Cvoid}, Ref(o))
    end

    @testset "a record's fields are read where the object is" begin
        k = _CXO.Kind("Pair", 16, 8, true, true, true, "e", "c", "m")
        k.ender = @cfunction(_cxo_end, Cvoid, (Ptr{Cvoid},)); k.process = _CXO._PROCESS[]
        T = @NamedTuple{a::UInt64, b::UInt64}
        o = _cxo_make(k, 91; T)
        GC.@preserve o unsafe_store!(Ptr{UInt64}(pointer_from_objref(o)), 92, 2)
        @test o.a == 91 && o.b == 92
        @test propertynames(o) == (:a, :b)
        @test o[] === (a = UInt64(91), b = UInt64(92))
        @test_throws ErrorException o.kind            # the holder's own fields are not the record's
        _CXO.release!(o)
        @test_throws ArgumentError o.a
    end

    @testset "the functions are found in the library, once per process" begin
        k = _CXO.Kind("box", 16, 8, true, true, true, "replibuild_own_end_999999", "c", "m")
        lib = abspath(first(filter(p -> occursin("libjulia", basename(p)), Base.Libc.Libdl.dllist())))
        err = try _CXO.resolve!(k, lib); nothing catch e; e end
        @test err isa ErrorException && occursin("replibuild_own_end_999999", err.msg) &&
              occursin("RepliBuild.build and RepliBuild.wrap again", err.msg)
        @test k.process == 0 && k.ender == C_NULL     # nothing half-found is kept
        # A kind found by another process (a wrapper precompiled into a package) is looked up again.
        k2 = _cxo_kind("box")
        k2.process = _CXO._PROCESS[] + 2
        @test_throws ErrorException _CXO.resolve!(k2, lib)
        @test _CXO._PROCESS[] != 0
    end
end

# ── The wrapper ──────────────────────────────────────────────────────────────

@testset "The wrapper holds, copies and moves what the build's compiler described" begin
    vec = "vector<int, std::allocator<int> >"
    up = "unique_ptr<Tracked, std::default_delete<Tracked> >"
    facts(n, size, ends, copy) = Dict{String,Any}("size" => size, "align" => 8, "ends" => ends, "copy" => copy,
        "move" => true, "bitwise" => !ends, "end_symbol" => "replibuild_own_end_$n",
        "copy_symbol" => "replibuild_own_copy_$n", "move_symbol" => "replibuild_own_move_$n", "from" => "f")
    own = Dict{String,Any}(vec => facts(0, 24, true, true), up => facts(1, 8, true, false),
                           "Tracked" => facts(2, 8, true, true), "Copied" => facts(3, 16, false, true),
                           "Flags" => facts(4, 4, true, true))
    structs = Dict{String,Any}(
        "Tracked" => Dict{String,Any}("kind" => "class", "byte_size" => "0x8", "scope" => "",
            "members" => Any[Dict{String,Any}("name" => "box_", "c_type" => "int*", "julia_type" => "Ptr{Cint}", "offset" => "0x0", "size" => 8)]),
        "Copied" => Dict{String,Any}("kind" => "struct", "byte_size" => "0x10", "scope" => "",
            "members" => Any[Dict{String,Any}("name" => "v", "c_type" => "int[4]", "julia_type" => "NTuple{4, Cint}", "offset" => "0x0", "size" => 16)]))
    ra = Dict{String,Any}(
        vec => Dict{String,Any}("byte_size" => "0x18", "pass" => "reference", "empty" => false),
        up => Dict{String,Any}("byte_size" => "0x8", "pass" => "reference", "empty" => false),
        "Tracked" => Dict{String,Any}("byte_size" => "0x8", "pass" => "reference", "empty" => false),
        "Copied" => Dict{String,Any}("byte_size" => "0x10", "pass" => "reference", "empty" => false))

    @testset "which type the compiler described" begin
        kind(c; ret = false, o = own) = _CXO_W._ownership_kind(c, o, structs, ra; ret)
        k = kind(vec; ret = true)
        @test k !== nothing && k.id == 0 && k.size == 24 && k.ends && k.copies && k.key == vec
        @test kind("const $vec") !== nothing                       # a qualifier is not another type
        @test kind(up).copies == false && kind(up).id == 1
        @test kind("Tracked").id == 2 && kind("Copied").ends == false
        # A pointer or a reference to one crosses as an address, not by value.
        @test kind("const $vec&") === nothing && kind("$vec*") === nothing && kind("Tracked&") === nothing
        # Plain data, and a type the build did not ask about.
        @test kind("int") === nothing && kind("list<int, std::allocator<int> >") === nothing
        # Metadata from before the question was asked.
        @test kind(vec; o = Dict{String,Any}()) === nothing
        @test _CXO_W._ownership_const(k) == "_RB_OWN_0"
    end

    @testset "the signature's spelling" begin
        spell(ct, c; kw...) = _CXO_W._by_address_ctype(ct, c; kw...)
        # A library class by value the wrapper returns as the object: the value, the object, or move of it.
        u = spell("Tracked", "Tracked"; owned_value = true)
        @test u == "Union{Tracked, RepliBuild.CxxObjects.Owned{Tracked}, RepliBuild.CxxObjects.Moved{Tracked}}"
        @test spell("Tracked", "Tracked") == "Tracked"
        # A container by value keeps the pointer's spelling, whatever is asked.
        @test spell("Any", vec; owned_value = true) == "Ptr{Cvoid}"
        # A reference to a smart pointer: also the object, and a Ref of it.
        r = spell("Ref{unique_ptr_Tracked}", "const $up&"; value_ref_bytes = 8, owned_ref = true)
        @test r == "Union{Ref{unique_ptr_Tracked}, Base.RefValue{NTuple{8, UInt8}}, " *
                   "RepliBuild.CxxObjects.Owned{NTuple{8, UInt8}}, Base.RefValue{RepliBuild.CxxObjects.Owned{NTuple{8, UInt8}}}}"
        @test spell("Ref{unique_ptr_Tracked}", "const $up&"; value_ref_bytes = 8) ==
              "Union{Ref{unique_ptr_Tracked}, Base.RefValue{NTuple{8, UInt8}}}"
        # Both stay themselves as a relaxed kind and at a strict position, and are types.
        bound = t -> _CXO_W._type_name_bound(t, Set(["Tracked"]))
        @test _CXO_W._relaxed_param_kind(u, bound) == u && _CXO_W._strict_param_spelling(u; bound) == u
        U = Union{_CXO_T, _CXO.Owned{_CXO_T}, _CXO.Moved{_CXO_T}}
        k = _cxo_kind("x"); o = _cxo_make(k, 1)
        @test o isa U && RepliBuild.move(o) isa U && o[] isa U && !(Ref(o) isa U) && !(3 isa U)
        _CXO.release!(o)
        # An owning type passes where the class's pointer is taken at a strict position.
        @test _CXO_W._strict_param_spelling("Ptr{Tracked}"; bound, owners = _ -> ["RepliBuild.CxxObjects.Owned{Tracked}"]) ==
              "Union{Ptr{Tracked}, Ref{Tracked}, AbstractVector{Tracked}, RepliBuild.CxxObjects.Owned{Tracked}}"
    end

    @testset "the constants are the ones the text names" begin
        kinds = Dict{Int,Any}(k.id => k for k in (_CXO_W._ownership_kind(c, own, structs, ra) for c in (vec, up, "Tracked")))
        chunk = _CXO_W._ownership_kind_chunk("x = _RB_OWN_2; y = _RB_OWN_0; z = _RB_OWN_0; w = MY_RB_OWN_1", kinds)
        lines = filter(l -> startswith(l, "const "), split(chunk, '\n'))
        @test length(lines) == 2
        @test lines[1] == "const _RB_OWN_0 = RepliBuild.CxxObjects.Kind($(repr(vec)), 24, 8, true, true, true, " *
                          "\"replibuild_own_end_0\", \"replibuild_own_copy_0\", \"replibuild_own_move_0\")"
        @test startswith(lines[2], "const _RB_OWN_2 = RepliBuild.CxxObjects.Kind(\"Tracked\", 8, 8, true, true, true, ")
        @test _CXO_W._ownership_kind_chunk("nothing here", kinds) == ""
        # A constant a call site names and nothing recorded is a generator bug, not a wrapper.
        @test_throws ErrorException _CXO_W._ownership_kind_chunk("_RB_OWN_7", kinds)
        # It evaluates to what it says.
        k = Core.eval(Module(), :(import RepliBuild; $(Meta.parse(String(lines[1]))); _RB_OWN_0))
        @test k isa _CXO.Kind && k.name == vec && k.size == 24 && k.ends && k.copies && k.end_symbol == "replibuild_own_end_0"
    end

    @testset "the docstring says what comes back" begin
        doc = "\"\"\"\n    f(x::Integer) -> NTuple{24, UInt8}\n\nWrapper for `f(int)`\n\n# Arguments\n- `x::Cint`\n\n# Returns\n- `NTuple{24, UInt8}`\n\n# Metadata\n\"\"\""
        out = _CXO_W._owned_return_doc(doc, "RepliBuild.CxxObjects.Owned{NTuple{24, UInt8}}")
        @test occursin("f(x::Integer) -> RepliBuild.CxxObjects.Owned{NTuple{24, UInt8}}\n", out)
        @test occursin("# Returns\n- `RepliBuild.CxxObjects.Owned{NTuple{24, UInt8}}`: the C++ object, held where C++ built it.", out)
        @test occursin("RepliBuild.release!(x)", out) && occursin("- `x::Cint`", out)
        @test !occursin("-> NTuple", out)
        @test _CXO_W._owned_return_doc("", "X") == ""
    end

    @testset "the wrapper text" begin
        fn(name, mangled, ret, retj, ps; class = "", method = false) = Dict{String,Any}(
            "name" => name, "mangled" => mangled, "demangled" => name,
            "class" => class, "is_method" => method, "is_vararg" => false, "is_noexcept" => false,
            "has_object_pointer" => nothing,
            "parameters" => Any[Dict{String,Any}("name" => n, "c_type" => c, "julia_type" => j) for (n, c, j) in ps],
            "return_type" => Dict{String,Any}("c_type" => ret, "julia_type" => retj))
        fns = Any[
            fn("iota", "_Z4iotai", vec, "Any", [("n", "int", "Cint")]),
            fn("boxed", "_Z5boxedi", up, "NTuple{8, UInt8}", [("v", "int", "Cint")]),
            fn("tracked", "_Z7trackedi", "Tracked", "Tracked", [("v", "int", "Cint")]),
            fn("copied", "_Z6copiedv", "Copied", "Copied", []),
            fn("take", "_Z4takeSt6vectorIiSaIiEE", "long", "Clong", [("v", vec, "Any")]),
            fn("adopt", "_Z5adoptSt10unique_ptrI7TrackedSt14default_deleteIS0_EE", "int", "Cint", [("t", up, "Any")]),
            fn("see", "_Z3see7Tracked", "int", "Cint", [("t", "Tracked", "Tracked")]),
            fn("sum", "_Z3sumRKSt6vectorIiSaIiEE", "long", "Clong", [("v", "const $vec&", "Ptr{Cvoid}")])]
        dir = mktempdir()
        write(joinpath(dir, "replibuild.toml"), """
        [project]
        name = "owned"
        root = "$(escape_string(dir))"
        [link]
        enable_lto = false
        [wrap]
        language = "cpp"
        [types]
        strictness = "warn"
        allow_unknown_structs = true
        [cache]
        enabled = false
        """)
        cfg = RepliBuild.ConfigurationManager.load_config(joinpath(dir, "replibuild.toml"))
        lib = abspath(first(filter(p -> occursin("libjulia", basename(p)), Base.Libc.Libdl.dllist())))
        gen(ownership) = (res = Base.invokelatest(_CXO_W.generate_introspective_module_cpp, cfg, lib,
                     merge(Dict{String,Any}("functions" => deepcopy(fns), "globals" => Dict{String,Any}(),
                                            "struct_definitions" => deepcopy(structs), "record_abi" => deepcopy(ra)),
                           ownership === nothing ? Dict{String,Any}() :
                           Dict{String,Any}("ownership" => Dict{String,Any}("types" => deepcopy(ownership), "skipped" => Any[]))),
                     "Owned", _CXO_W.create_type_registry(cfg), true);
                 res isa Tuple ? String(first(res)) : String(res))
        body(text, name) = (m = match(Regex("\\nfunction $(name)\\(.*?\\n\\s*end\\n", "s"), text); m === nothing ? "" : String(m.match))

        text = gen(own)
        # A return that has to be ended: the storage itself, born once C++ has built into it.
        b = body(text, "iota")
        @test occursin("_rb_ret = RepliBuild.CxxObjects.slot(NTuple{24, UInt8}, _RB_OWN_0, LIBRARY_PATH)", b)
        @test occursin("RepliBuild.JITManager.slot_address(_rb_ret)", b) && occursin("_thunk_inplace", b)
        @test occursin("return RepliBuild.CxxObjects.born!(_rb_ret)", b)
        @test !occursin("slot_value", b) && !occursin("return_slot", b)
        @test occursin("iota(n::Integer) -> RepliBuild.CxxObjects.Owned{NTuple{24, UInt8}}", text)
        @test occursin("slot(NTuple{8, UInt8}, _RB_OWN_1, LIBRARY_PATH)", body(text, "boxed"))
        @test occursin("slot(Tracked, _RB_OWN_2, LIBRARY_PATH)", body(text, "tracked"))
        # Nothing to end: the value, as before.
        b = body(text, "copied")
        @test occursin("RepliBuild.JITManager.return_slot(Copied)", b) && occursin("slot_value(_rb_ret)", b)
        @test !occursin("CxxObjects", b)

        # A container by value: a temporary the wrapper makes and settles, whether the call returns or throws.
        b = body(text, "take")
        @test occursin("function take(v::Any)", b)
        @test occursin("_rb_v1 = RepliBuild.CxxObjects.byvalue(NTuple{24, UInt8}, _RB_OWN_0, LIBRARY_PATH, v)", b)
        @test occursin(r"try GC\.@preserve _rb_v1 RepliBuild\.JITManager\.invoke\([^\n]*, _rb_v1\.ptr\) finally RepliBuild\.CxxObjects\.settle!\(_rb_v1\) end", b)
        @test !occursin("ByValue", b)                 # no lending of the caller's bytes
        b = body(text, "adopt")
        @test occursin("byvalue(NTuple{8, UInt8}, _RB_OWN_1, LIBRARY_PATH, t)", b) && occursin("settle!(_rb_v1)", b)
        # A library class by value: the thunk copies; the wrapper lends the value or the object.
        b = body(text, "see")
        @test occursin("function see(t::Union{Tracked, RepliBuild.CxxObjects.Owned{Tracked}, RepliBuild.CxxObjects.Moved{Tracked}})", b)
        @test occursin("_rb_v1 = RepliBuild.CxxObjects.lend(Tracked, _RB_OWN_2, LIBRARY_PATH, t)", b)
        @test occursin("_rb_v1.arg", b) && occursin("settle!(_rb_v1)", b)
        # A reference is a pointer position, as it was.
        b = body(text, "sum")
        @test occursin("pointer_arg(", b) && !occursin("CxxObjects", b)

        # One constant per kind the text names, defined before the functions, and the module parses.
        consts = [m.captures[1] for m in eachmatch(r"^const (_RB_OWN_\d+) = RepliBuild\.CxxObjects\.Kind\("m, text)]
        @test consts == ["_RB_OWN_0", "_RB_OWN_1", "_RB_OWN_2"]
        @test findfirst("const _RB_OWN_0", text).start < findfirst("function iota(", text).start
        ex = Meta.parseall(text)
        @test !any(e -> e isa Expr && e.head in (:incomplete, :error), ex.args)

        # Metadata from before the build asked: the wrapper it always was.
        old = gen(nothing)
        @test !occursin("CxxObjects", old) && !occursin("_RB_OWN_", old)
        @test occursin("RepliBuild.JITManager.return_slot(NTuple{24, UInt8})", body(old, "iota"))
        @test occursin("RepliBuild.JITManager.ByValue{NTuple{24, UInt8}}", body(old, "take"))
        @test occursin("function see(t::Tracked)", body(old, "see"))
        # ...and a type the build asked nothing about keeps it too.
        part = gen(Dict{String,Any}(up => own[up]))
        @test occursin("return_slot(NTuple{24, UInt8})", body(part, "iota")) && occursin("_RB_OWN_1", body(part, "boxed"))
        @test [m.captures[1] for m in eachmatch(r"^const (_RB_OWN_\d+) ="m, part)] == ["_RB_OWN_1"]
    end
end
