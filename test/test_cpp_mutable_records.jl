# =============================================================================
# C++ records emitted as `mutable struct` (library-free, CI-safe)
#
# GeneratorCpp emits a named union and a bitfield record as a `mutable struct`
# whose single field is the record's C bytes. In Julia such a type is a
# reference, so every place that moves one by value must go through its bytes:
#
#   X1  A Tier 2 return of one: `JITManager._invoke_call` allocated the sret
#       buffer as `Ref{T}()`, an undefined reference for a mutable `T`, and the
#       ccall conversion threw UndefRefError on every such call.
#   X5  An accessor for a member of such a type did `unsafe_load(Ptr{T}(p))`,
#       which reads C's bytes as an object reference.
#   X6  A bitfield wider than 32 bits: the getter loaded a `UInt32` and masked it
#       with a mask that does not fit, so it always threw, and there was no
#       setter.
#
# (abi_fuzz, 2026-10-18.) The accessors are evaluated from the generated text in
# a scratch module: they use only Base, so no library or toolchain is needed.
# =============================================================================

using Test
using RepliBuild
using Libdl

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

const _MW = RepliBuild.Wrapper

# Stands in for a Tier 2 thunk returning a 4-byte record: void f(T* sret, void** args).
_mut_fill_sret(sret::Ptr{Cvoid}, ::Ptr{Ptr{Cvoid}}) =
    (unsafe_store!(Ptr{NTuple{4,UInt8}}(sret), (0x11, 0x22, 0x33, 0x44)); nothing)

_mm(name, jt, ct, off, size) = Dict{String,Any}("name" => name, "julia_type" => jt, "c_type" => ct,
                                                "offset" => off, "size" => size)
_mbit(name, jt, ct, bits, at) = Dict{String,Any}("name" => name, "julia_type" => jt, "c_type" => ct,
                                                 "offset" => nothing, "size" => 4,
                                                 "bit_size" => bits, "data_bit_offset" => at)
_mrec(size, members...) = Dict{String,Any}("kind" => "struct", "byte_size" => "0x" * string(size, base = 16),
                                           "members" => Any[members...])

# A chunk of the generated module: from `head` to the first line that is just `end`.
function _chunk(code::AbstractString, head::AbstractString)
    i = findfirst(head, code)
    i === nothing && return ""
    rest = SubString(code, first(i))
    j = findfirst(r"\nend\n", rest)
    return j === nothing ? String(rest) : String(rest[1:last(j)])
end

@testset "C++ mutable records: returns, member reads, wide bitfields (X1, X5, X6)" begin
    dir = mktempdir()
    write(joinpath(dir, "replibuild.toml"), """
    [project]
    name = "cppmutsynth"
    root = "$(escape_string(dir))"
    [wrap]
    language = "cpp"
    [types]
    strictness = "warn"
    allow_unknown_structs = true
    [cache]
    enabled = false
    """)
    cfg = RepliBuild.ConfigurationManager.load_config(joinpath(dir, "replibuild.toml"))
    meta = Dict{String,Any}(
        "functions" => Any[],
        "globals" => Dict{String,Any}(),
        "struct_definitions" => Dict{String,Any}(
            # X5's shape: a bitfield record holding another as a plain member.
            "R1" => _mrec(4, _mbit("m1", "Cint", "int", 8, 0)),
            "R3" => _mrec(16, _mbit("m1", "UInt8", "char", 3, 0),
                              _mm("m2", "R1", "R1", "0x4", 4),
                              _mm("m4", "Cint", "int", "0x8", 4),
                              _mm("m5", "Cint", "int", "0xc", 4)),
            # X6: a packed 60-bit field, and a 33-bit one starting at bit 3.
            "R4" => _mrec(8, _mbit("m4", "Culonglong", "unsigned long long", 60, 0)),
            "W9" => _mrec(8, _mbit("a", "Cuint", "unsigned int", 3, 0),
                             _mbit("b", "Culonglong", "unsigned long long", 33, 3)),
        ))
    libref = abspath(first(filter(p -> occursin("libjulia", basename(p)), Libdl.dllist())))
    res = _MW.generate_introspective_module_cpp(cfg, libref, meta, "CppMutSynth",
                                                _MW.create_type_registry(cfg), true)
    code = res isa Tuple ? res[1] : res

    # Evaluate exactly the records and accessors under test.
    pieces = String[]
    for rec in ("R1", "R3", "R4", "W9")
        push!(pieces, _chunk(code, "mutable struct $rec\n"))
        push!(pieces, "$rec() = $rec(ntuple(i -> 0x00, sizeof($rec)))\n")
    end
    for f in ("get_m1(s::R1)", "get_m2(s::R3)", "get_m4(s::R4)", "set_m4!(s::R4",
              "get_b(s::W9)", "set_b!(s::W9", "get_a(s::W9)")
        c = _chunk(code, "function $f")
        @test (f, !isempty(c)) == (f, true)
        push!(pieces, c)
    end
    M = Module(:CppMutEval)
    Base.include_string(M, join(pieces, "\n"))

    @testset "X5: a mutable member is built from its bytes" begin
        bytes = ntuple(i -> i == 5 ? 0xaa : 0x00, 16)
        r3 = Base.invokelatest(M.R3, bytes)
        m2 = Base.invokelatest(M.get_m2, r3)
        @test m2 isa M.R1
        @test getfield(m2, 1) == (0xaa, 0x00, 0x00, 0x00)
        @test Base.invokelatest(M.get_m1, m2) == 0xaa
    end

    @testset "X6: bitfields wider than 32 bits read and write" begin
        v = 0x0fedcba987654321
        r4 = Base.invokelatest(M.R4, ntuple(i -> UInt8((v >> (8 * (i - 1))) & 0xff), 8))
        @test Base.invokelatest(M.get_m4, r4) === v
        Base.invokelatest(M.set_m4!, r4, 0xf123456789abcdef)       # bits above 60 are dropped
        @test Base.invokelatest(M.get_m4, r4) === 0x0123456789abcdef
        @test getfield(r4, 1)[8] == 0x01                            # the top nibble stays clear

        w = Base.invokelatest(M.W9)
        Base.invokelatest(M.set_b!, w, 0x1_2345_6789)               # all 33 bits
        @test Base.invokelatest(M.get_b, w) === 0x0000000123456789
        @test Base.invokelatest(M.get_a, w) == 0                    # its neighbour is untouched
        @test getfield(w, 1)[1] & 0x07 == 0x00
        Base.invokelatest(M.set_b!, w, -1)                          # modular, like C
        @test Base.invokelatest(M.get_b, w) === 0x00000001ffffffff
    end

    @testset "X1: a mutable record comes back through the sret slot" begin
        fptr = @cfunction(_mut_fill_sret, Cvoid, (Ptr{Cvoid}, Ptr{Ptr{Cvoid}}))
        # Windows goes through libJLCS's guard; elsewhere the call is direct.
        if !Sys.iswindows() || requires("Tier 2 sret through the libJLCS guard", :libJLCS)
            r = Base.invokelatest(RepliBuild.JITManager._invoke_call, fptr, M.R1, Ptr{Cvoid}[])
            @test r isa M.R1
            @test getfield(r, 1) == (0x11, 0x22, 0x33, 0x44)
        end
        # A mutable type with nothing inline to fill (a field that is a
        # reference) is refused before any call, not scribbled into.
        @test_throws ErrorException RepliBuild.JITManager._invoke_call(C_NULL, Base.RefValue{Any}, Ptr{Cvoid}[])
    end
end
