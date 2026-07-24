#!/usr/bin/env julia
# Slicer (llvmcall slicing M2) — fixture-gated tests.
#
# Slices functions out of the promoted slice_test module and asserts:
#   1. Structural invariant: every produced slice holds exactly ONE definition
#      (the target); everything reached is declared, internal constants may be
#      embedded, and the module verifies.
#   2. Boundary policy: mutable statics come through as declarations of their
#      promoted __rb_* names — never as embedded definitions.
#   3. Behavior through Base.llvmcall against the RTLD_GLOBAL-loaded .so,
#      including BOTH coherence directions of the cJSON divergence class:
#      Tier-3 writes / Tier-1 reads AND Tier-1 writes / Tier-3 reads.
#   4. Hazard reporting (setjmp family) and refusals (varargs target).
#   5. The slice cache round-trips.
#
# Usage: julia --project=. test/test_slicer.jl

using Test
using Libdl
using RepliBuild

const Slicer = RepliBuild.Slicer

const FIXTURE = joinpath(@__DIR__, "slice_test")
const TOML_PATH = joinpath(FIXTURE, "replibuild.toml")

RepliBuild.clean(TOML_PATH)
const LIB = RepliBuild.build(TOML_PATH)
const ABI_LL = joinpath(FIXTURE, "build", "slicetest_abi.ll")

# Slice declarations resolve against the .so via ORC process-symbol lookup.
Libdl.dlopen(LIB, Libdl.RTLD_NOW | Libdl.RTLD_GLOBAL)

const TARGETS = ["st_get_count", "st_bump", "st_apply", "st_call_op",
                 "st_guarded_div", "st_sum"]
const CACHE_DIR = joinpath(FIXTURE, ".replibuild_cache")

const R = Slicer.slice_library(ABI_LL; targets=TARGETS, cache_dir=CACHE_DIR)

# llvmcall wrappers (module-IR form; entry = target symbol)
@eval t1_get_count()      = Base.llvmcall(($(R["st_get_count"].ir), "st_get_count"), Clong, Tuple{})
@eval t1_bump(d::Clong)   = Base.llvmcall(($(R["st_bump"].ir), "st_bump"), Clong, Tuple{Clong}, d)
@eval t1_apply(op::Cint, x::Clong) =
    Base.llvmcall(($(R["st_apply"].ir), "st_apply"), Clong, Tuple{Cint,Clong}, op, x)
@eval t1_call_op(x::Clong) =
    Base.llvmcall(($(R["st_call_op"].ir), "st_call_op"), Clong, Tuple{Clong}, x)
@eval t1_guarded_div(a::Clong, b::Clong) =
    Base.llvmcall(($(R["st_guarded_div"].ir), "st_guarded_div"), Clong, Tuple{Clong,Clong}, a, b)

@testset "Slicer (slicing M2)" begin

@test isfile(ABI_LL)

@testset "Structural invariants" begin
    for name in TARGETS
        r = R[name]
        if name == "st_sum"
            @test !Slicer.sliced(r)
        else
            @test Slicer.sliced(r)
            # exactly one definition — the target
            @test length(collect(eachmatch(r"^define "m, r.ir))) == 1
            @test occursin("define", split(r.ir, "@$(name)(")[1])  # target is the define
            # a slice is small — kilobytes, not the library
            @test length(r.ir) < 100_000
        end
    end
end

@testset "Boundary policy" begin
    # Mutable static → declaration of the promoted name, no initializer
    r = R["st_get_count"]
    @test occursin("@__rb_slicetest_hidden_counter = external", r.ir)
    @test !occursin(r"global i64 \d", r.ir)  # no initializer anywhere
    @test r.n_declared_globals == 1
    @test r.n_declared_fns == 0
    @test r.n_embedded_constants == 0

    # Mutable fn-ptr slot declared in st_call_op's slice
    @test occursin("@__rb_slicetest_current_op = external", R["st_call_op"].ir)

    # st_apply: dispatch targets appear only as declarations (the promoted
    # names), never as definitions
    ra = R["st_apply"]
    for fn in ("op_double", "op_negate", "op_square")
        if occursin("__rb_slicetest_$fn", ra.ir)
            @test !occursin(Regex("^define .*@__rb_slicetest_$fn\\b", "m"), ra.ir)
        end
    end

    # Varargs target refused with the hazard recorded
    @test R["st_sum"].refusal !== nothing
    @test :target_varargs in R["st_sum"].hazards

    # setjmp family is a hazard, not a refusal
    @test Slicer.sliced(R["st_guarded_div"])
    @test :setjmp_family in R["st_guarded_div"].hazards
end

@testset "Behavior + state coherence through llvmcall" begin
    # Fresh process state: counter starts at 0
    @test t1_get_count() == 0

    # Tier-3 write / Tier-1 read
    @test ccall((:st_bump, LIB), Clong, (Clong,), 5) == 5
    @test t1_get_count() == 5

    # Tier-1 write / Tier-3 read — the cJSON divergence class, closed
    @test t1_bump(Clong(3)) == 8
    @test ccall((:st_get_count, LIB), Clong, ()) == 8

    # Symbol-level triangulation: dlsym pointer sees both tiers' writes
    h = Libdl.dlopen(LIB)
    counter_ptr = Ptr{Clong}(Libdl.dlsym(h, "__rb_slicetest_hidden_counter"))
    @test unsafe_load(counter_ptr) == 8

    # Dispatch through the const table (embedded or devirtualized — behavior is
    # the contract)
    @test t1_apply(Cint(0), Clong(21)) == 42
    @test t1_apply(Cint(1), Clong(21)) == -21
    @test t1_apply(Cint(2), Clong(9)) == 81

    # Mutable fn-ptr slot: Tier-3 writes the slot, Tier-1 dispatches through it
    @test t1_call_op(Clong(21)) == 42            # default: op_double
    ccall((:st_set_op, LIB), Cvoid, (Cint,), 2)  # → op_square
    @test t1_call_op(Clong(9)) == 81
    ccall((:st_set_op, LIB), Cvoid, (Cint,), 0)
    @test t1_call_op(Clong(9)) == 18

    # setjmp/longjmp across the JIT boundary
    @test t1_guarded_div(Clong(84), Clong(2)) == 42
    @test t1_guarded_div(Clong(1), Clong(0)) == -1
end

@testset "Slice cache round-trip" begin
    # Files exist for produced slices; refusals cache metadata only
    key_dirs = readdir(joinpath(CACHE_DIR, "slices"), join=true)
    @test length(key_dirs) == 1
    cached = readdir(only(key_dirs))
    @test "st_get_count.ll" in cached
    @test "st_sum.json" in cached
    @test !("st_sum.ll" in cached)

    # Second call serves identical results from cache
    R2 = Slicer.slice_library(ABI_LL; targets=TARGETS, cache_dir=CACHE_DIR)
    for name in TARGETS
        @test Slicer.sliced(R2[name]) == Slicer.sliced(R[name])
        Slicer.sliced(R[name]) && @test R2[name].ir == R[name].ir
        @test R2[name].refusal == R[name].refusal
    end
end

end  # top-level testset

# ── Real-library scale (uses the Hub lua build if present) ───────────────────

const LUA_ABI = "/home/john/Desktop/Projects/RepliBuild-Hub/packages/lua/build/lua_abi.ll"
const LUA_SO  = "/home/john/Desktop/Projects/RepliBuild-Hub/packages/lua/julia/liblua.so"

if isfile(LUA_ABI) && isfile(LUA_SO)
    @testset "Slicer: lua at scale" begin
        lua_targets = ["lua_gettop", "lua_settop", "lua_absindex", "lua_pushinteger",
                       "lua_tointegerx", "lua_type", "lua_pcallk", "luaL_openlibs",
                       "lua_close"]
        LR = Slicer.slice_library(LUA_ABI; targets=lua_targets)
        for t in lua_targets
            @test Slicer.sliced(LR[t])
            @test length(collect(eachmatch(r"^define "m, LR[t].ir))) == 1
        end

        Libdl.dlopen(LUA_SO, Libdl.RTLD_NOW | Libdl.RTLD_GLOBAL)
        @eval lt1_gettop(L::Ptr{Cvoid}) =
            Base.llvmcall(($(LR["lua_gettop"].ir), "lua_gettop"), Cint, Tuple{Ptr{Cvoid}}, L)
        @eval lt1_absindex(L::Ptr{Cvoid}, i::Cint) =
            Base.llvmcall(($(LR["lua_absindex"].ir), "lua_absindex"), Cint, Tuple{Ptr{Cvoid},Cint}, L, i)

        L = ccall((:luaL_newstate, LUA_SO), Ptr{Cvoid}, ())
        for i in 1:4
            ccall((:lua_pushinteger, LUA_SO), Cvoid, (Ptr{Cvoid}, Clonglong), L, i)
        end
        @test lt1_gettop(L) == 4
        @test lt1_absindex(L, Cint(-1)) == 4
        @test lt1_absindex(L, Cint(2)) == 2
        ccall((:lua_settop, LUA_SO), Cvoid, (Ptr{Cvoid}, Cint), L, 0)
        @test lt1_gettop(L) == 0
        ccall((:lua_close, LUA_SO), Cvoid, (Ptr{Cvoid},), L)
    end
else
    @info "lua at scale: skipped (Hub lua build artifacts not found)"
end
