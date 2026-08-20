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
#   6. The declared-symbol contract: `declares` names exactly what ORC must
#      resolve, including the promoted name of a hidden-visibility callee, and
#      the M3 pre-flight demotes a slice with an unresolvable declare instead of
#      letting it deadlock the JIT.
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
                 "st_guarded_div", "st_sum", "st_scaled", "st_table_at",
                 "st_sentinel", "st_is_sentinel"]
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
@eval t1_scaled(x::Clong) =
    Base.llvmcall(($(R["st_scaled"].ir), "st_scaled"), Clong, Tuple{Clong}, x)
@eval t1_table_at(i::Cint) =
    Base.llvmcall(($(R["st_table_at"].ir), "st_table_at"), Clong, Tuple{Cint}, i)

@testset "Slicer (slicing M2)" begin

@test isfile(ABI_LL)

@testset "Structural invariants" begin
    # st_sum: variadic target. st_sentinel/st_is_sentinel: reach an
    # address-significant internal constant. Both classes refuse by design —
    # asserted in their own testsets below.
    REFUSED = ("st_sum", "st_sentinel", "st_is_sentinel")
    for name in TARGETS
        r = R[name]
        if name in REFUSED
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

@testset "Address-significant internal constants are refused, not embedded" begin
    # Embedding is only sound when duplication is unobservable, which is
    # exactly what `unnamed_addr` asserts. ST_SENTINEL has its address
    # compared, so LLVM does NOT mark it — a second copy in the slice would
    # sit at a different address than the .so's and `st_is_sentinel` would
    # answer differently per tier. Same silent-divergence class as the cJSON
    # static, rotated from value identity onto address identity.
    ir = read(ABI_LL, String)
    @test occursin(r"^@ST_SENTINEL = internal constant"m, ir)          # no unnamed_addr
    @test occursin(r"^@OP_TABLE = internal unnamed_addr constant"m, ir) # the sound contrast

    for fn in ("st_sentinel", "st_is_sentinel")
        @test !Slicer.sliced(R[fn])
        @test occursin("ST_SENTINEL", R[fn].refusal)
        @test occursin("unnamed_addr", R[fn].refusal)
    end

    # The contrast still slices: OP_TABLE is address-insignificant, so
    # embedding it is sound and st_apply keeps its Tier-1 route.
    @test Slicer.sliced(R["st_apply"])
    @test occursin(r"^@OP_TABLE = internal unnamed_addr constant"m, R["st_apply"].ir)
end

@testset "Declared-symbol contract (M3 pre-flight input)" begin
    # `declares` is what the pre-flight dlsym-checks: exactly the names ORC
    # must resolve, intrinsics excluded (the backend lowers those).
    rg = R["st_get_count"]
    @test rg.declares == ["__rb_slicetest_hidden_counter"]
    @test !any(s -> startswith(s, "llvm."), R["st_apply"].declares)

    # Hidden-visibility callee: the slice binds the PROMOTED name, and the
    # bare `st_hidden_scale` — which is not in the dynsym — appears nowhere.
    rs = R["st_scaled"]
    @test Slicer.sliced(rs)
    @test "__rb_slicetest_st_hidden_scale" in rs.declares
    @test !("st_hidden_scale" in rs.declares)
    @test occursin("declare", rs.ir)
    @test !occursin(Regex("^define .*@st_hidden_scale\\b", "m"), rs.ir)

    # Same for the hidden CONST table: bound by declare (not local linkage, so
    # the Slicer will not embed it), therefore it had to be promoted.
    rt = R["st_table_at"]
    @test Slicer.sliced(rt)
    @test "__rb_slicetest_ST_HIDDEN_TABLE" in rt.declares
    @test occursin("@__rb_slicetest_ST_HIDDEN_TABLE = external", rt.ir)
    @test !occursin("[4 x i64] [i64 2", rt.ir)   # declared, never embedded

    # Every declared symbol of every produced slice resolves through the same
    # lookup ORC uses — i.e. the pre-flight passes for this fixture.
    for name in TARGETS
        Slicer.sliced(R[name]) || continue
        for sym in R[name].declares
            @test ccall(:dlsym, Ptr{Cvoid}, (Ptr{Cvoid}, Cstring), C_NULL, sym) != C_NULL
        end
    end
end

@testset "Behavior + state coherence through llvmcall" begin
    # STATE IS RELATIVE TO WHATEVER RAN BEFORE, DELIBERATELY.
    #
    # This asserted absolute values and opened with `t1_get_count() == 0`
    # ("fresh process state"), which is false in-suite: devtests runs
    # test_static_promotion.jl (§7b) first, over this same fixture in this same
    # process, and its single-copy proof deliberately writes absolute values —
    # `unsafe_store!(counter_ptr, 100)` then `M.st_bump(3)` leaves the counter
    # at 103, and `st_set_op(2)` leaves the op slot at op_square. That cost 7
    # in-suite failures against 154/154 standalone, and read as a Tier-1
    # regression when nothing was wrong with the slices.
    #
    # The subject here is COHERENCE — that a Tier-1 slice and a Tier-3 ccall
    # address ONE datum — which is a statement about deltas and cross-tier
    # agreement, never about the starting value. Asserting it that way makes
    # the testset independent of every prior file rather than of one particular
    # prior file, so a future §7b' cannot break it either. §7b's writes are not
    # cleanup it forgot: absolute stores ARE its proof mechanism, so the
    # invariant belongs here.
    #
    # The opening assertion is strictly STRONGER than the `== 0` it replaces:
    # that one only checked Tier 1 against a constant, this one checks both
    # tiers against each other before a single delta is taken.
    base = ccall((:st_get_count, LIB), Clong, ())
    @test t1_get_count() == base

    # Tier-3 write / Tier-1 read
    @test ccall((:st_bump, LIB), Clong, (Clong,), 5) == base + 5
    @test t1_get_count() == base + 5

    # Tier-1 write / Tier-3 read — the cJSON divergence class, closed
    @test t1_bump(Clong(3)) == base + 8
    @test ccall((:st_get_count, LIB), Clong, ()) == base + 8

    # Symbol-level triangulation: dlsym pointer sees both tiers' writes
    h = Libdl.dlopen(LIB)
    counter_ptr = Ptr{Clong}(Libdl.dlsym(h, "__rb_slicetest_hidden_counter"))
    @test unsafe_load(counter_ptr) == base + 8

    # Dispatch through the const table (embedded or devirtualized — behavior is
    # the contract)
    @test t1_apply(Cint(0), Clong(21)) == 42
    @test t1_apply(Cint(1), Clong(21)) == -21
    @test t1_apply(Cint(2), Clong(9)) == 81

    # Mutable fn-ptr slot: Tier-3 writes the slot, Tier-1 dispatches through it.
    # Set it EXPLICITLY rather than reading the initializer — "default" is only
    # the default in a fresh process, and §7b leaves it at op_square. Setting it
    # is also the honest shape for what this checks: every one of these is a
    # Tier-3 write followed by a Tier-1 dispatch, including the first.
    ccall((:st_set_op, LIB), Cvoid, (Cint,), 0)  # → op_double
    @test t1_call_op(Clong(21)) == 42
    ccall((:st_set_op, LIB), Cvoid, (Cint,), 2)  # → op_square
    @test t1_call_op(Clong(9)) == 81
    ccall((:st_set_op, LIB), Cvoid, (Cint,), 0)
    @test t1_call_op(Clong(9)) == 18

    # setjmp/longjmp across the JIT boundary
    @test t1_guarded_div(Clong(84), Clong(2)) == 42
    @test t1_guarded_div(Clong(1), Clong(0)) == -1

    # Call into a de-hidden LUAI_FUNC through the JIT. Un-promoted, this call
    # does not fail — it hangs (ORC waits on a symbol that never arrives), so
    # reaching this assertion at all is the regression signal.
    @test t1_scaled(Clong(7)) == 22
    @test t1_scaled(Clong(0)) == 1
    @test ccall((:st_scaled, LIB), Clong, (Clong,), 7) == 22

    # Load from a de-hidden const table through the JIT — same table the .so
    # reads, bound by symbol rather than duplicated into the slice.
    @test [t1_table_at(Cint(i)) for i in 0:3] == [2, 3, 5, 7]
    @test ccall((:st_table_at, LIB), Clong, (Cint,), 2) == 5
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
        @test R2[name].declares == R[name].declares   # pre-flight input survives
    end
end

@testset "M3 symbol pre-flight" begin
    preflight! = RepliBuild.Wrapper._tier1_preflight!

    # A slice whose declare list contains a symbol the process cannot resolve
    # must be dropped from the accepted set — not shipped to deadlock later.
    good = R["st_get_count"]
    bogus = Slicer.SliceResult("st_bogus", good.ir, Symbol[], nothing, 1, 0, 0,
                               ["__rb_slicetest_hidden_counter",
                                "__rb_slicetest_no_such_symbol"])
    results = Dict("st_get_count" => good, "st_bogus" => bogus)
    accepted = Set(["st_get_count", "st_bogus"])

    unresolved = @test_logs (:warn,) match_mode = :any preflight!(accepted, results, LIB)
    @test accepted == Set(["st_get_count"])
    @test unresolved["st_bogus"] == ["__rb_slicetest_no_such_symbol"]

    # An all-resolvable set passes untouched and logs nothing
    clean_accept = Set(["st_get_count"])
    @test isempty(@test_logs preflight!(clean_accept, results, LIB))
    @test clean_accept == Set(["st_get_count"])

    # An unloadable .so leaves every slice unverified → Tier 1 off wholesale
    dead = Set(["st_get_count"])
    @test_logs (:warn,) match_mode = :any preflight!(dead, results, "/nonexistent/libnope.so")
    @test isempty(dead)

    # The check is scoped to the library and its DT_NEEDED chain, NOT to
    # `dlsym(RTLD_DEFAULT, …)`. Resolving process-wide would verify a slice
    # against every library loaded in the wrap session — an earlier wrap, or
    # the previous `.so` after an edit — and those symbols do not exist in the
    # consumer's process, so the slice ships and deadlocks the JIT there.
    #
    # `jl_gc_collect` is a real, resolvable symbol in THIS process (libjulia)
    # that libslicetest.so does not supply. Process-wide it passes; scoped it
    # must not, and it must be reported as a stray rather than as missing.
    @test RepliBuild.Wrapper._symbol_resolves_via(C_NULL, "jl_gc_collect")
    stray = Slicer.SliceResult("st_stray", good.ir, Symbol[], nothing, 1, 0, 0,
                               ["__rb_slicetest_hidden_counter", "jl_gc_collect"])
    sr = Dict("st_stray" => stray)
    acc = Set(["st_stray"])
    u = @test_logs (:warn,) (:warn,) match_mode = :any preflight!(acc, sr, LIB)
    @test isempty(acc)
    @test u["st_stray"] == ["jl_gc_collect"]

    # And the pre-flight must not leave the library loaded: it opens
    # RTLD_LOCAL, so the symbols never enter the namespace
    # `dlsym(RTLD_DEFAULT, …)` searches, and it closes the handle when done.
    # An RTLD_GLOBAL load that is never closed is what contaminated later
    # wraps in the same session. Probe with a private COPY at a unique path —
    # this file's own top-level dlopen of LIB is RTLD_GLOBAL and would mask
    # the difference.
    lib2 = joinpath(mktempdir(), "libslicetest_preflight_probe.so")
    cp(LIB, lib2)
    @test !any(==(lib2), Libdl.dllist())
    acc2 = Set(["st_get_count"])
    @test isempty(@test_logs preflight!(acc2, Dict("st_get_count" => good), lib2))
    @test acc2 == Set(["st_get_count"])      # resolved through the handle …
    @test !any(==(lib2), Libdl.dllist())     # … and not left behind
end

end  # top-level testset

# ── Scale + fan-out, self-contained (src/slice_scale.c) ──────────────────────
#
# This was "Slicer: lua at scale" and read the RepliBuild-Hub lua build by
# absolute path. A core-engine test must not depend on which library version
# happens to be built in another repo: lua 5.5.1 made luaL_openlibs a macro, the
# plain symbol stopped existing, the slice was refused "function not found in
# module", and this file went red for a reason with nothing to do with slicing.
# Hub rebuilds are the integration test. This is the mechanic.
#
# What it keeps from the lua version — and states more sharply, because the
# fixture is ours and the numbers are therefore knowable:
#   * breadth: many targets sliced out of one module in one call
#   * DECLARATIONS-ONLY at fan-out: st_sc_hub reaches all 73 functions of the TU
#     through two layers, so reachability-following would emit the whole module;
#     one `define` plus `declare`s is the contract, and the size RATIO measures it
#   * live llvmcall against the same .so Tier 3 calls
# What it adds: an oracle independent of the other tier (the closed form), and a
# check that the call graph it claims to test actually survived -O2.

@testset "Slicer: scale + fan-out" begin
    leaves = ["st_sc_leaf_$(lpad(i, 2, '0'))" for i in 0:63]
    mids   = ["st_sc_mid_$k" for k in 0:7]
    scale_targets = vcat(["st_sc_hub"], mids, leaves)

    SR = Slicer.slice_library(ABI_LL; targets=scale_targets)
    for t in scale_targets
        @test Slicer.sliced(SR[t])
        @test length(collect(eachmatch(r"^define "m, SR[t].ir))) == 1
    end

    # THE FAN-OUT MUST BE REAL, OR EVERYTHING BELOW IS VACUOUS. [link]
    # optimization_level is "2"; without `noinline` in slice_scale.c the leaves
    # fold into the mids and the mids into the hub, and a slice of an inlined-flat
    # hub would trivially satisfy "one define" while testing nothing. Assert the
    # edges survived, in the module and in the slice.
    hub_ir = SR["st_sc_hub"].ir
    @test all(m -> occursin(m, hub_ir), mids)          # hub still CALLS the mids
    for m in mids
        @test occursin(Regex("^declare[^\n]*@$m\\(", "m"), hub_ir)   # …as declares
    end
    @test !occursin(r"^define[^\n]*@st_sc_mid_"m, hub_ir)        # …never bodies
    @test !occursin(r"^define[^\n]*@st_sc_leaf_"m, hub_ir)       # nor two layers down

    # The declarations-only property, quantified. The hub's transitive closure is
    # the whole TU, so a reachability slicer's output would be within a constant
    # factor of the module; declarations-only keeps it kilobytes.
    module_bytes = filesize(ABI_LL)
    @test sizeof(hub_ir) * 20 < module_bytes
    @test sizeof(hub_ir) < 60_000

    # Live behaviour through llvmcall on the slice, against the RTLD_GLOBAL .so.
    @eval sc1_hub(v::Clong) =
        Base.llvmcall(($(SR["st_sc_hub"].ir), "st_sc_hub"), Clong, Tuple{Clong}, v)
    @eval sc1_mid0(v::Clong) =
        Base.llvmcall(($(SR["st_sc_mid_0"].ir), "st_sc_mid_0"), Clong, Tuple{Clong}, v)

    # Closed form (slice_scale.c): leaf_i(v) = 3v + 100 + i, hub(v) = 192v + 8416.
    # Tier-1 vs Tier-3 agreement alone would only prove COHERENCE — both tiers can
    # be wrong together, which is the whole reason the eightbyte-coercion bugs went
    # unnoticed for so long. The algebra is the independent oracle.
    for v in (Clong(0), Clong(1), Clong(-7), Clong(1000))
        @test sc1_hub(v) == 192v + 8416
        @test sc1_hub(v) == ccall((:st_sc_hub, LIB), Clong, (Clong,), v)
    end
    @test sc1_mid0(Clong(2)) == sum(3 * 2 + 100 + i for i in 0:7)
    @test sc1_mid0(Clong(2)) == ccall((:st_sc_mid_0, LIB), Clong, (Clong,), 2)
end
