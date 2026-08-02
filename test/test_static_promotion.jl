#!/usr/bin/env julia
# Static-promotion pass (llvmcall slicing contract, M1) — fixture-gated tests.
#
# Builds test/slice_test/ (two TUs, O2) and asserts:
#   1. Promotion decisions are exact: mutable statics + static functions get
#      exported `__rb_slicetest_*` names; const statics and the public API
#      don't; the same-named static in the second TU gets the linker's
#      uniquified name recorded faithfully.
#   2. The .so's dynamic symbol table matches the map (nm -D).
#   3. Single-copy-of-state: dlsym of a promoted counter IS the live datum the
#      exported API mutates (write through the pointer, read via the API).
#   4. The promoted map lands in compilation_metadata.json.
#   5. The generated wrapper dlopens RTLD_GLOBAL, never wraps __rb_* symbols,
#      and its API still behaves.
#   6. The LUAI_FUNC class — external linkage + hidden visibility — is promoted
#      too. Linkage alone doesn't identify it, and `default<O2>` never
#      internalizes it, so before this it survived as a `define hidden`: absent
#      from the dynsym, invisible to dlsym, and a JIT deadlock for any slice
#      that declared it. Regression lock for that fix.
#
# Usage: julia --project=. test/test_static_promotion.jl

using Test
using JSON
using Libdl
using RepliBuild

const FIXTURE = joinpath(@__DIR__, "slice_test")
const TOML_PATH = joinpath(FIXTURE, "replibuild.toml")

RepliBuild.clean(TOML_PATH)
const LIB = RepliBuild.build(TOML_PATH)
const metadata = JSON.parsefile(joinpath(FIXTURE, "julia", "compilation_metadata.json"))

@testset "Static promotion (slicing M1)" begin

@test isfile(LIB)

@testset "Promotion map decisions" begin
    @test haskey(metadata, "promoted_symbols")
    promoted = metadata["promoted_symbols"]

    # Mutable statics promoted
    @test haskey(promoted, "hidden_counter")
    @test promoted["hidden_counter"] == "__rb_slicetest_hidden_counter"
    @test haskey(promoted, "current_op")
    @test haskey(promoted, "st_env")

    # Same-named static from slice_b.c: linker uniquified it, map records truth
    dup_keys = filter(k -> startswith(k, "hidden_counter"), collect(keys(promoted)))
    @test length(dup_keys) == 2

    # Address-taken static functions promoted
    for fn in ("op_double", "op_negate", "op_square")
        @test haskey(promoted, fn)
        @test startswith(promoted[fn], "__rb_slicetest_")
    end

    # LUAI_FUNC class: external linkage + hidden visibility. Not internal, so
    # the linkage-only rule missed it; not in the dynsym either, so a slice
    # declaring it deadlocks the JIT. Promotion must cover it.
    @test haskey(promoted, "st_hidden_scale")
    @test promoted["st_hidden_scale"] == "__rb_slicetest_st_hidden_scale"

    # LUAI_DDEF class: const + external + hidden. Const alone does NOT exempt a
    # global from promotion — only *internal* const does (those get embedded).
    @test haskey(promoted, "ST_HIDDEN_TABLE")
    @test promoted["ST_HIDDEN_TABLE"] == "__rb_slicetest_ST_HIDDEN_TABLE"

    # Const static table NOT promoted; public API NOT renamed
    @test !haskey(promoted, "OP_TABLE")
    for api in ("st_bump", "st_get_count", "st_apply", "st_sum", "st_guarded_div",
                "st_scaled", "st_table_at")
        @test !haskey(promoted, api)
    end
end

@testset "Dynamic symbol table matches" begin
    dynsyms = read(`nm -D --defined-only $LIB`, String)
    for (old, new) in metadata["promoted_symbols"]
        @test occursin(new, dynsyms)
    end
    # Un-promoted statics must not leak into dynsym under their bare names
    @test !occursin(r"\bOP_TABLE\b", dynsyms)
    # Promoted DATA symbols are data, not text
    @test occursin(r" [BD] __rb_slicetest_hidden_counter\b", dynsyms)

    # The de-hidden LUAI_FUNC lands in the dynsym as TEXT under its promoted
    # name only — dlsym-able (so slices bind), bare name gone (so it is not API)
    @test occursin(r" T __rb_slicetest_st_hidden_scale\b", dynsyms)
    # (`_` is a word char, so \b cannot match inside the __rb_ name)
    @test !occursin(r"\bst_hidden_scale\b", dynsyms)

    # De-hidden LUAI_DDEF table: read-only data, exported under __rb_ only
    @test occursin(r" R __rb_slicetest_ST_HIDDEN_TABLE\b", dynsyms)
    @test !occursin(r"\bST_HIDDEN_TABLE\b", dynsyms)
end

@testset "Single copy of state (dlsym ↔ API coherence)" begin
    h = Libdl.dlopen(LIB, Libdl.RTLD_NOW | Libdl.RTLD_GLOBAL)
    counter_ptr = Ptr{Clong}(Libdl.dlsym(h, "__rb_slicetest_hidden_counter"))
    @test counter_ptr != C_NULL

    # API writes, symbol reads
    @test ccall((:st_bump, LIB), Clong, (Clong,), 5) == 5
    @test unsafe_load(counter_ptr) == 5

    # Symbol writes, API reads — THE single-copy proof
    unsafe_store!(counter_ptr, 100)
    @test ccall((:st_get_count, LIB), Clong, ()) == 100

    # slice_b's same-named counter is a different datum
    @test ccall((:st_b_bump, LIB), Clong, (Clong,), 7) == 7
    @test unsafe_load(counter_ptr) == 100
    @test ccall((:st_b_get, LIB), Clong, ()) == 7

    # Promoted fn-ptr slot: st_set_op stores a promoted function's address
    ccall((:st_set_op, LIB), Cvoid, (Cint,), 2)   # op_square
    @test ccall((:st_call_op, LIB), Clong, (Clong,), 9) == 81
    slot = Ptr{Ptr{Cvoid}}(Libdl.dlsym(h, "__rb_slicetest_current_op"))
    @test unsafe_load(slot) == Libdl.dlsym(h, "__rb_slicetest_op_square")

    # De-hidden helper still callable through its public entry, and the
    # promoted symbol is the same code dlsym hands back
    @test ccall((:st_scaled, LIB), Clong, (Clong,), 7) == 22
    scale_ptr = Libdl.dlsym(h, "__rb_slicetest_st_hidden_scale")
    @test scale_ptr != C_NULL
    @test ccall(scale_ptr, Clong, (Clong,), 7) == 22

    # De-hidden const table: the exported symbol IS the datum the API reads
    @test ccall((:st_table_at, LIB), Clong, (Cint,), 2) == 5
    table_ptr = Ptr{Clong}(Libdl.dlsym(h, "__rb_slicetest_ST_HIDDEN_TABLE"))
    @test unsafe_load(table_ptr, 3) == 5

    # Behavior sanity on the survivors
    @test ccall((:st_apply, LIB), Clong, (Cint, Clong), 1, 21) == -21
    @test ccall((:st_guarded_div, LIB), Clong, (Clong, Clong), 84, 2) == 42
    @test ccall((:st_guarded_div, LIB), Clong, (Clong, Clong), 1, 0) == -1
end

@testset "Wrapper integration" begin
    wrapper = RepliBuild.wrap(TOML_PATH)
    @test isfile(wrapper)
    wrapper_text = read(wrapper, String)

    # __init__ loads the .so RTLD_GLOBAL (slice declarations resolve via ORC)
    @test occursin("RTLD_GLOBAL", wrapper_text)
    # Promoted symbols are slice surface, never API surface.
    #
    # Since 2026-08-02 they DO appear in one place: the TIER1_DECLARES table,
    # which records what each slice binds so the load-time check can verify it
    # against whatever library is actually present. That is the opposite of
    # leaking them into the API — it is what lets a wrapper detect a library
    # that lacks them. So the assertion is now about the surfaces that matter
    # rather than the raw string.
    # Scan CODE only: docstrings and comments discuss `__rb_*` by design (the
    # generated explanation of the load-time check names it), and prose cannot
    # leak a symbol into the API.
    wrapper_code = replace(wrapper_text, r"\"\"\".*?\"\"\""s => "")
    wrapper_code = join((first(split(l, '#')) for l in split(wrapper_code, '\n')), "\n")
    stray = [strip(l) for l in split(wrapper_code, '\n')
             if occursin("__rb_", l) && !occursin(r"^\s*\"[A-Za-z0-9_.]+\" => ", l)]
    @test isempty(stray)
    @test !occursin(r"^export .*__rb_"m, wrapper_text)          # never exported
    @test !occursin(r"ccall\(\(:__rb_", wrapper_text)           # never called directly
    @test !occursin(r"^function __rb_"m, wrapper_text)          # never defined as API
    # ...and it IS present as slice-binding data, which the guard depends on.
    @test occursin("TIER1_DECLARES", wrapper_text)
    @test occursin("__rb_slicetest_hidden_counter", wrapper_text)

    include(wrapper)
    M = Slicetest
    @test isdefined(M, :st_bump)
    @test isdefined(M, :st_apply)
    @test !isdefined(M, :op_double)         # statics don't become API
    @test isdefined(M, :st_scaled)
    @test !isdefined(M, :st_hidden_scale)   # nor do de-hidden LUAI_FUNCs
    @test M.st_scaled(7) == 22
    @test M.st_table_at(2) == 5

    # Wrapper API drives the same single copy dlsym sees
    before = ccall((:st_get_count, LIB), Clong, ())
    M.st_bump(3)
    @test ccall((:st_get_count, LIB), Clong, ()) == before + 3

    # After __init__, promoted symbols resolve through the GLOBAL namespace —
    # exactly the lookup path slice declarations will take at JIT time.
    p = ccall(:dlsym, Ptr{Cvoid}, (Ptr{Cvoid}, Cstring), C_NULL, "__rb_slicetest_hidden_counter")
    @test p != C_NULL
end

end  # testset
