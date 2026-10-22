#!/usr/bin/env julia
# test/test_aot_thunks.jl — the AOT thunk path, end to end, through the real pipeline.
#
# `[compile] aot_thunks = true` compiles the Tier-2 thunks into
# lib<name>_thunks.<dlext> beside the library, as the LAST STAGE OF WRAP
# (`ThunkBuilder.build_aot_thunks`, handed the thunk set the generator has just
# decided by `Wrapper.wrap_library`). The wrapper then dlopens that library and
# resolves every thunk address in `__init__` (`_THUNK_SLOTS`) instead of
# building an MLIR JIT. Before this file the path had no end-to-end coverage:
# stress_test/verify.jl hand-rolls its own link, and test_introspection.jl only
# checks that a failed AOT pass leaves no stale library behind.
#
# Fixture: test/aot_test/ (C++, no STL; see the header of src/aot_api.cpp).
# discover → build → wrap, in two passes:
#   1. cold — build compiles no thunks, and the first wrap compiles exactly the
#      set it derived. AOT used to run at build, before any wrap had derived
#      that set, and guessed it: every function in the metadata. Here the guess
#      compiled, with thunks for the Tier-3 extern "C" factories no wrapper
#      binds. A package whose guess did not compile or link got a refused or
#      JIT first wrap and needed a second build and wrap;
#   2. again — a second build and wrap derive and compile the same set.
# After each pass a child process loads the wrapper and calls through it
# (aot_test/probe_aot.jl). A C++ throw that escapes a thunk ends the process,
# and that has to take the child, not the suite: on Windows a void-returning
# thunk that threw did exactly that until `_invoke_void` routed the AOT forms
# through libJLCS's guard (probe `throw_void`).
#
# Nothing here loads the fixture's libraries in THIS process. A DLL this process
# holds open cannot be replaced, so pass 2's relink would fail on Windows. The
# exports are read from the file instead.
#
# Pass 1 is the regression test for the stage's move: with AOT back in build,
# the cold thunks library holds the guessed thunks, and pass 1 fails.
#
# devtests includes every file into one `Main`, hence the `aot_`/`AOT_` names.

using Test
using Libdl
using TOML
using RepliBuild

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

if requires("AOT thunks end to end", :libJLCS, :clangxx)

const AOT_DIR   = joinpath(@__DIR__, "aot_test")
const AOT_PROBE = joinpath(AOT_DIR, "probe_aot.jl")

# Every function the probe calls through a thunk, by mangled name (Itanium on
# both hosts). Each must be in the manifest, bound to an AOT slot in the
# wrapper, and exported by the thunks library.
const AOT_EXPECTED_THUNKS = [
    "_ZN11Accumulator3addEi",            # plain members
    "_ZNK11Accumulator5totalEv",
    "_ZNK11Accumulator7as_vec2Ev",       # member returning a record by value
    "_Z9pair_swap6Pair32",               # records by value: 8, 16, 24 bytes
    "_Z10vec2_scale4Vec2d",
    "_Z8vec2_dot4Vec2S_",
    "_Z10vec3_cross4Vec3S_",
    "_ZNK5Shape4areaEv",                 # virtual
    "_ZNK5Shape5sidesEv",
    "_ZNK6Square4areaEv",
    "_ZNK5Shape10area_timesEd",
    "_ZN11Accumulator11checked_addEi",   # throwing, one return shape each
    "_ZN11Accumulator9set_totalEi",
    "_Z10safe_ratiodd",
    "_Z11make_squared",
    "_Z9vec2_unit4Vec2",
    "_Z9throw_inti",
]

# Every case probe_aot.jl prints. `check_aot_probe` also requires the reverse —
# no printed case left unasserted — so a case added to the script cannot be
# silently ignored here.
const AOT_PROBES = [
    "aot_loaded", "dispatch_tier", "member_fn", "member_byvalue_return",
    "byvalue_pair32", "byvalue_vec2", "byvalue_vec3", "virtual_override", "no_jit",
    "throw_i32", "throw_f64", "throw_ptr", "throw_sret", "throw_non_std", "throw_void",
    "clean_after_throws",
]

aot_ciface(mangled) = "_mlir_ciface_$(mangled)_thunk"

function aot_clean_fixture()
    for name in ("build", "julia", ".replibuild_cache", ".debug")
        p = joinpath(AOT_DIR, name)
        ispath(p) && RepliBuild._rm_tree(p)
    end
end

# discover regenerates the toml from scratch, and `aot_thunks` is not a key it
# preserves, so it is set after every discover — the same way devtests seeds
# its curated fixture config.
function enable_aot!(toml)
    doc = TOML.parsefile(toml)
    get!(doc, "compile", Dict{String,Any}())["aot_thunks"] = true
    open(io -> TOML.print(io, doc), toml, "w")
end

aot_thunks_path(lib) = joinpath(dirname(lib),
    replace(basename(lib), "." * Libdl.dlext => "_thunks." * Libdl.dlext))

# The thunks library's exported names, read from the file: the PE export
# directory on Windows, the dynamic symbol table elsewhere.
function aot_exported_names(path)
    names = RepliBuild.Compiler._pe_exported_names(path)
    names === nothing || return names
    return RepliBuild.Compiler._nm_dynamic_names(read(`nm -D --defined-only $path`, String))
end

aot_manifest_set(lib) = Set{String}(RepliBuild._read_json(
    joinpath(dirname(lib), "thunk_manifest.json"))["function_thunks"])

# What wrap left on disk, and what the wrapper binds. Returns the mangled names
# behind the wrapper's thunk slots.
function check_aot_artifacts(lib, wrapper)
    thunks = aot_thunks_path(lib)
    @test isfile(thunks)
    exports = isfile(thunks) ? aot_exported_names(thunks) : Set{String}()

    manifest = joinpath(dirname(lib), "thunk_manifest.json")
    @test isfile(manifest)
    listed = isfile(manifest) ? aot_manifest_set(lib) : Set{String}()

    txt = read(wrapper, String)

    # The wrapper is bound to the thunks library, not to the JIT. A JIT wrapper
    # bakes an empty THUNKS_LIBRARY_PATH and calls
    # `JITManager.invoke("_mlir_ciface_…")`; its virtuals add `get_jit_thunk`.
    m = match(r"const THUNKS_LIBRARY_PATH = let baked = (\"[^\"\n]*\")", txt)
    @test m !== nothing
    baked = m === nothing ? "" : Meta.parse(m.captures[1])
    @test !isempty(baked) && isfile(baked) && isfile(thunks) && realpath(baked) == realpath(thunks)
    @test count("JITManager.invoke_aot_ptr(", txt) >= length(AOT_EXPECTED_THUNKS)
    @test !occursin("JITManager.invoke(", txt)
    @test !occursin("initialize_global_jit", txt)
    @test !occursin("get_jit_thunk", txt)

    slots = [String(s.captures[1]) for s in
             eachmatch(r"\(_FPTR_\w+, \"(_mlir_ciface_\w+_thunk)\"\)", txt)]
    @test !isempty(slots)
    @test allunique(slots)
    for mangled in AOT_EXPECTED_THUNKS
        @test aot_ciface(mangled) in slots     # bound by the wrapper
        @test aot_ciface(mangled) in exports   # defined by the thunks library
        @test mangled in listed                # recorded in the manifest
    end
    # Everything the wrapper binds, the library defines (wrap's own
    # `_assert_aot_thunks_present` refuses otherwise; this is the independent
    # check) and the manifest records. Manifest ⊇ slots, not equality: the
    # manifest is wrap's intent, and the dedup folds a deleting destructor (D0)
    # into the same Julia method as its D2, so D0 has no slot.
    @test isempty(setdiff(slots, exports))
    slot_mangled = Set{String}(chopsuffix(chopprefix(s, "_mlir_ciface_"), "_thunk") for s in slots)
    @test issubset(slot_mangled, listed)
    return slot_mangled
end

# Load the wrapper in a child and call through it.
function check_aot_probe(wrapper)
    r = run_isolated(`$AOT_PROBE $wrapper`; timeout = 600)
    v = probe_verdicts(r.output)
    @test r.ok
    @test haskey(v, "PROBE_DONE")
    for name in AOT_PROBES
        @test startswith(get(v, name, "no PROBE line"), "PASS")
    end
    @test sort!(collect(setdiff(keys(v), ("PROBE_DONE",)))) == sort(AOT_PROBES)
end

@testset "AOT thunks end to end" begin
    # discover registers the project in the RepliBuild registry; keep that out
    # of the real ~/.replibuild.
    mktempdir() do home
    withenv("REPLIBUILD_HOME" => home) do
        aot_clean_fixture()
        toml = RepliBuild.discover(AOT_DIR, force = true)
        enable_aot!(toml)
        @test RepliBuild.ConfigurationManager.load_config(toml).compile.aot_thunks

        lib = ""
        cold_slots = Set{String}()

        @testset "cold: build compiles no thunks, the first wrap compiles its own set" begin
            lib = RepliBuild.build(toml)
            @test isfile(lib)
            @test !isfile(aot_thunks_path(lib))      # no AOT pass in build
            @test !isfile(joinpath(dirname(lib), "thunk_manifest.json"))   # no wrap yet
            wrapper = RepliBuild.wrap(toml)
            cold_slots = check_aot_artifacts(lib, wrapper)

            # Exactly the set this wrap derived, on its first run. The build-time
            # pass guessed instead, and its guess also emitted thunks for the
            # Tier-3 extern "C" factories, which no wrapper binds.
            built = Set(n for n in aot_exported_names(aot_thunks_path(lib))
                        if startswith(n, "_mlir_ciface_"))
            @test built == Set(aot_ciface(m) for m in aot_manifest_set(lib))
            @test !any(n -> occursin("aot_new_", n), built)
            check_aot_probe(wrapper)
        end

        @testset "a second build and wrap compile the same set" begin
            thunks = aot_thunks_path(lib)
            before = aot_manifest_set(lib)
            @test RepliBuild.build(toml) == lib
            # Removed here so that its presence below is this wrap's doing.
            # `_rm_tree`: on Windows a scanner may still hold the fresh DLL.
            RepliBuild._rm_tree(thunks)
            wrapper = RepliBuild.wrap(toml)
            @test isfile(thunks)
            built = Set(n for n in aot_exported_names(thunks) if startswith(n, "_mlir_ciface_"))
            @test built == Set(aot_ciface(m) for m in before)
            @test aot_manifest_set(lib) == before   # wrap derives the same set again
            steady_slots = check_aot_artifacts(lib, wrapper)
            @test steady_slots == cold_slots
            check_aot_probe(wrapper)
        end
    end
    end
end

end # requires
