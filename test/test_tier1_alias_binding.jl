# =============================================================================
# Tier 1 binds a library's own symbols by their `__rb_` alias, never by name
#
# A slice is declarations-only, and ORC resolves a `declare` process-wide: it
# takes the first definition of that name in the process, and the pre-flight
# only asks whether the library *has* the symbol. So a slice of a function that
# called another PUBLIC function of its own library bound whichever library
# defined that name first. Julia loads its own `libz.so.1` at startup, and the
# Hub's zlib (1.3.1, `[wrap.tier1] enable = true`) ran `gzclose` as a slice
# whose `gzclose_w` was Julia's zlib 1.3.2, freeing a handle the other zlib
# allocated: `double free or corruption (!prev)` (2026-10-24, Linux).
#
# The fix has two halves, and this file checks each and then both together:
#   * `Compiler._alias_exports!` (run by `_promote_statics_libllvm` when Tier 1
#     is on) gives every exported definition the exported LLVM alias
#     `__rb_<lib>_<name>`, recorded in `exported_aliases`;
#   * `Slicer._claim_rb_aliases!` reads them back, and the closure walk makes
#     every reached exported definition declare its alias, or refuses.
# Live: `tier1_alias_test` names its public functions after zlib exports, calls
# them from one another, and returns sentinels; a decoy library with the same
# names is loaded RTLD_GLOBAL before the wrapper. Each Tier 1 call must return
# the fixture's sentinel. Needs the C bucket (a build); the live half runs in a
# child, because the failure mode with real zlib is heap corruption.
# =============================================================================

using Test
using JSON
using Libdl
using RepliBuild

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

const TAB_DIR = joinpath(@__DIR__, "tier1_alias_test")
const TAB_C = RepliBuild.Compiler
const TAB_SLICER = RepliBuild.Slicer

# One of everything the alias pass must classify. `api` reaches every kind of
# symbol a slice can declare: public function, public mutable and constant
# globals, a hidden and an internal function (promoted), a plain declaration.
const TAB_PROBE_IR = """
@pub_state = global i64 0
@pub_tab = constant [2 x i64] [i64 5, i64 7]
@tls_state = thread_local global i64 0
@mut = internal global i64 0

define i64 @helper(i64 %x) {
  %r = add i64 %x, 1
  ret i64 %r
}
define hidden i64 @hidden_fn() { ret i64 2 }
define internal i64 @static_fn() { ret i64 1 }
define weak i64 @weak_fn() { ret i64 3 }
define dllexport i64 @exported_fn() { ret i64 4 }
declare i64 @free_like(ptr)

define i64 @api(i64 %i) {
  %a = load i64, ptr @pub_state
  %p = getelementptr [2 x i64], ptr @pub_tab, i64 0, i64 %i
  %b = load i64, ptr %p
  %c = call i64 @helper(i64 %a)
  %d = call i64 @hidden_fn()
  %e = call i64 @static_fn()
  %f = call i64 @free_like(ptr @mut)
  %s1 = add i64 %a, %b
  %s2 = add i64 %s1, %c
  %s3 = add i64 %s2, %d
  %s4 = add i64 %s3, %e
  %s5 = add i64 %s4, %f
  ret i64 %s5
}
define i64 @api_tls() {
  %v = load i64, ptr @tls_state
  ret i64 %v
}
define i64 @api_weak() {
  %v = call i64 @weak_fn()
  ret i64 %v
}
define i64 @reads_mut() {
  %v = load i64, ptr @mut
  ret i64 %v
}
"""

@testset "Exported aliases: one per exported definition, and only with Tier 1" begin
    dir = mktempdir()
    in_ll = joinpath(dir, "probe.ll")
    write(in_ll, TAB_PROBE_IR)

    aliased = joinpath(dir, "probe_abi.ll")
    aliases = Dict{String,String}()
    promoted = TAB_C._promote_statics_libllvm(in_ll, aliased, "probe"; aliases)
    @test promoted !== nothing
    # The rename map is what it was: internals and hidden symbols, nothing public.
    @test Set(keys(something(promoted, Dict()))) == Set(["mut", "hidden_fn", "static_fn"])
    # Exactly the exported definitions get an alias. Not: the thread-local global
    # (the C API cannot alias one), the weak definition (hazard-demoted anyway),
    # the declaration, nor anything promoted or internal.
    @test aliases == Dict("api" => "__rb_probe_api", "api_tls" => "__rb_probe_api_tls",
                          "api_weak" => "__rb_probe_api_weak", "reads_mut" => "__rb_probe_reads_mut",
                          "helper" => "__rb_probe_helper", "exported_fn" => "__rb_probe_exported_fn",
                          "pub_state" => "__rb_probe_pub_state", "pub_tab" => "__rb_probe_pub_tab")
    out = read(aliased, String)
    @test occursin(r"^@__rb_probe_helper = alias i64 \(i64\), ptr @helper$"m, out)
    @test occursin(r"^@__rb_probe_pub_state = alias i64, ptr @pub_state$"m, out)
    @test occursin(r"^@__rb_probe_pub_tab = alias \[2 x i64\], ptr @pub_tab$"m, out)
    # A dllexport definition gets a dllexport alias, so on PE the alias reaches
    # the export directory whenever its aliasee does.
    @test occursin(r"^@__rb_probe_exported_fn = dllexport alias"m, out)
    @test !occursin("__rb_probe_tls_state", out)
    @test !occursin("__rb_probe_weak_fn", out)
    @test count(" alias ", out) == length(aliases)

    # Without the request (Tier 1 off) the module is exactly what promotion
    # alone makes of it: no alias anywhere.
    plain = joinpath(dir, "probe_plain_abi.ll")
    promoted_plain = TAB_C._promote_statics_libllvm(in_ll, plain, "probe")
    @test promoted_plain == promoted
    @test !occursin(" alias ", read(plain, String))
end

@testset "Slicer: a reached exported definition is declared by its alias" begin
    dir = mktempdir()
    in_ll = joinpath(dir, "probe.ll")
    write(in_ll, TAB_PROBE_IR)
    aliased = joinpath(dir, "probe_abi.ll")
    TAB_C._promote_statics_libllvm(in_ll, aliased, "probe"; aliases = Dict{String,String}())
    R = TAB_SLICER.slice_library(aliased; targets = ["api", "api_tls", "api_weak"])

    r = R["api"]
    @test TAB_SLICER.sliced(r)
    @test !TAB_SLICER.errored(r)
    # The library's own symbols by alias; the promoted names as they were; a
    # symbol the module only declares by its name, since the library does not
    # define it either.
    @test r.declares == ["__rb_probe_helper", "__rb_probe_hidden_fn", "__rb_probe_mut",
                         "__rb_probe_pub_state", "__rb_probe_pub_tab", "__rb_probe_static_fn",
                         "free_like"]
    @test occursin(r"^declare i64 @__rb_probe_helper\(i64\)$"m, r.ir)
    @test occursin(r"^@__rb_probe_pub_state = external global i64$"m, r.ir)
    @test occursin(r"^@__rb_probe_pub_tab = external constant \[2 x i64\]$"m, r.ir)
    @test occursin("call i64 @__rb_probe_helper(i64 %a)", r.ir)
    @test !occursin(r"@helper\b", r.ir)
    @test !occursin(r"@pub_state\b", r.ir)
    @test !occursin(r"@pub_tab\b", r.ir)
    # No alias survives into a slice, and the target is still the one definition.
    @test !occursin(" alias ", r.ir)
    @test count(r"^define "m, r.ir) == 1
    @test r.n_declared_fns == 4
    @test r.n_declared_globals == 3

    # A reached exported definition with no alias refuses the slice, as a verdict
    # (cached, not a Slicer error), and says what to do.
    t = R["api_tls"]
    @test !TAB_SLICER.sliced(t)
    @test !TAB_SLICER.errored(t)
    @test occursin("exported definition 'tls_state'", something(t.refusal, ""))
    @test occursin("__rb_", something(t.refusal, ""))
    @test occursin("Rebuild", something(t.refusal, ""))

    # A weak definition is not exported under its own name in the sense that
    # matters (another definition may replace it), carries the `:weak` hazard
    # that demotes it, and needs no alias.
    w = R["api_weak"]
    @test TAB_SLICER.sliced(w)
    @test :weak in w.hazards
    @test w.declares == ["weak_fn"]
end

@testset "Slicer: a module built before the alias pass refuses, never errors" begin
    dir = mktempdir()
    in_ll = joinpath(dir, "probe.ll")
    write(in_ll, TAB_PROBE_IR)
    plain = joinpath(dir, "probe_plain_abi.ll")
    TAB_C._promote_statics_libllvm(in_ll, plain, "probe")
    r = TAB_SLICER.slice_function(plain, "api")
    @test !TAB_SLICER.sliced(r)
    @test !TAB_SLICER.errored(r)
    @test occursin("has no `__rb_` alias", something(r.refusal, ""))
    @test occursin("project_hash", something(r.refusal, ""))
    # The same walk still refuses what it always refused: an internal mutable
    # global in an unpromoted module (reached by a target that reaches nothing
    # public, since the walk reports the first refusal it meets).
    raw = TAB_SLICER.slice_function(in_ll, "reads_mut")
    @test !TAB_SLICER.sliced(raw)
    @test occursin("not static-promoted", something(raw.refusal, ""))
end

# ── Live: the fixture, a decoy with its names, and the Tier 1 calls ──────────

"Run `RepliBuild.wrap` with its stdout captured; returns `(wrapper, printed)`."
function tab_wrap_logged(toml_path)
    out_path, out_io = mktemp()
    w = try
        redirect_stdout(out_io) do
            RepliBuild.wrap(toml_path)
        end
    finally
        close(out_io)
    end
    printed = read(out_path, String)
    rm(out_path; force = true)
    return w, printed
end

@testset "Tier 1 alias binding, live" begin
    toml_path = joinpath(TAB_DIR, "replibuild.toml")
    src = joinpath(TAB_DIR, "src", "zcollide.c")
    inc = joinpath(TAB_DIR, "include")

    # Regenerated each run: it holds absolute, machine-local paths.
    write(toml_path, """
    [project]
    name = "zcollide"
    root = "$(escape_string(TAB_DIR))"

    [compile]
    flags = ["-O2", "-fPIC"]
    source_files = ["$(escape_string(src))"]
    include_dirs = ["$(escape_string(inc))"]

    [link]
    # Slicing, not the whole-module embed.
    enable_lto = false
    optimization_level = "2"

    [binary]
    type = "shared"

    [wrap]
    language = "c"

    [wrap.tier1]
    enable = true

    [types]
    strictness = "warn"
    allow_unknown_structs = true
    allow_function_pointers = true

    [cache]
    enabled = true
    """)
    RepliBuild.clean(toml_path)

    lib = RepliBuild.build(toml_path)
    @test isfile(lib)
    wrapper, printed = tab_wrap_logged(toml_path)
    @test wrapper == joinpath(TAB_DIR, "julia", "Zcollide.jl")
    @test isfile(wrapper)

    @testset "every function slices and pre-flights" begin
        # The pre-pass summary. Any demotion would show as a parenthetical; an
        # errored or unresolved count is a bug in the pass or the alias export.
        m = match(r"tier1: (\d+) slices accepted(?: \((.*?)\))?"m, printed)
        @test m !== nothing
        if m !== nothing
            @test parse(Int, m.captures[1]) == 9
            @test m.captures[2] === nothing
        end
        @test occursin(r"tier1: 9 slices emitted \(9 functions\)", printed)
    end

    metadata = JSON.parsefile(joinpath(TAB_DIR, "julia", "compilation_metadata.json"))
    aliases = get(metadata, "exported_aliases", Dict())
    public = ["zc_set_seed", "gzclose_w", "gzclose", "gzclose_r", "adler32_z",
              "adler32", "z_verbose", "zc_verbosity", "zc_set_verbosity", "z_errmsg", "zc_table_at"]

    @testset "the metadata and the dynamic symbol table carry the aliases" begin
        @test haskey(metadata, "exported_aliases")
        for name in public
            @test get(aliases, name, nothing) == "__rb_zcollide_$name"
        end
        # The rename map stays what it was: the hidden state, nothing public.
        @test get(get(metadata, "promoted_symbols", Dict()), "zc_seed", nothing) == "__rb_zcollide_zc_seed"
        @test !any(haskey(get(metadata, "promoted_symbols", Dict()), n) for n in public)
        @test !haskey(aliases, "zc_seed")
        # What the loader binds: on ELF the dynamic symbol table, on PE the
        # export directory. Both the public names and their aliases are there.
        runtime = TAB_C._runtime_symbol_names(lib)
        @test runtime !== nothing
        if runtime !== nothing
            for name in public
                @test name in runtime
                @test "__rb_zcollide_$name" in runtime
            end
        end
        # The API is still the public names only: no alias is wrapped.
        text = read(wrapper, String)
        @test !occursin(r"^function __rb_"m, text)
        @test !occursin(r"ccall\(\(:__rb_", text)
    end

    @testset "the shipped slices declare the aliases" begin
        text = read(wrapper, String)
        @test occursin("\"gzclose\" => [\"__rb_zcollide_gzclose_w\"]", text)
        @test occursin("\"adler32\" => [\"__rb_zcollide_adler32_z\"]", text)
        @test occursin("\"zc_verbosity\" => [\"__rb_zcollide_z_verbose\"]", text)
        @test occursin("\"zc_table_at\" => [\"__rb_zcollide_z_errmsg\"]", text)
        slices = joinpath(TAB_DIR, "julia", "slices")
        checked = 0
        for (fn, callee) in (("gzclose", "gzclose_w"), ("adler32", "adler32_z"))
            path = joinpath(slices, TAB_SLICER.slice_file_stem(fn) * ".ll")
            @test isfile(path)
            isfile(path) || continue
            ir = read(path, String)
            @test occursin(Regex("^declare[^\\n]*@__rb_zcollide_$callee\\(", "m"), ir)
            @test !occursin(Regex("@$callee\\b"), ir)
            @test !occursin(" alias ", ir)
            checked += 1
        end
        @test checked == 2
        # The declares recorded off the shipped IR agree with the wrapper's table.
        ir = read(joinpath(slices, "zc_table_at.ll"), String)
        @test RepliBuild.Wrapper._slice_declared_symbols(ir) == ["__rb_zcollide_z_errmsg"]
    end

    @testset "a decoy loaded first does not catch the calls" begin
        decoy_src = joinpath(TAB_DIR, "decoy", "zdecoy.c")
        decoy = joinpath(TAB_DIR, "decoy", "libzdecoy." * Libdl.dlext)
        rm(decoy; force = true)
        (out, rc) = TAB_C._clang_for_c_bucket("clang", ["-shared", "-fPIC", "-O2", "-o", decoy, decoy_src])
        @test rc == 0
        @test isfile(decoy)
        rc == 0 && isfile(decoy) || return

        r = run_isolated(`$(joinpath(TAB_DIR, "probe_tier1_alias.jl")) $wrapper $decoy`; timeout = 600)
        @test r.ok
        v = probe_verdicts(r.output)
        @test haskey(v, "PROBE_DONE")
        probes = ["collision_gzclose_w", "collision_gzclose", "collision_adler32_z",
                  "collision_z_verbose", "collision_z_errmsg",
                  "tier1_surface", "dispatch_tier",
                  "declares_gzclose_w", "declares_adler32_z", "declares_zc_set_seed",
                  "declares_gzclose", "declares_adler32", "declares_zc_verbosity",
                  "declares_zc_set_verbosity", "declares_zc_table_at", "declares_gzclose_r",
                  "no_public_declare",
                  "gzclose", "gzclose_r", "adler32", "table", "state", "spliced"]
        for name in probes
            @test startswith(get(v, name, "no PROBE line"), "PASS")
        end
        @test length(v) == length(probes) + 1     # PROBE_DONE, and nothing unlisted
    end
end
