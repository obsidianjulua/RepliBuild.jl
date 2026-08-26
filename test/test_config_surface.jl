# Config surface guard.
#
# A TOML key that parses and then does nothing is worse than one that does not
# exist: users set it, docs describe it, and nothing reads it. `[ingest]
# extra_link_libs` was declared, documented as "additional -l libs at load time",
# serialized back out, and asserted in a test — and no code ever turned it into a
# dlopen. The test passed because it only checked that the value round-tripped
# through the PARSER, which is true whether or not the feature exists. That is the
# same class CLAUDE.md already records for harvested configs: "a WRONG config does
# not fail the build ... so the package test must assert behaviour that depends on
# the probes."
#
# So this file does two things:
#   1. Every RepliBuildConfig field must be CONSUMED or explicitly RESERVED.
#   2. `extra_link_libs` is asserted by EXECUTING what the generator emits.

using Test
using RepliBuild
using Libdl

const CM_PATH  = joinpath(@__DIR__, "..", "src", "Builder", "ConfigurationManager.jl")
const SRC_ROOT = joinpath(@__DIR__, "..", "src")
const CFG = RepliBuild.ConfigurationManager

# Fields that are deliberately parsed-but-inert. Each needs a REASON, and the
# guard fails if one of these becomes consumed — a stale reservation is how this
# list would rot into a permanent excuse.
const RESERVED_UNUSED = Dict(
    ("binary",    "strip_symbols")    => "parsed and serialized; nothing strips",
    ("discovery", "enabled")          => "whole [discovery] section is inert",
    ("discovery", "walk_dependencies")=> "whole [discovery] section is inert",
    ("discovery", "max_depth")        => "whole [discovery] section is inert",
    ("discovery", "ignore_patterns")  => "whole [discovery] section is inert",
    ("discovery", "parse_ast")        => "whole [discovery] section is inert",
    ("wrap",      "enabled")          => "no is_wrap_enabled exists; wrapping is never gated",
    ("wrap",      "style")            => "basic style is selected by MISSING metadata, not by this key",
    ("workflow",  "stages")           => "only reachable via is_stage_enabled, which has no callers",
    ("project",   "uuid")             => "carried in TOML, never used to identify anything",
    ("paths",     "source")           => "sources come from [compile]/discovery, not this path",
    ("paths",     "include")          => "include dirs come from [compile].include_dirs",
    ("paths",     "cache")            => "the live cache path is [cache].directory",
)

"""
Functions inside ConfigurationManager where reading a field is NOT consumption:
parsing it, writing it back out, checking it is well-formed, displaying it, or
copying it into a new config. `save_config` alone touches every field there is, so
without this the guard would call the whole surface live.

Consumption is what `get_output_path` and `is_cache_enabled` do — the accessors the
rest of `src/` actually calls. Anything outside this prefix set counts, so a future
consumer that ignores the naming convention makes the guard over-report and demand
a reservation, which is the safe direction to be wrong in.
"""
const STRUCTURAL_FN = r"^(load|save|validate|print|show|create_default|with)"

_src_texts() = Dict(joinpath(r, f) => read(joinpath(r, f), String)
                    for (r, _, fs) in walkdir(SRC_ROOT) for f in fs if endswith(f, ".jl"))

_is_cm(path) = occursin("ConfigurationManager.jl", path)

"""
Enclosing top-level function name for each line matching `pat`.
"""
function _reads_by_function(text::AbstractString, pat::Regex)
    fns = String[]
    cur = ""
    for line in split(text, '\n')
        m = match(r"^\s*function\s+([A-Za-z_][\w!]*)", line)
        m === nothing || (cur = m.captures[1])
        m2 = match(r"^([A-Za-z_][\w!]*)\s*\([^)]*\)\s*=", line)
        m2 === nothing || (cur = m2.captures[1])
        occursin(pat, line) && push!(fns, cur)
    end
    unique(fns)
end

"""
Locals a file binds to `config.<sec>`, so `ingest = config.ingest; ingest.library`
counts as consuming `[ingest] library`.

Without this the guard demands a reservation for every field reached through an
alias — `Compiler.ingest_library` and `Wrapper.TypeRegistry` both open that way.
Note the bare-name shortcut is NOT an acceptable substitute: the only `.style` in
`src/` outside ConfigurationManager is `tooltip.style`, inside a JavaScript string
literal in DAGDiff, and it would have marked the inert `[wrap] style` live.
"""
function _alias_vars(text::AbstractString, sec::AbstractString)
    unique(m.captures[1] for m in
           eachmatch(Regex("(?m)^\\s*([a-z_]\\w*)\\s*=\\s*[\\w.]*\\.\\Q$sec\\E(?![\\w])"), text))
end

"""
Does `name` appear OUTSIDE ConfigurationManager as something other than an
`export` entry? An exported accessor with no callers is not consumption — that is
exactly `is_stage_enabled`'s situation.

Export lists here wrap across lines, and only the first carries the `export`
keyword; matching that keyword alone let a continuation line count as a call site,
which is precisely how `is_stage_enabled` read as consumed.
"""
function _called_outside_cm(name::AbstractString, texts)
    pat = Regex("(?<!\\w)\\Q$name\\E(?![\\w!])")
    for (p, t) in texts
        _is_cm(p) && continue
        in_export = false
        for line in split(t, '\n')
            s = rstrip(line)
            if occursin(r"^\s*export\b", s)
                in_export = endswith(s, ",")
                continue
            elseif in_export
                in_export = endswith(s, ",")
                continue
            end
            occursin(pat, s) && return true
        end
    end
    false
end

@testset "Config surface" begin

    @testset "Every config field is consumed or explicitly reserved" begin
        texts   = _src_texts()
        cmtext  = read(CM_PATH, String)
        outside = filter(p -> !_is_cm(p.first), texts)

        # Sections come from RepliBuildConfig itself, so a new section is picked up
        # without editing this test. Matching is on the QUALIFIED `section.field`:
        # `enabled` is a field of three different structs, and matching the bare
        # name marked all three live off one hit.
        sections = Tuple{String,DataType}[]
        for (nm, T) in zip(fieldnames(CFG.RepliBuildConfig), fieldtypes(CFG.RepliBuildConfig))
            for U in (T isa Union ? Base.uniontypes(T) : (T,))
                U === Nothing && continue
                U isa DataType && isstructtype(U) && parentmodule(U) === CFG &&
                    push!(sections, (String(nm), U))
            end
        end
        @test length(sections) >= 8

        checked = 0
        unaccounted = String[]
        stale = String[]

        for (sec, U) in sections, f in fieldnames(U)
            fname = String(f)
            key = (sec, fname)
            checked += 1

            qual = Regex("\\.\\Q$sec\\E\\.\\Q$fname\\E(?![\\w])")
            consumed = any(occursin(qual, t) for t in values(outside))

            if !consumed
                # Reached through a local bound to the section.
                for t in values(outside), v in _alias_vars(t, sec)
                    if occursin(Regex("(?<!\\w)\\Q$v\\E\\.\\Q$fname\\E(?![\\w])"), t)
                        consumed = true
                        break
                    end
                end
            end

            if !consumed
                # Second chance: read inside a CM accessor that IS called.
                for fn in _reads_by_function(cmtext, qual)
                    isempty(fn) && continue
                    occursin(STRUCTURAL_FN, fn) && continue
                    if _called_outside_cm(fn, texts)
                        consumed = true
                        break
                    end
                end
            end

            if consumed && haskey(RESERVED_UNUSED, key)
                push!(stale, "[$sec] $fname is now consumed — drop it from RESERVED_UNUSED")
            elseif !consumed && !haskey(RESERVED_UNUSED, key)
                push!(unaccounted,
                      "[$sec] $fname is parsed but never consulted. Either wire it up, " *
                      "delete it, or add it to RESERVED_UNUSED with a reason.")
            end
        end

        isempty(unaccounted) || (println("\nUNACCOUNTED:"); foreach(x -> println("  ", x), unaccounted))
        isempty(stale)       || (println("\nSTALE RESERVATIONS:"); foreach(x -> println("  ", x), stale))

        @test checked > 50
        @test isempty(unaccounted)
        @test isempty(stale)
    end

    @testset "extra_link_libs emits nothing when undeclared" begin
        W = RepliBuild.Wrapper
        @test W._extra_link_libs_snippet((ingest = nothing,)) == ""
        @test W._extra_link_libs_snippet(
            (ingest = CFG.IngestConfig("/x/libfoo.so", String[], String[]),)) == ""
    end

    @testset "extra_link_libs emits a dlopen per declared lib" begin
        W = RepliBuild.Wrapper
        s = W._extra_link_libs_snippet(
            (ingest = CFG.IngestConfig("/x/libfoo.so", String[], ["m", "pthread"]),))
        @test occursin("\"m\"", s)
        @test occursin("\"pthread\"", s)
        @test occursin("Libdl.dlopen", s)
        @test occursin("RTLD_GLOBAL", s)
        # `-l` naming: the wrapper must try libNAME, not just NAME.
        @test occursin("\"lib\" * _lib", s)
    end

    @testset "the emitted snippet actually loads libraries" begin
        # The assertion that would have caught the original defect: RUN what the
        # generator emits. A snippet that parses but never dlopens passes every
        # text check above and fails here.
        W = RepliBuild.Wrapper

        # A library already in this process is guaranteed dlopen-able by path,
        # which makes the success case deterministic. Naming it `-l` style would
        # not be: on glibc `/usr/lib/libm.so` is a linker SCRIPT, so `-lm` links
        # fine and `dlopen("libm.so")` fails — see the docstring note.
        real = first(filter(p -> occursin(".so", p) && isfile(p), Libdl.dllist()))
        good = W._extra_link_libs_snippet(
            (ingest = CFG.IngestConfig("/x", String[], [real]),))
        m = Module(:PreloadGood)
        Core.eval(m, :(using Libdl))
        @test_nowarn include_string(m, good)

        bad = W._extra_link_libs_snippet(
            (ingest = CFG.IngestConfig("/x", String[],
                                       ["replibuild_no_such_library_anywhere"]),))
        m2 = Module(:PreloadBad)
        Core.eval(m2, :(using Libdl))
        @test_logs (:warn,) match_mode=:any include_string(m2, bad)
    end

    @testset "both generators splice the preload ahead of the main dlopen" begin
        # Placement is the whole point: opened after the library, these resolve
        # nothing. Assert ordering in the generator source, since the snippet
        # itself cannot know where it lands.
        for gen in ("C/GeneratorC.jl", "Cpp/GeneratorCpp.jl")
            t = read(joinpath(SRC_ROOT, "Wrapper", gen), String)
            @test occursin("_preload_snippet = _extra_link_libs_snippet(config)", t)
            n_init = length(collect(eachmatch(r"function __init__\(\)\n", t)))
            n_pre  = length(collect(eachmatch(r"function __init__\(\)\n\$_preload_snippet", t)))
            @test n_init > 0
            @test n_pre == n_init
        end
    end
end
