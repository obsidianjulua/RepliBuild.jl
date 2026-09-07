# SysConfigGen.jl — generate a library's configure-time headers in house
#
# RepliBuild compiles all-sources-minus-excludes under ONE uniform flag set and
# never runs a configure step. That rules out every library whose headers are
# produced by feature detection (`configure_file` over a `config.h.in`) — the
# build cannot even start. libcurl was the first exception, unblocked by hand:
# run cmake once, copy `lib/curl_config.h` into the package, check it in (see
# packages/curl/config/README.md in the Hub).
#
# This module generalises that one-off. cmake's **configure** step is not its
# build step: it is a self-contained feature-detection pass that emits the
# generated headers and, with CMAKE_EXPORT_COMPILE_COMMANDS, a full record of
# how upstream intends to compile every translation unit. Both are capturable
# in seconds without a compiler ever touching a real source file.
#
# So this reads a build system rather than running one, and yields three things:
#
#   1. the generated headers (and configure-time generated sources) to check in,
#      laid out under the same -I roots upstream compiles against, and with
#      try_compile probe scratch told apart from real generated library code,
#   2. the exact -D / -I set upstream uses, for [compile],
#   3. a mechanical answer to the Hub's admission question — "does every source
#      compile under one flag set?" — from compile_commands.json instead of by
#      eye, plus the exclude list implied by the files a target never compiles.
#
# What it does NOT do: run a build. Libraries that defer generation to a build
# rule (libpng's pnglibconf.h comes out of an awk pipeline wired as a custom
# command) yield nothing here — `generated_headers` comes back empty, which is
# the honest signal to look for a shipped fallback or reach for `ingest()`.


# Moved out of RepliBuildTooling 2026-08-26. It never belonged there: a project
# whose headers come from feature detection cannot be BUILT without this, so it
# is a build-system capability, not an introspection tool. Tooling is for
# looking at what was produced; this produces something. The file was already
# self-contained (Dates + JSON, its own CMakeProbe), so the move was a lift.

module SysConfigGen

using Dates
using JSON

# The one thing this needed from its old host's namespace. In RepliBuildTooling
# it arrived via `using RepliBuild: execute, …`; here it is a sibling submodule,
# so the dependency is explicit and local rather than a round trip through the
# top-level package.
#
# `execute` runs the probe under `with_llvm_env` by default (its `use_llvm_env`
# keyword), which is the point: cmake's feature probes get answered by the same
# clang that will later compile the sources. A header captured from the system
# compiler can disagree with the toolchain that consumes it.
using ..BuildBridge: execute

export CMakeProbe, cmake_probe, capture_config, toml_fragment,
       main_target, uniform

# ============================================================================
# TYPES
# ============================================================================

"""
    CMakeTarget

One cmake library target and the flags it compiles its sources under.

# Fields
- `name::String` — cmake target name (e.g. `pcre2-8-shared`)
- `kind::Symbol` — `:shared` or `:static`, inferred from the defines
- `files::Vector{String}` — its TUs, source-relative (generated ones config-relative)
- `defines::Vector{String}` — `-D` flags
- `include_dirs::Vector{String}` — `-I` dirs, package-relative where possible
- `flag_sets::Int` — distinct flag sets *within* this target; 1 is the healthy case

`kind` is read off `<target>_EXPORTS`, which cmake defines only when building a
shared or module library — a more reliable tell than the target's name, but only
a heuristic: a project that rolls its own visibility switch instead of using
cmake's (libyaml's `YAML_DECLARE_EXPORT`) reads as `:static` even when
`BUILD_SHARED_LIBS=ON` produced it. The label is advisory; it only decides which
target [`main_target`](@ref) prefers when a project configures several, and a
single-target project is picked correctly either way.
"""
struct CMakeTarget
    name::String
    kind::Symbol
    files::Vector{String}
    defines::Vector{String}
    include_dirs::Vector{String}
    flag_sets::Int
end

"""
    CMakeProbe

The result of a configure-only cmake run over an upstream source tree.

# Fields
- `name::String` — project name (defaults to the source directory's basename)
- `source_dir::String` — the upstream checkout that was probed
- `build_dir::String` — scratch build tree cmake configured into
- `cmake_args::Vector{String}` — the arguments the probe was run with
- `generated_headers::Vector{String}` — build-dir-relative generated headers
- `generated_sources::Vector{String}` — build-dir-relative generated `.c`/`.cpp`
  that a configured target actually compiles
- `scratch_sources::Vector{String}` — build-dir-relative generated `.c`/`.cpp`
  that **no** configured target compiles
- `include_roots::Vector{String}` — build-dir-relative `-I` dirs inside the build
  tree, longest first; `""` is the build dir itself
- `targets::Vector{CMakeTarget}` — every library target cmake configured
- `tree_sources::Vector{String}` — every `.c`/`.cpp` in the checkout
- `cmake_version::String`
- `probed_at::DateTime`

A project routinely configures the *same* sources into several targets — shared,
static, and a compat shim is the common trio. That is not per-file flag
divergence and must not be read as one, so uniformity is judged per target (see
[`uniform`](@ref)) and [`main_target`](@ref) picks the one to build from.

`generated_sources` and `scratch_sources` are split on one mechanical test —
does any target in `compile_commands.json` compile the file — because a configure
step emits both kinds into the same tree and they must never be confused. cmake's
`try_compile` writes its probe source into the build tree and compiles it out of
band, so it is generated, it is a `.c`, and it has its own `main()`; capturing it
as library code and listing it in `[compile] source_files` plants a `main` symbol
in the shared object. `check_c_source_compiles` uses `CMakeFiles/CMakeTmp/` (which
[`_is_cmake_internal`](@ref) already drops), but a hand-rolled `try_compile` with
its own bindir — SUNDIALS' `POSIX_TIMER_TEST/ltest.c` — lands in plain sight.

The split needs `compile_commands.json`; without it (a generator that does not
emit one) there is no evidence, so nothing is classified and `scratch_sources` is
empty rather than guessed. Note the test is *uncompiled*, not *scratch*: a
generated `.c` that upstream `#include`s rather than compiles also lands in
`scratch_sources`, which is why [`capture_config`](@ref) reports what it skipped
and takes `capture_scratch=true` to override.
"""
struct CMakeProbe
    name::String
    source_dir::String
    build_dir::String
    cmake_args::Vector{String}
    generated_headers::Vector{String}
    generated_sources::Vector{String}
    scratch_sources::Vector{String}
    include_roots::Vector{String}
    targets::Vector{CMakeTarget}
    tree_sources::Vector{String}
    cmake_version::String
    probed_at::DateTime
end

"""
    uniform(target::CMakeTarget) -> Bool
    uniform(probe::CMakeProbe) -> Bool

True when a target compiles all of its sources under one flag set — the
admission test for RepliBuild's source-build pipeline. For a probe, this reports
on [`main_target`](@ref); other targets may still diverge, which is fine, since
only one target is ever built.
"""
uniform(t::CMakeTarget) = t.flag_sets == 1
function uniform(p::CMakeProbe)
    t = main_target(p)
    return t === nothing ? false : uniform(t)
end

"""
    main_target(probe::CMakeProbe; target="") -> Union{CMakeTarget,Nothing}

The target to build from: `target` by name if given, otherwise the shared
library with the most sources, falling back to the largest target of any kind.
Shared is preferred because it matches `[binary] type = "shared"`, and its
defines carry the visibility switches (`<target>_EXPORTS`) that decide whether
symbols land in the dynamic table at all.
"""
function main_target(p::CMakeProbe; target::String="")
    isempty(p.targets) && return nothing
    if !isempty(target)
        i = findfirst(t -> t.name == target, p.targets)
        i === nothing && error("main_target: no target '$target'. Available: " *
                               join([t.name for t in p.targets], ", "))
        return p.targets[i]
    end
    shared = filter(t -> t.kind === :shared, p.targets)
    pool = isempty(shared) ? p.targets : shared
    return pool[argmax(map(t -> length(t.files), pool))]
end

function Base.show(io::IO, t::CMakeTarget)
    print(io, "$(t.name) [$(t.kind)] — $(length(t.files)) TUs, ",
              uniform(t) ? "uniform" : "$(t.flag_sets) flag sets")
end

function Base.show(io::IO, p::CMakeProbe)
    println(io, "CMakeProbe: $(p.name)")
    println(io, "  Source:    $(p.source_dir)")
    println(io, "  Generated: $(length(p.generated_headers)) header(s), " *
                "$(length(p.generated_sources)) source(s)")
    for f in vcat(p.generated_headers, p.generated_sources)
        println(io, "               $f")
    end
    if !isempty(p.scratch_sources)
        println(io, "  Scratch:   $(length(p.scratch_sources)) generated source(s) " *
                    "no target compiles — not captured")
        for f in p.scratch_sources
            println(io, "               $f")
        end
    end
    mt = main_target(p)
    println(io, "  Targets:   $(length(p.targets))")
    for t in p.targets
        println(io, "    ", (mt !== nothing && t.name == mt.name) ? "* " : "  ", t)
    end
    if mt === nothing
        println(io, "  No library target found — nothing to build from.")
    else
        n_excl = length(setdiff(p.tree_sources, mt.files))
        println(io, "  Chosen:    $(mt.name) — ",
                    uniform(mt) ? "UNIFORM, admissible for the source build" :
                                  "NOT uniform, see flag sets above")
        println(io, "  Excludes:  $n_excl file(s) it never compiles")
        println(io, "  Defines:   $(join(mt.defines, " "))")
    end
    print(io, "  cmake $(p.cmake_version) @ $(Dates.format(p.probed_at, "yyyy-mm-dd HH:MM"))")
end

# ============================================================================
# INTERNALS
# ============================================================================

const _HEADER_EXTS = (".h", ".hpp", ".hh", ".hxx", ".inc", ".def")
const _SOURCE_EXTS = (".c", ".cpp", ".cc", ".cxx")

_has_ext(path, exts) = lowercase(splitext(path)[2]) in exts

# EVERY PATH THIS MODULE COMPARES, SPLITS OR STORES IS '/'-SEPARATED. The compile
# lines come out of compile_commands.json, which cmake always writes with forward
# slashes, and every pattern here is written against that spelling —
# `_is_cmake_internal`, `_capture_rel`, `_collapse_excludes`, `toml_fragment`'s
# `config_rel * "/"`. `walkdir`/`relpath`/`abspath` answer in the HOST separator,
# so on Windows the two spellings never meet and every comparison silently says
# "no": a real generated TU was classified as try_compile scratch and dropped
# from `[compile] source_files`, cmake's own CompilerIdC probe was NOT dropped,
# `_build_include_roots` and `_translate_includes` both returned empty, and
# `_rel_source` handed absolute paths downstream. None of it raised.
#
# HOST-CONDITIONAL on purpose, the same reason `Compiler._canon_path` is: a
# backslash is a legal character in a POSIX filename, so rewriting one off
# Windows would corrupt a real name.
_posix(p::AbstractString) = Sys.iswindows() ? replace(String(p), '\\' => '/') : String(p)

# `joinpath`, but always '/'-separated (see `_posix`) and keeping joinpath's
# empty-prefix behaviour: `cmake_probe` defaults `clone_rel` to `""`, and
# `"" * "/" * "lib"` is an ABSOLUTE path, which is not what a clone-relative
# name means. Used wherever the result is matched against a '/'-joined prefix
# downstream, which `joinpath` would spell natively on one side only.
_join_rel(a::AbstractString, b::AbstractString) =
    isempty(a) ? String(b) : string(a, "/", b)

# cmake writes its own scaffolding into the build tree: compiler-identification
# probe sources under CMakeFiles/, and whatever FetchContent pulled into _deps/.
# Both match the extension whitelist below, and neither is ours to capture.
_is_cmake_internal(rel::String) =
    startswith(rel, "CMakeFiles/") || contains(rel, "/CMakeFiles/") ||
    startswith(rel, "_deps/")      || contains(rel, "/_deps/")

function _walk_generated(build_dir::String)
    headers, sources = String[], String[]
    for (root, _, files) in walkdir(build_dir)
        for f in files
            rel = _posix(relpath(joinpath(root, f), build_dir))
            _is_cmake_internal(rel) && continue
            _has_ext(f, _HEADER_EXTS) && push!(headers, rel)
            _has_ext(f, _SOURCE_EXTS) && push!(sources, rel)
        end
    end
    return sort!(headers), sort!(sources)
end

# One compile_commands entry carries the file, the directory it is compiled in,
# and either a `command` string or a pre-split `arguments` array.
function _entry_args(entry::Dict)
    haskey(entry, "arguments") && return String.(entry["arguments"])
    # cmake does not emit embedded quotes for the flags we read, so a whitespace
    # split is sufficient here.
    return String.(filter(!isempty, split(strip(entry["command"]), r"\s+")))
end

# cmake writes objects to `CMakeFiles/<target>.dir/<path>.o`, which is the only
# place in compile_commands where the target name appears at all.
function _target_of(entry::Dict, args::Vector{String})
    out = get(entry, "output", "")
    if isempty(out)
        i = findfirst(==("-o"), args)
        out = (i !== nothing && i < length(args)) ? args[i+1] : ""
    end
    m = match(r"CMakeFiles/([^/]+)\.dir/", out)
    return m === nothing ? "" : String(m.captures[1])
end

# Strip everything file-specific so what remains is the flag signature: the
# compiler driver, the input, the -o output, and -c itself all vary per TU by
# construction and say nothing about whether the flag SET is uniform.
function _flag_signature(args::Vector{String})
    sig = String[]
    skip_next = false
    for (i, a) in enumerate(args)
        i == 1 && continue           # compiler driver
        if skip_next
            skip_next = false
            continue
        end
        if a == "-o" || a == "-c"
            a == "-o" && (skip_next = true)
            continue
        end
        _has_ext(a, _SOURCE_EXTS) && continue
        endswith(a, ".o") && continue
        push!(sig, a)
    end
    return sig
end

# A TU is either a checked-in source (name it relative to the clone, so the
# exclude list can be matched against clone-relative paths) or a generated one
# living in the scratch build tree — which will travel with the package once
# captured, so it is named relative to the package's config dir instead.
#
# The generated case shares `_capture_rel` with `capture_config` rather than
# re-deriving a basename: `[compile] source_files` names the file at the path the
# capture actually wrote it to, and one derivation is the only way those two
# cannot drift into naming a file that isn't there.
function _rel_source(file::String, source_dir::String, build_dir::String,
                     config_rel::String, roots::Vector{String})
    if startswith(file, source_dir * "/")
        return file[length(source_dir)+2:end]
    elseif startswith(file, build_dir * "/")
        return _join_rel(config_rel, _capture_rel(file[length(build_dir)+2:end], roots))
    end
    return file
end

# Every -I / -isystem directory in a flag signature, in order. One derivation,
# because two consumers read it for different reasons — the TOML's include_dirs
# and the layout the captured headers have to be written in — and a header
# captured relative to a root the compile line never mentions is unreachable.
function _include_dirs(sig::Vector{String})
    out = String[]
    i = 1
    while i <= length(sig)
        a = sig[i]
        dir = if a == "-I" || a == "-isystem"
            i += 1
            i <= length(sig) ? sig[i] : ""
        elseif startswith(a, "-I")
            a[3:end]
        else
            ""
        end
        i += 1
        isempty(dir) || push!(out, dir)
    end
    return out
end

# -I dirs point at absolute paths inside the checkout or the scratch build tree.
# The build-tree ones are exactly the generated headers we are about to capture,
# so they become the package's config dir; the source ones become clone-relative
# so they survive a fresh clone at a different path.
function _translate_includes(sig::Vector{String}, source_dir::String,
                             build_dir::String, config_rel::String,
                             clone_rel::String)
    out = String[]
    for dir in _include_dirs(sig)
        ad = rstrip(_posix(abspath(dir)), '/')
        mapped = if ad == build_dir || startswith(ad, build_dir * "/")
            config_rel                      # generated headers travel with the package
        elseif ad == source_dir
            ""                              # the clone root; the resolver adds this
        elseif startswith(ad, source_dir * "/")
            _join_rel(clone_rel, ad[length(source_dir)+2:end])
        else
            continue                        # system / external dep, not ours to pin
        end
        !isempty(mapped) && mapped ∉ out && push!(out, mapped)
    end
    return out
end

# The build-tree -I dirs, build-dir-relative, longest first. These are the roots
# a generated header's include FORM is relative to: upstream passes
# `-I<build>/include` and writes `#include <sundials/sundials_config.h>`, so the
# header captured out of `<build>/include/sundials/` has to keep the `sundials/`
# level or `-Iconfig` resolves nothing. Longest first because a build tree
# routinely carries both `-I<build>/include` and `-I<build>`, and only the
# deepest matching root reproduces the spelling the sources use.
#
# `""` denotes the build dir itself and is therefore always last — it matches
# everything, so it must only be reached once the real roots have been tried.
function _build_include_roots(dirs, build_dir::String)
    roots = String[]
    for d in dirs
        ad = rstrip(_posix(abspath(d)), '/')
        r = if ad == build_dir
            ""
        elseif startswith(ad, build_dir * "/")
            ad[length(build_dir)+2:end]
        else
            continue
        end
        r ∉ roots && push!(roots, r)
    end
    sort!(roots; by = r -> (-length(r), r))
    return roots
end

# Where a captured file goes, relative to the package's config dir: its path
# under the deepest include root that contains it. With no matching root there is
# no evidence about the include form, so it degrades to the historical basename
# rather than to a new wrong answer.
#
# Both sides are '/'-separated — see `_posix`, which is where that is now made
# true on Windows rather than assumed. Until 2026-09-07 `rel` reached here in the
# host separator while the roots came '/'-spelled out of compile_commands.json,
# so no root ever matched and every file silently took the basename branch.
function _capture_rel(rel::String, roots::Vector{String})
    for r in roots
        isempty(r) && return rel
        startswith(rel, r * "/") && return rel[length(r)+2:end]
    end
    return basename(rel)
end

function _tree_sources(source_dir::String)
    out = String[]
    for (root, dirs, files) in walkdir(source_dir)
        filter!(d -> d != ".git", dirs)
        for f in files
            _has_ext(f, _SOURCE_EXTS) || continue
            push!(out, _posix(relpath(joinpath(root, f), source_dir)))
        end
    end
    return sort!(out)
end

# ============================================================================
# PROBE
# ============================================================================

"""
    cmake_probe(source_dir; args=String[], name="", build_dir="",
                generator="Ninja", shared=true, policy_floor=true,
                config_rel="config", clone_rel="", use_llvm_env=true) -> CMakeProbe

Run cmake's **configure** step over `source_dir` and read back what it generated
and how it intends to compile. No build is run and no compiler touches a real
source file; the cost is the feature-detection pass alone (single-digit seconds
for every library tried so far).

`args` are extra `-D` arguments — the place to turn off tests, tools, examples
and optional dependencies. Keeping that set lean is the whole game: a default
configure of libcurl enables brotli, zstd, nghttp2, idn2, psl, libssh2, c-ares
and krb5, and every one becomes a link library that can break `use(...)` on its
next bump.

Keyword notes:
- `shared` adds `-DBUILD_SHARED_LIBS=ON`, matching `[binary] type = "shared"`.
  Projects with their own `FOO_SHARED`/`FOO_STATIC` switches usually configure
  both target kinds anyway; that is expected, and `main_target` picks the shared
  one rather than mistaking it for flag divergence.
- `policy_floor` adds `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`. cmake 4.x refuses a
  `cmake_minimum_required` below 3.5 outright, a configure-time hard error on
  plenty of still-current releases (libyaml 0.2.5, for one).
- `clone_rel` is where the checkout will live relative to the package dir, used
  to render include paths for the TOML — for a `[dependencies]` git clone that
  is `.replibuild_cache/deps/<dep-name>`.
- `use_llvm_env` runs cmake under RepliBuild's LLVM environment, so the feature
  probes are answered by the same clang that will compile the sources. This is
  the one place the capture improves on doing it by hand: a header captured from
  the system compiler can disagree with the toolchain that consumes it.

The probe is read-only with respect to `source_dir`.
"""
function cmake_probe(source_dir::String;
                     args::Vector{String}=String[],
                     name::String="",
                     build_dir::String="",
                     generator::String="Ninja",
                     shared::Bool=true,
                     policy_floor::Bool=true,
                     config_rel::String="config",
                     clone_rel::String="",
                     use_llvm_env::Bool=true)

    source_dir = String(rstrip(_posix(abspath(source_dir)), '/'))
    isdir(source_dir) || error("cmake_probe: source dir not found: $source_dir")
    isfile(joinpath(source_dir, "CMakeLists.txt")) ||
        error("cmake_probe: no CMakeLists.txt in $source_dir — not a cmake project. " *
              "If the top level is a wrapper, point at the subdirectory that has one.")

    isempty(name) && (name = basename(source_dir))
    isempty(build_dir) && (build_dir = mktempdir(; prefix="rbcapture_$(name)_"))
    build_dir = String(rstrip(_posix(abspath(build_dir)), '/'))
    ispath(build_dir) && rm(build_dir; recursive=true, force=true)
    mkpath(build_dir)

    cmake_args = String["-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"]
    shared && push!(cmake_args, "-DBUILD_SHARED_LIBS=ON")
    policy_floor && push!(cmake_args, "-DCMAKE_POLICY_VERSION_MINIMUM=3.5")
    append!(cmake_args, args)

    full = String["-S", source_dir, "-B", build_dir, "-G", generator]
    append!(full, cmake_args)

    output, exitcode = execute("cmake", full; use_llvm_env=use_llvm_env)
    if exitcode != 0
        tail = join(last(split(output, '\n'), 25), '\n')
        error("cmake_probe: configure failed for '$name' (exit $exitcode).\n$tail")
    end

    ver_out, _ = execute("cmake", String["--version"]; use_llvm_env=false)
    cmake_version = let m = match(r"cmake version (\S+)", ver_out)
        m === nothing ? "unknown" : String(m.captures[1])
    end

    headers, gen_sources = _walk_generated(build_dir)

    # ── read back the intended compilation, grouped by target ─────────────
    ccpath = joinpath(build_dir, "compile_commands.json")
    raw = Tuple{String,String,Vector{String}}[]   # (target, abs file, flag signature)
    compiled_gen = Set{String}()                  # build-dir-relative TUs some target compiles
    inc_roots = String[]
    if isfile(ccpath)
        # use_mmap=false: a live mmap blocks deletion on Windows and is released
        # only at GC — and this one maps a file inside the cmake build tree that
        # `capture_config` removes moments later. See Builder/ThunkBuilder.jl.
        for e in JSON.parsefile(ccpath; use_mmap=false)
            file = get(e, "file", "")
            isempty(file) && continue
            eargs = _entry_args(e)
            tgt = _target_of(e, eargs)
            isempty(tgt) && (tgt = "unknown")
            sig = _flag_signature(eargs)
            afile = _posix(abspath(file))
            startswith(afile, build_dir * "/") &&
                push!(compiled_gen, afile[length(build_dir)+2:end])
            # Union across every target, not just main_target: a generated header
            # can be reachable only from another target's include path, and it is
            # still the include form that decides where the file has to land.
            for r in _build_include_roots(_include_dirs(sig), build_dir)
                r ∉ inc_roots && push!(inc_roots, r)
            end
            push!(raw, (tgt, afile, sig))
        end
    end
    sort!(inc_roots; by = r -> (-length(r), r))

    # Naming a generated TU needs the complete root set, so it happens after the
    # scan rather than inside it — a root discovered by a later entry still has
    # to govern an earlier file's path.
    by_target = Dict{String,Vector{Tuple{String,Vector{String}}}}()  # target => [(file, sig)]
    for (tgt, afile, sig) in raw
        push!(get!(by_target, tgt, Tuple{String,Vector{String}}[]),
              (_rel_source(afile, source_dir, build_dir, config_rel, inc_roots), sig))
    end

    # No compile_commands means no evidence about which generated sources are
    # library code, so classify nothing rather than guess — dropping a real
    # source would fail the build far from here.
    scratch = String[]
    if !isempty(by_target)
        scratch     = filter(s -> s ∉ compiled_gen, gen_sources)
        gen_sources = filter(s -> s ∈ compiled_gen, gen_sources)
    end

    targets = CMakeTarget[]
    for (tname, entries) in by_target
        sigs = unique(last.(entries))
        # Report from the dominant signature so a target that does diverge still
        # yields its majority flags rather than an arbitrary one.
        counts = Dict(s => count(==(s), last.(entries)) for s in sigs)
        main_sig = argmax(s -> counts[s], sigs)
        defs = filter(startswith("-D"), main_sig)
        kind = any(d -> endswith(d, "_EXPORTS"), defs) ? :shared : :static
        push!(targets, CMakeTarget(
            tname, kind, sort!(unique(first.(entries))), defs,
            _translate_includes(main_sig, source_dir, build_dir, config_rel, clone_rel),
            length(sigs)))
    end
    sort!(targets; by = t -> (-length(t.files), t.name))

    return CMakeProbe(name, source_dir, build_dir, cmake_args,
                      headers, gen_sources, scratch, inc_roots,
                      targets, _tree_sources(source_dir),
                      cmake_version, now())
end

# ============================================================================
# SYSCONFIG
# ============================================================================

"""
    capture_config(probe::CMakeProbe, out_dir; sources=true, layout=:auto,
                   capture_scratch=false) -> Vector{String}

Copy the probe's generated headers (and, with `sources=true`, its generated
`.c`/`.cpp`) into `out_dir`, and write a `SYSCONFIG.md` recording exactly how they
were produced. Returns the paths written.

`probe.scratch_sources` — generated sources no configured target compiles, which
is overwhelmingly cmake `try_compile` probe scratch carrying its own `main()` —
are **not** captured. Pass `capture_scratch=true` for the rare generated `.c` that
upstream `#include`s rather than compiles; SYSCONFIG.md always lists what was
skipped, so the decision is visible rather than silent.

`layout` decides where each captured file lands under `out_dir`:

- `:auto` (default) — under the deepest build-tree `-I` root that contains it, so
  the package reproduces the include form upstream compiles with. A header
  generated into `<build>/include/foo/cfg.h` with `-I<build>/include` on the
  compile line is included as `<foo/cfg.h>`, so it is written to
  `out_dir/foo/cfg.h` and resolves under a single `-Iout_dir`. A build tree that
  generates everything at its root — the common case, and pcre2's — reduces
  exactly to basenames.
- `:flat` — basenames only, regardless of include form.
- `:build_tree` — the raw build-dir-relative path.

Two sources landing on one destination is a hard error under every layout: the
`force=true` copy would otherwise ship one file under another's name, and a
config header is not something to get silently wrong.

The copied files are **build artifacts checked into the package on purpose**,
and they pin the package to this machine's feature detection at this upstream
commit — the same single-target pin libcurl's config carries. SYSCONFIG.md records
the cmake arguments so a version bump can regenerate and diff: a changed
`SIZEOF_*` or a vanished `USE_*` is real news, not noise.
"""
function capture_config(probe::CMakeProbe, out_dir::String;
                        sources::Bool=true, layout::Symbol=:auto,
                        capture_scratch::Bool=false)
    layout in (:auto, :flat, :build_tree) ||
        error("capture_config: layout must be :auto, :flat or :build_tree, got :$layout")

    out_dir = abspath(out_dir)
    picked = copy(probe.generated_headers)
    if sources
        append!(picked, probe.generated_sources)
        capture_scratch && append!(picked, probe.scratch_sources)
    end

    if isempty(picked)
        @warn """
              capture_config: '$(probe.name)' generated nothing at configure time.
              Its headers are probably emitted by a BUILD rule (a custom command or
              a script pipeline) rather than by configure_file. Check the upstream
              tree for a shipped fallback (libpng ships scripts/pnglibconf.h.prebuilt),
              otherwise this library wants RepliBuild.ingest() instead."""
        return String[]
    end

    mkpath(out_dir)
    written = String[]
    claimed = Dict{String,String}()   # dst => the rel that got there first
    for rel in picked
        sub = layout === :flat       ? basename(rel) :
              layout === :build_tree ? rel :
                                       _capture_rel(rel, probe.include_roots)
        dst = joinpath(out_dir, sub)
        if haskey(claimed, dst)
            error("capture_config: '$(claimed[dst])' and '$rel' both map to " *
                  "'$sub' under layout=:$layout. Copying would ship one under " *
                  "the other's name — re-run with layout=:build_tree.")
        end
        claimed[dst] = rel
        mkpath(dirname(dst))
        cp(joinpath(probe.build_dir, rel), dst; force=true)
        chmod(dst, 0o644)
        push!(written, dst)
    end

    open(joinpath(out_dir, "SYSCONFIG.md"), "w") do io
        println(io, "# Captured cmake configure output — checked in on purpose\n")
        println(io, "`$(probe.name)` cannot be compiled from a bare checkout: these files are")
        println(io, "produced by cmake's configure step (`configure_file` over a template), and")
        println(io, "RepliBuild compiles all-sources-minus-excludes under one flag set without")
        println(io, "ever running a configure. So they have to already exist.\n")
        println(io, "Captured by `RepliBuild.SysConfigGen.cmake_probe` + `capture_config`.\n")
        println(io, "## Files\n")
        println(io, "Paths are relative to this directory, which is the one `include_dirs`")
        println(io, "points at. Under `layout=:auto` they keep the include form upstream")
        println(io, "compiles with, so `-I` on this directory resolves them unchanged.\n")
        for w in written
            println(io, "- `$(_posix(relpath(w, out_dir)))`")
        end
        if !isempty(probe.scratch_sources)
            println(io, "\n## Skipped — generated, but no target compiles them\n")
            println(io, capture_scratch ?
                "Captured anyway (`capture_scratch=true`); confirm each is real library code." :
                "Not captured. Overwhelmingly cmake `try_compile` probe scratch, which has")
            capture_scratch || println(io,
                "its own `main()` — never add one to `[compile] source_files`.\n")
            capture_scratch && println(io)
            for s in probe.scratch_sources
                println(io, "- `$s`")
            end
        end
        println(io, "\n## Regenerating (required on any version bump)\n")
        println(io, "```julia")
        println(io, "using RepliBuild, RepliBuild.SysConfigGen")
        println(io, "p = cmake_probe(\"<checkout>\";")
        println(io, "                name=\"$(probe.name)\",")
        argl = filter(a -> !startswith(a, "-DCMAKE_") && a != "-DBUILD_SHARED_LIBS=ON",
                      probe.cmake_args)
        println(io, "                args=[", join(map(a -> "\"$a\"", argl),
                                                   ",\n                      "), "])")
        println(io, "capture_config(p, \"config\"", layout === :auto ? "" : "; layout=:$layout",
                    capture_scratch ? "; capture_scratch=true" : "", ")")
        println(io, "```\n")
        println(io, "Full cmake argument set used:\n")
        println(io, "```")
        foreach(a -> println(io, a), probe.cmake_args)
        println(io, "```\n")
        println(io, "## The pin this implies\n")
        println(io, "A snapshot of **this machine's** feature detection at the pinned commit.")
        println(io, "It travels with the package, so the build is reproducible — but it is a")
        println(io, "single-target pin, which matches the Hub. Regenerate on a version bump and")
        println(io, "diff: a changed `SIZEOF_*` or a vanished `USE_*` is real news.\n")
        println(io, "Captured from `$(probe.source_dir)` on ",
                    Dates.format(probe.probed_at, "yyyy-mm-dd"),
                    ", cmake $(probe.cmake_version), ", Sys.MACHINE, ".")
    end

    return written
end

# ============================================================================
# TOML PROPOSAL
# ============================================================================

# The resolver already prunes these directory names during source discovery, so
# proposing them as excludes would be redundant noise in the manifest.
const _RESOLVER_PRUNED = ("test", "tests", "testes", "example", "examples",
                          "fuzzing", "build", "doc", "docs")

# Collapse never-compiled files into directory patterns where a whole directory
# is dead. RepliBuild matches `exclude` as a SUBSTRING of the clone-relative
# path, so a coarse pattern is both shorter and safer than a list of names —
# libcurl's package notes the failure mode of the alternative: excluding
# `src/tool_*` by name let three other CLI files through, the build reported
# success, and the .so then failed to dlopen on a symbol whose definition had
# been excluded. Half-excluding a program is worse than not excluding it at all.
function _collapse_excludes(compiled::Vector{String}, uncompiled::Vector{String})
    isempty(uncompiled) && return String[]
    live = Set{String}()
    for f in compiled
        d = dirname(f)
        while true
            push!(live, d)
            isempty(d) && break
            # `dirname` is idempotent at a filesystem root — "/" on POSIX,
            # "C:\\" on Windows — so `isempty` alone only terminates for a
            # RELATIVE path. Everything here is meant to be relative; when a
            # caller upstream failed to relativize, this spun the CPU forever
            # instead of saying so. Stop at the fixed point either way.
            parent = dirname(d)
            parent == d && break
            d = parent
        end
    end

    patterns, covered = String[], Set{String}()
    for f in uncompiled
        f in covered && continue
        parts = split(dirname(f), '/'; keepempty=false)
        # Shallowest ancestor directory containing no compiled source at all.
        idx = findfirst(i -> join(parts[1:i], '/') ∉ live, eachindex(parts))
        if idx === nothing
            push!(patterns, f)
            push!(covered, f)
        else
            pat = join(parts[1:idx], '/') * "/"
            pat ∉ patterns && push!(patterns, pat)
            for g in uncompiled
                startswith(g, pat) && push!(covered, g)
            end
        end
    end
    # Drop what the resolver prunes on its own.
    filter!(p -> !(rstrip(p, '/') in _RESOLVER_PRUNED), patterns)
    return sort!(patterns)
end

"""
    toml_fragment(probe::CMakeProbe; target="", language="c", link_libraries=String[],
                 config_rel="config", shim_headers=String[]) -> String

Render the probe as `replibuild.toml` fragments for one target — `[compile]`
flags and include dirs from what upstream actually passes, plus the `exclude`
list implied by the sources that target never compiles.

This is a **proposal, not a manifest**. It is the structural half only; the
residue half (`[wrap.varargs]`, `[wrap.macros]`) cannot come from a build system,
because it is precisely what the preprocessor erased before the build system
ever saw it. Those entries are still earned one at a time from wrap failures.

Read every line before pasting. In particular decide, per library, what to do
with the defines that describe cmake's build rather than the library: `NDEBUG`
is usually noise, but `<target>_EXPORTS` may be load-bearing for symbol
visibility, and dropping it can empty the dynamic symbol table.
"""
function toml_fragment(probe::CMakeProbe;
                      target::String="",
                      language::String="c",
                      link_libraries::Vector{String}=String[],
                      config_rel::String="config",
                      shim_headers::Vector{String}=String[])
    t = main_target(probe; target=target)
    t === nothing && error("toml_fragment: probe found no library targets")

    io = IOBuffer()
    # Generated sources live in the package's config dir, not the clone, so they
    # are listed explicitly rather than being excluded as never-compiled.
    gen_files = filter(f -> startswith(f, config_rel * "/"), t.files)
    clone_files = filter(f -> !startswith(f, config_rel * "/"), t.files)
    excludes = _collapse_excludes(clone_files, setdiff(probe.tree_sources, clone_files))

    println(io, "# ── proposed by RepliBuild.SysConfigGen.toml_fragment ───────────────────")
    println(io, "# cmake $(probe.cmake_version), probed $(Dates.format(probe.probed_at, "yyyy-mm-dd")).")
    println(io, "# Target '$(t.name)' [$(t.kind)]: $(length(t.files)) TUs.")
    if uniform(t)
        println(io, "# ONE flag set across all of them — admissible for the source build.")
    else
        println(io, "# WARNING: $(t.flag_sets) distinct flag sets WITHIN this target.")
        println(io, "# RepliBuild compiles under one uniform set, so this does NOT drop in")
        println(io, "# as-is: narrow the configure, exclude the minority group, or treat")
        println(io, "# the library as ingest-only.")
    end
    if length(probe.targets) > 1
        others = join([x.name for x in probe.targets if x.name != t.name], ", ")
        println(io, "# Other targets configured (not built here): $others")
    end
    println(io)

    if !isempty(excludes)
        println(io, "# exclude: every .c/.cpp in the tree this target never compiles.")
        println(io, "# Substring-matched on the clone-relative path; verify each entry.")
        println(io, "exclude = [", join(map(e -> "\"$e\"", excludes), ", "), "]")
        println(io)
    end

    println(io, "[compile]")
    println(io, "flags = [", join(map(f -> "\"$f\"", vcat(["-O2", "-fPIC"], t.defines)), ", "), "]")
    println(io, "parallel = true")
    if !isempty(t.include_dirs)
        println(io, "include_dirs = [", join(map(d -> "\"$d\"", t.include_dirs), ", "), "]")
    end
    if !isempty(gen_files)
        println(io, "# Configure-time generated source(s), captured into $config_rel/.")
        println(io, "# The resolver only walks the clone, so these are added by hand.")
        println(io, "source_files = [", join(map(f -> "\"$f\"", gen_files), ", "), "]")
    end
    println(io)

    println(io, "[link]")
    println(io, "enable_lto = false")
    println(io, "optimization_level = \"2\"")
    if !isempty(link_libraries)
        println(io, "link_libraries = [", join(map(l -> "\"$l\"", link_libraries), ", "), "]")
    end
    println(io)

    println(io, "[binary]")
    println(io, "type = \"shared\"")
    println(io)

    println(io, "[wrap]")
    println(io, "language = \"$language\"")
    if !isempty(shim_headers)
        println(io, "shim_headers = [", join(map(h -> "\"$h\"", shim_headers), ", "), "]")
    else
        println(io, "# shim_headers = [...]   # the public header(s) users include")
    end

    return String(take!(io))
end

end # module SysConfigGen
