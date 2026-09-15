# VisibilityProbe.jl — does this library annotate its exports?
#
# `-fvisibility=hidden` is how you stop a wrapper carrying a library's internals:
# the compiler marks every definition hidden unless the author annotated it
# `visibility("default")`, and hidden symbols never reach `.dynsym`, so `dlsym`
# — and therefore the generated wrapper — sees only the public surface.
#
# That is a free, exact, author-supplied answer to "what is the API", for a
# library that annotates. For one that does not, the same flag hides EVERYTHING,
# `[link] promote_statics` reads hidden as internal and renames the whole API to
# `__rb_*`, and the wrapper generates nothing. `verify_wrap_surface` refuses that
# build rather than shipping it (Compiler.jl) — but a refusal at build time is a
# late and expensive way to learn a fact about a library.
#
# So: measure it first, per package, once. Compile the package's real sources
# under its real flags with `-fvisibility=hidden` added, and read the visibility
# off the emitted IR. A definition that keeps default visibility under that flag
# is one the author annotated; that set IS the library's declaration of its
# public API, computed by the compiler rather than parsed out of headers.
#
# THE VERDICT IS PER-CONFIGURATION, NOT PER-LIBRARY. An export macro is often
# conditional — cJSON's `CJSON_PUBLIC` expands to
# `__attribute__((visibility("default")))` only when `CJSON_API_VISIBILITY` is
# defined (cJSON.h:74-77) and to a bare `type` otherwise, so Hub cjson measures
# 0 of 92 annotated purely because its `[compile] flags` do not set that `-D`.
# The probe therefore measures the configuration that will actually be built,
# which is the useful question, and a `-D` can flip the answer. Same family as
# the harvested-config pin: the measurement is a snapshot of one configuration.
#
# Measured on the Hub 2026-09-15 (each from its own package dir):
#
#   box2d3   422 / 660 annotated (63.9%)   B2_API carries the attribute
#   pcre2     78 / 105 annotated (74.3%)   PCRE2_EXPORT carries it
#   cjson      0 /  92                     CJSON_PUBLIC inert without the -D
#   lua        0 / 361                     LUA_API is a bare `extern`
#   zlib       0 / 100                     no export macro at all
#
# NOTHING MEASURED IS NOT "NOTHING ANNOTATED". Run from the wrong directory,
# pcre2's 29 sources all fail to compile (its harvested `config/` is on the
# include path RELATIVE to the package dir, as `build()` cd's there) and a
# naive reading calls a 74%-annotated library unannotated. The probe cd's like
# `build()` does, and a probe that compiled nothing is a hard error rather than
# a verdict — same discipline as SysConfigGen's "with no compile_commands.json
# there is no evidence, so nothing is classified".

module VisibilityProbe

using ..ConfigurationManager
using ..Compiler: _clang_for_c_bucket, generate_macro_shims
using ..DependencyResolver
import JSON

export VisibilityReport, visibility_probe, annotates_exports

# ============================================================================
# TYPES
# ============================================================================

"""
What one probe run learned about a package.

* `external_defs` — function definitions with non-local linkage, i.e. everything
  that could reach the dynamic symbol table.
* `exported` — those that keep **default** visibility under `-fvisibility=hidden`.
  The library's own declaration of its API, as the compiler resolved it.
* `wrapped_today` / `retained` / `dropped` — only populated when the package has
  a built `compilation_metadata.json` to compare against. `dropped` is the
  decision-grade number: the functions currently in the wrapper that turning the
  flag on would remove.
* `verdict` — `:annotated`, `:unannotated`. (A run that measured nothing errors
  rather than returning a verdict.)
"""
struct VisibilityReport
    package::String
    sources::Int
    compiled::Int
    failed::Vector{String}
    external_defs::Int
    exported::Vector{String}
    library_exported::Vector{String}
    wrapped_today::Int
    retained::Vector{String}
    dropped::Vector{String}
    verdict::Symbol
end

# RepliBuild's OWN macro shims carry RB_SHIM_EXPORT, which on ELF is
# `visibility("default")` — unconditionally, by construction, so that
# `[wrap.macros]` constants survive a hidden build. They are therefore ALWAYS in
# `exported` and say nothing whatsoever about whether the library annotates.
#
# Counting them toward the verdict is not a rounding error, it inverts the answer:
# Hub lua measured `:annotated` at 89/450 on the first sweep with every one of the
# 89 a shim and zero library functions annotated — a verdict that would have sent
# someone to enable the flag on a package whose entire 156-function API the flag
# erases. The verdict is computed on the library's own definitions only; shims
# stay in `exported` because they genuinely do survive, which is what `retained`
# and `dropped` must reflect.
const _SHIM_PREFIX = "replibuild_shim_"
_is_shim(name::AbstractString) = startswith(name, _SHIM_PREFIX)

"""Did the library annotate its exports under the configuration probed?"""
annotates_exports(r::VisibilityReport) = r.verdict === :annotated

# ============================================================================
# IR SCAN
# ============================================================================

# LLVM prints `define [linkage] [visibility] [dllstorage] [cconv] [attrs] <ty> @name`.
# Both sets are matched as WHOLE TOKENS taken from the text BEFORE the `@`, never
# as substrings of the line: a function may legitimately be named `hidden_thing`,
# and a return type may be spelled `range(i32 -2147483647, -2147483648)`.
const _LOCAL_LINKAGE  = ("private", "internal")
const _NONDEFAULT_VIS = ("hidden", "protected")

"""
Classify one `define` line → `(:default | :hidden, name)`, or `nothing` when the
line is not an externally-visible function definition.
"""
function _classify_define(line::AbstractString)
    startswith(line, "define ") || return nothing
    at = findfirst('@', line)
    at === nothing && return nothing
    tokens = split(SubString(line, 1, at - 1))
    any(t -> t in _LOCAL_LINKAGE, tokens) && return nothing
    # LLVM quotes any symbol whose name needs escaping, and a quoted name may
    # contain whitespace — so the quoted form must be matched to its closing quote,
    # not with an optional quote and a no-whitespace class (which silently truncates
    # `@"odd name"` to `odd` and then never matches the DWARF spelling).
    rest = SubString(line, at)
    m = match(r"^@\"([^\"]*)\"", rest)
    m === nothing && (m = match(r"^@([^\s(]+)", rest))
    m === nothing && return nothing
    vis = any(t -> t in _NONDEFAULT_VIS, tokens) ? :hidden : :default
    return (vis, String(m.captures[1]))
end

function _scan_ir(path::AbstractString)
    total = 0
    exported = String[]
    for line in eachline(path)
        c = _classify_define(line)
        c === nothing && continue
        total += 1
        c[1] === :default && push!(exported, c[2])
    end
    return (total, exported)
end

# ============================================================================
# PROBE
# ============================================================================

"""
    visibility_probe(toml_path; metadata=nothing) -> VisibilityReport

Compile the package's sources under its own `[compile]` flags plus
`-fvisibility=hidden`, and report which definitions keep default visibility.

Runs from the package directory (`build()` does, and relative `include_dirs`
depend on it). Compiles to IR only — no link, no DWARF, no promotion — so it
costs one compile pass and nothing else. This is a **probe**: run it once per
package, and again on a version bump, the way `harvest.jl` is run.

Errors rather than returning a verdict when no source compiled — a probe that
measured nothing has found a broken invocation, not an unannotated library.
"""
function visibility_probe(toml_path::AbstractString; metadata::Union{Nothing,String}=nothing)
    toml_abs = abspath(String(toml_path))
    isfile(toml_abs) || error("VisibilityProbe: no such config: $toml_abs")
    pkg_dir = dirname(toml_abs)

    return cd(pkg_dir) do
        config = ConfigurationManager.load_config(toml_abs)
        config = DependencyResolver.resolve_dependencies(config)

        # The macro-shim TU is generated by `build()`, not listed in the config, and
        # it carries RB_SHIM_EXPORT — on ELF `visibility("default")`, precisely so
        # `[wrap.macros]` constants survive a hidden build. Probe the same source set
        # the build compiles or every shim reads as a function the flag would drop,
        # which is a false positive in the one list a human is meant to act on.
        sources = generate_macro_shims(config, ConfigurationManager.get_source_files(config))
        sources = filter(sources) do s
            endswith(s, ".c") || endswith(s, ".cpp") ||
                endswith(s, ".cc") || endswith(s, ".cxx")
        end
        isempty(sources) && error("""
            VisibilityProbe: $(config.project.name) has no sources to probe.

            Nothing was resolved from [dependencies] and [compile] source_files is
            empty, so there is no compilation to read visibility off.
            """)

        base  = vcat(ConfigurationManager.get_compile_flags(config), ["-fvisibility=hidden"])
        incs  = ["-I$d" for d in ConfigurationManager.get_include_dirs(config)]
        defs  = ["-D$k=$v" for (k, v) in config.compile.defines]

        total = 0
        exported = String[]
        failed = String[]
        compiled = 0

        scratch = mktempdir()
        try
            for (i, src) in enumerate(sources)
                is_c = endswith(src, ".c")
                # Same C/C++ split the real compile makes — a C++ -std= flag
                # applied to a C TU is an error, not a warning.
                flags = is_c ? filter(f -> !startswith(f, "-std=c++") &&
                                           !startswith(f, "-std=gnu++"), base) : base
                cc = is_c ? "clang" : "clang++"
                ll = joinpath(scratch, string(i, "_", basename(src), ".ll"))
                (_, rc) = _clang_for_c_bucket(cc, vcat(["-S", "-emit-llvm"], flags,
                                                       incs, defs, ["-o", ll, src]))
                if rc != 0 || !isfile(ll)
                    push!(failed, src)
                    continue
                end
                compiled += 1
                (t, e) = _scan_ir(ll)
                total += t
                append!(exported, e)
            end
        finally
            rm(scratch; recursive=true, force=true)
        end

        if compiled == 0
            error("""
            VisibilityProbe: $(config.project.name) — NOT ONE of $(length(sources)) sources compiled.

            This is an unmeasured library, NOT an unannotated one, and the two must
            not be confused: reading this as "annotates nothing" would rule out a
            library that annotates everything.

            The usual cause is a missing include root. A package whose generated
            headers are checked into `config/` lists it in [compile] include_dirs
            RELATIVE to the package directory (pcre2 is the type case) — so the
            probe must run from there, as it does, and the clone must be resolved.

            First failure: $(first(failed))
            """)
        end
        isempty(failed) || @warn "VisibilityProbe: $(length(failed)) of $(length(sources)) sources did not compile; the measurement is partial" first_failure=first(failed)

        exported = sort!(unique!(exported))

        # Compare against what the wrapper carries today, when there is one to read.
        # That turns an abstract ratio into the number the decision needs: what does
        # turning the flag on actually remove?
        wrapped = String[]
        md_path = metadata === nothing ?
                  joinpath(pkg_dir, "julia", "compilation_metadata.json") : metadata
        if isfile(md_path)
            try
                md = JSON.parsefile(md_path; use_mmap=false)
                fs = get(md, "functions", nothing)
                if fs isa Vector
                    for f in fs
                        f isa Dict || continue
                        sym = get(f, "mangled", "")
                        isempty(sym) && (sym = get(f, "name", ""))
                        isempty(sym) || push!(wrapped, sym)
                    end
                end
            catch e
                @debug "VisibilityProbe: metadata unreadable" exception=e
            end
        end
        unique!(wrapped)
        expset  = Set(exported)
        retained = sort!(filter(in(expset), wrapped))
        dropped  = sort!(filter(!in(expset), wrapped))

        library_exported = filter(!_is_shim, exported)

        return VisibilityReport(config.project.name, length(sources), compiled, failed,
                                total, exported, library_exported,
                                length(wrapped), retained, dropped,
                                isempty(library_exported) ? :unannotated : :annotated)
    end
end

# ============================================================================
# DISPLAY
# ============================================================================

function Base.show(io::IO, ::MIME"text/plain", r::VisibilityReport)
    shims = length(r.exported) - length(r.library_exported)
    pct = r.external_defs == 0 ? 0.0 :
          round(100 * length(r.library_exported) / r.external_defs, digits=1)
    println(io, "VisibilityReport: ", r.package)
    println(io, "  sources compiled   : ", r.compiled, "/", r.sources,
            isempty(r.failed) ? "" : "  ($(length(r.failed)) FAILED — partial)")
    println(io, "  external defs      : ", r.external_defs)
    println(io, "  library annotated  : ", length(r.library_exported), "  ($pct%)")
    shims > 0 && println(io, "  macro shims        : ", shims,
                         "  (always annotated — excluded from the verdict)")
    println(io, "  verdict            : ", r.verdict)
    if r.wrapped_today > 0
        println(io, "  wrapped today      : ", r.wrapped_today)
        println(io, "    retained         : ", length(r.retained))
        println(io, "    DROPPED          : ", length(r.dropped))
        if !isempty(r.dropped)
            for d in first(r.dropped, 6)
                println(io, "      → ", d)
            end
            length(r.dropped) > 6 && println(io, "      … and ",
                                             length(r.dropped) - 6, " more")
        end
    end
    println(io)
    if r.verdict === :unannotated
        println(io, "  Do NOT set -fvisibility=hidden for this package as configured.")
        println(io, "  Every definition would go hidden, promote_statics would rename the")
        println(io, "  whole API to __rb_*, and verify_wrap_surface would refuse the build.")
        println(io, "  Check whether upstream's export macro is gated behind a -D before")
        println(io, "  concluding the library annotates nothing (cJSON_API_VISIBILITY is")
        println(io, "  the type case) — adding that define can flip this verdict.")
    else
        println(io, "  -fvisibility=hidden is viable here. REVIEW THE DROPPED LIST above")
        println(io, "  before setting it: those are functions the wrapper carries today,")
        println(io, "  and the probe cannot tell an internal you are glad to lose from a")
        println(io, "  public function upstream forgot to annotate.")
    end
end

end # module VisibilityProbe
