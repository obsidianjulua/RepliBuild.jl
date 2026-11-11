#!/usr/bin/env julia
# =============================================================================
# c_ccall_coverage — how much of each Hub C library a wrapper can call
# =============================================================================
#
# A HAND-RUN DEVELOPER TOOL, NOT A TEST. It reads the Hub (beside this repo, or
# the path given), which no test may do, and is wired into no suite. It builds
# and wraps nothing: it measures the wrappers and libraries already on disk, so
# their `BUILD_GENERATOR` is the version measured.
#
#     julia --project=. test/tools/c_ccall_coverage.jl [<hub>] [--all]
#
# For every C package in the Hub's index.toml (`--all`: every package whose
# replibuild.toml says `language = "c"`, indexed or not):
#
#   exported  the functions the shipped `julia/*.so` gives the loader: `nm -D
#             --defined-only`, types T/W/i, ELF version suffix cut, less
#             `__rb_*` (static promotion's aliases) and `_init`/`_fini`.
#   excluded  the exported names the recipe's `[wrap] exclude_symbols` drops,
#             matched with RepliBuild's own `_glob_matcher` as wrap matches
#             them (glfw's `_glfw*`: internal functions in dynsym because the
#             build cannot use -fvisibility=hidden).
#   covered   the exported, not excluded, names that are a `ccall` or `@ccall`
#             target somewhere in the wrapper's PARSED code: a docstring never
#             counts, an ABI trap stub calls nothing and never counts, and a
#             `[wrap.varargs]` `@ccall` counts.
#
# Each miss is labelled:
#   trap         the wrapper's stub says "cannot call '<name>'"
#   via sibling  the wrapper defines `<name>` and calls `<name>_` or
#                `replibuild_shim_<name>` instead (zlib's `gzgetc`, a macro
#                over `gzgetc_`): callable under its name, missed by symbol
#   varargs      variadic in the metadata and not declared
#   unwrapped    in the metadata, no call
#   not in metadata
#
# Prints a Markdown table per package, the totals (raw, net of exclusions, and
# with sibling-reached names counted), and the misses by label.
#
# ELF only (`nm -D`): a Windows Hub's libraries are PE, whose API is the export
# directory; this refuses to run there rather than measure the wrong table.
# =============================================================================

using TOML
using RepliBuild

Sys.iswindows() && error("c_ccall_coverage reads ELF dynamic symbols (nm -D); run it on Linux")

const ARGV = filter(a -> !startswith(a, "--"), ARGS)
const HUB = isempty(ARGV) ? normpath(joinpath(@__DIR__, "..", "..", "..", "RepliBuild-Hub")) : ARGV[1]
isfile(joinpath(HUB, "index.toml")) || error("no index.toml under $HUB: pass the Hub's path")

const INDEX = TOML.parsefile(joinpath(HUB, "index.toml"))
const INDEXED = Set(k for (k, v) in INDEX if get(v, "language", "") == "c")

function c_packages()
    pkgs = sort(collect(INDEXED))
    "--all" in ARGS || return pkgs
    for d in readdir(joinpath(HUB, "packages"))
        t = joinpath(HUB, "packages", d, "replibuild.toml")
        isfile(t) || continue
        lang = get(get(TOML.parsefile(t), "wrap", Dict()), "language", "")
        (lang == "c" && !(d in INDEXED)) && push!(pkgs, d)
    end
    return sort(pkgs)
end

function exported_functions(so)
    out = Set{String}()
    for l in eachline(`nm -D --defined-only $so`)
        f = split(l)
        length(f) >= 3 || continue
        f[2] in ("T", "W", "i") || continue
        n = String(first(split(f[3], '@')))
        (startswith(n, "__rb_") || n in ("_init", "_fini")) && continue
        push!(out, n)
    end
    return out
end

_callee(x) = x isa QuoteNode ? string(x.value) : x isa Symbol ? string(x) : x isa String ? x : nothing

# Every `ccall((:name, lib), …)` and `@ccall lib.name(…)::T` in the parsed code.
function call_targets!(s::Set{String}, ex)
    ex isa Expr || return s
    if ex.head == :call && ex.args[1] === :ccall && length(ex.args) >= 2
        t = ex.args[2]
        n = t isa Expr && t.head == :tuple ? _callee(t.args[1]) : _callee(t)
        n === nothing || push!(s, n)
    elseif ex.head == :macrocall && ex.args[1] === Symbol("@ccall")
        for a in ex.args[2:end]
            c = a isa Expr && a.head == :(::) ? a.args[1] : a
            if c isa Expr && c.head == :call
                f = c.args[1]
                n = f isa Expr && f.head == :. ? _callee(f.args[2]) : _callee(f)
                n === nothing || push!(s, n)
            end
        end
    end
    foreach(a -> call_targets!(s, a), ex.args)
    return s
end

function measure(p)
    dir = joinpath(HUB, "packages", p)
    toml = joinpath(dir, "replibuild.toml")
    wrapper = joinpath(dir, "julia", RepliBuild.ConfigurationManager.wrapper_module_name(toml) * ".jl")
    sos = isdir(joinpath(dir, "julia")) ?
          filter(f -> endswith(f, ".so"), readdir(joinpath(dir, "julia"); join = true)) : String[]
    (isfile(wrapper) && length(sos) == 1) ||
        return (; p, ok = false, note = "wrapper $(isfile(wrapper) ? "present" : "missing"), $(length(sos)) .so")

    src = read(wrapper, String)
    ast = Meta.parseall(src)
    any(e -> e isa Expr && e.head in (:incomplete, :error), ast.args) &&
        return (; p, ok = false, note = "wrapper does not parse")
    targets = call_targets!(Set{String}(), ast)

    all_exported = exported_functions(only(sos))
    patterns = String[String(x) for x in get(get(TOML.parsefile(toml), "wrap", Dict()), "exclude_symbols", String[])]
    (matches, _) = RepliBuild._glob_matcher(patterns)
    excluded = Set(n for n in all_exported if matches(n))
    expected = setdiff(all_exported, excluded)

    defined = Set(m.captures[1] for m in eachmatch(r"^function ([A-Za-z_][A-Za-z0-9_!]*)\("m, src))
    md = RepliBuild.Compiler._read_json(joinpath(dir, "julia", "compilation_metadata.json"))
    fns = get(md, "functions", [])
    vararg = Set(String(get(f, "name", "")) for f in fns if get(f, "is_vararg", false))
    known = Set(String(get(f, "name", "")) for f in fns)
    label(m) = (m in defined && (m * "_" in targets || "replibuild_shim_" * m in targets)) ? "via sibling" :
               occursin("cannot call '$m'", src) ? "trap" :
               m in vararg ? "varargs" :
               m in known ? "unwrapped" : "not in metadata"
    misses = [(m, label(m)) for m in sort(collect(setdiff(expected, targets)))]
    gen = match(r"BUILD_GENERATOR = \"([^\"]*)\"", src)
    return (; p, ok = true, hit = length(expected) - length(misses), expected = length(expected),
            excluded = length(excluded), misses, generator = gen === nothing ? "?" : gen.captures[1],
            indexed = p in INDEXED)
end

rows = map(measure, c_packages())
ok = [r for r in rows if r.ok]
pct(a, b) = b == 0 ? "—" : string(round(100a / b; digits = 2))

println("| Package | Covered / exported | % | Misses | Excluded by recipe | Generator |")
println("|---|---|---|---|---|---|")
for r in rows
    if !r.ok
        println("| $(r.p) | — | — | $(r.note) | | |")
        continue
    end
    note = r.indexed ? "" : " (not indexed)"
    println("| $(r.p)$note | $(r.hit)/$(r.expected) | $(pct(r.hit, r.expected)) | ",
            "$(r.expected - r.hit) | $(r.excluded == 0 ? "" : r.excluded) | $(r.generator) |")
end

hit = sum(r -> r.hit, ok; init = 0)
expected = sum(r -> r.expected, ok; init = 0)
excluded = sum(r -> r.excluded, ok; init = 0)
sibling = sum(r -> count(m -> m[2] == "via sibling", r.misses), ok; init = 0)
println()
println("packages measured: $(length(ok)) of $(length(rows))")
println("exported (raw):                $hit/$(expected + excluded) = $(pct(hit, expected + excluded))%")
println("net of recipe exclusions:      $hit/$expected = $(pct(hit, expected))%  ($(expected - hit) misses, $excluded excluded)")
println("sibling-reached counted:       $(hit + sibling)/$expected = $(pct(hit + sibling, expected))%  ($(expected - hit - sibling) misses)")

println("\nmisses by label:")
for r in ok, k in ("trap", "varargs", "unwrapped", "not in metadata", "via sibling")
    ms = [m for (m, l) in r.misses if l == k]
    isempty(ms) || println("  $(r.p) [$k] $(length(ms)): ", join(ms, ", "))
end
