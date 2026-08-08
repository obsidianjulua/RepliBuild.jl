"""
    RepliBuild.Debug

Static inspection of the code the JIT actually emitted.

Every function here reads **artifacts on disk** — no live process, no JIT engine,
no debugger, no breakpoint that has to resolve. That is the whole point: the
dynamic path (gdb on a running thunk) needs a process wedged at exactly the right
moment, and the static path needs a file. They answer the same questions.

Two artifacts, both under `<pkg>/.debug`, both written by the JIT:

  * `mlir/jlcs_<hash>.mlir` — the module text, written whenever a library's
    engine is initialized. This is the debugger's source file, not a dump.
  * `obj/<lib>.o` — the emitted object, written only when
    `REPLIBUILD_JIT_OBJDUMP` is set, because MLIR's object cache has to be
    requested before the engine exists and costs the memory to retain every
    emitted object for its lifetime.

The object carries the same DWARF the JIT registers with gdb, so
[`disassemble`](@ref) interleaves generated MLIR with machine code exactly as
`disassemble /s` does at a live prompt.

# Capturing

```julia
ENV["REPLIBUILD_JIT_OBJDUMP"] = "1"   # BEFORE the wrapper is loaded
using MyWrapper
```

The engine is created once per library per process and cached, so setting this
after the first Tier-2 call is too late — [`object_path`](@ref) will report the
object missing rather than silently returning stale bytes.
"""
module Debug

export debug_root, object_path, mlir_sources, has_object,
       thunks, mlir_body, disassemble, dwarf, walk

# `objdump`/`llvm-dwarfdump` come from the same binutils/LLVM install the build
# already requires; they are resolved at call time so a missing tool names
# itself rather than failing this module's load.
const _OBJDUMP  = "objdump"
const _DWARFDUMP = "llvm-dwarfdump"

const _CAPTURE_HINT = """
Set REPLIBUILD_JIT_OBJDUMP before the wrapper is loaded and re-run:

    REPLIBUILD_JIT_OBJDUMP=1 julia --project=. <script>

The object cache must exist when the engine is created, and engines are cached
per library per process — enabling it afterwards cannot recover the object."""

# =============================================================================
# Locating artifacts
# =============================================================================

"""
    debug_root(path) -> String

The `.debug` directory for a wrapped library.

Accepts anything a caller plausibly has in hand: the package directory, `"."`
while standing in it, the wrapper `.so`, the `.debug` directory, a directory
under it (`.debug/mlir`), or one of the artifacts themselves. Everything at or
below a `.debug` component resolves to that component — the first version only
matched `.debug` as an exact basename, so `.debug/mlir` re-appended and produced
`…/.debug/mlir/.debug/mlir`, an error that blamed a missing directory for a path
bug.
"""
function debug_root(path::AbstractString)
    p = abspath(isempty(path) ? "." : path)
    # At or under a `.debug` component: that component IS the answer. Covers the
    # directory, its subdirectories, and any file inside them.
    parts = splitpath(p)
    i = findlast(==(".debug"), parts)
    i === nothing || return joinpath(parts[1:i]...)
    if isfile(p)
        # A wrapper .so lives at <pkg>/julia/lib*.so, so its `.debug` is two
        # levels up; any other file is treated as a sibling of one.
        d = dirname(p)
        return basename(d) == "julia" ? joinpath(dirname(d), ".debug") :
                                        joinpath(d, ".debug")
    end
    return joinpath(p, ".debug")
end

"""
    object_path(path) -> String

Path to the emitted object for a wrapped library, or `""` when none was written.

Returns the single `.o` under `<pkg>/.debug/obj`. When a package somehow has
several, the newest wins — engines are per library, so more than one means an
older capture was left behind, and the stale one is never the answer.
"""
function object_path(path::AbstractString)
    dir = joinpath(debug_root(path), "obj")
    isdir(dir) || return ""
    objs = filter(f -> endswith(f, ".o"), readdir(dir; join=true))
    isempty(objs) && return ""
    length(objs) == 1 && return objs[1]
    return argmax(mtime, objs)
end

"""
    has_object(path) -> Bool

Whether an emitted object was captured for this library.
"""
has_object(path::AbstractString) = !isempty(object_path(path))

"""
    mlir_sources(path) -> Vector{String}

The generated MLIR files for a wrapped library, newest first.

These exist after any JIT initialization — unlike the object, they need no
environment variable, because the parser has to name the buffer regardless.
"""
function mlir_sources(path::AbstractString)
    dir = joinpath(debug_root(path), "mlir")
    isdir(dir) || return String[]
    srcs = filter(f -> endswith(f, ".mlir"), readdir(dir; join=true))
    return sort(srcs; by=mtime, rev=true)
end

# A missing artifact and a mistyped path produce the same empty directory, and
# the second is far more common at a REPL. Distinguish them: if the `.debug`
# root itself is absent, lead with the resolution rather than blaming the JIT.
function _path_hint(path::AbstractString, root::AbstractString)
    isdir(root) && return ""
    return """


        `.debug` does not exist here — check the path before the artifact:
        `$path` resolved to `$root`
        Accepted: a package directory, "." inside one, a wrapper .so, `.debug`,
        or anything under it."""
end

function _require_object(path::AbstractString)
    obj = object_path(path)
    if isempty(obj)
        root = debug_root(path)
        error("""
            No emitted object under $(joinpath(root, "obj")).

            $_CAPTURE_HINT$(_path_hint(path, root))""")
    end
    return obj
end

function _require_mlir(path::AbstractString)
    srcs = mlir_sources(path)
    if isempty(srcs)
        root = debug_root(path)
        error("""
            No generated MLIR under $(joinpath(root, "mlir")).

            This is written at JIT initialization, so an empty directory means the
            library's Tier-2 engine never came up in the process that last ran —
            or `clean()` removed it and nothing has re-run since.$(_path_hint(path, root))""")
    end
    return srcs[1]
end

# =============================================================================
# Reading them
# =============================================================================

"""
    thunks(path) -> Vector{String}

Every thunk symbol in the generated MLIR, sorted.

These are the names to break on in gdb and to pass as `symbol` to
[`disassemble`](@ref). Read from the MLIR rather than the object's symbol table
so this answers before any object has been captured.
"""
function thunks(path::AbstractString)
    src = _require_mlir(path)
    names = String[]
    for line in eachline(src)
        m = match(r"^func\.func @([A-Za-z0-9_]+_thunk)\b", line)
        m === nothing || push!(names, m.captures[1])
    end
    return sort!(unique!(names))
end

"""
    mlir_body(path, symbol) -> String

The generated MLIR for one thunk — the `func.func` block, verbatim.

This is what the machine code below it was lowered from, and reading the two
together is the point of [`walk`](@ref).
"""
function mlir_body(path::AbstractString, symbol::AbstractString)
    src = _require_mlir(path)
    lines = readlines(src)
    start = findfirst(l -> occursin(Regex("^func\\.func @\\Q$symbol\\E\\b"), l), lines)
    if start === nothing
        avail = thunks(path)
        error("""
            No thunk `$symbol` in $(basename(src)).
            $(length(avail)) available; closest by name:
              $(join(_closest(symbol, avail), "\n  "))""")
    end
    # The block ends at the first line that is exactly a closing brace, which is
    # how the generator emits it; nested regions are indented.
    stop = findnext(l -> l == "}", lines, start)
    stop === nothing && (stop = length(lines))
    return join(lines[start:stop], "\n")
end

_closest(want, have; n=5) =
    first(sort(have; by = h -> _dist(lowercase(want), lowercase(h))), min(n, length(have)))

# Cheap ranking, not a real edit distance: shared-prefix length dominates, and
# ties break on how far the lengths differ. Good enough to surface the symbol a
# typo meant, which is all this is for.
_dist(a, b) = -length(_common_prefix(a, b)) + abs(length(a) - length(b)) / 100
function _common_prefix(a, b)
    i = 0
    while i < min(ncodeunits(a), ncodeunits(b)) && codeunit(a, i + 1) == codeunit(b, i + 1)
        i += 1
    end
    return a[1:i]
end

"""
    disassemble(path; symbol="", source=true) -> String

Disassemble the emitted object, with the generated MLIR interleaved.

This is `disassemble /s` without the debugger. `symbol` restricts output to one
thunk (see [`thunks`](@ref)); omitted, the whole object comes back. `source=false`
drops the interleaving for a plain instruction listing.

Source resolution needs no working directory: the object's `DW_AT_comp_dir` is
absolute, so the `.mlir` is found from anywhere — including from a different
package's tree.
"""
function disassemble(path::AbstractString; symbol::AbstractString="", source::Bool=true)
    obj = _require_object(path)
    args = [source ? "-dS" : "-d"]
    isempty(symbol) || push!(args, "--disassemble=$symbol")
    push!(args, obj)
    out = try
        read(`$_OBJDUMP $args`, String)
    catch e
        error("`$_OBJDUMP` failed on $obj: $e")
    end
    # objdump prints the requested symbol's header even when it matched nothing,
    # so an unknown name yields a plausible-looking stub. Catch that here rather
    # than letting a caller read an empty listing as an empty function.
    if !isempty(symbol) && !occursin(symbol, out)
        error("`$symbol` is not in $(basename(obj)). Use `thunks(...)` to list what is.")
    end
    return out
end

"""
    dwarf(path; section="info") -> String

Dump a DWARF section of the emitted object via `llvm-dwarfdump`.

`section` is the suffix of the dwarfdump flag: `"info"`, `"line"`, `"abbrev"`,
and so on. `"line"` is the one that shows the address→MLIR-line table the
debugger and [`disassemble`](@ref) both read.
"""
function dwarf(path::AbstractString; section::AbstractString="info")
    obj = _require_object(path)
    try
        return read(`$_DWARFDUMP --debug-$section $obj`, String)
    catch e
        error("`$_DWARFDUMP --debug-$section` failed on $obj: $e")
    end
end

"""
    walk(path, symbol) -> String

One thunk, both views: the generated MLIR and the machine code it became.

The intended entry point for "what does this thunk actually do" — it needs only
a package path and a name, and it answers without starting anything.
"""
function walk(path::AbstractString, symbol::AbstractString)
    body = mlir_body(path, symbol)
    asm  = disassemble(path; symbol=symbol)
    return """
    ═══ MLIR ═══════════════════════════════════════════════════════════════
    $body

    ═══ EMITTED ════════════════════════════════════════════════════════════
    $asm"""
end

end # module Debug
