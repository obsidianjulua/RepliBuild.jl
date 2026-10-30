# Builder/GraphBuild.jl — compile and link every target of a build graph with
# RepliBuild's own toolchain.
#
# The graph (`BuildGraph.TargetGraph`) is what the project's build system
# resolved: which targets exist, which sources each compiles under which flags,
# defines and include path, and the order. This module does not re-derive any
# of that. It compiles every translation unit under its own source's compile
# group, then links one library per library target, in `topological_order`,
# through the machinery a single-library build uses (`Compiler`): the per-file
# IR cache, `link_optimize_ir`, `create_library`. Wrapping and the TOML are a
# later step; this one leaves a directory per library holding what metadata
# extraction needs.
#
# What each target kind becomes:
#
# - `:shared`, `:module`: one library each, `<out>/<name>/lib/lib<name>.<dlext>`.
#   A library is linked against the libraries of the `:shared` targets it
#   reaches (through its `:object`, `:static` and `:interface` dependencies too,
#   and transitively through those shared targets' own), deps first, plus its
#   `link_libraries`. A `:module` is never linked against: CMake refuses that,
#   so a dependency on one only orders the build.
# - `:object`: compiled under its own groups, never linked on its own; its IR
#   folds into every library that links its objects. CMake says which by
#   listing the object files (the object target's `artifacts`) as sources of
#   the consumer; an object target with no artifacts (a graph written by hand)
#   folds wherever it is reached, as a static target does.
# - `:static`: folded into each library that reaches it through `:object`,
#   `:static` and `:interface` targets, unless it is behind a `:shared` or
#   `:module` target that library depends on: that library folded it, and a
#   second copy would split its state. Dependencies may be listed directly or,
#   as CMake's File API lists them, transitively; the answer is the same.
#   Not archived: one IR module per library keeps the facts the build reads off
#   the linked IR (`nounwind`, `byval`) and the DWARF covering every function
#   the library exports, which an archive linked as native objects would not.
#   The difference from the build system's own link is member selection: every
#   function of the static target lands in each library that folds it, not only
#   the ones it calls. A static target no library folds (a project whose only
#   library is static, or one only executables use) is linked as a library of
#   its own, since a wrapper needs one to load.
# - `:executable`, `:utility`, `:interface`: nothing is built. Utility targets
#   are the build's generators (`GraphGenerators` runs them); interface targets
#   only pass their dependencies and link inputs on to whoever depends on them.
#
# One LLVM per module: a library whose every translation unit is C (folded ones
# included) is built in the C bucket (Julia's JLL clang, in-process link);
# anything else in the C++ bucket (system clang for its C files too, external
# llvm-link/opt). An object or static target folded into libraries of both
# buckets is compiled once per bucket, under `<out>/<name>/ir/<bucket>/`.

module GraphBuild

using Dates: DateTime
using TOML
import SHA

import ..BuildGraph
import ..BuildGraph: TargetGraph, GraphTarget, CompileGroup, IncludeDir, GraphSource,
                     topological_order
import ..ConfigurationManager
import ..ConfigurationManager: RepliBuildConfig
import ..Compiler
import ..BuildBridge

export plan_graph, build_graph, GraphPlan, LibraryPlan, TUCompile, TargetResult

"The compiler driver for each language this build compiles."
const _COMPILERS = Dict("C" => "clang", "CXX" => "clang++")

"The `-x` spelling of each language this build compiles."
const _CLANG_LANGUAGE = Dict("C" => "c", "CXX" => "c++")

"""
Languages whose sources are left out of the library, and said so: Windows
resource scripts (`RC`) carry version information and no code, so there is no
IR to make of them. Every other language that is not C or C++ is refused.
"""
const _SKIPPED_LANGUAGES = ("RC",)

"""
    TUCompile

One translation unit compiled to IR: `source` under the compile group `group`
of the target `owner`, for a library in `bucket` (`:c` or `:cpp`). `args` is
the complete argument list for `compiler`, ending in `-o <ir_file> <source>`.
`include_dirs` are the group's include directories in order, whose headers the
cache key covers.
"""
struct TUCompile
    source::String
    owner::String
    group::Int
    language::String
    bucket::Symbol
    compiler::String
    args::Vector{String}
    ir_file::String
    include_dirs::Vector{String}
end

"""
    LibraryPlan

One library to link: target `id`/`name` of `kind`, built in `bucket` into
`library` under `dir`. `tus` are its own translation units, then those of each
target in `folded`, in topological order. `link_dependencies` are the ids of
the `:shared` targets it links against, deps first, found as `link_names` in
`link_dirs` (their files are `link_files`). `link_libraries` are its non-target link inputs as the graph
spells them, and `link_args` everything `create_library` gets on top of the
dependencies: relative rpaths or the import library, then `link_libraries` as
link-line arguments. `skipped_sources` are sources left out by language.
"""
struct LibraryPlan
    id::String
    name::String
    kind::Symbol
    bucket::Symbol
    dir::String
    library::String
    tus::Vector{TUCompile}
    folded::Vector{String}
    link_dependencies::Vector{String}
    link_names::Vector{String}
    link_dirs::Vector{String}
    link_files::Vector{String}
    link_libraries::Vector{String}
    link_args::Vector{String}
    include_dirs::Vector{String}
    system_include_dirs::Vector{String}
    defines::Vector{String}
    skipped_sources::Vector{String}
end

"""
    GraphPlan

What `build_graph` will do, decided without running a tool: the `libraries` to
link in topological order, every distinct translation-unit compile across them
(`compiles`), and, for each target in scope that builds no library of its own,
why (`notes`, `id => text`). `scope` is every target the plan covers, in
topological order.
"""
struct GraphPlan
    graph::TargetGraph
    out_dir::String
    scope::Vector{String}
    libraries::Vector{LibraryPlan}
    compiles::Vector{TUCompile}
    notes::Vector{Pair{String,String}}
end

"""
    TargetResult

What `build_graph` did with one target. `status` is `:built` (a library),
`:folded` (compiled into the libraries `folded_into`) or `:skipped` (`note`
says why). For a built library, `dir` holds it (`library`), its link
intermediates and `graph_target.toml`; `sources` and `ir_files` are every
translation unit linked into it, and `linked_ir` the module(s) the library was
linked from, which is where `nounwind` and `byval` facts are read.
"""
struct TargetResult
    id::String
    name::String
    kind::Symbol
    status::Symbol
    dir::String
    library::String
    bucket::Symbol
    sources::Vector{String}
    ir_files::Vector{String}
    linked_ir::Vector{String}
    link_dependencies::Vector{String}
    folded_into::Vector{String}
    note::String
end

_posix(p::AbstractString) = replace(String(p), '\\' => '/')

function _posix_abs(p::AbstractString)
    s = _posix(abspath(p))
    return (length(s) > 1 && endswith(s, '/') && !endswith(s, ":/")) ? s[1:end-1] : s
end

# ── Planning ─────────────────────────────────────────────────────────────────

"A path as a key: `/`-separated, and on Windows case-folded (`Compiler._canon_path`)."
_canon(p::AbstractString) = Compiler._canon_path(_posix(p))

"""
    _closures(order) -> Dict{String,Set{String}}

Every target each target depends on, directly or not. A graph may list
dependencies either way (CMake's File API lists them transitively; a graph
written by hand usually directly), and `_reach` gives the same answer for both.
"""
function _closures(order::Vector{GraphTarget})
    c = Dict{String,Set{String}}()
    for t in order
        s = Set{String}()
        for d in t.dependencies
            push!(s, d)
            union!(s, c[d])
        end
        c[t.id] = s
    end
    return c
end

"""
    _reach(byid, pos, closures, t) -> (; folded, linked, contributors)

What target `t` takes from the targets it depends on, each in topological order.

`linked`: every `:shared` target `t` is linked against and loaded after,
reached through anything but `:module`, `:executable` and `:utility` targets
(which only order the build), including what those shared targets link.

`folded`: the `:object` and `:static` targets whose IR goes into `t`'s library.
A static target is folded when it is reached through `:object`, `:static` and
`:interface` targets and is not behind a `:shared` or `:module` target `t`
depends on: that library folded it, and a second copy here would split its
state. An object target folds where the build system links its objects: into
a target that lists one of its `artifacts` as a source (as CMake reports a
consumer of an OBJECT library), `t` or a static target `t` folds. An object
target with no artifacts (a graph written by hand) folds as a static one does.

`contributors`: the folded and interface targets whose `link_libraries`
become `t`'s.
"""
function _reach(byid::AbstractDict, pos::AbstractDict, closures::AbstractDict, t::GraphTarget)
    behind = Set{String}()
    for b in closures[t.id]
        byid[b].kind in (:shared, :module) && union!(behind, closures[b])
    end
    candidates = Set{String}()
    interfaces = Set{String}()
    linked = Set{String}()
    seen = Set{Tuple{String,Bool}}()
    stack = Tuple{String,Bool}[(d, true) for d in t.dependencies]
    while !isempty(stack)
        (id, folding) = pop!(stack)
        (id, folding) in seen && continue
        push!(seen, (id, folding))
        d = byid[id]
        if d.kind in (:object, :static, :interface)
            f = folding && !(id in behind)
            f && push!(d.kind == :interface ? interfaces : candidates, id)
            append!(stack, ((x, f) for x in d.dependencies))
        elseif d.kind == :shared
            push!(linked, id)
            append!(stack, ((x, false) for x in d.dependencies))
        end
    end
    statics = [id for id in candidates if byid[id].kind == :static]
    listed = Set{String}()
    for u in vcat(t.id, statics), s in byid[u].sources
        s.group == 0 && push!(listed, _canon(s.path))
    end
    objects = String[]
    for id in closures[t.id]
        o = byid[id]
        o.kind == :object || continue
        keep = isempty(o.artifacts) ? id in candidates : any(a -> _canon(a) in listed, o.artifacts)
        keep && push!(objects, id)
    end
    bypos(s) = sort!(collect(s); by = id -> pos[id])
    folded = bypos(vcat(statics, objects))
    return (; folded, linked = bypos(linked), contributors = bypos(vcat(folded, collect(interfaces))))
end

"""
    _library_ids(order, byid, pos, closures, only) -> Vector{String}

The targets that become libraries, in topological order. With `only ===
nothing`: every `:shared` and `:module` target, and every `:static` target that
none of those (nor another such static target) folds. With `only` (names or
ids): those targets, which must be `:shared`, `:module` or `:static`, and the
`:shared` targets they link against.
"""
function _library_ids(order::Vector{GraphTarget}, byid::AbstractDict, pos::AbstractDict,
                      closures::AbstractDict, only)
    if only === nothing
        libs = [t.id for t in order if t.kind in (:shared, :module)]
        folded_somewhere = Set{String}()
        for id in libs
            union!(folded_somewhere, _reach(byid, pos, closures, byid[id]).folded)
        end
        # Dependents before dependencies, so a static target folded by a
        # standalone static target is not made a library of its own.
        for t in Iterators.reverse(order)
            (t.kind == :static && !(t.id in folded_somewhere)) || continue
            push!(libs, t.id)
            union!(folded_somewhere, _reach(byid, pos, closures, t).folded)
        end
        return sort!(libs; by = id -> pos[id])
    end
    wanted = only isa AbstractString ? [String(only)] : String[string(x) for x in only]
    isempty(wanted) && error("GraphBuild: `only` names no target")
    libs = Set{String}()
    for w in wanted
        i = findfirst(t -> t.id == w, order)
        i === nothing && (i = findfirst(t -> t.name == w, order))
        i === nothing && error("GraphBuild: `only` names '$w', which is not a target id or name in the graph")
        t = order[i]
        t.kind in (:shared, :module, :static) ||
            error("GraphBuild: `only` names '$(t.name)', a :$(t.kind) target, which builds no library; " *
                  "name the :shared, :module or :static target that uses it")
        push!(libs, t.id)
        union!(libs, _reach(byid, pos, closures, t).linked)
    end
    return sort!(collect(libs); by = id -> pos[id])
end

"""
    _compile_reads(t, s) -> Bool

Whether compiling `t` reads the source `s`: a translation unit in a language
this build compiles, or a header (by extension) the target lists. The other
generated entries a build system lists are not read here: an OBJECT library's
object files on its consumer (its IR is folded instead), CMake's `.rule`
files, an export `.def` the link writes.
"""
function _compile_reads(t::GraphTarget, s::GraphSource)
    s.group > 0 && return !(t.compile_groups[s.group].language in _SKIPPED_LANGUAGES)
    return occursin(r"\.(h|hh|hpp|hxx|h\+\+|inc|inl|ipp|tcc|tpp)$"i, s.path)
end

"A target name usable as a directory name on every host."
_valid_dir_name(name::AbstractString) =
    !isempty(name) && !(name in (".", "..")) && !occursin(r"[/\\:*?\"<>|\x00-\x1f]", name)

"""
    _has_full_debug_info(flags) -> Bool

Whether the last debug-level flag in `flags` asks for full debug info. DWARF is
where the wrapper's types come from, so a group without it (no `-g`, or `-g0`,
`-g1`, `-gline-tables-only`, … last) gets `-g` after its own flags.
"""
function _has_full_debug_info(flags::AbstractVector{<:AbstractString})
    for f in Iterators.reverse(flags)
        occursin(r"^-(g|g2|g3|ggdb|ggdb2|ggdb3|gdwarf(-\d+)?)$", f) && return true
        occursin(r"^-(g0|g1|ggdb0|ggdb1|gline-tables-only|gline-directives-only|gmlt)$", f) && return false
    end
    return false
end

"""
    _is_pic(flags) -> Bool

Whether the last relocation-model flag in `flags` is `-fPIC`/`-fpic`. Every
translation unit here lands in a shared library, but a build system compiles an
object or static target, and an executable-style default, without it.
"""
function _is_pic(flags::AbstractVector{<:AbstractString})
    for f in Iterators.reverse(flags)
        f in ("-fPIC", "-fpic") && return true
        f in ("-fPIE", "-fpie", "-fno-pic", "-fno-PIC", "-fno-pie", "-fno-PIE") && return false
    end
    return false
end

"""
    _compile_args(group, source, ir_file; windows) -> Vector{String}

The clang arguments that compile `source` to `ir_file` under `group`: the
group's flags in order, then `-g` and (off Windows) `-fPIC` where the flags do
not already end up there, the include path in order (`-I`, or `-isystem` for a
system directory), the defines, the group's language as `-x`, and the output
and input. `-x` makes the build system's language decide, not the extension.
"""
function _compile_args(grp::CompileGroup, source::String, ir_file::String; windows::Bool)
    args = String["-S", "-emit-llvm"]
    append!(args, grp.flags)
    _has_full_debug_info(grp.flags) || push!(args, "-g")
    (windows || _is_pic(grp.flags)) || push!(args, "-fPIC")
    for inc in grp.includes
        p = _posix(inc.path)
        inc.is_system ? push!(args, "-isystem", p) : push!(args, "-I" * p)
    end
    for d in grp.defines
        push!(args, "-D" * d)
    end
    push!(args, "-x", _CLANG_LANGUAGE[grp.language], "-o", ir_file, source)
    return args
end

"Where target `name`'s IR for `bucket` lives."
_ir_dir(out::String, name::String, bucket::Symbol) = string(out, "/", name, "/ir/", bucket)

"The library file of target `name`."
_library_file(out::String, name::String; windows::Bool) =
    string(out, "/", name, "/lib/lib", name, windows ? ".dll" : ".so")

_is_library_file(x::AbstractString) = occursin(r"\.(a|so|dll|lib|dylib|tbd)$|\.so\.\d", x)

"""
    _link_input(x) -> String

One `link_libraries` entry as a link-line argument: a flag (`-pthread`,
`-Wl,…`) and a path to a library file go as they are, a bare name as `-l<name>`.
"""
function _link_input(x::AbstractString)
    startswith(x, "-") && return String(x)
    (occursin('/', x) || occursin('\\', x) || _is_library_file(x)) && return String(x)
    return "-l" * x
end

"""
    _link_args(out, name, dep_names, link_libraries; windows) -> Vector{String}

What a library's link takes beyond its dependencies' `-L`/`-l` (which
`create_library` writes from `[link] link_dirs`/`link_libraries`, with an
absolute rpath for each on ELF). On ELF, an rpath relative to `\$ORIGIN` for
each dependency, so the output directory can move; on PE, which has no rpath,
an import library beside the DLL, which any MinGW linker can link against.
Then the target's own link inputs.
"""
function _link_args(out::String, name::String, dep_names::Vector{String},
                    link_libraries::Vector{String}; windows::Bool)
    args = String[]
    if windows
        push!(args, string("-Wl,--out-implib,", out, "/", name, "/lib/lib", name, ".dll.a"))
    else
        for d in dep_names
            push!(args, string("-Wl,-rpath,\$ORIGIN/../../", d, "/lib"))
        end
    end
    append!(args, _link_input.(link_libraries))
    return args
end

"""
    plan_graph(g, out_dir; only = nothing, windows = Sys.iswindows()) -> GraphPlan

Decide everything `build_graph` does without running a tool: which targets
become libraries (see the module comment), which translation units each links
and with which arguments each compiles, and what each library links against.
`only` limits the build to the named targets (names or ids) and the shared
targets they link against.

Refused, with an error naming the targets: a target whose `generated` sources
that a compile reads (`_compile_reads`) do not exist yet (run the build's
generators first), a source in a language other than C, C++ and RC, a library
with no translation unit, and two targets whose names are not distinct
directory names on every host.
"""
function plan_graph(g::TargetGraph, out_dir::AbstractString; only = nothing,
                    windows::Bool = Sys.iswindows())::GraphPlan
    out = _posix_abs(out_dir)
    order = topological_order(g)
    pos = Dict(t.id => i for (i, t) in enumerate(order))
    byid = Dict(t.id => t for t in order)

    closures = _closures(order)
    lib_ids = _library_ids(order, byid, pos, closures, only)
    reaches = Dict(id => _reach(byid, pos, closures, byid[id]) for id in lib_ids)

    # Every target that gets a directory: the libraries and what they fold.
    with_dir = Set{String}(lib_ids)
    for id in lib_ids
        union!(with_dir, reaches[id].folded)
    end
    dir_targets = [t for t in order if t.id in with_dir]
    bad = [t.name for t in dir_targets if !_valid_dir_name(t.name)]
    isempty(bad) || error("GraphBuild: target name(s) not usable as a directory name: " *
                          join(repr.(bad), ", "))
    by_lower = Dict{String,Vector{String}}()
    for t in dir_targets
        push!(get!(by_lower, lowercase(t.name), String[]), t.name)
    end
    clashes = sort!([join(v, " / ") for v in values(by_lower) if length(v) > 1])
    isempty(clashes) || error("GraphBuild: target names that are one directory on a case-folding " *
                              "file system: " * join(clashes, "; "))

    missing_generated = Pair{String,Vector{String}}[]
    unsupported = String[]
    for t in dir_targets
        gone = [_posix(s.path) for s in t.sources
                if s.generated && _compile_reads(t, s) && !isfile(s.path)]
        isempty(gone) || push!(missing_generated, t.name => gone)
        for s in t.sources
            s.group == 0 && continue
            lang = t.compile_groups[s.group].language
            (haskey(_COMPILERS, lang) || lang in _SKIPPED_LANGUAGES) && continue
            push!(unsupported, "$(t.name): $(_posix(s.path)) ($lang)")
        end
    end
    if !isempty(missing_generated)
        error("GraphBuild: $(length(missing_generated)) target(s) list generated sources that do not " *
              "exist yet. A step of the project's build writes them; run its generators first.\n" *
              join(["  $(n): " * join(v, ", ") for (n, v) in missing_generated], "\n"))
    end
    if !isempty(unsupported)
        error("GraphBuild: sources in a language RepliBuild does not compile (only C and C++ " *
              "become IR; RC is left out):\n  " * join(unsupported, "\n  "))
    end

    libraries = LibraryPlan[]
    compiles = TUCompile[]
    compiled = Set{String}()
    for id in lib_ids
        t = byid[id]
        r = reaches[id]
        members = vcat([t], [byid[x] for x in r.folded])
        units = Tuple{GraphTarget,GraphSource,CompileGroup}[]
        skipped = String[]
        for m in members, s in m.sources
            s.group == 0 && continue
            grp = m.compile_groups[s.group]
            if grp.language in _SKIPPED_LANGUAGES
                push!(skipped, _posix(s.path))
            else
                push!(units, (m, s, grp))
            end
        end
        isempty(units) && error("GraphBuild: target '$(t.name)' (:$(t.kind)) has no C or C++ " *
                                "translation unit, its own or folded, to link into a library")
        bucket = all(u -> u[3].language == "C", units) ? :c : :cpp

        tus = TUCompile[]
        seen_ir = Set{String}()
        include_dirs = String[]
        system_include_dirs = String[]
        defines = String[]
        for (m, s, grp) in units
            src = _posix(s.path)
            ir = _posix(Compiler._ir_output_path(_ir_dir(out, m.name, bucket), src))
            ir in seen_ir && continue   # one source listed twice
            push!(seen_ir, ir)
            tu = TUCompile(src, m.id, s.group, grp.language, bucket, _COMPILERS[grp.language],
                           _compile_args(grp, src, ir; windows), ir,
                           [_posix(i.path) for i in grp.includes])
            push!(tus, tu)
            if !(ir in compiled)
                push!(compiled, ir)
                push!(compiles, tu)
            end
            for i in grp.includes
                p = _posix(i.path)
                dest = i.is_system ? system_include_dirs : include_dirs
                p in dest || push!(dest, p)
            end
            for d in grp.defines
                d in defines || push!(defines, d)
            end
        end

        link_libraries = copy(t.link_libraries)
        for c in r.contributors, x in byid[c].link_libraries
            x in link_libraries || push!(link_libraries, x)
        end
        dep_names = [byid[d].name for d in r.linked]
        push!(libraries, LibraryPlan(
            t.id, t.name, t.kind, bucket, string(out, "/", t.name),
            _library_file(out, t.name; windows), tus, r.folded, r.linked, dep_names,
            [string(out, "/", n, "/lib") for n in dep_names],
            [_library_file(out, n; windows) for n in dep_names], link_libraries,
            _link_args(out, t.name, dep_names, link_libraries; windows),
            include_dirs, system_include_dirs, defines, skipped))
    end

    folded_into = Dict{String,Vector{String}}()
    for lib in libraries, f in lib.folded
        push!(get!(folded_into, f, String[]), lib.name)
    end
    scope = only === nothing ? [t.id for t in order] :
            [t.id for t in order if t.id in with_dir]
    notes = Pair{String,String}[]
    for id in scope
        id in lib_ids && continue
        t = byid[id]
        note = haskey(folded_into, id) ? "folded into " * join(folded_into[id], ", ") :
               t.kind == :executable ? "executables are not built" :
               t.kind == :utility ? "a utility target runs build steps; GraphGenerators runs those" :
               t.kind == :interface ? "an interface target compiles nothing" :
               "no library built here uses it"
        push!(notes, id => note)
    end
    return GraphPlan(g, out, scope, libraries, compiles, notes)
end

# ── Compiling ────────────────────────────────────────────────────────────────

"""
    _include_digest(dir) -> String

The SHA-256 of every header under `dir` (`Compiler._feed_include_headers!`).
Computed once per directory per build: many groups share one include path.
"""
function _include_digest(dir::AbstractString)::String
    ctx = SHA.SHA256_CTX()
    Compiler._key_feed!(ctx, "replibuild-graph-include-v1")
    Compiler._feed_include_headers!(ctx, dir)
    return bytes2hex(SHA.digest!(ctx))
end

"""
    _tu_fingerprint(tu, toolchain, include_digests) -> String

The compile fingerprint of one translation unit, written to its IR's `.key`
sidecar and checked by `Compiler.needs_recompile`: the compiler, every argument
but the output and input (so moving the output directory recompiles nothing),
the headers of each include directory, and the toolchain identity of its
bucket. SHA-256 through `_key_feed!`, as every persisted key.
"""
function _tu_fingerprint(tu::TUCompile, toolchain::AbstractString,
                         include_digests::AbstractDict)::String
    ctx = SHA.SHA256_CTX()
    Compiler._key_feed!(ctx, "replibuild-graph-tu-v1")
    Compiler._key_feed!(ctx, tu.compiler)
    for a in @view tu.args[1:end-3]   # all but `-o <ir_file> <source>`
        Compiler._key_feed!(ctx, a)
    end
    for d in tu.include_dirs
        Compiler._key_feed!(ctx, include_digests[d])
    end
    Compiler._key_feed!(ctx, toolchain)
    return bytes2hex(SHA.digest!(ctx))
end

"""
    _foreach_parallel(f, items; ntasks)

`f(x)` for every item, `ntasks` at a time. The tasks are spawned, so they spread
over every Julia thread, and each mostly waits on a compiler process, so they
overlap even on one thread.
"""
function _foreach_parallel(f, items; ntasks::Int = max(Threads.nthreads(), Sys.CPU_THREADS))
    isempty(items) && return nothing
    ch = Channel{eltype(items)}(length(items))
    foreach(x -> put!(ch, x), items)
    close(ch)
    @sync for _ in 1:min(ntasks, length(items))
        Threads.@spawn for x in ch
            f(x)
        end
    end
    return nothing
end

"""
    _compile_one(tu, fingerprint) -> (ok, exitcode, output)

Compile one translation unit. The C bucket goes through
`Compiler._clang_for_c_bucket` (Julia's JLL clang), the C++ bucket through
`BuildBridge.execute` (the system clang); either way the toolchain environment
rides on the command. A stale IR file and key are removed first, so a failed
compile cannot leave an old module that looks current.
"""
function _compile_one(tu::TUCompile, fingerprint::String)
    mkpath(dirname(tu.ir_file))
    rm(tu.ir_file; force = true)
    rm(Compiler._compile_key_path(tu.ir_file); force = true)
    (output, exitcode) = tu.bucket == :c ? Compiler._clang_for_c_bucket(tu.compiler, tu.args) :
                                           BuildBridge.execute(tu.compiler, tu.args)
    ok = exitcode == 0 && isfile(tu.ir_file)
    ok && Compiler.write_compile_key(tu.ir_file, fingerprint)
    return (ok, exitcode, output)
end

"""
    _compile_all(compiles; force = false) -> Int

Compile every translation unit whose IR is missing, older than its source, or
from another fingerprint, in parallel; returns how many were compiled. Every
failure is reported, after every compile has run, so one build shows them all;
the ones that succeeded are cached for the next.
"""
function _compile_all(compiles::Vector{TUCompile}; force::Bool = false)::Int
    isempty(compiles) && return 0
    toolchain = Dict(b => Compiler._toolchain_identity(b) for b in unique(tu.bucket for tu in compiles))
    dirs = unique(d for tu in compiles for d in tu.include_dirs)
    digests = Vector{String}(undef, length(dirs))
    Threads.@threads for i in eachindex(dirs)
        digests[i] = _include_digest(dirs[i])
    end
    include_digests = Dict(zip(dirs, digests))
    fps = [_tu_fingerprint(tu, toolchain[tu.bucket], include_digests) for tu in compiles]
    todo = [i for i in eachindex(compiles)
            if Compiler.needs_recompile(compiles[i].source, compiles[i].ir_file, !force, fps[i])]
    if isempty(todo)
        println("  compile: $(length(compiles)) cached")
        return 0
    end
    println("  compile: $(length(todo))/$(length(compiles)) translation units " *
            "($(length(compiles) - length(todo)) cached)")
    failures = Tuple{String,Int,String}[]
    lk = ReentrantLock()
    _foreach_parallel(todo) do i
        (ok, code, output) = _compile_one(compiles[i], fps[i])
        ok || lock(() -> push!(failures, (compiles[i].source, code, output)), lk)
    end
    if !isempty(failures)
        sort!(failures)
        detail = join(["  $(src) (exit $code)\n    " *
                       join(last(split(strip(output), '\n'), 12), "\n    ")
                       for (src, code, output) in first(failures, 10)], "\n")
        error("GraphBuild: $(length(failures)) of $(length(todo)) translation unit(s) failed to compile" *
              (length(failures) > 10 ? " (the first 10):" : ":") * "\n" * detail)
    end
    return length(todo)
end

# ── Linking ──────────────────────────────────────────────────────────────────

"""
    _library_config(lib; optimization_level = "2", fallback = false) -> RepliBuildConfig

The config `Compiler.link_optimize_ir` and `Compiler.create_library` link one
library with, built by the same `parse_*` functions a replibuild.toml goes
through: root `lib.dir`, link intermediates in `build/`, the library in `lib/`,
the bucket's language, its shared dependencies as `[link] link_dirs` and
`link_libraries` (which `create_library` turns into `-L`/`-l`, an rpath per
directory on ELF, and the DLLs it opens before the load check on Windows).
LTO bitcode and static promotion are off: they serve Tier 1, which graph builds
do not wire.
"""
function _library_config(lib::LibraryPlan; optimization_level::AbstractString = "2",
                         fallback::Bool = false)::RepliBuildConfig
    data = Dict{String,Any}(
        "project" => Dict{String,Any}("name" => lib.name, "root" => lib.dir,
                                      "uuid" => "00000000-0000-0000-0000-000000000000"),
        "paths" => Dict{String,Any}("output" => "lib", "build" => "build", "cache" => "cache"),
        "compile" => Dict{String,Any}("source_files" => [tu.source for tu in lib.tus],
                                      "include_dirs" => copy(lib.include_dirs),
                                      "flags" => String[]),
        "link" => Dict{String,Any}("optimization_level" => String(optimization_level),
                                   "enable_lto" => false, "promote_statics" => false,
                                   "fallback" => fallback,
                                   "link_libraries" => copy(lib.link_names),
                                   "link_dirs" => copy(lib.link_dirs)),
        "binary" => Dict{String,Any}("type" => "shared", "output_name" => basename(lib.library)),
        "wrap" => Dict{String,Any}("language" => lib.bucket == :c ? "c" : "cpp"),
        "cache" => Dict{String,Any}("enabled" => true, "directory" => "cache"),
    )
    CM = ConfigurationManager
    wrap = CM.parse_wrap_config(data)
    return RepliBuildConfig(
        CM.parse_project_config(data, lib.dir * "/replibuild.toml"),
        CM.parse_paths_config(data),
        CM.parse_discovery_config(data),
        CM.parse_compile_config(data),
        CM.parse_link_config(data, wrap.language),
        CM.parse_binary_config(data),
        wrap,
        CM.parse_llvm_config(data),
        CM.parse_workflow_config(data),
        CM.parse_cache_config(data),
        CM.parse_dependencies_config(data),
        CM.parse_types_config(data),
        nothing,
        "",
        DateTime(2000))
end

"""
    _link_levels(libraries) -> Vector{Vector{LibraryPlan}}

The libraries in waves: each after every library it links against. One wave
links in parallel.
"""
function _link_levels(libraries::Vector{LibraryPlan})
    level = Dict{String,Int}()
    for lib in libraries
        level[lib.id] = 1 + maximum((level[d] for d in lib.link_dependencies); init = 0)
    end
    n = maximum(values(level); init = 0)
    return [[lib for lib in libraries if level[lib.id] == k] for k in 1:n]
end

"""
    _link_one(lib; optimization_level, fallback) -> (library, linked_ir)

Link one library: `link_optimize_ir` over every translation unit's IR, then
`create_library` with the per-TU list as its source-location fallback and the
library's own link arguments.
"""
function _link_one(lib::LibraryPlan; optimization_level::AbstractString, fallback::Bool)
    config = _library_config(lib; optimization_level, fallback)
    # link_optimize_ir writes here and does not create it (compile_to_ir does,
    # in a single-library build); llvm-link with nowhere to write falls back.
    mkpath(ConfigurationManager.get_build_path(config))
    ir_files = [tu.ir_file for tu in lib.tus]
    linked = Compiler.link_optimize_ir(config, ir_files, lib.name)
    linked_ir = _posix.(linked isa String ? [linked] : linked)
    path = Compiler.create_library(config, linked, basename(lib.library);
                                   per_tu_ir = ir_files, extra_link_args = lib.link_args)
    return (_posix(path), linked_ir)
end

"""
    _write_manifest(lib, library, linked_ir)

`<dir>/graph_target.toml`: what metadata extraction and wrapping need to take
this library on alone (its sources, include path, defines, bucket and linked
IR) and what it was linked against. Keys sorted, so the file is a function of
the build.
"""
function _write_manifest(lib::LibraryPlan, library::String, linked_ir::Vector{String})
    data = Dict{String,Any}(
        "id" => lib.id, "name" => lib.name, "kind" => string(lib.kind),
        "bucket" => string(lib.bucket), "library" => library, "linked_ir" => linked_ir,
        "sources" => [tu.source for tu in lib.tus],
        "per_tu_ir" => [tu.ir_file for tu in lib.tus],
        "include_dirs" => lib.include_dirs, "system_include_dirs" => lib.system_include_dirs,
        "defines" => lib.defines, "folded" => lib.folded,
        "link_dependencies" => lib.link_dependencies,
        "link_dependency_libraries" => lib.link_files,
        "link_libraries" => lib.link_libraries, "skipped_sources" => lib.skipped_sources,
    )
    open(lib.dir * "/graph_target.toml", "w") do io
        TOML.print(io, data; sorted = true)
    end
    return nothing
end

"""
    build_graph(g, out_dir; only = nothing, optimization_level = "2",
                fallback = false, force = false) -> Vector{TargetResult}

Compile every translation unit of `g` that a library needs, in parallel, each
under its own compile group; then link one library per library target into
`<out_dir>/<name>/`, in topological order (libraries that do not depend on each
other link in parallel). Returns one result per target in scope, in
topological order. See `plan_graph` for `only` and for what is refused before
anything runs.

`optimization_level` is the link-time `opt` and codegen level, `fallback` the
C bucket's `[link] fallback`, and `force` recompiles every translation unit
regardless of the IR cache.
"""
function build_graph(g::TargetGraph, out_dir::AbstractString; only = nothing,
                     optimization_level::AbstractString = "2", fallback::Bool = false,
                     force::Bool = false)::Vector{TargetResult}
    start = time()
    plan = plan_graph(g, out_dir; only)
    println("RepliBuild graph | $(length(plan.libraries)) libraries, " *
            "$(length(plan.compiles)) translation units")
    _compile_all(plan.compiles; force)

    linked = Dict{String,Tuple{String,Vector{String}}}()
    for wave in _link_levels(plan.libraries)
        failures = Pair{String,String}[]
        lk = ReentrantLock()
        # On this thread, not spawned: the C bucket links and optimizes on
        # Julia's own libLLVM in this process. External links still overlap,
        # each task waiting on its own tools.
        asyncmap(wave; ntasks = Sys.CPU_THREADS) do lib
            try
                r = _link_one(lib; optimization_level, fallback)
                lock(() -> linked[lib.id] = r, lk)
            catch e
                lock(() -> push!(failures, lib.name => sprint(showerror, e)), lk)
            end
            nothing
        end
        if !isempty(failures)
            sort!(failures)
            error("GraphBuild: $(length(failures)) librar$(length(failures) == 1 ? "y" : "ies") " *
                  "failed to link:\n" * join(["  $n: $m" for (n, m) in failures], "\n"))
        end
    end
    for lib in plan.libraries
        _write_manifest(lib, linked[lib.id]...)
    end
    println("  done: $(round(time() - start, digits = 2))s")
    return _results(plan, linked)
end

function _results(plan::GraphPlan, linked::AbstractDict)
    libs = Dict(lib.id => lib for lib in plan.libraries)
    notes = Dict(plan.notes)
    out = TargetResult[]
    for id in plan.scope
        t = BuildGraph.target_by_id(plan.graph, id)
        if haskey(libs, id)
            lib = libs[id]
            (library, linked_ir) = linked[id]
            push!(out, TargetResult(id, t.name, t.kind, :built, lib.dir, library, lib.bucket,
                                    [tu.source for tu in lib.tus], [tu.ir_file for tu in lib.tus],
                                    linked_ir, lib.link_dependencies, String[], ""))
        else
            into = [lib.name for lib in plan.libraries if id in lib.folded]
            push!(out, TargetResult(id, t.name, t.kind, isempty(into) ? :skipped : :folded,
                                    "", "", :none, String[], String[], String[], String[],
                                    into, notes[id]))
        end
    end
    return out
end

end # module GraphBuild
