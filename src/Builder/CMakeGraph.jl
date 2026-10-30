# Builder/CMakeGraph.jl — read the build graph a configured CMake project resolved.
#
# CMake's configure step resolves everything RepliBuild needs to compile a
# multi-library project: which targets exist, what each compiles with which
# flags, defines and include path (inherited PUBLIC/INTERFACE usage included),
# what each links, what is generated during the build, and the order. Its File
# API writes that resolution to JSON when a query file asks for it. This module
# asks for `codemodel-v2`, runs the configure step (never the build), and reads
# the reply into the build-graph contract (`BuildGraph.TargetGraph`), the way
# DWARFParser reads types from what Clang resolved rather than from a source AST.
#
# The reply's shapes, as CMake 4.4 writes them (checked against a live reply;
# `test/tools/gen_cmake_graph_fixtures.jl` regenerates the vendored ones):
#
# - A `compileCommandFragments` entry is a FRAGMENT of the command line in the
#   generator's native shell encoding, not one argument: `CMAKE_<LANG>_FLAGS_<CONFIG>`
#   arrives whole (`-O3 -DNDEBUG`), and `CMAKE_<LANG>_STANDARD_LIBRARIES` is one
#   `libraries` fragment of ten `-l` flags on MinGW. The contract's `flags` and
#   `link_libraries` hold arguments, so each fragment is split by the rules of
#   the shell that would have run it (`_split_fragment`), in order, and never
#   joined with its neighbours.
# - An OBJECT library linked into another target appears in that target's
#   `sources` as its object files (generated, no compile group), each the same
#   file as one of the object library's `artifacts`, plus a `dependencies` edge.
#   Its objects are never a link fragment.
# - A custom command's output listed as a target's source is a generated source
#   of that target, beside a generated `<output>.rule` source the command hangs
#   on. The command line itself is not in the codemodel.
# - Source paths inside the top-level source directory are relative to it;
#   artifact paths inside the top-level build directory are relative to that;
#   a relative path in a link fragment is relative to the directory the link
#   runs in (the top build dir under Ninja, the target's build dir otherwise).
#   Paths can carry `./` segments (`CMakeFiles/helpers.dir/./x.c.obj` as an
#   artifact, `CMakeFiles/helpers.dir/x.c.obj` as the consumer's source), so
#   every path is normalised before it is compared or stored.
# - `dependencies` is transitive, and its order is not stable between two
#   configures of one project, so it is stored sorted.

module CMakeGraph

import .._read_json
using ..BuildBridge: execute
using ..BuildGraph: IncludeDir, CompileGroup, GraphSource, GraphTarget, TargetGraph

export cmake_graph, read_cmake_graph

# ============================================================================
# PATHS
# ============================================================================

# Every path this module stores is absolute and '/'-separated (the contract's
# rule). Host-conditional for the reason `SysConfigGen._posix` gives: a
# backslash is a legal character in a POSIX file name.
_posix(p::AbstractString) = Sys.iswindows() ? replace(String(p), '\\' => '/') : String(p)

# Absolute in either host's spelling, whichever host reads it: a reply written
# on Windows says `C:/…`, and the vendored fixtures are read on Linux too.
_is_abs(p::AbstractString) = startswith(p, "/") || occursin(r"^[A-Za-z]:/", p)

# Lexical normalisation of a '/'-separated path: drop `.` and empty segments,
# fold `..` into its parent. Pure string work, so it answers the same on every
# host (`normpath` answers in the host's separator).
function _normalize(p::AbstractString)::String
    s = _posix(p)
    m = match(r"^(//|[A-Za-z]:/|/)", s)
    root = m === nothing ? "" : String(m.match)
    parts = String[]
    for seg in split(s[ncodeunits(root)+1:end], '/')
        (isempty(seg) || seg == ".") && continue
        if seg == ".."
            if !isempty(parts) && parts[end] != ".."
                pop!(parts)
                continue
            end
            isempty(root) || continue      # `/..` is `/`
        end
        push!(parts, String(seg))
    end
    out = root * join(parts, "/")
    return isempty(out) ? "." : out
end

# `p` as an absolute path: as written when it already is one, else against `base`.
_abs(p::AbstractString, base::AbstractString) =
    (q = _posix(p); _normalize(_is_abs(q) ? q : string(base, "/", q)))

# ============================================================================
# FRAGMENTS
# ============================================================================

# Whether the reply's fragments are in Windows command-line encoding. CMake
# encodes for the shell that runs the build step: Windows rules for Ninja,
# MinGW Makefiles and the IDE generators on a Windows host, POSIX rules for
# MSYS and Unix Makefiles and for every generator elsewhere. The reply is read
# on the host that configured it, except for vendored test fixtures, which
# the tests choose explicitly.
_windows_shell(generator::AbstractString) =
    Sys.iswindows() && !occursin("MSYS", generator) && !occursin("Unix", generator)

_split_fragment(s::AbstractString, windows::Bool) =
    windows ? _split_windows(s) : _split_posix(s)

# Windows command-line rules (`CommandLineToArgvW`): blanks separate outside
# quotes, `"` toggles quoting, and a backslash is literal unless a run of them
# ends at a `"`: 2n backslashes then `"` give n backslashes and a toggle, 2n+1
# give n backslashes and a literal `"`. A path written with `\` survives whole.
function _split_windows(s::AbstractString)::Vector{String}
    args = String[]
    buf = IOBuffer()
    inquote = false
    started = false
    cs = collect(s)
    i, n = 1, length(cs)
    while i <= n
        c = cs[i]
        if c == '\\'
            j = i
            while j <= n && cs[j] == '\\'
                j += 1
            end
            nb = j - i
            if j <= n && cs[j] == '"'
                print(buf, '\\'^(nb ÷ 2))
                if isodd(nb)
                    print(buf, '"')
                    j += 1
                end
            else
                print(buf, '\\'^nb)
            end
            started = true
            i = j
        elseif c == '"'
            inquote = !inquote
            started = true
            i += 1
        elseif (c == ' ' || c == '\t') && !inquote
            started && push!(args, String(take!(buf)))
            started = false
            i += 1
        else
            print(buf, c)
            started = true
            i += 1
        end
    end
    inquote && error("CMakeGraph: unterminated quote in command fragment: $s")
    started && push!(args, String(take!(buf)))
    return args
end

# POSIX shell word rules, which is what CMake writes for a POSIX shell: blanks
# separate, `\x` is `x` outside quotes, `'…'` is literal, and inside `"…"` a
# backslash escapes only `\`, `"`, `$` and a backquote. No expansion happens:
# CMake escapes `$` itself, so a `$` that reaches here is literal.
function _split_posix(s::AbstractString)::Vector{String}
    args = String[]
    buf = IOBuffer()
    started = false
    cs = collect(s)
    i, n = 1, length(cs)
    while i <= n
        c = cs[i]
        if c == ' ' || c == '\t' || c == '\n'
            started && push!(args, String(take!(buf)))
            started = false
            i += 1
        elseif c == '\\'
            i < n && print(buf, cs[i+1])
            started = true
            i += 2
        elseif c == '\''
            j = findnext(==('\''), cs, i + 1)
            j === nothing && error("CMakeGraph: unterminated quote in command fragment: $s")
            print(buf, String(cs[i+1:j-1]))
            started = true
            i = j + 1
        elseif c == '"'
            i += 1
            closed = false
            while i <= n
                d = cs[i]
                if d == '"'
                    closed = true
                    i += 1
                    break
                elseif d == '\\' && i < n && cs[i+1] in ('\\', '"', '$', '`')
                    print(buf, cs[i+1])
                    i += 2
                else
                    print(buf, d)
                    i += 1
                end
            end
            closed || error("CMakeGraph: unterminated quote in command fragment: $s")
            started = true
        else
            print(buf, c)
            started = true
            i += 1
        end
    end
    started && push!(args, String(take!(buf)))
    return args
end

# ============================================================================
# REPLY
# ============================================================================

# The File API's `type` spellings. A lookup table only; never iterated.
const _KINDS = Dict(
    "SHARED_LIBRARY"    => :shared,
    "STATIC_LIBRARY"    => :static,
    "OBJECT_LIBRARY"    => :object,
    "MODULE_LIBRARY"    => :module,
    "EXECUTABLE"        => :executable,
    "UTILITY"           => :utility,
    "INTERFACE_LIBRARY" => :interface,
)

# The current reply index: of every `index-*.json` and `error-*.json`, the one
# with the largest name once the prefix is cut (the File API's rule). When that
# is an error index, the last configure failed and its replies are older than
# the build tree, so nothing is read.
function _reply_index(reply_dir::String)::String
    isdir(reply_dir) || error("read_cmake_graph: no File API reply at $reply_dir. The build " *
                              "directory was not configured with a codemodel-v2 query " *
                              "(`cmake_graph` writes one before it configures).")
    names = filter(f -> occursin(r"^(index|error)-.+\.json$", f), readdir(reply_dir))
    isempty(names) && error("read_cmake_graph: $reply_dir holds no reply index")
    latest = names[argmax(map(f -> replace(f, r"^(index|error)-" => ""), names))]
    startswith(latest, "error-") &&
        error("read_cmake_graph: the last configure of this build directory failed " *
              "($reply_dir/$latest); fix the configure and run it again")
    return joinpath(reply_dir, latest)
end

# The codemodel-v2 object the index names: the reply to the shared query file
# `codemodel-v2`, else any codemodel object of major version 2 it lists (a
# client query, which this module does not write, answers that way).
function _codemodel_entry(index::Dict, index_path::String)::Dict
    entry = get(get(index, "reply", Dict()), "codemodel-v2", nothing)
    if entry === nothing
        for o in get(index, "objects", Any[])
            if get(o, "kind", "") == "codemodel" && get(get(o, "version", Dict()), "major", 0) == 2
                entry = o
                break
            end
        end
    end
    entry === nothing && error("read_cmake_graph: $index_path answers no codemodel-v2 query")
    haskey(entry, "error") &&
        error("read_cmake_graph: cmake could not answer codemodel-v2: $(entry["error"])")
    major = get(get(entry, "version", Dict()), "major", 0)
    major == 2 || error("read_cmake_graph: codemodel version $major, expected 2")
    return entry
end

function _configuration(cm::Dict, config::String)::Dict
    cfgs = get(cm, "configurations", Any[])
    names = [String(c["name"]) for c in cfgs]
    shown = isempty(names) ? "none" : join(map(repr, names), ", ")
    if isempty(config)
        length(cfgs) == 1 && return only(cfgs)
        error("read_cmake_graph: the build directory has $(length(cfgs)) configurations " *
              "($shown); name one with `config`")
    end
    i = findfirst(==(config), names)
    i === nothing && error("read_cmake_graph: no configuration $(repr(config)); available: $shown")
    return cfgs[i]
end

function _compile_group(g::Dict, source_dir::String, windows::Bool)::CompileGroup
    flags = String[]
    for f in get(g, "compileCommandFragments", Any[])
        append!(flags, _split_fragment(f["fragment"], windows))
    end
    defines = String[String(d["define"]) for d in get(g, "defines", Any[])]
    includes = IncludeDir[IncludeDir(_abs(i["path"], source_dir), get(i, "isSystem", false) === true)
                          for i in get(g, "includes", Any[])]
    return CompileGroup(String(g["language"]), flags, defines, includes)
end

# The `libraries` arguments of a target's link step that are not another
# target's artifact. A path argument is made absolute against the directory
# the link runs in; a flag (`-lfoo`, `-Wl,…`) and a bare file name (a library
# by name) are kept as written.
function _link_libraries(t::Dict, link_dir::String, windows::Bool,
                         artifacts::Set{String})::Vector{String}
    link = get(t, "link", nothing)
    link === nothing && return String[]
    libs = String[]
    for f in get(link, "commandFragments", Any[])
        get(f, "role", "") == "libraries" || continue
        for a in _split_fragment(f["fragment"], windows)
            if startswith(a, "-")
                push!(libs, a)
                continue
            end
            p = _abs(a, link_dir)
            p in artifacts && continue
            q = _posix(a)
            push!(libs, (_is_abs(q) || occursin('/', q)) ? p : a)
        end
    end
    return libs
end

"""
    read_cmake_graph(build_dir; config="", windows_shell=nothing) -> BuildGraph.TargetGraph

Read the File API `codemodel-v2` reply of a CMake build directory that was
configured with that query (see [`cmake_graph`](@ref)) into a `TargetGraph`.

`config` names the configuration to read. A single-config generator writes
one (named after `CMAKE_BUILD_TYPE`, possibly `""`) and it is taken without
asking; a multi-config generator writes several, and reading one without
naming it is an error that lists them.

Targets come in the reply's order. Each target's `flags` and `link_libraries`
are arguments, split out of the reply's command fragments by the rules of the
shell that runs the build; `link_libraries` keeps only the `libraries` role and
drops any argument that is an artifact of a target in the graph. Every path is
absolute and `/`-separated, resolved the way the File API documents it.
`source_dir` and `build_dir` of the result are the ones the reply records.

`windows_shell` overrides which encoding the fragments are split by. Left
`nothing`, it is the one the reading host and the reply's generator imply,
which is right whenever the reply is read where it was configured; a reply
carried to another host (a vendored test fixture) says which it was written in.
"""
function read_cmake_graph(build_dir::AbstractString; config::AbstractString="",
                          windows_shell::Union{Nothing,Bool}=nothing)::TargetGraph
    reply_dir = joinpath(String(build_dir), ".cmake", "api", "v1", "reply")
    index_path = _reply_index(reply_dir)
    index = _read_json(index_path)
    entry = _codemodel_entry(index, index_path)
    cm_path = joinpath(dirname(index_path), entry["jsonFile"])
    cm = _read_json(cm_path)

    paths = cm["paths"]
    src_root = _normalize(paths["source"])
    bld_root = _normalize(paths["build"])
    generator = String(get(get(get(index, "cmake", Dict()), "generator", Dict()), "name", ""))
    windows = windows_shell === nothing ? _windows_shell(generator) : windows_shell
    ninja = occursin("Ninja", generator)

    cfg = _configuration(cm, String(config))
    refs = get(cfg, "targets", Any[])
    raw = Dict{String,Any}[_read_json(joinpath(dirname(cm_path), r["jsonFile"])) for r in refs]

    # Every artifact first: a link argument is dropped when it names one of
    # any target's, which may come later in the list.
    artifacts = Set{String}()
    for t in raw, a in get(t, "artifacts", Any[])
        push!(artifacts, _abs(a["path"], bld_root))
    end

    targets = GraphTarget[]
    for t in raw
        name = String(t["name"])
        type = String(t["type"])
        kind = get(_KINDS, type, nothing)
        kind === nothing && error("read_cmake_graph: target '$name' has type $type, " *
                                  "which the build graph has no kind for")
        groups = CompileGroup[_compile_group(g, src_root, windows) for g in get(t, "compileGroups", Any[])]
        sources = GraphSource[GraphSource(_abs(s["path"], src_root),
                                          haskey(s, "compileGroupIndex") ? Int(s["compileGroupIndex"]) + 1 : 0,
                                          get(s, "isGenerated", false) === true)
                              for s in get(t, "sources", Any[])]
        # Sorted: CMake lists them in an order that moves between two configures
        # of one project (helpers, core, ext_version, then core, ext_version,
        # helpers), and a graph must be a function of the build.
        deps = sort!(unique!(String[String(d["id"]) for d in get(t, "dependencies", Any[])]))
        target_build = _abs(get(get(t, "paths", Dict()), "build", "."), bld_root)
        libs = _link_libraries(t, ninja ? bld_root : target_build, windows, artifacts)
        arts = String[_abs(a["path"], bld_root) for a in get(t, "artifacts", Any[])]
        push!(targets, GraphTarget(String(t["id"]), name, kind, sources, groups, deps, libs, arts))
    end
    return TargetGraph(src_root, bld_root, targets)
end

# ============================================================================
# CONFIGURE
# ============================================================================

"""
    cmake_graph(source_dir; build_dir, generator="Ninja", options=String[],
                build_type="Release", policy_floor=true, use_llvm_env=true,
                config="") -> BuildGraph.TargetGraph

Configure `source_dir` into `build_dir` with a File API `codemodel-v2` query
and read the reply ([`read_cmake_graph`](@ref)). Only the configure step runs;
nothing is built, and `build_dir` is where the project's build-time generated
files would land.

The configure runs the way `SysConfigGen.cmake_probe` runs it: in RepliBuild's
toolchain environment (`use_llvm_env`), so CMake identifies the clang that will
compile the sources, with `-DCMAKE_BUILD_TYPE=<build_type>` (omitted when
empty) and, with `policy_floor`, `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`, which
cmake 4.x needs for projects that declare an older minimum. `options` come
last, so they override both. An existing `build_dir` is reconfigured in place,
never deleted.
"""
function cmake_graph(source_dir::AbstractString;
                     build_dir::AbstractString,
                     generator::String="Ninja",
                     options::Vector{String}=String[],
                     build_type::String="Release",
                     policy_floor::Bool=true,
                     use_llvm_env::Bool=true,
                     config::String="")::TargetGraph
    src = String(rstrip(_posix(abspath(source_dir)), '/'))
    isfile(joinpath(src, "CMakeLists.txt")) ||
        error("cmake_graph: no CMakeLists.txt in $src: not a cmake project")
    bld = String(rstrip(_posix(abspath(build_dir)), '/'))

    query_dir = joinpath(bld, ".cmake", "api", "v1", "query")
    mkpath(query_dir)
    touch(joinpath(query_dir, "codemodel-v2"))

    args = String["-S", src, "-B", bld, "-G", generator]
    isempty(build_type) || push!(args, "-DCMAKE_BUILD_TYPE=$build_type")
    policy_floor && push!(args, "-DCMAKE_POLICY_VERSION_MINIMUM=3.5")
    append!(args, options)

    output, exitcode = execute("cmake", args; use_llvm_env=use_llvm_env)
    if exitcode != 0
        tail = join(last(split(output, '\n'), 25), '\n')
        error("cmake_graph: configure failed for $src (exit $exitcode).\n$tail")
    end
    return read_cmake_graph(bld; config=config)
end

end # module CMakeGraph
