#!/usr/bin/env julia
# =============================================================================
# Regenerate test/fixtures/cmake_graph/ — vendored CMake File API replies
# =============================================================================
#
# THIS IS A DEVELOPER TOOL, NOT A TEST. It is wired into no suite. It needs
# cmake, ninja and a C/C++ compiler (RepliBuild's toolchain environment), and
# with `--curl` the network or an existing checkout. Run it by hand after a
# change to `RepliBuild.CMakeGraph` or to the fixture project, then commit the
# fixtures:
#
#     julia --project=. test/tools/gen_cmake_graph_fixtures.jl
#     julia --project=. test/tools/gen_cmake_graph_fixtures.jl --curl
#     julia --project=. test/tools/gen_cmake_graph_fixtures.jl --curl=<checkout>
#
# `graphfix` configures test/fixtures/cmake_graph/project (a PUBLIC include
# inherited through a link edge, a target-private define, an OBJECT library
# linked into a shared one, a source and a header written by custom commands,
# a link library that is no target). `--curl` also configures libcurl at the
# commit the Hub's curl recipe pins, cloned into a temporary directory (or the
# checkout named, whose HEAD must be that commit): its `libcurl_shared`
# compiles almost nothing and links the ~190 objects of `libcurl_object`, the
# edge compile_commands.json cannot express (`SysConfigGen._object_library_hint`).
#
# Each fixture is the reply trimmed to what `read_cmake_graph` reads, with
# stable file names, sorted keys, and the configured source and build roots
# written as `@SOURCE@` and `@BUILD@` (the test substitutes its own). curl's is
# also trimmed to the targets `libcurl_shared` depends on. Before writing, the
# tool reads the trimmed fixture back through `read_cmake_graph` and refuses
# to write when that differs from reading the live reply, when a machine path
# survived the substitution, or when the fixture no longer shows the shapes
# test_cmake_graph.jl asserts on (a vacuous fixture).

using RepliBuild
import JSON
import TOML

const CG = RepliBuild.CMakeGraph
const BG = RepliBuild.BuildGraph
const TEST_DIR = dirname(@__DIR__)   # this file lives in test/tools/
const OUT = joinpath(TEST_DIR, "fixtures", "cmake_graph")

# The Hub's packages/curl/replibuild.toml pins this tag and commit.
const CURL_URL = "https://github.com/curl/curl.git"
const CURL_TAG = "curl-8_21_0"
const CURL_COMMIT = "68720b4837284335b2d63cb358f8f6ce65f5bc55"
# Lean configure: no tool, tests, docs or optional dependencies. SHARE_LIB_OBJECT
# is ON by default only on Windows; set so the edge exists on every host.
const CURL_OPTIONS = String[
    "-DBUILD_SHARED_LIBS=ON", "-DBUILD_STATIC_LIBS=ON", "-DSHARE_LIB_OBJECT=ON",
    "-DBUILD_CURL_EXE=OFF", "-DBUILD_TESTING=OFF", "-DBUILD_EXAMPLES=OFF",
    "-DBUILD_LIBCURL_DOCS=OFF", "-DBUILD_MISC_DOCS=OFF", "-DENABLE_CURL_MANUAL=OFF",
    "-DCURL_USE_LIBPSL=OFF", "-DUSE_LIBIDN2=OFF", "-DCURL_ZLIB=OFF", "-DCURL_BROTLI=OFF",
    "-DCURL_ZSTD=OFF", "-DUSE_NGHTTP2=OFF", "-DCURL_USE_LIBSSH2=OFF",
    "-DCURL_DISABLE_LDAP=ON", "-DCURL_ENABLE_SSL=OFF",
]

refuse(name, why) = error("$name: refusing to write a vacuous fixture: $why")

# ── trimming ─────────────────────────────────────────────────────────────────
# A spec maps each kept key to `nothing` (keep the value whole) or to a spec
# applied to the value (a dict, or each element of an array).

pick(x, ::Nothing) = x
pick(x::AbstractVector, spec::Dict) = Any[pick(e, spec) for e in x]
function pick(x::AbstractDict, spec::Dict)
    out = Dict{String,Any}()
    for (k, sub) in spec
        haskey(x, k) && (out[k] = pick(x[k], sub))
    end
    return out
end

const TARGET_SPEC = Dict{String,Any}(
    "id" => nothing, "name" => nothing, "type" => nothing, "paths" => nothing,
    "artifacts" => Dict{String,Any}("path" => nothing),
    "dependencies" => Dict{String,Any}("id" => nothing),
    "sources" => Dict{String,Any}("path" => nothing, "compileGroupIndex" => nothing,
                                  "isGenerated" => nothing),
    "compileGroups" => Dict{String,Any}(
        "language" => nothing,
        "compileCommandFragments" => Dict{String,Any}("fragment" => nothing),
        "defines" => Dict{String,Any}("define" => nothing),
        "includes" => Dict{String,Any}("path" => nothing, "isSystem" => nothing)),
    "link" => Dict{String,Any}("commandFragments" => Dict{String,Any}("fragment" => nothing,
                                                                       "role" => nothing)),
)

# The reader reads only `libraries` fragments; the rest can hold machine paths
# in the host's spelling (curl's `libcurl.def` arrives with backslashes).
# `dependencies` is written sorted: CMake's order moves between two configures
# of one project, and the reader sorts it anyway.
function trim_target(t)
    out = pick(t, TARGET_SPEC)
    if haskey(out, "link")
        frags = get(out["link"], "commandFragments", Any[])
        out["link"]["commandFragments"] = filter(f -> get(f, "role", "") == "libraries", frags)
    end
    haskey(out, "dependencies") && sort!(out["dependencies"]; by = d -> d["id"])
    return out
end

# ── deterministic JSON ───────────────────────────────────────────────────────
# Sorted keys, and a dict or array of scalars on one line, so a regenerated
# fixture diffs by content and stays small.

scalar(x) = !(x isa AbstractDict || x isa AbstractVector)

function jstr(io, s::AbstractString)
    print(io, '"')
    for c in s
        if c == '"' || c == '\\'
            print(io, '\\', c)
        elseif c < ' '
            print(io, "\\u", string(UInt16(c); base=16, pad=4))
        else
            print(io, c)
        end
    end
    print(io, '"')
end

function jwrite(io, x, ind::Int=0)
    if x isa AbstractString
        jstr(io, x)
    elseif x isa Bool || x isa Integer
        print(io, x)
    elseif x === nothing
        print(io, "null")
    elseif x isa AbstractDict
        ks = sort!(collect(String, keys(x)))
        if isempty(ks)
            print(io, "{}")
        elseif all(k -> scalar(x[k]), ks)
            print(io, "{")
            for (i, k) in enumerate(ks)
                i > 1 && print(io, ", ")
                jstr(io, k); print(io, ": "); jwrite(io, x[k])
            end
            print(io, "}")
        else
            println(io, "{")
            for (i, k) in enumerate(ks)
                print(io, " "^(ind + 2)); jstr(io, k); print(io, ": "); jwrite(io, x[k], ind + 2)
                println(io, i < length(ks) ? "," : "")
            end
            print(io, " "^ind, "}")
        end
    elseif x isa AbstractVector
        if isempty(x)
            print(io, "[]")
        elseif all(scalar, x)
            print(io, "[")
            for (i, e) in enumerate(x)
                i > 1 && print(io, ", ")
                jwrite(io, e)
            end
            print(io, "]")
        else
            println(io, "[")
            for (i, e) in enumerate(x)
                print(io, " "^(ind + 2)); jwrite(io, e, ind + 2)
                println(io, i < length(x) ? "," : "")
            end
            print(io, " "^ind, "]")
        end
    else
        error("jwrite: cannot write $(typeof(x))")
    end
end

jtext(x) = sprint(io -> (jwrite(io, x); println(io)))

# ── roots ────────────────────────────────────────────────────────────────────

# Replace each root only where a path segment ends, longest first, so a root
# that prefixes a sibling (`…/project` and `…/project2`) is not cut into it.
function substitute_roots(text::AbstractString, pairs::Vector{Pair{String,String}})
    for (root, ph) in sort(pairs; by = p -> -length(first(p)))
        text = replace(text, Regex("\\Q" * root * "\\E(?=[/\"])") => ph)
    end
    return text
end

# A machine path that survived: a drive letter (either slash, JSON-escaped or
# not), or one of the POSIX homes and temp roots, or this host's own.
function leaked_paths(text::AbstractString)
    hits = String[]
    for m in eachmatch(r"[A-Za-z]:(/|\\\\)[^\"]*|/(home|Users|tmp|var/folders)/[^\"]*", text)
        push!(hits, m.match)
    end
    for root in (CG._posix(homedir()), CG._posix(tempdir()))
        occursin(root, text) && push!(hits, root)
    end
    return unique(hits)
end

# ── reading a fixture back ───────────────────────────────────────────────────

# Lay the fixture's reply out under `<root>/build/.cmake/api/v1/reply` with the
# placeholders replaced, exactly as test_cmake_graph.jl does, and read it.
function read_fixture(reply_files::Dict{String,String}, windows_shell::Bool)
    root = mktempdir()
    S = CG._posix(joinpath(root, "src"))
    B = CG._posix(joinpath(root, "build"))
    reply = joinpath(B, ".cmake", "api", "v1", "reply")
    mkpath(reply)
    for (f, text) in reply_files
        write(joinpath(reply, f), replace(text, "@SOURCE@" => S, "@BUILD@" => B))
    end
    return CG.read_cmake_graph(B; windows_shell=windows_shell), S, B
end

# Every field of a target as plain values, with `f` applied to every string.
function flat(t::BG.GraphTarget, f)
    return (f(t.id), f(t.name), t.kind,
            [(f(s.path), s.group, s.generated) for s in t.sources],
            [(f(g.language), map(f, g.flags), map(f, g.defines),
              [(f(i.path), i.is_system) for i in g.includes]) for g in t.compile_groups],
            map(f, t.dependencies), map(f, t.link_libraries), map(f, t.artifacts))
end

# ── one fixture ──────────────────────────────────────────────────────────────

function closure(g::BG.TargetGraph, roots::Vector{String})
    keep = Set{String}()
    todo = [only(t.id for t in g.targets if t.name == r) for r in roots]
    while !isempty(todo)
        id = pop!(todo)
        id in keep && continue
        push!(keep, id)
        append!(todo, BG.target_by_id(g, id).dependencies)
    end
    return keep
end

"""
Configure `source_dir`, trim its reply, check it, and write it to
`test/fixtures/cmake_graph/<name>/`. `roots` limits the fixture to those
targets and what they depend on (empty: every target). `check(g, raw, S, B)`
gets the fixture as read back, the trimmed target JSON by name, and the roots
it was read under, and refuses a vacuous one.
"""
function make_fixture(name::String, source_dir::String; options=String[], roots=String[],
                      upstream=nothing, check)
    build_dir = mktempdir()
    @info "$name: configuring $source_dir"
    live = CG.cmake_graph(source_dir; build_dir=build_dir, options=options)

    reply_dir = joinpath(build_dir, ".cmake", "api", "v1", "reply")
    index_path = CG._reply_index(reply_dir)
    index = RepliBuild._read_json(index_path)
    entry = CG._codemodel_entry(index, index_path)
    cm = RepliBuild._read_json(joinpath(reply_dir, entry["jsonFile"]))
    cfg = CG._configuration(cm, "")
    generator = String(index["cmake"]["generator"]["name"])
    windows = CG._windows_shell(generator)

    keep = isempty(roots) ? Set(t.id for t in live.targets) : closure(live, roots)
    files = Dict{String,String}()
    refs = Any[]
    kept_names = String[]
    for r in cfg["targets"]
        r["id"] in keep || continue
        t = trim_target(RepliBuild._read_json(joinpath(reply_dir, r["jsonFile"])))
        fname = "target-$(r["name"]).json"
        haskey(files, fname) && refuse(name, "two targets named $(r["name"])")
        files[fname] = jtext(t)
        push!(kept_names, String(r["name"]))
        push!(refs, Dict{String,Any}("id" => r["id"], "name" => r["name"], "jsonFile" => fname))
    end
    files["codemodel-v2.json"] = jtext(Dict{String,Any}(
        "kind" => "codemodel", "version" => cm["version"], "paths" => cm["paths"],
        "configurations" => Any[Dict{String,Any}("name" => cfg["name"], "targets" => refs)]))
    files["index-0.json"] = jtext(Dict{String,Any}(
        "cmake" => Dict{String,Any}("generator" => pick(index["cmake"]["generator"],
                                                         Dict{String,Any}("name" => nothing,
                                                                          "multiConfig" => nothing)),
                                    "version" => Dict{String,Any}("string" => index["cmake"]["version"]["string"])),
        "reply" => Dict{String,Any}("codemodel-v2" => Dict{String,Any}(
            "kind" => "codemodel", "version" => entry["version"], "jsonFile" => "codemodel-v2.json"))))

    roots_map = Pair{String,String}[live.source_dir => "@SOURCE@", live.build_dir => "@BUILD@"]
    for (f, text) in files
        files[f] = substitute_roots(text, roots_map)
        leaks = leaked_paths(files[f])
        isempty(leaks) || refuse(name, "$f keeps machine paths: $(join(first(leaks, 5), ", "))")
    end

    # Lossless: the trimmed fixture, read under other roots, is the live read.
    g, S, B = read_fixture(files, windows)
    reroot(s) = substitute_roots(s, Pair{String,String}[live.source_dir => S, live.build_dir => B])
    want = [flat(t, reroot) for t in live.targets if t.id in keep]
    got = [flat(t, identity) for t in g.targets]
    if want != got
        for (w, h) in zip(want, got)
            w == h || @error "$name: target $(w[2]) reads differently from the fixture" live=w fixture=h
        end
        refuse(name, "the trimmed fixture does not read back as the live reply")
    end
    raw = Dict(k => JSON.parse(replace(files["target-$k.json"], "@SOURCE@" => S, "@BUILD@" => B);
                               dicttype=Dict{String,Any})
               for k in kept_names)
    check(g, raw, S, B)

    dest = joinpath(OUT, name)
    ispath(dest) && rm(dest; recursive=true)
    mkpath(joinpath(dest, "reply"))
    for f in sort!(collect(keys(files)))
        write(joinpath(dest, "reply", f), files[f])
    end
    meta = Dict{String,Any}(
        "cmake_version" => String(index["cmake"]["version"]["string"]),
        "generator" => generator,
        "windows_shell" => windows,
        "host" => string(Sys.MACHINE),
        "options" => options,
    )
    upstream === nothing || (meta["upstream"] = upstream)
    open(joinpath(dest, "FIXTURE.toml"), "w") do io
        println(io, "# Written by test/tools/gen_cmake_graph_fixtures.jl; regenerate, do not edit.")
        println(io, "# `windows_shell` is the encoding the reply's command fragments are in.")
        TOML.print(io, meta; sorted=true)
    end
    total = sum(filesize(joinpath(dest, "reply", f)) for f in keys(files))
    @info "$name: wrote $(length(files)) reply files ($(total) bytes) to $dest"
    rm(build_dir; recursive=true, force=true)
end

# ── what each fixture must show ──────────────────────────────────────────────

function by_name(g::BG.TargetGraph)
    d = Dict{String,BG.GraphTarget}()
    for t in g.targets
        d[t.name] = t
    end
    return d
end

function check_graphfix(g, raw, S, B)
    n = "graphfix"
    t = by_name(g)
    for (name, kind) in ("core" => :shared, "ext" => :shared, "helpers" => :object,
                         "ext_version" => :utility)
        (haskey(t, name) && t[name].kind === kind) || refuse(n, "no $kind target '$name'")
    end
    core, ext, helpers, ver = t["core"], t["ext"], t["helpers"], t["ext_version"]
    any(f -> occursin(' ', f["fragment"]),
        raw["core"]["compileGroups"][1]["compileCommandFragments"]) ||
        refuse(n, "no compile fragment of core holds several arguments")
    any(i -> i.path == S * "/core/include", only(ext.compile_groups).includes) ||
        refuse(n, "core's PUBLIC include is not in ext's compile group")
    ("EXT_LEVEL=2" in only(ext.compile_groups).defines &&
     !("EXT_LEVEL=2" in only(core.compile_groups).defines)) ||
        refuse(n, "ext's private define is missing or leaked into core")
    all(d -> d in ext.dependencies, (core.id, helpers.id, ver.id)) ||
        refuse(n, "ext does not depend on core, helpers and ext_version")
    any(s -> s.path == only(helpers.artifacts) && s.group == 0 && s.generated, ext.sources) ||
        refuse(n, "helpers' object file is not a source of ext")
    any(s -> s.path == B * "/gen/table.cpp" && s.group > 0 && s.generated, core.sources) ||
        refuse(n, "core compiles no generated source")
    any(s -> s.path == B * "/gen/ext_version.h.rule" && s.generated, ver.sources) ||
        refuse(n, "ext_version holds no rule for the generated header")
    "-lgraphfix_syslib" in ext.link_libraries ||
        refuse(n, "ext does not link graphfix_syslib")
    frags = [CG._split_fragment(f["fragment"], false) for f in raw["ext"]["link"]["commandFragments"]]
    any(a -> CG._abs(a, B) in core.artifacts, reduce(vcat, frags; init=String[])) ||
        refuse(n, "ext's link step names no artifact of core, so the exclusion is not exercised")
    any(l -> CG._abs(l, B) in core.artifacts, ext.link_libraries) &&
        refuse(n, "an artifact of core survived into ext's link libraries")
end

function check_curl(g, raw, S, B)
    n = "curl"
    t = by_name(g)
    (haskey(t, "libcurl_shared") && haskey(t, "libcurl_object")) ||
        refuse(n, "no libcurl_shared or libcurl_object target")
    shared, obj = t["libcurl_shared"], t["libcurl_object"]
    (shared.kind === :shared && obj.kind === :object) || refuse(n, "unexpected target kinds")
    obj.id in shared.dependencies || refuse(n, "libcurl_shared does not depend on libcurl_object")
    length(obj.artifacts) >= 100 || refuse(n, "libcurl_object has $(length(obj.artifacts)) objects")
    objsrc = Set(s.path for s in shared.sources if s.group == 0 && s.generated)
    issubset(Set(obj.artifacts), objsrc) ||
        refuse(n, "libcurl_shared does not list libcurl_object's objects as sources")
    count(s -> s.group > 0, shared.sources) < 5 ||
        refuse(n, "libcurl_shared compiles its own sources: not the OBJECT-library case")
end

# ── main ─────────────────────────────────────────────────────────────────────

function curl_checkout(dir::Union{Nothing,String})
    if dir === nothing
        dir = mktempdir()
        @info "curl: fetching $CURL_TAG ($CURL_COMMIT) into $dir"
        run(`git -C $dir init -q`)
        run(`git -C $dir remote add origin $CURL_URL`)
        run(`git -C $dir fetch -q --depth 1 origin $CURL_COMMIT`)
        run(`git -C $dir checkout -q FETCH_HEAD`)
    end
    head = readchomp(`git -C $dir rev-parse HEAD`)
    head == CURL_COMMIT || error("curl checkout $dir is at $head, not the pinned $CURL_COMMIT")
    return dir
end

function main(args)
    make_fixture("graphfix", joinpath(OUT, "project"); check=check_graphfix)
    i = findfirst(a -> a == "--curl" || startswith(a, "--curl="), args)
    if i !== nothing
        dir = curl_checkout(args[i] == "--curl" ? nothing : String(args[i][ncodeunits("--curl=")+1:end]))
        make_fixture("curl", dir; options=CURL_OPTIONS, roots=["libcurl_shared"],
                     upstream=Dict("url" => CURL_URL, "tag" => CURL_TAG, "commit" => CURL_COMMIT),
                     check=check_curl)
    end
end

main(ARGS)
