# Tools.jl — where the non-LLVM command-line tools come from.
#
# RepliBuild runs a handful of programs that are not part of an LLVM install:
# GNU binutils (readelf, objdump, nm, c++filt) on every build, git for
# `[dependencies]`, cmake and ninja for graph mode and SysConfigGen. Each used
# to be whatever that name meant on PATH, which is where "the objdump on PATH
# is not GNU" and "no readelf on this box" came from.
#
# This module is the one answer to "which program is `nm`". Per tool, in order:
#
#   1. `REPLIBUILD_TOOL_<NAME>` (a path): used as given, never probed.
#   2. The JLL's copy (Binutils_jll, Git_jll, CMake_jll, Ninja_jll,
#      LLD_unified_jll) and the
#      one on PATH, in the order `REPLIBUILD_TOOLS` names: `jll` (the default
#      off Windows) or `system` (the default on Windows, where the supported
#      host is MSYS2 and the JLL copies are unverified).
#   3. The bare name, so a machine with neither fails by naming the tool.
#
# A JLL copy counts only if it STARTS (`--version` exits 0): its shared
# libraries are not all in its own artifact, and a copy that cannot load is
# passed over for the system's, not reported as the tool failing.
#
# The environment a JLL copy needs rides on the `Cmd` (`cmd`, `with_env`),
# never on `ENV`, like the toolchain's (LLVMEnvironment.llvm_cmd). cmake and
# ninja get none: they start the system compiler, and Julia's library
# directory ahead of the system's on the loader path would hand that compiler
# Julia's libstdc++.

module Tools

import Binutils_jll, Git_jll, CMake_jll, Ninja_jll, LLD_unified_jll, LLVM_unified_jll

export tool_path, tool_source, tool_cmd

"""
One tool's JLL copy: the module, the name of its `<product>_path` binding, the
suffix of its `REPLIBUILD_TOOL_<suffix>` override, and whether the JLL's
library directories go on the loader path of the process (`libpath`).
"""
struct Provider
    jll::Module
    product::Symbol
    override::String
    libpath::Bool
end

const PROVIDERS = Dict{String, Provider}(
    "readelf" => Provider(Binutils_jll, :readelf_path, "READELF", true),
    "objdump" => Provider(Binutils_jll, :objdump_path, "OBJDUMP", true),
    "nm"      => Provider(Binutils_jll, :nm_path, "NM", true),
    "c++filt" => Provider(Binutils_jll, :cxxfilt_path, "CXXFILT", true),
    "git"     => Provider(Git_jll, :git_path, "GIT", true),
    "cmake"   => Provider(CMake_jll, :cmake_path, "CMAKE", false),
    "ninja"   => Provider(Ninja_jll, :ninja_path, "NINJA", false),
    # The C bucket's linker (Compiler._bundled_linker): LLD at the version of
    # Julia's own LLVM, which is the bundled clang's.
    "ld.lld"  => Provider(LLD_unified_jll, :ld_lld_path, "LD_LLD", true),
)

"What a tool resolved to. `source` is `:override`, `:jll`, `:system` or `:missing`."
struct Resolved
    name::String
    path::String
    source::Symbol
end

const _LOCK = ReentrantLock()
const _CACHE = Dict{Tuple{String, String, String, String}, Resolved}()
# JLL paths handed out, back to the tool they are: a caller that kept the path
# (`_dumper_candidates`) still gets the copy's environment when it runs it.
const _JLL_PATHS = Dict{String, String}()

"The variable the loader reads library directories from on this host."
const LIBPATH_KEY = Sys.iswindows() ? "PATH" :
                    Sys.isapple()   ? "DYLD_FALLBACK_LIBRARY_PATH" : "LD_LIBRARY_PATH"
const _PATHSEP = Sys.iswindows() ? ";" : ":"

"""
    order() -> Symbol

`:jll` or `:system`: which copy of a tool is tried first. `REPLIBUILD_TOOLS`
sets it; anything else there is an error, not a silent default.
"""
function order()::Symbol
    v = lowercase(strip(get(ENV, "REPLIBUILD_TOOLS", "")))
    isempty(v) && return Sys.iswindows() ? :system : :jll
    v in ("jll", "system") && return Symbol(v)
    error("REPLIBUILD_TOOLS is $(repr(v)); it takes `jll` or `system`")
end

"Whether `name` is a tool this module resolves."
known(name::AbstractString) = haskey(PROVIDERS, name)

function _jll_path(p::Provider)::Union{String, Nothing}
    try
        p.jll.is_available() || return nothing
        path = getfield(p.jll, p.product)
        return (path isa AbstractString && isfile(path)) ? String(path) : nothing
    catch
        return nothing
    end
end

# What the JLL copy of `name` needs in its environment, as pairs to lay over
# the command's own.
function _jll_env(name::String, p::Provider, base)::Vector{Pair{String, String}}
    env = Pair{String, String}[]
    if p.libpath
        dirs = try String(p.jll.LIBPATH[]) catch; "" end
        if !isempty(dirs)
            cur = _env_get(base, LIBPATH_KEY)
            push!(env, LIBPATH_KEY => (isempty(cur) ? dirs : dirs * _PATHSEP * cur))
        end
    end
    if name == "git"
        # What Git.jl sets for the same binary: git finds its helper programs
        # (git-remote-https), its templates and a CA bundle beside itself only
        # when told where they are.
        root = try String(Git_jll.artifact_dir) catch; "" end
        if !isempty(root)
            # Git for Windows keeps both under `mingw64`.
            for prefix in (root, joinpath(root, "mingw64"))
                core = joinpath(prefix, "libexec", "git-core")
                templates = joinpath(prefix, "share", "git-core", "templates")
                isdir(core) || continue
                push!(env, "GIT_EXEC_PATH" => core)
                isdir(templates) && push!(env, "GIT_TEMPLATE_DIR" => templates)
                break
            end
        end
        cert = joinpath(dirname(Sys.BINDIR), "share", "julia", "cert.pem")
        (isfile(cert) && !haskey(base, "GIT_SSL_CAINFO")) && push!(env, "GIT_SSL_CAINFO" => cert)
    elseif name == "cmake"
        # `-G Ninja` looks for ninja on PATH. Last, so a system ninja still wins.
        ninja = resolve("ninja")
        if ninja.source === :jll
            cur = _env_get(base, "PATH")
            dir = dirname(ninja.path)
            push!(env, "PATH" => (isempty(cur) ? dir : cur * _PATHSEP * dir))
        end
    end
    return env
end

_env_of(cmd::Cmd) = cmd.env === nothing ? ENV :
    Dict{String, String}(String(k) => String(v) for (k, v) in (split(e, '='; limit = 2) for e in cmd.env) )

# `cmd` with `pairs` laid over its environment (the process's, if it has none
# of its own). Windows names are case-insensitive and its own spelling is
# `Path`: a second entry spelled `PATH` beside it is two variables in the
# block the child gets, and which one it reads is not ours to choose. So an
# entry replaces whatever spelling is there (LLVMEnvironment._overlay_env
# does the same for the toolchain's).
function _overlay(cmd::Cmd, pairs::Vector{Pair{String, String}})::Cmd
    isempty(pairs) && return cmd
    env = Dict{String, String}(String(k) => String(v) for (k, v) in _env_of(cmd))
    for (k, v) in pairs
        if Sys.iswindows()
            for existing in collect(keys(env))
                uppercase(existing) == uppercase(k) && delete!(env, existing)
            end
        end
        env[k] = v
    end
    return setenv(cmd, env)
end

# One variable of an environment, whatever its case on Windows.
function _env_get(base, key::String)::String
    v = get(base, key, nothing)
    v === nothing || return String(v)
    Sys.iswindows() || return ""
    for (k, val) in base
        uppercase(String(k)) == uppercase(key) && return String(val)
    end
    return ""
end

function _starts(name::String, p::Provider, path::String)::Bool
    return _probe(_overlay(`$path --version`, _jll_env(name, p, ENV))) !== nothing
end

"How long a tool gets to answer `--version` before it is taken not to start."
const PROBE_SECONDS = 20.0

"""
    _probe(cmd; timeout = PROBE_SECONDS) -> Union{String,Nothing}

Run `cmd` and return what it printed, or `nothing` when it could not be
started, exited non-zero, or did not finish in `timeout` seconds. The limit is
the point: a program whose library is missing does not always exit. On a
Windows machine with nobody at it, the loader's "DLL was not found" box waits
for a click that never comes, and a probe without a limit waits with it.
"""
function _probe(cmd::Cmd; timeout::Real = PROBE_SECONDS)::Union{String, Nothing}
    out = tempname()
    try
        proc = open(out, "w") do io
            run(pipeline(ignorestatus(cmd); stdin = devnull, stdout = io, stderr = devnull); wait = false)
        end
        if timedwait(() -> process_exited(proc), float(timeout); pollint = 0.05) === :timed_out
            kill(proc, Base.SIGKILL)
            wait(proc)
            return nothing
        end
        return (proc.exitcode == 0 && proc.termsignal == 0) ? read(out, String) : nothing
    catch
        return nothing
    finally
        rm(out; force = true)
    end
end

"""
    resolve(name) -> Resolved

Which program `name` is on this machine, by the order in this file's header.
Cached per (name, override, order, PATH): a test that changes one of them gets
a fresh answer.
"""
function resolve(name::AbstractString)::Resolved
    name = String(name)
    p = get(PROVIDERS, name, nothing)
    p === nothing && error("Tools.resolve: $(repr(name)) is not a tool this module resolves " *
                           "(known: $(join(sort!(collect(keys(PROVIDERS))), ", ")))")
    override = get(ENV, "REPLIBUILD_TOOL_" * p.override, "")
    ord = order()
    key = (name, override, String(ord), get(ENV, "PATH", ""))
    return lock(_LOCK) do
        get!(_CACHE, key) do
            isempty(override) || return Resolved(name, override, :override)
            for src in (ord === :jll ? (:jll, :system) : (:system, :jll))
                if src === :jll
                    path = _jll_path(p)
                    if path !== nothing && _starts(name, p, path)
                        _JLL_PATHS[path] = name
                        return Resolved(name, path, :jll)
                    end
                else
                    w = Sys.which(name)
                    w === nothing || return Resolved(name, String(w), :system)
                end
            end
            return Resolved(name, name, :missing)
        end
    end
end

"The program to run for `name`: an absolute path, or the bare name when none was found."
tool_path(name::AbstractString)::String = resolve(name).path

"`:override`, `:jll`, `:system` or `:missing`."
tool_source(name::AbstractString)::Symbol = resolve(name).source

"Whether `name` resolved to a program at all."
have(name::AbstractString)::Bool = resolve(name).source !== :missing

# The tool a program stands for: a known bare name, or a JLL path this module
# handed out. `nothing` for anything else, which runs as it is.
function _tool_of(program::AbstractString)::Union{String, Nothing}
    known(program) && return String(program)
    return lock(() -> get(_JLL_PATHS, String(program), nothing), _LOCK)
end

"""
    with_env(name, cmd) -> Cmd

`cmd` with what the resolved copy of `name` needs laid over its environment.
A system copy or an override needs nothing and `cmd` comes back as it is.
"""
function with_env(name::AbstractString, cmd::Cmd)::Cmd
    r = resolve(name)
    r.source === :jll || return cmd
    return _overlay(cmd, _jll_env(r.name, PROVIDERS[r.name], _env_of(cmd)))
end

"""
    tool_cmd(cmd) -> Cmd

`cmd` with its program resolved: `` tool_cmd(`git -C \$dir status`) `` runs
the git this module chose, in the environment that copy needs. A command
whose program is not one of this module's tools is returned unchanged.
"""
function tool_cmd(cmd::Cmd)::Cmd
    isempty(cmd.exec) && return cmd
    name = _tool_of(cmd.exec[1])
    name === nothing && return cmd
    r = resolve(name)
    out = Cmd(Cmd(vcat(r.path, cmd.exec[2:end])); ignorestatus = cmd.ignorestatus,
              env = cmd.env, dir = cmd.dir)
    return with_env(name, out)
end

# ── LLVM's own tools at Julia's LLVM ────────────────────────────────────────
#
# `llvm-link` and `opt` are not in PROVIDERS: by name they are the C++
# bucket's, from the system LLVM that compiled the IR. The C bucket's external
# pipeline (`[link] fallback = true`) needs them at the major of Julia's own
# libLLVM instead, which is what LLVM_unified_jll ships. They are handed out
# as a directory (LLVMEnvironment.c_toolchain_bin_dir), and a program run
# from that directory gets the JLL's library path (`with_dir_env`): it loads
# Julia's libLLVM, which is not beside it.

const _BUNDLED_DIRS = Dict{String, String}()         # tools directory => library path
const _MATCHED_LLVM = Ref{Union{Missing, Nothing, String}}(missing)

"""
    matched_llvm_bin() -> Union{String,Nothing}

The directory of LLVM_unified_jll's tools, when it has an `llvm-link` and an
`opt` that start and report the major of `Base.libllvm_version`. `nothing`
where the JLL has no build, or its tools do not start.
"""
function matched_llvm_bin()::Union{String, Nothing}
    lock(_LOCK) do
        _MATCHED_LLVM[] === missing || return _MATCHED_LLVM[]
        _MATCHED_LLVM[] = try
            _find_matched_llvm()
        catch
            nothing
        end
    end
end

function _find_matched_llvm()::Union{String, Nothing}
    LLVM_unified_jll.is_available() || return nothing
    dir = dirname(String(LLVM_unified_jll.opt_path))
    exe = Sys.iswindows() ? ".exe" : ""
    libdirs = String(LLVM_unified_jll.LIBPATH[])
    cur = _env_get(ENV, LIBPATH_KEY)
    env = [LIBPATH_KEY => (isempty(cur) ? libdirs : libdirs * _PATHSEP * cur)]
    for tool in ("llvm-link", "opt")
        path = joinpath(dir, tool * exe)
        isfile(path) || return nothing
        out = read(pipeline(_overlay(`$path --version`, env); stdin = devnull, stderr = devnull), String)
        m = match(r"LLVM version (\d+)\.", out)
        (m !== nothing && parse(Int, m.captures[1]) == Base.libllvm_version.major) || return nothing
    end
    _BUNDLED_DIRS[dir] = libdirs
    return dir
end

"Whether `program` is run from a directory this module handed out."
in_bundled_dir(program::AbstractString)::Bool =
    lock(() -> haskey(_BUNDLED_DIRS, dirname(String(program))), _LOCK)

"`cmd` with the library path its program's bundled directory needs, or `cmd`."
function with_dir_env(cmd::Cmd)::Cmd
    isempty(cmd.exec) && return cmd
    libdirs = lock(() -> get(_BUNDLED_DIRS, dirname(cmd.exec[1]), nothing), _LOCK)
    libdirs === nothing && return cmd
    cur = _env_get(_env_of(cmd), LIBPATH_KEY)
    return _overlay(cmd, [LIBPATH_KEY => (isempty(cur) ? libdirs : libdirs * _PATHSEP * cur)])
end

"Every tool's resolution, sorted by name, for `check_environment`."
report()::Vector{Resolved} = [resolve(n) for n in sort!(collect(keys(PROVIDERS)))]

end # module Tools
