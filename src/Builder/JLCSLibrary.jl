# JLCSLibrary.jl — where libJLCS, the dialect library, comes from.
#
# Two sources:
#
#   * a LOCAL BUILD, src/mlir/build/libJLCS.<dlext>, from src/mlir/build.sh
#     against the system LLVM/MLIR (what `Pkg.build` makes when it can);
#   * the ARTIFACT `JLCS` in Artifacts.toml: a self-contained libJLCS built by
#     deps/jlcs_jll/build_tarballs.jl with LLVM and MLIR linked in, so it needs
#     neither on this machine. Lazy: downloaded the first time it is needed.
#
# A local build wins when it exists: it is what someone working on the dialect
# just built. `REPLIBUILD_JLCS` overrides the order: `local` or `artifact`
# forces one source, anything else is the path of a library to use as given.
#
# The answer is decided at run time, never baked into the precompile cache: a
# depot can move, and a local build can appear after the package was
# precompiled. Once a library has been found the answer is kept for the
# process, since only one libJLCS may ever be loaded in it (two copies are two
# exception buffers; see JITManager.open_thunks_library).

module JLCSLibrary

import ..SRC_DIR
import Artifacts, LazyArtifacts, Libdl
import Zlib_jll, Zstd_jll

const LOCAL_PATH = joinpath(SRC_DIR, "mlir", "build", "libJLCS." * Libdl.dlext)

# The dialect ABI this RepliBuild generates for and calls: the same file
# libJLCS is compiled with (src/mlir/CMakeLists.txt), so a local build from
# this checkout always agrees. A library from anywhere else (the artifact, a
# JLL, a build left over from an older checkout) is asked before it is used.
const ABI_VERSION_FILE = joinpath(SRC_DIR, "mlir", "ABI_VERSION")
include_dependency(ABI_VERSION_FILE)
const ABI_VERSION = parse(Int, strip(read(ABI_VERSION_FILE, String)))
const ARTIFACTS_TOML = joinpath(dirname(SRC_DIR), "Artifacts.toml")
const ARTIFACT_NAME = "JLCS"

"""
Where libJLCS is. `source` is `:override`, `:local`, `:artifact`, or `:missing`
(then `path` is the local build's, the place an error should name). `note` says
why an artifact was passed over, for the doctor and for `check_library`.
"""
struct Location
    path::String
    source::Symbol
    note::String
end

const _FOUND = Ref{Union{Nothing, Location}}(nothing)
const _LOCK = ReentrantLock()
const _DOWNLOAD_FAILED = Ref{Union{Nothing, String}}(nothing)

function _mode()
    v = strip(get(ENV, "REPLIBUILD_JLCS", ""))
    isempty(v) && return (:auto, "")
    lowercase(v) in ("local", "artifact") && return (Symbol(lowercase(v)), "")
    return (:path, String(v))
end

"The artifact's tree hash for this host, or `nothing` when Artifacts.toml has none."
function artifact_hash()
    isfile(ARTIFACTS_TOML) || return nothing
    return try
        Artifacts.artifact_hash(ARTIFACT_NAME, ARTIFACTS_TOML)
    catch
        nothing
    end
end

# The library inside an installed artifact directory.
function _artifact_library(dir::AbstractString)::Union{String, Nothing}
    for sub in ("lib", "bin")
        p = joinpath(dir, sub, "libJLCS." * Libdl.dlext)
        isfile(p) && return p
    end
    return nothing
end

"""
    artifact_location(; install) -> Location

The artifact's library. With `install = false` nothing is downloaded: an
artifact that exists for this host but is not on disk is `:missing` with a
note saying so.
"""
function artifact_location(; install::Bool)::Location
    hash = artifact_hash()
    hash === nothing &&
        return Location(LOCAL_PATH, :missing, "no prebuilt libJLCS for this platform")
    if !Artifacts.artifact_exists(hash)
        install ||
            return Location(LOCAL_PATH, :missing, "a prebuilt libJLCS is available and not yet downloaded")
        # One attempt per process: every Tier 2 call asks, and a machine that
        # is offline would otherwise wait on the network each time.
        _DOWNLOAD_FAILED[] === nothing || return Location(LOCAL_PATH, :missing, _DOWNLOAD_FAILED[])
        try
            LazyArtifacts.ensure_artifact_installed(ARTIFACT_NAME, ARTIFACTS_TOML)
        catch e
            _DOWNLOAD_FAILED[] = "the prebuilt libJLCS could not be downloaded: " *
                                 first(split(sprint(showerror, e), '\n'))
            return Location(LOCAL_PATH, :missing, _DOWNLOAD_FAILED[])
        end
    end
    lib = _artifact_library(Artifacts.artifact_path(hash))
    lib === nothing &&
        return Location(LOCAL_PATH, :missing, "the JLCS artifact holds no libJLCS." * Libdl.dlext)
    return Location(lib, :artifact, "")
end

# The order, with its inputs passed in: `artifact` is called only when the
# artifact is the source the mode and the local build leave.
function _decide(mode::Symbol, given::AbstractString, local_path::AbstractString,
                 artifact::Function)::Location
    if mode === :path
        return isfile(given) ? Location(given, :override, "") :
               Location(given, :missing, "REPLIBUILD_JLCS names a file that does not exist")
    elseif mode === :local
        return Location(local_path, isfile(local_path) ? :local : :missing, "")
    elseif mode === :artifact
        return artifact()
    end
    return isfile(local_path) ? Location(local_path, :local, "") : artifact()
end

"""
    locate(; install = true) -> Location

Where libJLCS is, by the order in this file's header. `install = false` is for
probes (the doctor, a test's prerequisite check): it never downloads.
"""
function locate(; install::Bool = true)::Location
    lock(_LOCK) do
        _FOUND[] === nothing || return _FOUND[]
        mode, given = _mode()
        loc = _decide(mode, given, LOCAL_PATH, () -> artifact_location(; install))
        loc.source === :missing && return loc
        # Before anything opens the file, by `ccall` or by path.
        loc.source === :artifact && _preload_artifact_dependencies()
        return _FOUND[] = loc
    end
end

"""
    path(; install = true) -> String

The path of libJLCS, or of the local build when there is none anywhere. With
`install = true` (a caller about to use the library) one that speaks another
dialect ABI is an error here, before anything opens it; a probe
(`install = false`) gets the path and asks `abi_problem` itself.
"""
function path(; install::Bool = true)::String
    loc = locate(; install)
    install && verify_abi!(loc)
    return loc.path
end

"""
    library_abi_version(path) -> Union{Int,Nothing}

What the libJLCS at `path` reports as its dialect ABI version: `0` for one
that loads and has no `jlcs_abi_version` (built before the check existed),
`nothing` for one that does not load at all, which is the loader's story to
tell and not a version.
"""
function library_abi_version(path::AbstractString)::Union{Int, Nothing}
    # The flags the `ccall`s' own open uses: the first open of a library
    # decides how its symbols bind.
    h = Libdl.dlopen(path, Libdl.RTLD_LAZY | Libdl.RTLD_DEEPBIND; throw_error = false)
    h === nothing && return nothing
    f = Libdl.dlsym(h, :jlcs_abi_version; throw_error = false)
    return f === nothing ? 0 : Int(ccall(f, Cint, ()))
end

"""
    abi_problem(loc) -> Union{String,Nothing}

Why the library at `loc` must not be used, or `nothing`. The generator's IR
and the calls into the C API are for dialect ABI `ABI_VERSION` exactly; with
any other, the failure is a wrong call at run time, not an error.
"""
function abi_problem(loc::Location; expected::Int = ABI_VERSION,
                     got::Union{Int, Nothing} = loc.source === :missing ? nothing :
                                                library_abi_version(loc.path))::Union{String, Nothing}
    loc.source === :missing && return nothing
    (got === nothing || got == expected) && return nothing
    said = got == 0 ? "has no dialect ABI version (it was built before RepliBuild checked one)" :
                      "speaks dialect ABI $got"
    fix = if loc.source === :local
        "Rebuild it from this checkout: cd \"$(dirname(dirname(loc.path)))\" && ./build.sh " *
        "(or Pkg.build(\"RepliBuild\")). Or set REPLIBUILD_JLCS=artifact to use the prebuilt library."
    elseif loc.source === :artifact
        "The prebuilt library and this RepliBuild are from different releases: update RepliBuild, " *
        "or build the dialect locally (src/mlir/build.sh), which takes precedence."
    else
        "REPLIBUILD_JLCS names it: point it at a library built from this RepliBuild's src/mlir, or unset it."
    end
    return "libJLCS at $(loc.path) $said; this RepliBuild speaks dialect ABI $expected. $fix"
end

const _ABI_VERDICTS = Dict{String, Union{String, Nothing}}()

"Throw `abi_problem`'s reason for `loc`, asked once per library per process."
function verify_abi!(loc::Location)
    loc.source === :missing && return nothing
    problem = lock(_LOCK) do
        get!(() -> abi_problem(loc), _ABI_VERDICTS, loc.path)
    end
    problem === nothing || error(problem)
    return nothing
end

"Whether a libJLCS is on disk now (never downloads)."
available()::Bool = locate(; install = false).source !== :missing

# What the artifact's library needs loaded before it: LLVMSupport's compression
# libraries. The loader matches them by soname against what the process has
# open, so opening Julia's own copies first means no system zlib or zstd is
# looked for.
function _preload_artifact_dependencies()
    for lib in (Zlib_jll.libz, Zstd_jll.libzstd)
        try
            Libdl.dlopen(lib)
        catch
        end
    end
    return nothing
end

"""
The piece of a `Libdl.LazyLibraryPath` that is libJLCS: stringified when the
library is first opened, which is when the location is decided (and the
artifact downloaded if that is the source).
"""
struct LoadPath end
# `path()` verifies the dialect ABI, so a `ccall` never opens a library that
# speaks another one: the error is raised here, every time, until it is fixed.
Base.string(::LoadPath) = path()
Base.print(io::IO, p::LoadPath) = print(io, string(p))

end # module JLCSLibrary
