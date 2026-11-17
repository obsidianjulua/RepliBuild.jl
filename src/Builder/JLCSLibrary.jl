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

"The path of libJLCS, or of the local build when there is none anywhere."
path(; install::Bool = true)::String = locate(; install).path

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
Base.string(::LoadPath) = locate().path
Base.print(io::IO, p::LoadPath) = print(io, string(p))

end # module JLCSLibrary
