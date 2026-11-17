# test_jlcs_library.jl — where libJLCS comes from.
#
# The dialect library used to be one path, src/mlir/build/libJLCS.<dlext>,
# baked into MLIRNative as a constant. It now has two sources, a local build
# and the prebuilt `JLCS` artifact (LLVM and MLIR linked in), chosen when the
# library is first needed (RepliBuild.JLCSLibrary): a local build wins, and
# REPLIBUILD_JLCS forces `local`, `artifact`, or a path.
#
# No toolchain and no download: the order is tested on `_decide` with its
# inputs passed in, and the probes run with `install = false`.

using Test
using RepliBuild
import Libdl, TOML

const _JL = RepliBuild.JLCSLibrary

@testset "REPLIBUILD_JLCS: a source by name, or a path" begin
    @test withenv(() -> _JL._mode(), "REPLIBUILD_JLCS" => nothing) == (:auto, "")
    @test withenv(() -> _JL._mode(), "REPLIBUILD_JLCS" => "") == (:auto, "")
    @test withenv(() -> _JL._mode(), "REPLIBUILD_JLCS" => "Local") == (:local, "")
    @test withenv(() -> _JL._mode(), "REPLIBUILD_JLCS" => " artifact ") == (:artifact, "")
    @test withenv(() -> _JL._mode(), "REPLIBUILD_JLCS" => "/opt/x/libJLCS.so") == (:path, "/opt/x/libJLCS.so")
end

@testset "a local build wins, the artifact is the fallback" begin
    mktempdir() do dir
        built = joinpath(dir, "libJLCS.so"); touch(built)
        absent = joinpath(dir, "absent", "libJLCS.so")
        asked = Ref(0)
        prebuilt = _JL.Location(joinpath(dir, "artifact.so"), :artifact, "")
        artifact() = (asked[] += 1; prebuilt)
        none() = (asked[] += 1; _JL.Location(absent, :missing, "no prebuilt libJLCS for this platform"))

        # A local build: the artifact is not even asked for (asking can download).
        @test _JL._decide(:auto, "", built, artifact) == _JL.Location(built, :local, "")
        @test asked[] == 0
        # No local build: the artifact.
        @test _JL._decide(:auto, "", absent, artifact) == prebuilt
        @test asked[] == 1
        # Neither: missing, with the artifact's reason.
        loc = _JL._decide(:auto, "", absent, none)
        @test loc.source === :missing
        @test occursin("no prebuilt", loc.note)

        # Forced sources ignore the other.
        @test _JL._decide(:artifact, "", built, artifact) == prebuilt
        asked[] = 0
        @test _JL._decide(:local, "", absent, artifact).source === :missing
        @test _JL._decide(:local, "", built, artifact).source === :local
        @test asked[] == 0

        # A path is used as given, and a path that is not there says so.
        @test _JL._decide(:path, built, absent, artifact) == _JL.Location(built, :override, "")
        gone = _JL._decide(:path, absent, built, artifact)
        @test gone.source === :missing
        @test gone.path == absent
        @test occursin("does not exist", gone.note)
        @test asked[] == 0
    end
end

@testset "a probe never downloads and never throws" begin
    loc = _JL.artifact_location(; install = false)
    @test loc isa _JL.Location
    @test loc.source in (:artifact, :missing)
    if loc.source === :artifact
        @test isfile(loc.path)
    else
        @test !isempty(loc.note)
    end
    @test _JL.available() isa Bool
    @test RepliBuild.MLIRNative.libJLCS_path(; install = false) isa String
end

@testset "the location is decided at load, not baked into the precompile cache" begin
    lib = RepliBuild.MLIRNative.libJLCS
    @test lib isa Libdl.LazyLibrary
    @test !(lib isa AbstractString)
    # The local build's path is the answer when there is no library at all,
    # so `isfile(libJLCS_path())` stays the question "is there a libJLCS".
    @test endswith(_JL.LOCAL_PATH, joinpath("src", "mlir", "build", "libJLCS." * Libdl.dlext))
end

@testset "Artifacts.toml names a lazy JLCS for both Linux hosts" begin
    @test isfile(_JL.ARTIFACTS_TOML)
    entries = TOML.parsefile(_JL.ARTIFACTS_TOML)[_JL.ARTIFACT_NAME]
    @test entries isa Vector
    archs = Set{String}()
    for e in entries
        @test e["os"] == "linux"
        @test e["libc"] == "glibc"
        @test e["lazy"] === true          # never fetched at Pkg.add
        @test length(e["git-tree-sha1"]) == 40
        dl = only(e["download"])
        @test length(dl["sha256"]) == 64
        @test startswith(dl["url"], "https://")
        @test endswith(dl["url"], ".tar.gz")
        push!(archs, e["arch"])
    end
    @test archs == Set(["x86_64", "aarch64"])
end
