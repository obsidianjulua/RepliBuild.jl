# test_deps_build_gate.jl — `Pkg.build` builds libJLCS again when the one that
# is there is another dialect's.
#
# deps/build.jl answered "already built and loadable" for any libJLCS the
# loader would open. A checkout that moved to a newer dialect kept the library
# built from the older one: RepliBuild refused it (JLCSLibrary.verify_abi!),
# the doctor said to run `Pkg.build("RepliBuild")`, and that command did
# nothing (2026-11-28, a Windows box a month behind main). The script asks the
# library for its dialect ABI now, in a child, so the file is not held open
# over the build that replaces it.
#
# No toolchain: the old libJLCS is a library of Julia's own, which loads and
# has no `jlcs_abi_version`. deps/build.jl runs as Pkg runs it, in a process of
# its own, from a copy of the three files it reads in a scratch tree where
# there is no build.sh and no MSYS2: what it prints before it gives up is the
# gate's answer.

using Test
using Libdl
using RepliBuild

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

const _DBG_ROOT = pkgdir(RepliBuild)
const _DBG_BUILT = "already built and loadable"

# A tree shaped like the package around deps/build.jl, with `lib` as its
# local libJLCS. `abi = false` leaves out src/mlir/ABI_VERSION.
function _dbg_stage(dir::AbstractString, lib::AbstractString; abi::Bool = true)
    mlir = joinpath(dir, "src", "mlir")
    mkpath(joinpath(dir, "deps"))
    mkpath(joinpath(mlir, "build"))
    cp(joinpath(_DBG_ROOT, "deps", "build.jl"), joinpath(dir, "deps", "build.jl"))
    cp(joinpath(_DBG_ROOT, "src", "mlir", "windows_build.jl"), joinpath(mlir, "windows_build.jl"))
    abi && cp(joinpath(_DBG_ROOT, "src", "mlir", "ABI_VERSION"), joinpath(mlir, "ABI_VERSION"))
    # realpath: a local Linux build is a link to libJLCS.so.<version>.
    cp(realpath(lib), joinpath(mlir, "build", "libJLCS." * Libdl.dlext))
    return joinpath(dir, "deps", "build.jl")
end

# Run the staged script the way Pkg does. No MSYS2 at the root it is given and
# never the network, so a build that is wanted cannot start on either host.
function _dbg_run(script::AbstractString, dir::AbstractString)
    out = joinpath(dir, "out.txt")
    cmd = addenv(`$(Base.julia_cmd()) --startup-file=no $script`,
                 "MSYS2_ROOT" => joinpath(dir, "no-msys2"), "REPLIBUILD_JLCS" => "local")
    p = run(pipeline(ignorestatus(cmd); stdout = out, stderr = out))
    return (; exitcode = p.exitcode, output = read(out, String))
end

@testset "Pkg.build's gate asks the library for its dialect ABI" begin
    # Loads anywhere Julia runs, and is no libJLCS: it reports no ABI version.
    old = Libdl.dlpath(Base.libm_name)
    @test isfile(old)
    @test Libdl.dlsym(Libdl.dlopen(old), :jlcs_abi_version; throw_error = false) === nothing

    @testset "a library with no ABI version is not \"already built\"" begin
        mktempdir() do dir
            r = _dbg_run(_dbg_stage(dir, old), dir)
            @test r.exitcode == 0                       # the script never fails an install
            @test occursin("has no dialect ABI version", r.output)
            @test !occursin(_DBG_BUILT, r.output)
        end
    end

    # The control: the same file is "already built" when there is no ABI to
    # hold it to. So it loaded in the child, and above it was the comparison
    # that refused it, not a library that failed to open.
    @testset "the comparison is what refuses it" begin
        mktempdir() do dir
            r = _dbg_run(_dbg_stage(dir, old; abi = false), dir)
            @test r.exitcode == 0
            @test occursin(_DBG_BUILT, r.output)
            @test !occursin("dialect ABI", r.output)
        end
    end

    # A library built from this checkout is this checkout's dialect.
    loc = RepliBuild.JLCSLibrary.locate(; install = false)
    if loc.source === :local && RepliBuild.JLCSLibrary.abi_problem(loc) === nothing
        @testset "this checkout's own build is \"already built\"" begin
            mktempdir() do dir
                r = _dbg_run(_dbg_stage(dir, loc.path), dir)
                @test r.exitcode == 0
                @test occursin(_DBG_BUILT, r.output)
            end
        end
    else
        requires("Pkg.build's gate, this checkout's own build", :libJLCS) &&
            @info "libJLCS here is not a local build of this dialect; the gate's positive case did not run" source = loc.source
    end
end
