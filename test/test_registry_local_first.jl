# test_registry_local_first.jl — a project builds where it lives; the registry
# vendors that build and loads it by `use` and by `using`.
#
# `register` builds and wraps the project in place, copies the result to
# builds/<hash>/, and adds a Julia package around it to an environment. `use`
# follows an edit by rebuilding the project where it lives, and `using` in that
# environment loads the new build. Before this, the registry copied a project's
# sources into itself and built the copy: two projects with the same relative
# paths overwrote each other, and edits never reached `use`.
#
# A real C build: the c_test sources, copied to a temp dir so the edit below
# never touches the fixture, in a temp REPLIBUILD_HOME and a temp Julia
# environment. Pkg runs offline there (installed versions only).

using Test
using RepliBuild

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

const _LF_PR = RepliBuild.PackageRegistry

@testset "register vendors the project's own build; use follows edits" begin
    mktempdir() do sb
        proj = joinpath(sb, "c_test")
        mkpath(proj)
        for d in ("src", "include")
            cp(joinpath(@__DIR__, "c_test", d), joinpath(proj, d))
        end
        envdir = joinpath(sb, "env")
        withenv("REPLIBUILD_HOME" => joinpath(sb, "home"), "JULIA_PKG_OFFLINE" => "true") do
            toml = RepliBuild.discover(proj; force=true)
            entry = RepliBuild.register(toml; env=envdir)

            # Built in the project, vendored outside it.
            @test isfile(joinpath(proj, "julia", "CTest.jl"))
            vendored = _LF_PR._cached_build_dir(entry.build_hash)
            @test isfile(joinpath(vendored, "CTest.jl"))
            @test entry.julia_env == envdir

            m = RepliBuild.use("c_test"; verbose=false)
            @test Base.invokelatest(() -> m.add_i32(Int32(2), Int32(3))) == 5

            # An edit to the project reaches `use`: it rebuilds where the project
            # lives and vendors the new build. The old one stays loaded in this
            # process, from its own folder, so nothing it holds open is replaced.
            src = joinpath(proj, "src", "mathkit.c")
            edited = replace(read(src, String), "return a + b;" => "return a + b + 100;")
            @test edited != read(src, String)
            write(src, edited)
            m2 = RepliBuild.use("c_test"; verbose=false)
            @test Base.invokelatest(() -> m2.add_i32(Int32(2), Int32(3))) == 105
            entry2 = _LF_PR._load_index().entries["c_test"]
            @test entry2.build_hash != entry.build_hash

            # `using CTest`, in a child that sees only that environment, loads the new build.
            code = "using CTest; print(\"SUM \", CTest.add_i32(Int32(2), Int32(3)))"
            r = withenv("JULIA_LOAD_PATH" => join(["@", "@stdlib"], Sys.iswindows() ? ";" : ":")) do
                run_isolated(`--startup-file=no -e $code`; project=envdir, timeout=900)
            end
            @test r.ok
            @test occursin("SUM 105", r.output)
        end
    end
end
