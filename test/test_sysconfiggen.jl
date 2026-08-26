#!/usr/bin/env julia
# test/test_sysconfiggen.jl — the configure-step capture that lets RepliBuild
# build a library whose headers do not exist in git.
#
# Needs cmake, which is why this lives in devtests and not CI.
#
# Moved here from RepliBuildTooling 2026-08-26 with the code. A project whose
# headers come from feature detection cannot be BUILT without this, so it is a
# build capability, not an introspection tool.
#
# The end-to-end testset drives a self-contained generated CMakeLists (a
# configure_file, a live source, an orphan source and a tools/ dir), so it needs
# no network and no vendored library.

using Test
using RepliBuild

const SCG = RepliBuild.SysConfigGen

@testset "SysConfigGen" begin

@testset "sysconfig internals" begin
    # A flag signature must drop everything file-specific, or two TUs that
    # share a flag set would look divergent purely because their names differ.
    sig = SCG._flag_signature(["/usr/bin/cc", "-DFOO=1", "-I/inc", "-O2",
                             "-o", "CMakeFiles/x.dir/a.c.o", "-c", "src/a.c"])
    @test sig == ["-DFOO=1", "-I/inc", "-O2"]
    @test SCG._flag_signature(["cc", "-DA", "-o", "b.o", "-c", "b.cpp"]) == ["-DA"]

    # Target extraction, from the `output` field and from -o as a fallback.
    @test SCG._target_of(Dict("output" => "CMakeFiles/mylib.dir/src/a.c.o"), String[]) == "mylib"
    @test SCG._target_of(Dict{String,Any}(), ["cc", "-o", "CMakeFiles/z.dir/a.c.o"]) == "z"
    @test SCG._target_of(Dict("output" => "weird/path.o"), String[]) == ""

    @test SCG._is_cmake_internal("CMakeFiles/3.31/CompilerIdC/CMakeCCompilerId.c")
    @test SCG._is_cmake_internal("_deps/foo-src/x.h")
    @test !SCG._is_cmake_internal("config.h")

    # Include translation: build-tree -> the package's config dir, source
    # tree -> clone-relative, anything else (system/external) dropped.
    inc = SCG._translate_includes(["-I/src/lib", "-I/bld", "-I/usr/include", "-I/src"],
                                "/src", "/bld", "config", "deps/x")
    @test inc == ["deps/x/lib", "config"]

    # Exclusion collapsing prefers the shallowest wholly-dead directory. A
    # directory that still holds compiled sources must stay per-file, since a
    # coarse pattern there would silently drop live code.
    ex = SCG._collapse_excludes(["src/a.c", "src/b.c"],
                              ["src/skip.c", "tools/x.c", "tools/deep/y.c"])
    @test "tools/" in ex
    @test "src/skip.c" in ex
    @test !("src/" in ex)
    @test !("tools/deep/" in ex)      # already covered by tools/

    # Directory names the resolver prunes on its own are not proposed.
    @test isempty(SCG._collapse_excludes(["src/a.c"], ["tests/t.c"]))
end

# -- CMake harvest end-to-end, gated on cmake being installed ---------------
if Sys.which("cmake") === nothing
    @info "cmake not found — skipping cmake_probe end-to-end test."
else
    @testset "sysconfig end-to-end" begin
        src = mktempdir()
        write(joinpath(src, "CMakeLists.txt"), """
            cmake_minimum_required(VERSION 3.10)
            project(fixt C)
            set(FIXT_GREETING "hello")
            configure_file(fixt_config.h.in fixt_config.h)
            add_library(fixt \${LIBKIND} a.c b.c)
            target_include_directories(fixt PRIVATE \${CMAKE_CURRENT_BINARY_DIR})
            target_compile_definitions(fixt PRIVATE HAVE_CONFIG_H)
            """)
        write(joinpath(src, "fixt_config.h.in"),
              "#define FIXT_GREETING \"@FIXT_GREETING@\"\n")
        write(joinpath(src, "a.c"), "int a(void) { return 1; }\n")
        write(joinpath(src, "b.c"), "int b(void) { return 2; }\n")
        # Never referenced by any target — must show up as an exclusion.
        write(joinpath(src, "orphan.c"), "int orphan(void) { return 3; }\n")
        mkpath(joinpath(src, "tools"))
        write(joinpath(src, "tools", "cli.c"), "int main(void) { return 0; }\n")

        gen = Sys.which("ninja") === nothing ? "Unix Makefiles" : "Ninja"
        probe = SCG.cmake_probe(src; name="fixt", generator=gen,
                              clone_rel="deps/fixt", use_llvm_env=false)

        # The generated header is the whole point: it does not exist in the
        # source tree, only in the build tree, and only after configure.
        @test "fixt_config.h" in probe.generated_headers
        @test !isfile(joinpath(src, "fixt_config.h"))

        t = SCG.main_target(probe)
        @test t !== nothing
        @test t.name == "fixt"
        @test SCG.uniform(t)
        @test sort(t.files) == ["a.c", "b.c"]
        @test "-DHAVE_CONFIG_H" in t.defines
        @test "config" in t.include_dirs      # the build dir, remapped

        # Sources no target compiles become exclusions.
        frag = SCG.toml_fragment(probe; language="c")
        @test occursin("orphan.c", frag)
        @test occursin("tools/", frag)
        @test occursin("-DHAVE_CONFIG_H", frag)
        @test !occursin("\"a.c\"", frag)      # live source must not be excluded

        # Harvest copies the header out and leaves a provenance note.
        out = mktempdir()
        written = SCG.capture_config(probe, out)
        @test length(written) == 1
        @test isfile(joinpath(out, "fixt_config.h"))
        @test occursin("hello", read(joinpath(out, "fixt_config.h"), String))
        @test isfile(joinpath(out, "SYSCONFIG.md"))
        @test occursin("cmake_probe", read(joinpath(out, "SYSCONFIG.md"), String))

        rm(src; recursive=true, force=true)
        rm(out; recursive=true, force=true)
        rm(probe.build_dir; recursive=true, force=true)
    end
end
end
