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

    # Build-tree -I roots, longest first. `""` is the build dir itself and must
    # sort last: it matches everything, so reaching it before a deeper root would
    # flatten away the very level the include form depends on.
    @test SCG._build_include_roots(["/bld/include", "/bld", "/src/lib", "/usr/include"],
                                   "/bld") == ["include", ""]
    @test SCG._build_include_roots(["/bld/a/b", "/bld/a"], "/bld") == ["a/b", "a"]
    @test SCG._build_include_roots(["/src", "/usr/include"], "/bld") == String[]

    # A captured file is named relative to the deepest root containing it, so the
    # package reproduces the spelling upstream compiles with.
    @test SCG._capture_rel("include/sundials/sundials_config.h", ["include", ""]) ==
          "sundials/sundials_config.h"
    @test SCG._capture_rel("flat.h", ["include", ""]) == "flat.h"
    # A build tree that generates everything at its root — pcre2's shape — must
    # reduce to exactly the historical basenames.
    @test SCG._capture_rel("config.h", [""]) == "config.h"
    # No root evidence at all degrades to the historical basename, never to a
    # new answer.
    @test SCG._capture_rel("deep/dir/x.h", String[]) == "x.h"
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

    # -- try_compile scratch, and the include form of a generated header -------
    #
    # Both defects surfaced harvesting SUNDIALS (2026-09-06) and both are
    # structural — nothing below mentions SUNDIALS, and every class here is
    # produced by ordinary cmake:
    #
    #  1. A hand-rolled `try_compile` with its own bindir leaves its probe source
    #     in plain sight in the build tree (`check_c_source_compiles` hides it
    #     under CMakeFiles/CMakeTmp/, which _is_cmake_internal already drops).
    #     It is generated, it is a .c, and it has its own `main()` — captured as
    #     library code and listed in `[compile] source_files`, it plants a `main`
    #     symbol in the shared object.
    #
    #  2. A header generated under an include root is included by its path
    #     relative to that root (`<fixt/fixt_config.h>`), so flattening it to a
    #     basename makes `-Iconfig` resolve nothing and the build dies on a
    #     missing include.
    #
    # The fixture carries the flat shape (pcre2's) and the nested shape in the
    # same capture, because a real project has both and one rule has to serve
    # them. The decisive assertion is a compile in a subprocess against the
    # capture alone — the source tree and the build tree are removed from the
    # include path first, so nothing but the captured layout can answer.
    @testset "scratch classification + generated header layout" begin
        src = mktempdir()
        write(joinpath(src, "CMakeLists.txt"), """
            cmake_minimum_required(VERSION 3.10)
            project(fixt C)
            set(FIXT_GREETING "hello")
            # (a) included as <fixt/fixt_config.h> — generated UNDER an include root
            configure_file(fixt_config.h.in \${PROJECT_BINARY_DIR}/include/fixt/fixt_config.h)
            # (b) included as "flat.h" — generated AT the build root (pcre2's shape)
            configure_file(flat.h.in \${PROJECT_BINARY_DIR}/flat.h)
            # (c) same basename at both levels — distinct under :auto, a collision under :flat
            configure_file(flat.h.in \${PROJECT_BINARY_DIR}/dup.h)
            configure_file(flat.h.in \${PROJECT_BINARY_DIR}/include/fixt/dup.h)
            # (d) a real generated SOURCE the target compiles (pcre2_chartables.c's shape)
            configure_file(gen_table.c.dist \${PROJECT_BINARY_DIR}/gen_table.c COPYONLY)
            # (e) try_compile with its own bindir — scratch .c outside CMakeFiles/
            file(WRITE \${PROJECT_BINARY_DIR}/PROBE_TEST/ltest.c
                 "int main(void){ return 0; }\\n")
            try_compile(PROBE_OK \${PROJECT_BINARY_DIR}/PROBE_TEST
                        \${PROJECT_BINARY_DIR}/PROBE_TEST/ltest.c)
            add_library(fixt SHARED a.c b.c \${PROJECT_BINARY_DIR}/gen_table.c)
            target_include_directories(fixt PRIVATE
                \${PROJECT_BINARY_DIR}/include \${PROJECT_BINARY_DIR})
            """)
        write(joinpath(src, "fixt_config.h.in"),
              "#define FIXT_GREETING \"@FIXT_GREETING@\"\n")
        write(joinpath(src, "flat.h.in"), "#define FIXT_FLAT 1\n")
        write(joinpath(src, "gen_table.c.dist"), "const int fixt_table[2] = {1,2};\n")
        write(joinpath(src, "a.c"), """
            #include <fixt/fixt_config.h>
            #include "flat.h"
            const char *a(void) { return FIXT_GREETING; }
            """)
        write(joinpath(src, "b.c"), "int b(void) { return 2; }\n")

        gen = Sys.which("ninja") === nothing ? "Unix Makefiles" : "Ninja"
        probe = SCG.cmake_probe(src; name="fixt", generator=gen,
                                clone_rel="deps/fixt", use_llvm_env=false)

        # The fixture must actually produce both classes, or every assertion
        # below is vacuous — the failure mode a guard is most likely to rot into.
        @test isfile(joinpath(probe.build_dir, "PROBE_TEST", "ltest.c"))
        @test occursin("main(", read(joinpath(probe.build_dir, "PROBE_TEST", "ltest.c"), String))
        @test "include/fixt/fixt_config.h" in probe.generated_headers

        # (1) Classified on one mechanical test: does any target compile it.
        @test "PROBE_TEST/ltest.c" in probe.scratch_sources
        @test "PROBE_TEST/ltest.c" ∉ probe.generated_sources
        @test "gen_table.c" in probe.generated_sources        # real, and compiled
        @test "gen_table.c" ∉ probe.scratch_sources

        # Include roots come off the compile line, deepest first.
        @test probe.include_roots == ["include", ""]

        out = mktempdir()
        written = SCG.capture_config(probe, out)
        rel = sort(map(w -> relpath(w, out), written))

        # (2) Layout: nested where the include form is nested, flat where it is
        # flat, in the same capture.
        @test rel == ["dup.h", "fixt/dup.h", "fixt/fixt_config.h", "flat.h", "gen_table.c"]
        @test isfile(joinpath(out, "fixt", "fixt_config.h"))
        @test occursin("hello", read(joinpath(out, "fixt", "fixt_config.h"), String))

        # Scratch is not captured, and no captured source defines an entry point.
        @test !isfile(joinpath(out, "ltest.c"))
        @test isempty(filter(f -> occursin("main(", read(f, String)),
                             filter(f -> endswith(f, ".c"), written)))
        # ...and the skip is recorded rather than silent.
        @test occursin("ltest.c", read(joinpath(out, "SYSCONFIG.md"), String))

        # `[compile] source_files` must name the path the capture actually wrote
        # to — the two derivations are shared precisely so they cannot disagree.
        frag = SCG.toml_fragment(probe; language="c")
        @test occursin("\"config/gen_table.c\"", frag)
        @test !occursin("ltest.c", frag)
        for m in eachmatch(r"\"config/([^\"]+)\"", frag)
            @test isfile(joinpath(out, m.captures[1]))
        end

        # (3) THE TRACE: compile the consumer against the capture alone. The
        # source and build trees are off the include path, so only the captured
        # layout can resolve <fixt/fixt_config.h>. This is the compile that
        # failed with `fatal error: fixt/fixt_config.h: No such file or directory`.
        cc = Sys.which("cc")
        if cc === nothing
            @info "no cc — skipping the compile-against-capture trace."
        else
            probe_dir = mktempdir()
            cp(joinpath(src, "a.c"), joinpath(probe_dir, "a.c"))
            log = joinpath(probe_dir, "cc.log")
            ok = success(pipeline(`$cc -fsyntax-only -I$out $(joinpath(probe_dir, "a.c"))`;
                                  stdout=log, stderr=log))
            ok || @info "compile against capture failed" log=read(log, String)
            @test ok
            # Negative control, built by hand rather than through capture_config
            # so it stays the historical rule (basename, no collision guard) and
            # cannot drift with the code under test: the same compiler, the same
            # consumer, the same headers, laid out flat — and it must fail. That
            # is what proves the trace measures the layout, not the toolchain.
            flat = mktempdir()
            for w in written
                cp(w, joinpath(flat, basename(w)); force=true)
            end
            @test !success(pipeline(`$cc -fsyntax-only -I$flat $(joinpath(probe_dir, "a.c"))`;
                                    stdout=devnull, stderr=devnull))
            rm(probe_dir; recursive=true, force=true)
            rm(flat; recursive=true, force=true)
        end

        # :flat cannot express two generated headers sharing a basename, and
        # must say so rather than ship one under the other's name.
        @test_throws ErrorException SCG.capture_config(probe, mktempdir(); layout=:flat)
        @test_throws ErrorException SCG.capture_config(probe, mktempdir(); layout=:sideways)

        # The escape hatch for a generated .c upstream #includes rather than
        # compiles: opt in explicitly, and it is recorded as such.
        keep = mktempdir()
        SCG.capture_config(probe, keep; layout=:build_tree, capture_scratch=true)
        @test isfile(joinpath(keep, "PROBE_TEST", "ltest.c"))
        @test isfile(joinpath(keep, "include", "fixt", "fixt_config.h"))

        rm(src; recursive=true, force=true)
        rm(out; recursive=true, force=true)
        rm(keep; recursive=true, force=true)
        rm(probe.build_dir; recursive=true, force=true)
    end
end
end
