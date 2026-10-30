#!/usr/bin/env julia
# test/test_graph_build.jl — the planning half of the graph executor
# (`Builder/GraphBuild.jl`): which translation units compile with which
# arguments, what folds into which library, what each library links against,
# and what is refused. Needs no toolchain: graphs are built by hand and nothing
# is compiled. `test_graph_build_live.jl` (devtests) builds and calls one.

using Test
using RepliBuild

const BG = RepliBuild.BuildGraph
const GB = RepliBuild.GraphBuild

const GB_ROOT = replace(mktempdir(), '\\' => '/')
const GB_OUT = GB_ROOT * "/out"
gbsrc(p) = GB_ROOT * "/src/" * p

gbgroup(lang; flags = String[], defines = String[], includes = BG.IncludeDir[]) =
    BG.CompileGroup(lang, flags, defines, includes)
gbtarget(id, kind; name = id, sources = BG.GraphSource[], groups = BG.CompileGroup[],
         deps = String[], links = String[]) =
    BG.GraphTarget(id, name, kind, sources, groups, deps, links, String[])
gbgraph(ts...) = BG.TargetGraph(GB_ROOT * "/src", GB_ROOT * "/build", collect(ts))
unit(p, g = 1; generated = false) = BG.GraphSource(gbsrc(p), g, generated)
tu_for(lib, file) = only(filter(tu -> endswith(tu.source, "/" * file), lib.tus))
lib_named(plan, n) = only(filter(l -> l.name == n, plan.libraries))
argafter(args, flag) = args[findfirst(==(flag), args) + 1]
plan_error(f) = try f(); "" catch e; sprint(showerror, e) end

@testset "Graph build planning" begin

    @testset "each translation unit compiles under its own group" begin
        inc = BG.IncludeDir(GB_ROOT * "/a/include", false)
        sys = BG.IncludeDir(GB_ROOT * "/ext", true)
        gc = gbgroup("C"; flags = ["-std=gnu11", "-O2"], defines = ["A_ONLY", "LEVEL=2"], includes = [inc])
        gx = gbgroup("CXX"; flags = ["-std=gnu++17", "-g"], defines = ["X_ONLY=1"], includes = [inc, sys])
        a = gbtarget("a", :shared; sources = [unit("a.c", 1), unit("b.cpp", 2), unit("a.h", 0)],
                     groups = [gc, gx])
        lib = only(GB.plan_graph(gbgraph(a), GB_OUT; windows = false).libraries)

        @test length(lib.tus) == 2                    # the header is no translation unit
        @test lib.bucket == :cpp                      # one C++ unit: the C++ bucket

        c = tu_for(lib, "a.c")
        @test c.compiler == "clang" && c.language == "C" && c.bucket == :cpp
        @test c.args[1:2] == ["-S", "-emit-llvm"]
        @test c.args[end-2:end] == ["-o", c.ir_file, gbsrc("a.c")]
        @test "-DA_ONLY" in c.args && "-DLEVEL=2" in c.args
        @test !("-DX_ONLY=1" in c.args)
        @test "-std=gnu11" in c.args && !any(startswith("-std=gnu++"), c.args)
        @test argafter(c.args, "-x") == "c"
        @test "-I" * GB_ROOT * "/a/include" in c.args
        @test !("-isystem" in c.args)
        # The group's flags first, then what the IR pipeline needs on top.
        @test findfirst(==("-O2"), c.args) < findfirst(==("-g"), c.args)
        @test "-fPIC" in c.args
        @test startswith(c.ir_file, GB_OUT * "/a/ir/cpp/")

        x = tu_for(lib, "b.cpp")
        @test x.compiler == "clang++" && argafter(x.args, "-x") == "c++"
        @test "-DX_ONLY=1" in x.args && !("-DA_ONLY" in x.args)
        @test count(==("-g"), x.args) == 1            # its own -g, not a second
        @test argafter(x.args, "-isystem") == GB_ROOT * "/ext"
        @test x.include_dirs == [GB_ROOT * "/a/include", GB_ROOT * "/ext"]
        @test lib.include_dirs == [GB_ROOT * "/a/include"]
        @test lib.system_include_dirs == [GB_ROOT * "/ext"]
        @test lib.defines == ["A_ONLY", "LEVEL=2", "X_ONLY=1"]

        # PE has no position-dependent code to fix.
        libw = only(GB.plan_graph(gbgraph(a), GB_OUT; windows = true).libraries)
        @test !any(tu -> "-fPIC" in tu.args, libw.tus)
    end

    @testset "-g and -fPIC only where the group's flags do not end there" begin
        @test !GB._has_full_debug_info(String[])
        @test GB._has_full_debug_info(["-g"])
        @test GB._has_full_debug_info(["-O2", "-g3"])
        @test !GB._has_full_debug_info(["-g", "-gline-tables-only"])
        @test !GB._has_full_debug_info(["-g3", "-g0"])
        @test GB._has_full_debug_info(["-g0", "-gdwarf-5"])
        @test GB._is_pic(["-fPIC"]) && GB._is_pic(["-fpic"])
        @test !GB._is_pic(String[])
        @test !GB._is_pic(["-fPIE"])
        @test !GB._is_pic(["-fPIC", "-fno-pie"])
        g = gbgroup("C"; flags = ["-g0", "-fPIE"])
        args = GB._compile_args(g, gbsrc("x.c"), GB_OUT * "/x.ll"; windows = false)
        @test args[findfirst(==("-fPIE"), args) + 1:end][1:2] == ["-g", "-fPIC"]
    end

    @testset "the bucket is the library's: C only when every unit is C" begin
        cg = gbgroup("C")
        clib = gbtarget("clib", :shared; sources = [unit("c1.c"), unit("c2.c")], groups = [cg])
        lib = only(GB.plan_graph(gbgraph(clib), GB_OUT; windows = false).libraries)
        @test lib.bucket == :c
        @test all(tu -> tu.bucket == :c && tu.compiler == "clang", lib.tus)
        @test all(tu -> startswith(tu.ir_file, GB_OUT * "/clib/ir/c/"), lib.tus)

        # A C++ object library folded in moves the whole library, its C files
        # included, to the C++ bucket: one LLVM per module.
        xobj = gbtarget("xobj", :object; sources = [unit("x.cpp")], groups = [gbgroup("CXX")])
        mixed = gbtarget("mixed", :shared; sources = [unit("m.c")], groups = [cg], deps = ["xobj"])
        lib = only(GB.plan_graph(gbgraph(mixed, xobj), GB_OUT; windows = false).libraries)
        @test lib.bucket == :cpp
        @test tu_for(lib, "m.c").bucket == :cpp && tu_for(lib, "m.c").compiler == "clang"
    end

    @testset "object targets fold into every library that uses them, compiled once" begin
        obj = gbtarget("obj", :object; sources = [unit("o.cpp")], groups = [gbgroup("CXX"; defines = ["OBJ"])])
        b = gbtarget("b", :shared; sources = [unit("b.cpp")], groups = [gbgroup("CXX"; defines = ["B"])],
                     deps = ["obj"])
        c = gbtarget("c", :shared; sources = [unit("c.cpp")], groups = [gbgroup("CXX")], deps = ["obj"])
        plan = GB.plan_graph(gbgraph(c, b, obj), GB_OUT; windows = false)
        @test [l.name for l in plan.libraries] == ["b", "c"]
        lb = lib_named(plan, "b")
        @test lb.folded == ["obj"]
        @test [basename(tu.source) for tu in lb.tus] == ["b.cpp", "o.cpp"]   # own units first
        o = tu_for(lb, "o.cpp")
        @test o.owner == "obj"
        @test "-DOBJ" in o.args && !("-DB" in o.args)                       # the object's group
        @test startswith(o.ir_file, GB_OUT * "/obj/ir/cpp/")
        @test tu_for(lib_named(plan, "c"), "o.cpp").ir_file == o.ir_file
        @test length(plan.compiles) == 3                                    # o.cpp once
        @test Dict(plan.notes)["obj"] == "folded into b, c"

        # Folded into a C library and a C++ library, it is compiled for each.
        cobj = gbtarget("cobj", :object; sources = [unit("co.c")], groups = [gbgroup("C")])
        cl = gbtarget("cl", :shared; sources = [unit("cl.c")], groups = [gbgroup("C")], deps = ["cobj"])
        xl = gbtarget("xl", :shared; sources = [unit("xl.cpp")], groups = [gbgroup("CXX")], deps = ["cobj"])
        plan = GB.plan_graph(gbgraph(cl, xl, cobj), GB_OUT; windows = false)
        cos = [tu for tu in plan.compiles if endswith(tu.source, "/co.c")]
        @test sort([tu.bucket for tu in cos]) == [:c, :cpp]
        @test sort([dirname(tu.ir_file) for tu in cos]) == [GB_OUT * "/cobj/ir/c", GB_OUT * "/cobj/ir/cpp"]
    end

    @testset "static targets fold; one no library folds is a library itself" begin
        util = gbtarget("util", :static; sources = [unit("u.c")], groups = [gbgroup("C")])
        b = gbtarget("b", :shared; sources = [unit("b.c")], groups = [gbgroup("C")], deps = ["util"])
        plan = GB.plan_graph(gbgraph(b, util), GB_OUT; windows = false)
        @test [l.name for l in plan.libraries] == ["b"]
        @test lib_named(plan, "b").folded == ["util"]
        @test Dict(plan.notes)["util"] == "folded into b"

        # A chain of static targets with no shared target: the top one is the
        # library and folds the rest.
        s2 = gbtarget("s2", :static; sources = [unit("s2.c")], groups = [gbgroup("C")])
        s1 = gbtarget("s1", :static; sources = [unit("s1.c")], groups = [gbgroup("C")], deps = ["s2"])
        exe = gbtarget("tool", :executable; sources = [unit("main.c")], groups = [gbgroup("C")], deps = ["s1"])
        plan = GB.plan_graph(gbgraph(exe, s1, s2), GB_OUT; windows = false)
        @test [l.name for l in plan.libraries] == ["s1"]
        @test lib_named(plan, "s1").kind == :static
        @test [basename(tu.source) for tu in lib_named(plan, "s1").tus] == ["s1.c", "s2.c"]
        @test Dict(plan.notes)["tool"] == "executables are not built"
        @test Dict(plan.notes)["s2"] == "folded into s1"
    end

    zpath = GB_ROOT * "/ext/lib/libz.so"
    alpha = gbtarget("alpha", :shared; sources = [unit("alpha.cpp")], groups = [gbgroup("CXX")], links = ["m"])
    util = gbtarget("util", :static; sources = [unit("util.cpp")], groups = [gbgroup("CXX")],
                    deps = ["alpha"], links = ["-pthread", "m"])
    beta = gbtarget("beta", :shared; sources = [unit("beta.cpp")], groups = [gbgroup("CXX")],
                    deps = ["util"], links = [zpath])
    gamma = gbtarget("gamma", :shared; sources = [unit("gamma.cpp")], groups = [gbgroup("CXX")], deps = ["beta"])
    exe = gbtarget("exe", :executable; sources = [unit("main.cpp")], groups = [gbgroup("CXX")], deps = ["gamma"])
    gen = gbtarget("gen", :utility)
    mod = gbtarget("mod", :module; sources = [unit("mod.cpp")], groups = [gbgroup("CXX")])
    delta = gbtarget("delta", :shared; sources = [unit("delta.cpp")], groups = [gbgroup("CXX")],
                     deps = ["mod", "gen"])
    chain = gbgraph(exe, gamma, delta, beta, util, mod, gen, alpha)

    @testset "link order and inputs" begin
        plan = GB.plan_graph(chain, GB_OUT; windows = false)
        @test [l.name for l in plan.libraries] == ["alpha", "mod", "delta", "beta", "gamma"]

        lb = lib_named(plan, "beta")
        @test lb.folded == ["util"]
        @test lb.link_dependencies == ["alpha"]                     # reached through util
        @test lb.link_names == ["alpha"]
        @test lb.link_dirs == [GB_OUT * "/alpha/lib"]
        @test lb.link_files == [GB_OUT * "/alpha/lib/libalpha.so"]
        @test lb.library == GB_OUT * "/beta/lib/libbeta.so"
        @test lb.link_libraries == [zpath, "-pthread", "m"]          # its own, then util's
        @test lb.link_args == ["-Wl,-rpath,\$ORIGIN/../../alpha/lib", zpath, "-pthread", "-lm"]

        lg = lib_named(plan, "gamma")
        @test lg.folded == String[]                                 # util belongs to beta
        @test lg.link_dependencies == ["alpha", "beta"]             # deps first
        @test lg.link_args == ["-Wl,-rpath,\$ORIGIN/../../alpha/lib", "-Wl,-rpath,\$ORIGIN/../../beta/lib"]

        # A module and a utility only order the build.
        @test lib_named(plan, "delta").link_dependencies == String[]
        @test lib_named(plan, "mod").kind == :module
        @test Dict(plan.notes)["gen"] == "a utility target runs build steps; GraphGenerators runs those"
        @test Dict(plan.notes)["exe"] == "executables are not built"
        @test !any(tu -> endswith(tu.source, "main.cpp"), plan.compiles)

        @test [[l.name for l in w] for w in GB._link_levels(plan.libraries)] ==
              [["alpha", "mod", "delta"], ["beta"], ["gamma"]]

        # PE: no rpath; an import library beside the DLL.
        pw = GB.plan_graph(chain, GB_OUT; windows = true)
        lbw = lib_named(pw, "beta")
        @test lbw.library == GB_OUT * "/beta/lib/libbeta.dll"
        @test lbw.link_files == [GB_OUT * "/alpha/lib/libalpha.dll"]
        @test lbw.link_args == ["-Wl,--out-implib," * GB_OUT * "/beta/lib/libbeta.dll.a", zpath, "-pthread", "-lm"]
    end

    @testset "transitive dependencies, as CMake's File API lists them" begin
        # alpha folds the object library `base`, whose object file it lists as
        # a source (group 0, generated, one of base's artifacts); alpha also
        # folds the static `zip`. beta links alpha and folds its own static
        # `util`; its dependencies, listed transitively, name base and zip too.
        obj = GB_ROOT * "/build/CMakeFiles/base.dir/base.cpp.obj"
        base = BG.GraphTarget("base", "base", :object, [unit("base.cpp")], [gbgroup("CXX")],
                              String[], String[], [obj])
        zip = gbtarget("zip", :static; sources = [unit("zip.c")], groups = [gbgroup("C")])
        t_alpha = gbtarget("alpha", :shared;
                         sources = [unit("alpha.cpp"), BG.GraphSource(obj, 0, true)],
                         groups = [gbgroup("CXX")], deps = ["base", "zip"])
        t_util = gbtarget("util", :static; sources = [unit("util.cpp")], groups = [gbgroup("CXX")])
        t_beta = gbtarget("beta", :shared; sources = [unit("beta.cpp")], groups = [gbgroup("CXX")],
                        deps = ["alpha", "base", "util", "zip"])
        plan = GB.plan_graph(gbgraph(t_beta, t_util, t_alpha, zip, base), GB_OUT; windows = false)
        @test [l.name for l in plan.libraries] == ["alpha", "beta"]   # no refusal for the .obj
        @test lib_named(plan, "alpha").folded == ["base", "zip"]
        @test tu_for(lib_named(plan, "alpha"), "base.cpp").owner == "base"
        # A second copy of alpha's base and zip in beta would split their state.
        @test lib_named(plan, "beta").folded == ["util"]
        @test lib_named(plan, "beta").link_dependencies == ["alpha"]

        # An object library with artifacts folds only where they are listed:
        # linked for its usage requirements alone, it adds no code.
        user = gbtarget("user", :shared; sources = [unit("user.cpp")], groups = [gbgroup("CXX")],
                        deps = ["base"])
        plan = GB.plan_graph(gbgraph(user, base), GB_OUT; windows = false)
        @test lib_named(plan, "user").folded == String[]

        # A static target that only an executable dependency links is folded
        # (nothing else would provide it); one behind a shared target is not.
        tool = gbtarget("tool", :executable; sources = [unit("tool.c")], groups = [gbgroup("C")],
                        deps = ["zip"])
        lib = gbtarget("lib", :shared; sources = [unit("lib.c")], groups = [gbgroup("C")],
                       deps = ["tool", "zip"])
        @test lib_named(GB.plan_graph(gbgraph(lib, tool, zip), GB_OUT), "lib").folded == ["zip"]
    end

    @testset "link inputs: flags and paths as they are, names as -l" begin
        @test GB._link_input("m") == "-lm"
        @test GB._link_input("python3.11") == "-lpython3.11"
        @test GB._link_input("-Wl,--as-needed") == "-Wl,--as-needed"
        @test GB._link_input("/usr/lib/libz.so") == "/usr/lib/libz.so"
        @test GB._link_input("C:/x/libz.dll.a") == "C:/x/libz.dll.a"
        @test GB._link_input("libfoo.a") == "libfoo.a"
        @test GB._link_input("libssl.so.3") == "libssl.so.3"
    end

    @testset "only: the named libraries and what they link against" begin
        plan = GB.plan_graph(chain, GB_OUT; only = ["gamma"], windows = false)
        @test [l.name for l in plan.libraries] == ["alpha", "beta", "gamma"]
        @test plan.scope == ["alpha", "util", "beta", "gamma"]
        @test Dict(plan.notes)["util"] == "folded into beta"
        # A name or an id; a static target asked for is a library itself.
        named = gbtarget("t-1", :static; name = "tee", sources = [unit("t.c")], groups = [gbgroup("C")])
        plan = GB.plan_graph(gbgraph(named), GB_OUT; only = "tee", windows = false)
        @test only(plan.libraries).id == "t-1"
        @test occursin("'nobody'", plan_error(() -> GB.plan_graph(chain, GB_OUT; only = ["nobody"])))
        e = plan_error(() -> GB.plan_graph(chain, GB_OUT; only = "exe"))
        @test occursin("'exe'", e) && occursin(":executable", e)
    end

    @testset "the plan is a function of the graph, not of its listing order" begin
        shuffled = gbgraph(alpha, gen, util, mod, beta, delta, gamma, exe)
        a = GB.plan_graph(chain, GB_OUT; windows = false)
        b = GB.plan_graph(shuffled, GB_OUT; windows = false)
        @test [(l.name, l.link_args, [tu.args for tu in l.tus]) for l in a.libraries] ==
              [(l.name, l.link_args, [tu.args for tu in l.tus]) for l in b.libraries]
        @test [tu.ir_file for tu in a.compiles] == [tu.ir_file for tu in b.compiles]
    end

    @testset "refusals" begin
        # Generated sources that do not exist yet: named, compiled or not.
        gen_c = GB_ROOT * "/build/gen/made.c"
        gen_h = GB_ROOT * "/build/gen/made.h"
        g = gbtarget("needsgen", :shared;
                     sources = [unit("n.c"), BG.GraphSource(gen_c, 1, true), BG.GraphSource(gen_h, 0, true)],
                     groups = [gbgroup("C")])
        e = plan_error(() -> GB.plan_graph(gbgraph(g), GB_OUT))
        @test occursin("generated", e) && occursin("needsgen", e)
        @test occursin(gen_c, e) && occursin(gen_h, e)
        mkpath(dirname(gen_c)); write(gen_c, "int made(void) { return 1; }\n"); write(gen_h, "\n")
        @test length(only(GB.plan_graph(gbgraph(g), GB_OUT).libraries).tus) == 2
        # Generated entries no compile reads: a CMake `.rule` file, an export
        # `.def`, an object file. Not there, and not refused.
        r = gbtarget("rules", :shared;
                     sources = [unit("r.c"), BG.GraphSource(gen_c * ".rule", 0, true),
                                BG.GraphSource(GB_ROOT * "/build/exports.def", 0, true),
                                BG.GraphSource(GB_ROOT * "/build/other.o", 0, true)],
                     groups = [gbgroup("C")])
        @test length(only(GB.plan_graph(gbgraph(r), GB_OUT).libraries).tus) == 1
        @test GB._compile_reads(r, BG.GraphSource(GB_ROOT * "/build/x.HPP", 0, true))

        # A language with no IR: refused, except RC, which is left out and said so.
        asm = gbtarget("asmlib", :shared; sources = [unit("a.c", 1), unit("fast.S", 2)],
                       groups = [gbgroup("C"), gbgroup("ASM")])
        e = plan_error(() -> GB.plan_graph(gbgraph(asm), GB_OUT))
        @test occursin("asmlib", e) && occursin("fast.S", e) && occursin("ASM", e)
        rc = gbtarget("rclib", :shared; sources = [unit("a.c", 1), unit("ver.rc", 2)],
                      groups = [gbgroup("C"), gbgroup("RC")])
        lib = only(GB.plan_graph(gbgraph(rc), GB_OUT).libraries)
        @test lib.skipped_sources == [gbsrc("ver.rc")] && length(lib.tus) == 1

        # Nothing to link.
        hdr = gbtarget("hdronly", :shared; sources = [unit("h.h", 0)])
        @test occursin("hdronly", plan_error(() -> GB.plan_graph(gbgraph(hdr), GB_OUT)))

        # Names that are not one directory each on every host.
        one = gbtarget("Foo", :shared; sources = [unit("f1.c")], groups = [gbgroup("C")])
        two = gbtarget("foo", :shared; sources = [unit("f2.c")], groups = [gbgroup("C")])
        e = plan_error(() -> GB.plan_graph(gbgraph(one, two), GB_OUT))
        @test occursin("Foo", e) && occursin("foo", e) && occursin("case", e)
        slash = gbtarget("x", :shared; name = "a/b", sources = [unit("x.c")], groups = [gbgroup("C")])
        @test occursin("a/b", plan_error(() -> GB.plan_graph(gbgraph(slash), GB_OUT)))
    end

    @testset "the config a library links with" begin
        plan = GB.plan_graph(chain, GB_OUT; windows = false)
        lb = lib_named(plan, "beta")
        cfg = GB._library_config(lb; optimization_level = "3")
        CM = RepliBuild.ConfigurationManager
        @test cfg.wrap.language == :cpp
        @test cfg.link.link_dirs == lb.link_dirs && cfg.link.link_libraries == ["alpha"]
        @test !cfg.link.enable_lto && !cfg.link.promote_statics && !cfg.link.fallback
        @test replace(CM.get_build_path(cfg), '\\' => '/') == lb.dir * "/build"
        @test replace(CM.get_output_path(cfg), '\\' => '/') == lb.dir * "/lib"
        @test CM.get_library_name(cfg) == "libbeta.so"
        @test isempty(cfg.wrap.macros) && isempty(cfg.types.templates) && !cfg.wrap.tier1.enable
        # Codegen keeps the -O: the level reaches `_codegen_opt_flags`.
        @test RepliBuild.Compiler._codegen_opt_flags(cfg) == ["-O3", "-Xclang", "-disable-llvm-passes"]
        cl = only(GB.plan_graph(gbgraph(gbtarget("c1", :shared; sources = [unit("c.c")],
                                                 groups = [gbgroup("C")])), GB_OUT).libraries)
        @test GB._library_config(cl).wrap.language == :c
    end

    @testset "the compile fingerprint" begin
        inc = GB_ROOT * "/fp/include"
        mkpath(inc)
        write(inc * "/fp.h", "int fp(void);\n")
        t = gbtarget("fp", :shared; sources = [unit("fp.c")],
                     groups = [gbgroup("C"; flags = ["-O2"], includes = [BG.IncludeDir(inc, false)])])
        tu = only(only(GB.plan_graph(gbgraph(t), GB_OUT).libraries).tus)
        digests() = Dict(d => GB._include_digest(d) for d in tu.include_dirs)
        fp = GB._tu_fingerprint(tu, "toolchain", digests())
        @test fp == GB._tu_fingerprint(tu, "toolchain", digests())
        @test fp != GB._tu_fingerprint(tu, "other toolchain", digests())
        # Moving the output directory recompiles nothing.
        moved = only(only(GB.plan_graph(gbgraph(t), GB_ROOT * "/elsewhere").libraries).tus)
        @test moved.ir_file != tu.ir_file
        @test GB._tu_fingerprint(moved, "toolchain", digests()) == fp
        # A flag does; so does a header's contents.
        t2 = gbtarget("fp", :shared; sources = [unit("fp.c")],
                      groups = [gbgroup("C"; flags = ["-O3"], includes = [BG.IncludeDir(inc, false)])])
        tu2 = only(only(GB.plan_graph(gbgraph(t2), GB_OUT).libraries).tus)
        @test GB._tu_fingerprint(tu2, "toolchain", digests()) != fp
        write(inc * "/fp.h", "int fp(int);\n")
        @test GB._tu_fingerprint(tu, "toolchain", digests()) != fp
    end
end
