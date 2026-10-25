# test_codegen_flags.jl — IR is turned into machine code at the link level.
#
# create_library handed clang `-shared -fPIC` and the optimized IR, with no -O:
# the -O0 backend (FastISel + the fast register allocator) generated every
# shipped library, and on LLVM 23 crashed outright ("Cannot emit physreg copy
# instruction", llama.cpp on Windows). Codegen now runs at [link]
# optimization_level with the IR pipeline disabled, since that IR is already
# optimized. No toolchain.

using Test
using RepliBuild

const _CG_CM = RepliBuild.ConfigurationManager
const _CG_C  = RepliBuild.Compiler

function _cg_config(level::String)
    dir = mktempdir()
    toml = joinpath(dir, "replibuild.toml")
    write(toml, """
    [project]
    name = "cgprobe"
    root = "."
    [link]
    optimization_level = "$level"
    """)
    return _CG_CM.load_config(toml)
end

@testset "codegen opt flags" begin
    for (level, flag) in (("0", "-O0"), ("2", "-O2"), ("3", "-O3"), ("s", "-Os"), ("z", "-Oz"))
        @test _CG_C._codegen_opt_flags(_cg_config(level)) == [flag, "-Xclang", "-disable-llvm-passes"]
    end

    # Both IR -> machine code sites use it: the merged-module link and the per-TU
    # fallback. Read from the source, the way the config-surface guard does.
    src = read(joinpath(pkgdir(RepliBuild), "src", "Builder", "Compiler.jl"), String)
    @test occursin("_codegen_opt_flags(config)...,", src)                       # create_library
    @test occursin(r"vcat\(\[\"-c\", \"-fPIC\"\], _codegen_opt_flags\(config\)", src)  # per-TU fallback
end

# The same IR links to the same bytes. A PE image holds its link time, which lld
# and GNU ld take from the clock unless told not to, so every relink changed a
# wrapper's LIBRARY_SHA; ELF holds none. Each link of a shipped binary passes
# the flags. The relinks themselves are in test_reproducible_link.jl (devtests).
@testset "reproducible link flags" begin
    @test _CG_C._reproducible_link_flags(windows=true) == ["-Wl,--no-insert-timestamp"]
    @test _CG_C._reproducible_link_flags(windows=false) == String[]
    @test _CG_C._reproducible_link_flags() == _CG_C._reproducible_link_flags(windows=Sys.iswindows())

    # A function's text runs to the first `end` in column 0.
    body(file, name) = something(match(Regex("\\nfunction $name\\(.*?\\nend\\n", "s"),
                                       read(joinpath(pkgdir(RepliBuild), "src", "Builder", file), String)),
                                 (; match = "")).match
    @test occursin("_reproducible_link_flags()", body("Compiler.jl", "create_library"))
    @test occursin("_reproducible_link_flags()", body("Compiler.jl", "create_executable"))
    @test occursin("Compiler._reproducible_link_flags()", body("ThunkBuilder.jl", "build_aot_thunks"))

    # A library binds its own exported symbols to itself. On ELF the dynamic
    # linker resolves a library's own PLT calls and GOT loads through the
    # process first, so a library loaded earlier with the same name took them
    # (Julia's own libz, for the Hub's zlib); `-Bsymbolic` makes the library
    # search itself first. A DLL never goes through its import table for its
    # own symbols, so PE needs nothing. Every library link passes the flags.
    @test _CG_C._self_binding_link_flags(windows=false) == ["-Wl,-Bsymbolic"]
    @test _CG_C._self_binding_link_flags(windows=true) == String[]
    @test _CG_C._self_binding_link_flags() == _CG_C._self_binding_link_flags(windows=Sys.iswindows())
    @test occursin("_self_binding_link_flags()", body("Compiler.jl", "create_library"))
end

# One LLVM per module. A C++ package routed its .c files to the JLL clang 18 and
# linked them with LLVM 23's llvm-link, which rejected their debug info and
# dropped it with a warning: every ggml.c function wrapped from an inferred
# signature (ggml_version -> Cint).
@testset "one toolchain per module" begin
    clean = "llvm-link: linking 3 modules\n"
    @test _CG_C._assert_debug_info_kept(clean, "llvm-link") === nothing
    dirty = """
            Function-local variables are not allowed in a DICompileUnit's global variables list
            warning: ignoring invalid debug info in C:\\pkg\\build\\ggml.c-4b130895.ll
            warning: ignoring invalid debug info in /pkg/build/quants.c-1.ll
            """
    err = try
        _CG_C._assert_debug_info_kept(dirty, "llvm-link"); nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("2 module(s)", err.msg)
    @test occursin("ggml.c-4b130895.ll", err.msg) && occursin("quants.c-1.ll", err.msg)

    # The C bucket is chosen per package, not per file extension.
    src = read(joinpath(pkgdir(RepliBuild), "src", "Builder", "Compiler.jl"), String)
    @test occursin("(output, exitcode) = config.wrap.language == :c ?\n        _clang_for_c_bucket(compiler, cmd_args) :\n        BuildBridge.execute(compiler, cmd_args)", src)

    # A C++ package's IR cache is keyed on the system compiler too.
    cfg(lang) = (d = mktempdir(); t = joinpath(d, "replibuild.toml");
                 write(t, "[project]\nname = \"fp\"\nroot = \".\"\n[wrap]\nlanguage = \"$lang\"\n");
                 _CG_CM.load_config(t))
    @test _CG_C.compute_compile_fingerprint(cfg("c")) != _CG_C.compute_compile_fingerprint(cfg("cpp"))
    @test !isempty(_CG_C._system_clang_identity())
end

# libJLCS is pinned to the LLVM it was built against; check_environment used to
# report it ready as long as the file existed.
@testset "libJLCS drift is detected" begin
    stale = RepliBuild.EnvironmentDoctor._jlcs_stale
    @test !stale("23.1.3", "23.1.5")        # patch bump: same SONAME pin
    @test stale("23.1.3", "24.1.0")         # major upgrade
    @test stale("22.1.8", "22.2.0")         # minor moves the Linux SONAME
    @test !stale("", "24.1.0")              # unstamped artifact: no evidence, load probe decides
    @test !stale("23.1.3", "")              # no llvm-config
end
