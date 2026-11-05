# test/test_compile_never_ran.jl — a compile that did not run is a failure, not
# the last build's IR.
#
# `compile_single_to_ir` judged success by `isfile(ir_file)`. A compiler that
# fails deletes its `-o` output (the clang driver does), but one that never
# started deletes nothing: SPAWN_FAILED from a command line past Windows'
# 32,767 characters, or a JLL clang the OS refuses to run (Smart App Control,
# 2026-10-18). The previous build's IR then passed for this build's, was
# stamped with the NEW fingerprint, and was linked; the build reported success,
# and every later build with those flags took the stale IR from the cache
# (audit 2026-11-05, B3, reproduced through a 40,618-character include path).
#
# The trigger here is a define of 200,000 characters: past Windows' whole
# command line and Linux's 128 KiB per argument, so the spawn itself fails on
# both hosts and no compiler, JLL or system, has to exist. Negative-checked by
# restoring `success = isfile(ir_file)` without the removal: every assertion on
# the IR, the key and the batch error fails.

using Test
using RepliBuild

const _CNR_C  = RepliBuild.Compiler
const _CNR_CM = RepliBuild.ConfigurationManager
const _CNR_BB = RepliBuild.BuildBridge

_cnr_toml_path(p) = replace(p, '\\' => '/')

function _cnr_project(dir, sources)
    mkpath(joinpath(dir, "src"))
    paths = String[]
    for (i, s) in enumerate(sources)
        p = joinpath(dir, "src", "f$i.c")
        write(p, s)
        push!(paths, p)
    end
    toml = joinpath(dir, "replibuild.toml")
    write(toml, """
    [project]
    name = "neverran"
    root = "$(_cnr_toml_path(dir))"

    [compile]
    source_files = [$(join(["\"$(_cnr_toml_path(p))\"" for p in paths], ", "))]

    [compile.defines]
    PAD = "$("x"^200_000)"

    [cache]
    enabled = true

    [wrap]
    language = "c"
    """)
    return (_CNR_CM.load_config(toml), paths)
end

# The previous build's output, as a compile that does not run would find it.
function _cnr_stale!(cfg, src)
    ir = _CNR_C._ir_output_path(_CNR_C.get_build_path(cfg), src)
    mkpath(dirname(ir))
    write(ir, "; the previous build's IR\n")
    write(ir * ".key", "the previous build's fingerprint")
    return ir
end

@testset "A compile that never ran is a failure" begin
    @testset "compile_single_to_ir" begin
        mktempdir() do d
            (cfg, (src,)) = _cnr_project(d, ["int f1(void) { return 1; }\n"])
            ir = _cnr_stale!(cfg, src)
            (path, ok, code) = _CNR_C.compile_single_to_ir(cfg, src)
            @test path == ir
            @test code == _CNR_BB.SPAWN_FAILED        # the trigger did what it is for
            @test !ok
            @test !isfile(ir)                         # the old IR is not this build's
            @test !isfile(ir * ".key")                # and nothing claims it is current
        end
    end

    # Both batch paths: one file compiles sequentially, two in `Threads.@threads`.
    for n in (1, 2)
        @testset "compile_to_ir, $n source(s)" begin
            mktempdir() do d
                (cfg, srcs) = _cnr_project(d, ["int f$i(void) { return $i; }\n" for i in 1:n])
                irs = [_cnr_stale!(cfg, s) for s in srcs]
                @test_throws ErrorException _CNR_C.compile_to_ir(cfg, srcs)
                @test !any(isfile, irs)
                fp = _CNR_C.compute_compile_fingerprint(cfg)
                # The key the old code stamped is what made the stale IR a cache
                # hit for this config on every later build.
                @test all(i -> _CNR_C.needs_recompile(srcs[i], irs[i], true, fp), eachindex(irs))
            end
        end
    end
end
