# test_jlcs_artifact_live.jl — the prebuilt libJLCS, loaded and run.
#
# The `JLCS` artifact is a libJLCS with LLVM and MLIR linked in
# (deps/jlcs_jll/build_tarballs.jl). This runs the dialect through it in a
# child with REPLIBUILD_JLCS=artifact, whatever local build is beside it: the
# library must come from the artifact store, load with every symbol resolved,
# depend on no LLVM or MLIR library, and parse, lower and JIT a module.
#
# Skipped where there is no artifact for the host or it cannot be fetched
# (offline, or a release not yet published).

using Test
using RepliBuild

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

haskey(ENV, "REPLIBUILD_HOME") || (ENV["REPLIBUILD_HOME"] = mktempdir())

const _JAL_LOC = RepliBuild.JLCSLibrary.artifact_location(; install = true)

if _JAL_LOC.source !== :artifact
    @testset "prebuilt libJLCS, live" begin
        @info "skipped: no prebuilt libJLCS here" reason = _JAL_LOC.note
        @test_skip false
    end
else
    @testset "prebuilt libJLCS, live" begin
        code = """
        using RepliBuild, Libdl
        M = RepliBuild.MLIRNative
        loc = RepliBuild.JLCSLibrary.locate()
        println("PROBE source: ", loc.source === :artifact ? "PASS" : "FAIL \$(loc.source)")
        p = M.libJLCS_path()
        println("PROBE path: ", p == $(repr(_JAL_LOC.path)) ? "PASS" : "FAIL \$p")
        h = Libdl.dlopen(p, Libdl.RTLD_NOW | Libdl.RTLD_LOCAL; throw_error = false)
        println("PROBE rtld_now: ", h === nothing ? "FAIL does not load" : "PASS")
        J = RepliBuild.JLCSLibrary
        v = J.library_abi_version(p)
        println("PROBE abi: ", v == J.ABI_VERSION ? "PASS" : "FAIL library \$v, RepliBuild \$(J.ABI_VERSION)")
        ok = try
            redirect_stdout(devnull) do
                M.test_dialect()
            end
            true
        catch e
            println("test_dialect threw: ", sprint(showerror, e))
            false
        end
        println("PROBE dialect: ", ok ? "PASS" : "FAIL")
        sum = Ref{Int32}(0)
        rc = ccall((:jlcs_selftest_jit_add, M.libJLCS), Cint, (Int32, Int32, Ptr{Int32}), 40, 2, sum)
        println("PROBE selftest: ", (rc == 0 && sum[] == 42) ? "PASS" : "FAIL rc=\$rc sum=\$(sum[])")
        loaded = filter(l -> occursin(r"lib(LLVM|MLIR)[-.]", basename(l)) && !occursin("julia", l), Libdl.dllist())
        println("PROBE no_system_llvm: ", isempty(loaded) ? "PASS" : "FAIL \$loaded")
        println("PROBE_DONE")
        """
        r = withenv("REPLIBUILD_JLCS" => "artifact") do
            run_isolated(`-e $code`; timeout = 300)
        end
        v = probe_verdicts(r.output)
        @test r.ok
        @test haskey(v, "PROBE_DONE")
        for name in ("source", "path", "rtld_now", "abi", "dialect", "selftest", "no_system_llvm")
            @test startswith(get(v, name, "no PROBE line"), "PASS")
        end
    end
end
