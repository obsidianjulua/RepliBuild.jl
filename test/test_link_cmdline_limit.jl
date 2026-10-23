# test_link_cmdline_limit.jl — a C++ link with more inputs than one Windows
# command line holds.
#
# llamacpp with ggml-vulkan has 384 translation units. From the Hub's checkout
# depth its llvm-link command line came to ~39,000 characters, past the 32,767
# CreateProcess accepts: llvm-link never started, and the clang fallback failed
# with ENAMETOOLONG. Links now pass a long input list in a response file
# (Compiler._args_with_response_file; the writer itself is pinned in
# test_response_files.jl).
#
# Two halves:
#   1. The real readers. llvm-link and clang read a response file by different
#      rules (Windows and GNU), so a forced one goes through both, on every host,
#      from a directory with a space in its name.
#   2. The build. Enough one-line TUs at a deep enough path that the link passes
#      the limit on Windows. Elsewhere it is an ordinary build, which must not
#      write a response file.

using Test
using RepliBuild
import Libdl

isdefined(@__MODULE__, :TestSupport) || include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

if requires("Link command-line limit", :clangxx, :llvm_link)

@testset "llvm-link and clang read the response file" begin
    Cm = RepliBuild.Compiler
    BB = RepliBuild.BuildBridge
    mktempdir() do root
        dir = joinpath(root, "with space")
        mkpath(dir)
        irs = String[]
        for i in 1:3
            src = joinpath(dir, "f$i.cpp")
            write(src, "extern \"C\" int rsp_f$i() { return $i; }\n")
            ir = joinpath(dir, "f$i.ll")
            (out, code) = BB.execute("clang++", ["-S", "-emit-llvm", "-O0", src, "-o", ir])
            @test code == 0
            push!(irs, ir)
        end
        @test count(isfile, irs) == 3

        # threshold 0 and windows = true: the @file branch on every host.
        linked = joinpath(dir, "linked.ll")
        args = Cm._args_with_response_file(["-S"], irs, ["-o", linked], joinpath(dir, "link.rsp");
                                           windows = true, threshold = 0)
        @test count(startswith('@'), args) == 1 && !any(in(irs), args)
        (out, code) = BB.execute(RepliBuild.LLVMEnvironment.resolve_tool("llvm-link", :cpp), args)
        @test code == 0
        text = isfile(linked) ? read(linked, String) : ""
        @test all(i -> occursin(Regex("define [^\\n]*@rsp_f$i\\("), text), 1:3)

        lib = joinpath(dir, "librsp." * Libdl.dlext)
        args = Cm._args_with_response_file(["-shared", "-fPIC"], irs, ["-o", lib], joinpath(dir, "lib.rsp");
                                           windows = true, threshold = 0)
        (out, code) = BB.execute("clang++", args)
        @test code == 0
        if isfile(lib)
            h = Libdl.dlopen(lib)
            try
                @test all(i -> Libdl.dlsym(h, Symbol("rsp_f$i"); throw_error = false) !== nothing, 1:3)
            finally
                Libdl.dlclose(h)
            end
        else
            @test isfile(lib)
        end
    end
end

@testset "a C++ build whose link passes the command-line limit" begin
    n = 200
    mktempdir() do root
        # Each IR path is <proj>/build/u_NNNN.cpp-<8 hex>.ll. A 170-character
        # project path makes 200 of them ~40,000 characters, and keeps the
        # longest file the build writes (an .ll.tmp beside them) under MAX_PATH.
        proj = joinpath(root, "p" * "x"^max(1, 170 - length(root) - 2))
        mkpath(joinpath(proj, "src"))
        names = [string("u_", lpad(i, 4, '0')) for i in 1:n]
        for nm in names
            write(joinpath(proj, "src", nm * ".cpp"), "extern \"C\" int $(nm)_v() { return 1; }\n")
        end
        toml = joinpath(proj, "replibuild.toml")
        write(toml, """
        [project]
        name = "rsp_limit"
        root = "."

        [compile]
        flags = ["-O0"]
        source_files = [$(join(("\"src/$nm.cpp\"" for nm in names), ", "))]

        [link]
        enable_lto = false

        [binary]
        type = "shared"

        [wrap]
        language = "cpp"
        """)

        lib = RepliBuild.build(toml)
        @test isfile(lib)

        rsp    = joinpath(proj, "build", "rsp_limit_link.rsp")
        linked = joinpath(proj, "build", "rsp_limit_linked.ll")
        @test isfile(linked)              # llvm-link ran: no fallback to a clang-side link
        if Sys.iswindows()
            @info "Windows: the link passes the command-line limit and takes the response file"
            @test isfile(rsp) && length(readlines(rsp)) == n
        else
            @info "Not Windows: an ordinary link, no response file"
            @test !ispath(rsp)
        end

        h = Libdl.dlopen(lib)
        try
            @test all(nm -> Libdl.dlsym(h, Symbol(nm * "_v"); throw_error = false) !== nothing, names)
        finally
            Libdl.dlclose(h)
        end
    end
end

end # requires
