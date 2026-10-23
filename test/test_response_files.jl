# test_response_files.jl — a link longer than one Windows command line.
#
# CreateProcess accepts at most 32,767 characters of command line, and a link
# names every translation unit. llamacpp with ggml-vulkan has 384, whose IR paths
# came to ~39,000 characters from the Hub's checkout depth. llvm-link never
# started; BuildBridge reported that as "exit 1", so the warning blamed llvm-link;
# and the clang fallback, handed the same list, died with ENAMETOOLONG. A long
# input list now goes to a response file (Compiler._args_with_response_file), and
# a program that never started comes back as BuildBridge.SPAWN_FAILED with a
# message that says so.
#
# No toolchain: the writer is pure, and a missing program is enough for a spawn
# failure. test_link_cmdline_limit.jl (devtests) runs the real llvm-link and
# clang over these files and builds a project past the limit.

using Test
using RepliBuild

const _RSP_C  = RepliBuild.Compiler
const _RSP_BB = RepliBuild.BuildBridge

@testset "link inputs go to a response file past the command-line limit" begin
    mktempdir() do dir
        # Windows spellings (drive, backslashes, a space), at the depth of a Hub
        # checkout: the case the writer is for.
        build = "C:\\Users\\someone\\Desktop\\Projects\\RepliBuild Hub\\packages\\llamacpp\\build"
        inputs = [string(build, "\\unit_", lpad(i, 4, '0'), ".cpp-0123abcd.ll") for i in 1:400]
        head = ["-S"]
        tail = ["-o", build * "\\llamacpp_linked.ll"]
        rsp  = joinpath(dir, "sub dir", "big_link.rsp")

        # The 400 inputs really do pass the limit, so the cases below are not vacuous.
        @test _RSP_C._cmdline_chars(vcat(head, inputs, tail)) > 32_767

        @testset "a command line that fits is left alone" begin
            few = inputs[1:3]
            @test _RSP_C._args_with_response_file(head, few, tail, rsp; windows = true) ==
                  vcat(head, few, tail)
            @test !ispath(rsp)
        end

        @testset "off Windows the inputs stay on the command line" begin
            @test _RSP_C._args_with_response_file(head, inputs, tail, rsp; windows = false) ==
                  vcat(head, inputs, tail)
            @test !ispath(rsp)
        end

        @testset "on Windows they move to an @file and the flags stay" begin
            args = _RSP_C._args_with_response_file(head, inputs, tail, rsp; windows = true)
            @test args == vcat(head, ["@" * rsp], tail)
            @test _RSP_C._cmdline_chars(args) < 32_767
            lines = readlines(rsp)
            @test length(lines) == length(inputs)
            # One double-quoted, /-separated path per line: what Windows-rule
            # llvm-link and GNU-rule clang both read as the same path.
            @test all(l -> startswith(l, '"') && endswith(l, '"') && count(==('"'), l) == 2, lines)
            @test !any(l -> occursin('\\', l), lines)
            @test [l[2:end-1] for l in lines] == [replace(f, '\\' => '/') for f in inputs]
        end

        @testset "POSIX paths are written as they are" begin
            # The encoding, not the decision: threshold 0 takes the @file branch.
            posix = [string("/home/someone/hub dir/build/unit_", i, ".ll") for i in 1:3]
            rsp2 = joinpath(dir, "posix.rsp")
            args = _RSP_C._args_with_response_file(String[], posix, String[], rsp2;
                                                   windows = true, threshold = 0)
            @test args == ["@" * rsp2]
            @test [l[2:end-1] for l in readlines(rsp2)] == posix
        end

        @testset "a path a response file cannot carry is refused" begin
            for bad in ("C:\\x\\a\"b.ll", "C:\\x\\a\nb.ll")
                err = try
                    _RSP_C._args_with_response_file(head, vcat(inputs, [bad]), tail,
                                                    joinpath(dir, "bad.rsp"); windows = true)
                    nothing
                catch e
                    e
                end
                @test err isa ErrorException
                @test occursin("response file", err.msg)
            end
        end
    end

    # Every link that names a list of inputs goes through the writer. Read from the
    # source, the way test_codegen_flags.jl pins the codegen call sites.
    src = read(joinpath(pkgdir(RepliBuild), "src", "Builder", "Compiler.jl"), String)
    @test occursin("_args_with_response_file([\"-S\"], ir_files, [\"-o\", linked_ir],", src)  # llvm-link
    @test occursin("_args_with_response_file(cmd_head, files, cmd_tail, rsp_path)", src)      # create_library
    @test occursin("_args_with_response_file(cmd_head, objs, cmd_tail, rsp_path)", src)       # per-TU objects
    @test occursin("_args_with_response_file(String[], files, cmd_tail,", src)                # create_executable
    @test !occursin("vcat(cmd_head, files, cmd_tail)", src)
    @test !occursin("vcat(cmd_head, objs, cmd_tail)", src)
end

@testset "a program that never started is reported as one" begin
    (out, code) = _RSP_BB.execute("replibuild-no-such-program", ["--version"])
    @test code == _RSP_BB.SPAWN_FAILED
    @test startswith(out, "could not start replibuild-no-such-program")
    @test occursin("ENOENT", out)

    # A program that ran and failed keeps its own exit code. Without the LLVM
    # environment, which is for LLVM tools, not for starting Julia.
    julia = Base.julia_cmd().exec[1]
    (_, code) = _RSP_BB.execute(julia, ["--startup-file=no", "-e", "exit(3)"]; use_llvm_env = false)
    @test code == 3

    if Sys.iswindows()
        # The case that started it: a command line CreateProcess refuses.
        (out, code) = _RSP_BB.execute(julia, [repeat("x", 40_000)]; use_llvm_env = false)
        @test code == _RSP_BB.SPAWN_FAILED
        @test occursin("ENAMETOOLONG", out)
        @test occursin(r"\b4\d{4} characters of command line", out)   # sized...
        @test length(out) < 1_000                                       # ...not echoed back
    end
end
