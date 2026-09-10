# Toolchain install advice — one answer per question.
#
# The same fact used to be stated in several places and drift independently:
#
#   * the LLVM floor:  EnvironmentDoctor's own `const = 21`, build.sh's
#     `LLVM_MIN_MAJOR=21`, and a third implicit copy in LLVMEnvironment's prefix
#     ladder — which is how that ladder ended up probing LLVM 20 down to 15
#     against a minimum of 21.
#   * the Arch install line: two spellings in build.sh, a third in the doctor.
#   * the Debian/Ubuntu line: three spellings in build.sh alone, one of them
#     (`mlir-21-dev`) a hardcoded version sitting beside a `${LLVM_MIN_MAJOR}`
#     interpolation in the same file.
#
# Nothing here checks that the advice is CORRECT — no test can know what a
# distro calls its packages this month. It checks that there is exactly one of
# each, which is what makes fixing the advice a single edit.

using Test
using RepliBuild

const _BUILD_SH = joinpath(dirname(dirname(pathof(RepliBuild))), "src", "mlir", "build.sh")
const _DOCTOR   = RepliBuild.EnvironmentDoctor

@testset "Toolchain advice has one source" begin

    @test isfile(_BUILD_SH)
    sh = read(_BUILD_SH, String)

    code = [l for l in eachsplit(sh, '\n') if !startswith(strip(l), "#")]

    @testset "build.sh derives the LLVM floor, never restates it" begin
        # An assignment from a literal is the second copy coming back. The
        # derivation line assigns from a $(...) substitution, so it is exempt.
        restated = [l for l in code if match(r"^LLVM_MIN_MAJOR\s*=\s*[0-9]", strip(l)) !== nothing]
        @test isempty(restated) || error("build.sh restates the LLVM floor: $restated")
        @test occursin("MIN_LLVM_VERSION", sh)          # reads the Julia const
        @test occursin("LLVM_ENV_JL", sh)

        # And the derivation actually yields the Julia value on this checkout —
        # a regex that silently stops matching would otherwise go unnoticed
        # until someone's build failed with an empty floor.
        derived = readchomp(pipeline(`sed -n 's/^const MIN_LLVM_VERSION = \([0-9][0-9]*\).*/\1/p'
                                      $(joinpath(dirname(_BUILD_SH), "..", "Builder", "LLVMEnvironment.jl"))`,
                                     `head -1`))
        @test !isempty(derived)
        @test parse(Int, derived) == RepliBuild.LLVMEnvironment.MIN_LLVM_VERSION
    end

    @testset "no hardcoded package versions in build.sh" begin
        # `mlir-21-dev`, `llvm-22-dev`, `llvm.sh 21` — each is a version that
        # cannot follow the floor. Comments are exempt; they describe history.
        pinned = [l for l in code
                  if match(r"\b(?:mlir|llvm)-[0-9]+-dev\b", l) !== nothing ||
                     match(r"llvm\.sh\s+[0-9]+", l) !== nothing]
        @test isempty(pinned) || error("build.sh pins a package version: $pinned")
    end

    @testset "one install hint, and the failure paths all use it" begin
        @test count(l -> occursin(r"^install_hint\(\)\s*\{", l), code) == 1
        @test count(l -> strip(l) == "install_hint", code) >= 3

        # Package-manager lines may appear ONLY inside the hint function. A
        # second one anywhere else is the divergence this file exists to stop.
        inside = false
        strays = String[]
        for line in code
            if occursin(r"^install_hint\(\)\s*\{", line)
                inside = true; continue
            end
            if inside
                strip(line) == "}" && (inside = false)
                continue
            end
            match(r"\b(?:yay -S|apt install|dnf install|pacman -S)\b", line) === nothing ||
                push!(strays, String(line))
        end
        @test isempty(strays) || error("install advice outside install_hint(): $strays")
    end

    @testset "shell hint and Julia advice agree" begin
        # The doctor's advice is the reference; build.sh must not invent its own
        # spelling of the same instruction. Compared on package sets rather than
        # whole lines, so formatting/padding differences do not fail the test.
        advice = join(_DOCTOR._LLVM_ADVICE, "\n")

        pkgs(text, pat) = Set(m.captures[1] for m in eachmatch(pat, text))
        arch_pat = r"yay -S ([a-z0-9 .+-]+)"
        fed_pat  = r"dnf install ([a-z0-9 .+-]+)"

        for (label, pat) in (("Arch", arch_pat), ("Fedora", fed_pat))
            sh_set = pkgs(sh, pat)
            jl_set = pkgs(advice, pat)
            @test !isempty(sh_set)
            @test !isempty(jl_set)
            @test Set(split(strip(only(sh_set)))) == Set(split(strip(only(jl_set)))) ||
                  error("$label install line differs between build.sh and EnvironmentBuild advice:\n" *
                        "  build.sh: $(only(sh_set))\n  doctor:   $(only(jl_set))")
        end

        # The Debian/Ubuntu line is version-bearing on both sides; assert they
        # name the same installer, and that the doctor's carries the floor.
        @test occursin("apt.llvm.org/llvm.sh", sh)
        @test occursin("apt.llvm.org/llvm.sh", advice)
        @test occursin(string(RepliBuild.LLVMEnvironment.MIN_LLVM_VERSION), advice)
    end

    @testset "the floor has one definition in Julia" begin
        @test _DOCTOR.MIN_LLVM_VERSION === RepliBuild.LLVMEnvironment.MIN_LLVM_VERSION
    end
end
