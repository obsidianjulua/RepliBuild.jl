# LLVM prefix search — version gating and newest-wins.
#
# The list this guards used to run `/usr/lib/llvm-20` down to `llvm-15`: every
# entry BELOW the 21 minimum, and 21+ absent entirely. Two consequences, both
# silent. A Debian box with LLVM 22 at /usr/lib/llvm-22 had its only usable
# install walked straight past. And the loop version-checked nothing at all, so
# the first prefix carrying clang++/llvm-config won — on a distro whose default
# /usr LLVM is too old, that was the answer, and the failure surfaced later
# inside CMake naming a missing header rather than here naming a version.
#
# Driven over FABRICATED prefixes: no LLVM needed, and the assertions do not
# depend on what this machine happens to have installed.

using Test
using RepliBuild

const L = RepliBuild.LLVMEnvironment

# A prefix that looks like a real LLVM install and reports `major` from its
# llvm-config. `nothing` writes an llvm-config that answers garbage.
function fake_prefix(root, name, major)
    p = joinpath(root, name)
    mkpath(joinpath(p, "bin")); mkpath(joinpath(p, "lib")); mkpath(joinpath(p, "include"))
    cfg = joinpath(p, "bin", "llvm-config")
    body = major === nothing ? "not a version" : "$(major).1.8"
    write(cfg, "#!/bin/sh\necho '$body'\n")
    chmod(cfg, 0o755)
    touch(joinpath(p, "bin", "clang++"))
    return p
end

@testset "LLVM prefix search" begin

    @testset "_llvm_major_of" begin
        mktempdir() do d
            @test L._llvm_major_of(joinpath(fake_prefix(d, "v22", 22), "bin", "llvm-config")) == 22
            @test L._llvm_major_of(joinpath(fake_prefix(d, "v14", 14), "bin", "llvm-config")) == 14
            # Unparseable and missing both score 0 — "cannot be asked" is not
            # "is fine", which is how an unusable prefix used to get accepted.
            @test L._llvm_major_of(joinpath(fake_prefix(d, "junk", nothing), "bin", "llvm-config")) == 0
            @test L._llvm_major_of(joinpath(d, "nope", "llvm-config")) == 0
        end
    end

    @testset "below the minimum is refused" begin
        mktempdir() do d
            old = fake_prefix(d, "old", L.MIN_LLVM_VERSION - 1)
            @test L._best_llvm_prefix([old], "") === nothing
        end
    end

    @testset "newest wins, not first-listed" begin
        mktempdir() do d
            # The regression shape: an acceptable-but-older prefix listed FIRST,
            # a newer one after it. The old loop returned the first hit.
            older = fake_prefix(d, "llvm-21", L.MIN_LLVM_VERSION)
            newer = fake_prefix(d, "llvm-22", L.MIN_LLVM_VERSION + 1)
            @test L._best_llvm_prefix([older, newer], "") == newer
            @test L._best_llvm_prefix([newer, older], "") == newer
        end
    end

    @testset "too-old default does not shadow a supported versioned tree" begin
        mktempdir() do d
            usr  = fake_prefix(d, "usr", 14)                        # distro default
            good = fake_prefix(d, "usr-lib-llvm-22", L.MIN_LLVM_VERSION + 1)
            @test L._best_llvm_prefix([usr, good], "") == good      # was `usr`
        end
    end

    @testset "incomplete prefixes are skipped" begin
        mktempdir() do d
            ok = fake_prefix(d, "complete", L.MIN_LLVM_VERSION)

            no_lib = fake_prefix(d, "nolib", L.MIN_LLVM_VERSION)
            rm(joinpath(no_lib, "lib"); recursive=true)
            @test L._best_llvm_prefix([no_lib], "") === nothing

            no_clang = fake_prefix(d, "noclang", L.MIN_LLVM_VERSION)
            rm(joinpath(no_clang, "bin", "clang++"))
            @test L._best_llvm_prefix([no_clang], "") === nothing

            @test L._best_llvm_prefix([no_lib, no_clang, ok], "") == ok
            @test L._best_llvm_prefix(String[], "") === nothing
        end
    end

    @testset "the generated ladder covers the supported range and nothing below" begin
        # Reproduces the list the resolver builds for Debian-style layouts.
        ladder = ["/usr/lib/llvm-$v"
                  for v in L.MAX_PROBED_LLVM_VERSION:-1:L.MIN_LLVM_VERSION]
        @test "/usr/lib/llvm-$(L.MIN_LLVM_VERSION)" in ladder
        @test "/usr/lib/llvm-$(L.MAX_PROBED_LLVM_VERSION)" in ladder
        # The exact entries that used to be there, and must never come back.
        for v in 15:(L.MIN_LLVM_VERSION - 1)
            @test !("/usr/lib/llvm-$v" in ladder)
        end
        @test L.MAX_PROBED_LLVM_VERSION > L.MIN_LLVM_VERSION
    end

    @testset "the minimum has ONE definition" begin
        # EnvironmentDoctor used to declare its own `= 21`. Three copies of this
        # fact is how the ladder drifted to probing 20 down to 15.
        @test RepliBuild.EnvironmentDoctor.MIN_LLVM_VERSION === L.MIN_LLVM_VERSION
    end
end
