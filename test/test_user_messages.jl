# test_user_messages.jl — what RepliBuild tells a user is true, and names the fix.
#
# A walkthrough of the user docs as a new user and an audit of the messages
# (2026-10-21) found advice that sent people the wrong way:
#
#   * `search` ended with "Install: RepliBuild.use(...)", which fails for every
#     name nobody registered — every Hub package on a fresh install;
#   * the doctor reported clang64\bin missing from PATH as a stale build, and its
#     summary promised C++ builds on a box with no clang++;
#   * `discover(path, build=true, wrap=true)` did nothing on a project that
#     already had its replibuild.toml;
#   * `wrap` asked "Did you compile with -g flag?" (RepliBuild always passes -g)
#     and fell back to a wrapper its own guard refuses;
#   * the build's load check called a Smart App Control block, or a DLL missing
#     from PATH, "unresolved symbols".
#
# Pure functions and file operations: no toolchain.

using Test
using RepliBuild

const _PR = RepliBuild.PackageRegistry
const _ED = RepliBuild.EnvironmentDoctor
const _CO = RepliBuild.Compiler

@testset "User-facing messages" begin

    @testset "search: the Hub steps register before they use" begin
        steps = join(_PR._HUB_USE_STEPS, "\n")
        @test occursin("does not install", steps)
        @test occursin("RepliBuild.register(", steps)
        @test occursin("packages/<name>/replibuild.toml", steps)
        @test first(findfirst("register(", steps)) < first(findfirst("use(", steps))
        @test !occursin("Install:", steps)
    end

    @testset "use: a missing name points at register and the Hub" begin
        mktempdir() do home
            withenv("REPLIBUILD_HOME" => home) do
                err = try
                    _PR.use("no_such_pkg_for_this_test"; verbose=false)
                    nothing
                catch e
                    e
                end
                @test err isa ErrorException
                msg = sprint(showerror, err)
                @test occursin("not in registry", msg)
                @test occursin("RepliBuild.register(", msg)
                @test occursin("RepliBuild-Hub", msg)
            end
        end
    end

    @testset "discover keeps an existing replibuild.toml and still builds" begin
        mktempdir() do dir
            toml = joinpath(dir, "replibuild.toml")
            text = """
            [project]
            name = "keepme"
            root = "."

            [compile]
            source_files = ["no_such_file.c"]

            [wrap]
            language = "c"
            """
            write(toml, text)
            mktempdir() do home
                withenv("REPLIBUILD_HOME" => home) do
                    # Without build=true nothing else runs.
                    @test RepliBuild.discover(dir) == toml
                    # With it, build runs against the kept file — which refuses
                    # the missing source. It used to return before building.
                    err = try
                        RepliBuild.discover(dir; build=true, wrap=true)
                        nothing
                    catch e
                        e
                    end
                    @test err !== nothing
                    err === nothing || @test occursin("no_such_file.c", sprint(showerror, err))
                end
            end
            @test read(toml, String) == text          # kept as written, not rescanned
        end
    end

    @testset "wrap without compilation_metadata.json says to build" begin
        mktempdir() do dir
            toml = joinpath(dir, "replibuild.toml")
            write(toml, """
            [project]
            name = "nometa"
            root = "."

            [compile]
            source_files = []

            [wrap]
            language = "c"
            """)
            cfg = RepliBuild.ConfigurationManager.load_config(toml)
            lib = joinpath(dir, "libnometa." * Base.Libc.Libdl.dlext)
            write(lib, "not a library")
            err = try
                RepliBuild.Wrapper.wrap_library(cfg, lib)
                nothing
            catch e
                e
            end
            @test err isa ErrorException
            msg = sprint(showerror, err)
            @test occursin("compilation_metadata.json", msg)
            @test occursin("RepliBuild.build(", msg)
            @test !occursin("Did you compile", msg)
        end
    end

    @testset "the build's load check names the Windows causes" begin
        lib = "C:/proj/julia/libfoo.dll"
        sac = _CO._load_failure_message(lib,
            "could not load library \"$lib\"\nAn Application Control policy has blocked this file.")
        @test occursin("Smart App Control", sac)
        @test occursin("3077", sac)
        @test !occursin("unresolved symbols", sac)

        nomod = _CO._load_failure_message(lib,
            "could not load library \"$lib\"\nThe specified module could not be found.")
        @test occursin("could not be found", nomod)
        @test occursin("PATH", nomod)
        @test !occursin("unresolved symbols", nomod)

        unres = _CO._load_failure_message("/proj/julia/libfoo.so",
            "/proj/julia/libfoo.so: undefined symbol: dl_error")
        @test occursin("unresolved symbols", unres)
        @test occursin("link_libraries", unres)
    end

    @testset "doctor: clang64\\bin off PATH is told apart from a stale build" begin
        mktempdir() do root
            bin = joinpath(root, "clang64", "bin")
            mkpath(bin)
            # No CLANG64 libraries there: no evidence, so no claim.
            @test _ED._clang64_bin_off_path(windows=true, root=root, path="") === nothing
            write(joinpath(bin, "libc++.dll"), "")
            @test _ED._clang64_bin_off_path(windows=true, root=root, path="C:\\Windows") == bin
            # On PATH, in either slash style, any case, trailing separator: fine.
            spelled = uppercase(replace(bin, '/' => '\\')) * "\\"
            @test _ED._clang64_bin_off_path(windows=true, root=root,
                                            path="C:\\Windows;" * spelled) === nothing
            @test _ED._clang64_bin_off_path(windows=true, root=root, path=bin) === nothing
            # Off Windows the question does not arise.
            @test _ED._clang64_bin_off_path(windows=false, root=root, path="") === nothing
        end
        # The install advice names the PATH entry, the other half of "not found".
        @test any(l -> occursin("PATH", l), _ED._LLVM_ADVICE_WINDOWS)
    end

    @testset "doctor: the MSYS2 environment is read from clang++'s location" begin
        @test _ED._msys2_env("C:\\msys64\\clang64\\bin\\clang++.exe") == "clang64"
        @test _ED._msys2_env("C:/msys64/mingw64/bin/clang++.exe") == "mingw64"
        @test _ED._msys2_env("C:/msys64/usr/bin/clang++.exe") == "msys"
        @test _ED._msys2_env("C:/LLVM/bin/clang++.exe") == ""
        @test _ED._msys2_env("C:/tools/clang++.exe") == ""

        good = _ED._check_msys2("C:/msys64/clang64/bin/clang++.exe")
        @test good.found && good.meets_requirement
        bad = _ED._check_msys2("C:/msys64/mingw64/bin/clang++.exe")
        @test bad.found && !bad.meets_requirement
        @test occursin("MINGW64", bad.message)
        @test !_ED._check_msys2("").found
    end

    @testset "doctor: the summary claims only what its probes proved" begin
        probe(name, ok) = _ED.ToolStatus(name, false, ok, "", "", ok, "")
        c_ok = [probe("DWARF dumper", true), probe("symbol reader", true),
                probe("clang (JLL)", true), probe("sysroot", true)]
        summary(tools, tier2) =
            join(_ED._plain.(_ED._summary_lines(_ED.ToolchainStatus(tools, true, true, tier2))), "\n")

        no_cxx = summary(vcat(c_ok, [probe("llvm-config", false), probe("clang++", false)]), false)
        @test occursin("C wrappers will build", no_cxx)
        @test !occursin("C and C++", no_cxx)
        @test occursin("C++ builds unavailable", no_cxx)
        @test occursin("noexcept", no_cxx)

        cxx = summary(vcat(c_ok, [probe("llvm-config", true), probe("clang++", true)]), false)
        @test occursin("C++ builds —", cxx)
        @test !occursin("C++ builds unavailable", cxx)
    end

    @testset "doctor: dialect advice is runnable, llvm-dwarfdump is probed" begin
        advice = join(_ED.jlcs_build_advice(), "\n")
        @test occursin("build.sh", advice)
        @test !occursin('\\', advice)                 # bash strips backslashes
        @test occursin(Sys.iswindows() ? "MSYS2 CLANG64" : "Pkg.build", advice)

        # Probed under the bare name C++ wrappers run it by, as a Tier 2 probe.
        t = _ED._check_llvm_dwarfdump()
        @test t.name == "llvm-dwarfdump"
        @test t.section == _ED.SEC_CPP
        status = _ED.check_environment(verbose=false)
        @test any(p -> p.name == "llvm-dwarfdump", status.tools)
    end
end
