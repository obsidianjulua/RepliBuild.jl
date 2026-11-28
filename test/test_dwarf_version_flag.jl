# test_dwarf_version_flag.jl — a Windows compile asks for DWARF 5.
#
# The metadata is read from DWARF, and the facts read since the summer assume
# version 5, clang's default on Linux: `DW_AT_deleted` and `DW_AT_defaulted`
# exist from 5 on. clang's default for a mingw target is 4. There a deleted
# copy constructor looked callable, a class that can be neither copied nor
# moved was passed by value instead of refused, and a deleted constructor was
# planned into the forced unit (2026-11-28: the first full run on a Windows
# box since those facts were read; test_byvalue_class_live.jl and
# test_inline_ctors_live.jl are the live halves).
#
# So `-gdwarf-5` follows the project's flags on Windows, in the list the
# compile fingerprint hashes, unless the project names a version. Other hosts
# keep their own default: the flag there would move every fingerprint and
# nothing else. No toolchain; driven with both spellings of the host.

using Test
using RepliBuild

const _DV_CM = RepliBuild.ConfigurationManager

@testset "DWARF 5 on Windows, the host's default elsewhere" begin
    f = _DV_CM.dwarf_version_flags

    @test f(["-O2"]; windows = true) == ["-O2", "-gdwarf-5"]
    @test f(["-O2"]; windows = false) == ["-O2"]
    @test f(String[]; windows = true) == ["-gdwarf-5"]
    @test f(String[]; windows = false) == String[]

    # A version the project names is the project's, either spelling.
    @test f(["-gdwarf-4", "-O2"]; windows = true) == ["-gdwarf-4", "-O2"]
    @test f(["-O2", "-gdwarf"]; windows = true) == ["-O2", "-gdwarf"]
    @test f(["-gdwarf-5"]; windows = true) == ["-gdwarf-5"]
    # A debug level is not a version, and neither is a flag that only starts like one.
    @test f(["-g", "-g3", "-ggdb"]; windows = true) == ["-g", "-g3", "-ggdb", "-gdwarf-5"]
    @test f(["-gdwarf-aranges"]; windows = true) == ["-gdwarf-aranges", "-gdwarf-5"]

    # The default is this host's.
    @test f(["-O2"]) == f(["-O2"]; windows = Sys.iswindows())

    @testset "in the list the fingerprint hashes" begin
        mktempdir() do dir
            cfg(name, body) = begin
                p = joinpath(dir, name)
                write(p, "[project]\nname = \"$(splitext(name)[1])\"\nroot = \".\"\n\n[compile]\n$body")
                _DV_CM.load_config(p)
            end
            plain = cfg("a.toml", "flags = [\"-O2\"]\nvisibility = \"hidden\"\n")
            @test _DV_CM.get_compile_flags(plain; windows = true) == ["-O2", "-fvisibility=hidden", "-gdwarf-5"]
            @test _DV_CM.get_compile_flags(plain; windows = false) == ["-O2", "-fvisibility=hidden"]
            @test _DV_CM.get_compile_flags(plain) == _DV_CM.get_compile_flags(plain; windows = Sys.iswindows())

            named = cfg("b.toml", "flags = [\"-O2\", \"-gdwarf-4\"]\n")
            @test _DV_CM.get_compile_flags(named; windows = true) == ["-O2", "-gdwarf-4"]

            # A project that names a version is another compile: the per-file
            # IR cache must not serve one for the other.
            fp = RepliBuild.Compiler.compute_compile_fingerprint
            @test fp(cfg("c.toml", "flags = [\"-O2\"]\n"); toolchain = "t") !=
                  fp(cfg("d.toml", "flags = [\"-O2\", \"-gdwarf-4\"]\n"); toolchain = "t")
        end
    end
end
