#!/usr/bin/env julia
# test/test_windows_port.jl — host-format leftovers the punch list still named
# after the Windows gate opened.
#
# No toolchain required. These are the silent-wrong or silent-ignore classes
# a Hub rebuild would otherwise rediscover: library extension, system-header
# provenance, GNU-vs-LLVM dumper identity, the empty-DWARF diagnostic, path
# escaping in wrap_basic, and LLVM_CONFIG without `.exe`.

using Test
using Libdl
using RepliBuild

const C = RepliBuild.Compiler
const CM = RepliBuild.ConfigurationManager
const LE = RepliBuild.LLVMEnvironment

@testset "Windows port leftovers" begin

    @testset "get_library_name uses the host shared-lib suffix" begin
        mktempdir() do dir
            toml = joinpath(dir, "replibuild.toml")
            write(toml, """
            [project]
            name = "portprobe"
            root = "$(escape_string(dir))"
            [binary]
            type = "shared"
            """)
            cfg = CM.load_config(toml)
            @test CM.get_library_name(cfg) == "libportprobe." * Libdl.dlext
        end
    end

    @testset "_is_system_decl_file sees toolchain headers on this host" begin
        @test C._is_system_decl_file("/usr/include/stdio.h")
        @test C._is_system_decl_file("/usr/lib/clang/18/include/stddef.h")
        @test C._is_system_decl_file("/include/c++/v1/vector")
        @test !C._is_system_decl_file("")
        @test !C._is_system_decl_file("/home/foo/proj/include/foo.h")
        @test !C._is_system_decl_file("C:/Projects/RepliBuild.jl/test/c_test/include/mathkit.h")

        @test C._is_system_decl_file("C:\\msys64\\clang64\\include\\stdio.h")
        @test C._is_system_decl_file("C:/msys64/clang64/include/c++/v1/vector")
        @test C._is_system_decl_file("C:/msys64/ucrt64/include/stdio.h")
        @test C._is_system_decl_file("C:/Program Files/LLVM/include/stddef.h")
        @test C._is_system_decl_file("C:/Program Files (x86)/Windows Kits/10/Include/10.0.22621.0/ucrt/stdio.h")
    end

    @testset "GNU binutils identity is not a GNU substring" begin
        @test C._is_gnu_binutils("GNU objdump (GNU Binutils) 2.44\nCopyright (C) 2025")
        @test C._is_gnu_binutils("GNU readelf (GNU Binutils for Debian) 2.42")
        @test C._is_gnu_binutils("GNU objdump (GNU Binutils) 2.42")
        @test !C._is_gnu_binutils("LLVM version 22.1.8\n  compatible with GNU objdump")
        @test !C._is_gnu_binutils("llvm-objdump, compatible with GNU objdump")
        @test !C._is_gnu_binutils("")
    end

    @testset "empty-DWARF guard names the tool, not readelf_tool" begin
        dumper = (tool = "C:/msys64/mingw64/bin/objdump.exe",
                  dwarf = "--dwarf=",
                  sections = "-h")
        msg = C._dwarf_format_mismatch_error(dumper, "libfoo.dll", 8192)
        @test occursin("objdump.exe", msg)
        @test occursin("--dwarf=info", msg)
        @test !occursin("readelf_tool", msg)
        @test occursin("format mismatch", msg)

        dumper_elf = (tool = "readelf", dwarf = "--debug-dump=", sections = "-S")
        msg_elf = C._dwarf_format_mismatch_error(dumper_elf, "libfoo.so", 9000)
        @test occursin("readelf", msg_elf)
        @test occursin("--debug-dump=info", msg_elf)
    end

    @testset "wrap_basic escapes library paths with repr" begin
        src = read(joinpath(@__DIR__, "..", "src", "Wrapper", "Generator.jl"), String)
        @test occursin("repr(abspath(lib_path))", src)
        @test !occursin("raw\"\$(abspath(lib_path))\"", src)
    end

    @testset "LLVM_CONFIG path is probed with .exe and as-given" begin
        mktempdir() do dir
            name = Sys.iswindows() ? "llvm-config.exe" : "llvm-config"
            fake = joinpath(dir, name)
            write(fake, "x")
            @test LE._resolve_tool_path(fake) == abspath(fake)
            if Sys.iswindows()
                @test LE._resolve_tool_path(joinpath(dir, "llvm-config")) == abspath(fake)
            end
            @test LE._resolve_tool_path("") === nothing
            @test LE._resolve_tool_path(joinpath(dir, "no-such-tool")) === nothing
        end
    end

    @testset "generated _slice_symbols_resolve does not ccall(:dlsym)" begin
        chunk = RepliBuild.Wrapper._tier1_registry_chunk(
            ["crc32"], Dict("crc32" => ["crc32_z"]))
        @test occursin("Libdl.dlsym", chunk)
        @test occursin("LIB_HANDLE", chunk)
        @test occursin("throw_error = false", chunk)
        i = findfirst("function _slice_symbols_resolve", chunk)
        @test i !== nothing
        @test !occursin(r"ccall\(\s*:dlsym", chunk[first(i):end])
    end

    @testset "environment doctor looks for libJLCS with host dlext" begin
        status = RepliBuild.check_environment(verbose=false)
        jlcs = only(t for t in status.tools if t.name == "libJLCS")
        expected = joinpath(RepliBuild.SRC_DIR, "mlir", "build", "libJLCS." * Libdl.dlext)
        @test jlcs.path == (isfile(expected) ? expected : "")
        @test jlcs.found == isfile(expected)
    end

    @testset "PE link flags do not emit rpath" begin
        flags = try
            LE.get_link_flags()
        catch
            nothing
        end
        if flags === nothing
            @info "no LLVM toolchain — skipping get_link_flags rpath check"
        elseif Sys.iswindows()
            @test !any(contains(f, "rpath") for f in flags)
        else
            @test any(contains(f, "rpath") for f in flags)
        end
    end
end
