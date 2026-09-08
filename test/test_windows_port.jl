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

    # Two facts here, and only one of them is host-independent. Asserting both
    # unconditionally is what made this file red on Linux while passing on
    # Windows — the reverse of the vacuous-green class, but the same cause: a
    # test that states one host's answer instead of the invariant.
    #
    # The PATTERN LIST is a plain substring set and can be driven on any host,
    # provided the input is spelled the way `_canon_path` would leave it —
    # forward slashes, lowercased. So a Linux CI run still guards the MSYS2 and
    # WinSDK entries.
    @testset "_is_system_decl_file pattern list (host-independent)" begin
        @test C._is_system_decl_file("/usr/include/stdio.h")
        @test C._is_system_decl_file("/usr/lib/clang/18/include/stddef.h")
        @test C._is_system_decl_file("/include/c++/v1/vector")
        @test !C._is_system_decl_file("")
        @test !C._is_system_decl_file("/home/foo/proj/include/foo.h")
        @test !C._is_system_decl_file("C:/Projects/RepliBuild.jl/test/c_test/include/mathkit.h")

        # The Windows entries, pre-canonicalised.
        @test C._is_system_decl_file("c:/msys64/clang64/include/stdio.h")
        @test C._is_system_decl_file("c:/msys64/clang64/include/c++/v1/vector")
        @test C._is_system_decl_file("c:/msys64/ucrt64/include/stdio.h")
        @test C._is_system_decl_file("c:/msys64/mingw64/include/stdio.h")
        @test C._is_system_decl_file("c:/program files/llvm/include/stddef.h")
        @test C._is_system_decl_file("c:/program files (x86)/windows kits/10/include/ucrt/stdio.h")
    end

    # The CANONICALISATION is host-conditional on purpose: `_canon_path` is the
    # identity off Windows, because a backslash is a legal character in a POSIX
    # filename and POSIX paths are case-sensitive — normalising either would
    # misclassify real project files. So the real-world spellings match only on
    # Windows, and the Linux arm pins that as deliberate rather than skipping it.
    @testset "_is_system_decl_file canonicalises host path spellings" begin
        real_world = ("C:\\msys64\\clang64\\include\\stdio.h",
                      "C:/Program Files/LLVM/include/stddef.h",
                      "C:/Program Files (x86)/Windows Kits/10/Include/10.0.22621.0/ucrt/stdio.h")
        for p in real_world
            @test C._is_system_decl_file(p) == Sys.iswindows()
        end
        @test C._canon_path("A\\B") == (Sys.iswindows() ? "a/b" : "A\\B")
    end

    @testset "GNU binutils identity is not a GNU substring" begin
        @test C._is_gnu_binutils("GNU objdump (GNU Binutils) 2.44\nCopyright (C) 2025")
        @test C._is_gnu_binutils("GNU readelf (GNU Binutils for Debian) 2.42")
        @test C._is_gnu_binutils("GNU objdump (GNU Binutils) 2.42")
        @test !C._is_gnu_binutils("LLVM version 22.1.8\n  compatible with GNU objdump")
        @test !C._is_gnu_binutils("llvm-objdump, compatible with GNU objdump")
        @test !C._is_gnu_binutils("")
    end

    @testset "the objdump Debug disassembles with is GNU, not llvm-objdump" begin
        # `Debug.disassemble` passes `--disassemble=<symbol>` — GNU spelling.
        # llvm-objdump answers `unknown argument` to it, and in a CLANG64 shell
        # the bare name IS llvm-objdump, so the bare name is not a safe default
        # here even though `objdump -p` happens to work on both.
        tool = C._gnu_objdump()
        @test !isempty(tool)
        @test occursin("objdump", lowercase(basename(tool)))
        # Debug must not resolve it independently — one answer, one trap closed.
        @test RepliBuild.Debug._objdump() == tool

        # Whatever it picked has to PASS the identity test, not merely exist.
        # Skips only when the machine has no objdump at all, which is the one
        # case `_gnu_objdump` cannot improve on.
        out, ec = try
            RepliBuild.BuildBridge.execute(tool, ["--version"])
        catch
            ("", 1)
        end
        if ec == 0
            @test C._is_gnu_binutils(out)
        else
            @info "no objdump on this machine — skipping the GNU identity check"
        end
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

    # ── AOT thunks must bind to the ONE libJLCS ──────────────────────────────
    #
    # A thunk that catches a C++ exception calls `jlcs_catch_current_exception`,
    # which writes into libJLCS's own `jlcs_exception_buffer`;
    # `_check_pending_exception` reads it back by ccall'ing into
    # `MLIRNative.libJLCS`. One buffer, or the exception is swallowed.
    #
    # ELF held that by RUNPATH. PE has none, so the loader — which searches the
    # loading module's directory and PATH, and libJLCS is in neither — could not
    # open the thunks DLL at all: `The specified module could not be found`,
    # naming the module it DID find rather than the dependency it did not.
    # Both halves are one requirement: open libJLCS, by its absolute path,
    # first. Vendoring a copy beside the thunks would satisfy the loader and
    # break the buffer, so the test is the ordering, not the presence.
    @testset "AOT thunks open libJLCS before the thunks library" begin
        @test isdefined(RepliBuild.JITManager, :open_thunks_library)

        jm = read(joinpath(RepliBuild.SRC_DIR, "IRGen", "JITManager.jl"), String)
        i = findfirst("function open_thunks_library", jm)
        @test i !== nothing
        body = jm[first(i):end]
        body = body[1:first(something(findfirst("\nend\n", body), (lastindex(body),)))]

        at_jlcs = findfirst("MLIRNative.libJLCS", body)
        at_thunks = findfirst("dlopen(thunks_path", body)
        @test at_jlcs !== nothing
        @test at_thunks !== nothing
        # Ordering IS the fix: libJLCS is already in the process when the
        # loader resolves the thunks library's import table.
        @test first(at_jlcs) < first(at_thunks)

        # Neither generator may go around it. A bare `dlopen` of the thunks
        # path is the regression, and it fails only on PE and only at load.
        for gen in (joinpath("Cpp", "GeneratorCpp.jl"), joinpath("C", "GeneratorC.jl"))
            src = read(joinpath(RepliBuild.SRC_DIR, "Wrapper", gen), String)
            @test !occursin("Libdl.dlopen(THUNKS_LIBRARY_PATH", src)
            @test occursin("open_thunks_library(THUNKS_LIBRARY_PATH)", src)
        end

        # And the failure has to name libJLCS, not just the file that opened.
        mktempdir() do dir
            missing_thunks = joinpath(dir, "libnope_thunks." * Libdl.dlext)
            err = try
                RepliBuild.JITManager.open_thunks_library(missing_thunks)
                nothing
            catch e
                sprint(showerror, e)
            end
            @test err !== nothing
            @test occursin("libJLCS", err)
            @test occursin("jlcs_catch_current_exception", err)
        end
    end

    @testset "a shim's dllexport must not cost the library its auto-export" begin
        # PE keeps mingw's auto-export only while NOTHING is explicitly
        # exported; one `dllexport` anywhere flips the whole image to
        # explicit-only. The macro shims now carry one (they must — see
        # generate_macro_shims), so a library that exports nothing of its own
        # needs `--export-all-symbols` handed back or its entire API leaves the
        # export directory, which is the list the wrapper reads on Windows.
        # Measured on a two-function probe DLL: {lib_a, lib_b, shim} → {shim}.
        #
        # `_pe_export_intent` is the decision and it reads IR text, so drive it
        # with IR text — no compiler, and the assertions hold on either host.
        mktempdir() do dir
            shim = joinpath(dir, "shim.ll")
            write(shim, "define dllexport i32 @replibuild_shim_X() {\n  ret i32 8\n}\n")
            plain = joinpath(dir, "plain.ll")
            write(plain, "define i32 @lib_a() {\n  ret i32 1\n}\n")
            explicit = joinpath(dir, "explicit.ll")
            write(explicit, "define dso_local dllexport ptr @pcre2_code_copy_8(ptr %0) {\n  ret ptr %0\n}\n")

            # Ours is the only export → auto-export has to be restored.
            i1 = C._pe_export_intent([plain, shim])
            @test i1.saw_shim
            @test !i1.saw_foreign_export

            # The library exports on its own account → leave the surface it
            # chose alone; forcing the flag would publish every internal.
            i2 = C._pe_export_intent([explicit, shim])
            @test i2.saw_shim
            @test i2.saw_foreign_export

            # No shims → nothing introduced, nothing to decide.
            i3 = C._pe_export_intent([plain])
            @test !i3.saw_shim
            @test !i3.saw_foreign_export

            # A dllexport on the shim's own line is OURS, not the library's —
            # the whole point of matching per line rather than per file.
            i4 = C._pe_export_intent([shim])
            @test i4.saw_shim
            @test !i4.saw_foreign_export
        end
    end
end
