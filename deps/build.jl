# Build the JLCS MLIR dialect at install time, when the machine can.
#
# The dialect library (`libJLCS.so`, `libJLCS.dll` on Windows) is not shipped —
# it links against the system MLIR and LLVM, so a prebuilt one would be wrong on
# any box whose LLVM differs, which is most of them. Until now the consequence landed on the user at the worst possible
# moment: `Pkg.add` succeeded, `using RepliBuild` succeeded, and the first call
# that reached Tier 2 raised "JLCS dialect library not found — build it first"
# from inside a read-only depot.
#
# THIS SCRIPT MUST NOT THROW. A `deps/build.jl` that fails takes `Pkg.add` down
# with it, and RepliBuild is perfectly usable without the dialect: Tier 3
# (plain `ccall` wrappers) is the whole product for C libraries, and Tier 2 is
# what needs MLIR. A machine with no system MLIR should install a working
# package and be told clearly what it does not have — not be refused.
#
# So every failure path here is a warning and an `exit 0`. The runtime guard in
# `MLIRNative.check_library` is still the backstop, and now it is a backstop
# rather than the only line of defence.

const MLIR_DIR   = joinpath(@__DIR__, "..", "src", "mlir")
const BUILD_DIR  = joinpath(MLIR_DIR, "build")
# CMake names the target by host convention: libJLCS.so, libJLCS.dll.
const LIB_PATH   = joinpath(BUILD_DIR, "libJLCS." * Base.Libc.Libdl.dlext)
const LOG_PATH   = joinpath(@__DIR__, "build.log")

# The MSYS2 command, shared with the doctor's advice.
include(joinpath(MLIR_DIR, "windows_build.jl"))

# How to build it by hand, for this host. The Windows package list is not
# repeated here: check_environment() prints it from the doctor's one copy.
function build_steps()
    if Sys.iswindows()
        return """
        To build it again, close the Julia sessions using RepliBuild, then run
            julia -e 'using Pkg; Pkg.build("RepliBuild")'
        or from PowerShell:
            $(jlcs_msys2_line(abspath(MLIR_DIR)))

        `RepliBuild.check_environment()` lists the CLANG64 packages it needs, and
        docs/src/install.md (Windows) has the setup."""
    end
    return """
    To build it later, once the tools are installed:
        julia -e 'using Pkg; Pkg.build("RepliBuild")'
    or by hand:
        cd "$(abspath(MLIR_DIR))" && ./build.sh

    Requires: cmake, a C++ toolchain, and the system MLIR/LLVM development
    packages, with `mlir-tblgen` and `llvm-config` on PATH. On Arch that is the
    `mlir` package; on Debian/Ubuntu (apt.llvm.org), `libmlir-<N>-dev` and
    `mlir-<N>-tools`. `RepliBuild.check_environment()` prints the full list."""
end

# What a failure should tell someone who has never heard of JLCS.
function advise(reason::AbstractString; detail::AbstractString = "")
    @warn """
    RepliBuild: the JLCS MLIR dialect was not built. $reason

    RepliBuild still installs and works. What is unavailable is **Tier 2** —
    the MLIR-JIT path used for C++ member functions and by-value aggregates.
    Tier 3 (`ccall`) wrappers, which is all a C library needs, are unaffected.

    $(build_steps())
    $(isempty(detail) ? "" : "\n" * detail)
    """
    exit(0)
end

# Already built AND still loadable.
#
# Existence alone is not enough, and this is the case that bites on a rolling
# distro: libJLCS.so links libMLIR.so.<version>, so an LLVM upgrade leaves a
# file that is present, correct-looking, and impossible to dlopen. Asking the
# loader is the only check that distinguishes "built" from "built against an
# LLVM that is gone".
function usable_library()
    isfile(LIB_PATH) || return false
    try
        Base.Libc.Libdl.dlopen(LIB_PATH, Base.Libc.Libdl.RTLD_LAZY)
        return true
    catch
        # On Windows the same failure has a second cause: the DLLs it imports
        # (libLLVM, libc++) are in CLANG64's bin, which must be on PATH.
        why = Sys.iswindows() ? "LLVM upgraded, or C:\\msys64\\clang64\\bin not on PATH?" :
                                "LLVM upgraded?"
        @info "RepliBuild: existing $(basename(LIB_PATH)) does not load ($why)."
        return false
    end
end

haveit(prog) = Sys.which(prog) !== nothing

# Run build.sh with its output in the log. True when it exited 0.
function run_logged(cmd::Cmd)
    try
        open(LOG_PATH, "w") do log
            run(pipeline(cmd; stdout = log, stderr = log))
        end
        return true
    catch
        return false
    end
end

# Built, but prove it loads before claiming success — a linkable-but-not-
# loadable library is exactly what this script exists to catch.
function finish(ok::Bool)
    if !ok || !isfile(LIB_PATH)
        advise("The build failed."; detail = "Build log: $LOG_PATH")
    end
    usable_library() ||
        advise("The build produced a library that will not load.";
               detail = "Build log: $LOG_PATH")
    println("RepliBuild: JLCS dialect built — $LIB_PATH")
end

# Windows: build.sh runs in a CLANG64 login shell started from here
# (windows_build.jl says why that is a CLANG64 shell). The toolchain is checked
# where that shell will look for it, not on this process's PATH.
function main_windows()
    root = msys2_root()
    bin = "$root/clang64/bin"
    version = try
        strip(read(`$bin/llvm-config.exe --version`, String))
    catch
        "unknown"
    end
    # libLLVM-<major>.dll is named by major alone, so a library built against
    # another minor still loads. build.sh stamps the version it built against.
    stamp = joinpath(BUILD_DIR, "jlcs_llvm_version")
    mm(v) = match(r"^\d+\.\d+", v)
    built = isfile(stamp) ? strip(read(stamp, String)) : ""
    stale = mm(built) !== nothing && mm(version) !== nothing &&
            mm(built).match != mm(version).match
    stale && println("RepliBuild: libJLCS.dll was built against LLVM $built; CLANG64 has $version. Rebuilding.")
    !stale && usable_library() &&
        return println("RepliBuild: JLCS dialect already built and loadable.")

    isfile(msys2_bash(root)) ||
        advise("MSYS2 was not found at $root (set MSYS2_ROOT if it is elsewhere).")
    missing = filter(t -> !isfile("$bin/$t.exe"), ["cmake", "mlir-tblgen", "llvm-config"])
    isempty(missing) ||
        advise("MSYS2 CLANG64 has no $(join(missing, ", ")) in $bin.")

    println("RepliBuild: building JLCS dialect against LLVM/MLIR $version in MSYS2 CLANG64 …")
    ok = run_logged(jlcs_msys2_cmd(MLIR_DIR; root))

    # libJLCS.dll imports libLLVM and libc++ from clang64\bin, which Windows
    # finds only through PATH. A library that built but cannot load for that
    # reason needs a PATH entry, not another build.
    if ok && isfile(LIB_PATH) && !usable_library()
        onpath = any(p -> lowercase(rstrip(replace(p, '\\' => '/'), '/')) == lowercase(bin),
                     split(get(ENV, "PATH", ""), ';'))
        onpath || advise("It built, but $bin is not on PATH, and libJLCS.dll loads its LLVM from there.";
                         detail = "Add $(replace(bin, '/' => '\\')) to the Windows PATH, then restart Julia.")
    end
    finish(ok)
end

function main()
    Sys.iswindows() && return main_windows()
    Sys.islinux() || advise("This platform is not supported; RepliBuild runs on Linux and Windows.")

    usable_library() && return println("RepliBuild: JLCS dialect already built and loadable.")

    missing = filter(!haveit, ["cmake", "mlir-tblgen", "llvm-config"])
    isempty(missing) ||
        advise("Missing from PATH: $(join(missing, ", ")).")

    version = try
        strip(read(`llvm-config --version`, String))
    catch
        "unknown"
    end
    println("RepliBuild: building JLCS dialect against LLVM/MLIR $version …")

    # Output goes to a log rather than the terminal: a CMake build is a few
    # hundred lines and `Pkg.add` is not the place for them. The path is named
    # in the warning, so a failure is still diagnosable.
    finish(run_logged(`bash $(joinpath(MLIR_DIR, "build.sh"))`))
end

try
    main()
catch e
    # The last resort. Anything unanticipated above still must not fail the
    # install, so it degrades to the same warning as everything else.
    @warn "RepliBuild: dialect build skipped after an unexpected error." exception = e
    exit(0)
end
