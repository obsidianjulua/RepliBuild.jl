# Install

RepliBuild runs on **Linux** (ELF `.so`) and **Windows** (PE `.dll`, under
`x86_64-w64-windows-gnu` — mingw via MSYS2 CLANG64, not MSVC). Both read DWARF
through GNU binutils. macOS is refused at load: the AAPCS64 ABI classifier is
not built, so arm64 Mach-O has no struct-passing rules to apply. This manual
writes `.so`; on Windows read `.dll`.

```julia
using Pkg
Pkg.add("RepliBuild")
using RepliBuild
RepliBuild.check_environment()
```

`Pkg.add` installs the newest registered release. This manual is built from
`main`, so it can describe changes that release does not have yet:
`pkgversion(RepliBuild)` says which version you have, and the
[Changelog](release-notes.md) says what each one changed.

`check_environment()` probes every tool a build runs and prints how to install
whatever is missing. `status.ready` means C builds will work. `status.tier2_ready`
means C++ and the hard ABI cases (Tier 2) will work too.

**Julia.** CI tests Julia 1.12 and 1.13. `Project.toml` still admits 1.10 and
1.11, which nothing tests.

## C libraries

**Linux:** GNU binutils (`readelf`, `nm`), which almost every distribution
installs. Nothing else: compilation uses the Clang JLL, and linking runs
in-process on Julia's own libLLVM.

**Windows:** MSYS2 with its CLANG64 environment, and GNU `objdump` — see
[Windows](#Windows) below. The Clang JLL is a bare compiler artifact, with no
libc headers and no CRT, so it borrows the CLANG64 sysroot. It finds that through
the `clang` on `PATH`, then under `MSYS2_ROOT` (default `C:\msys64`);
`REPLIBUILD_C_SYSROOT` overrides both, and a candidate is accepted only if it
really holds `include/math.h`.

## C++ libraries

C++ needs a system LLVM/MLIR **21 or newer** (verified against 23.1.3 under
MSYS2 CLANG64, and 22.x on Linux), CMake 3.20+, and the JLCS dialect built
against that LLVM (below).

Put LLVM's `bin` directory on `PATH`, so that `llvm-config`, `clang++`,
`mlir-tblgen` and `llvm-dwarfdump` resolve to it. A C++ wrapper runs
`llvm-dwarfdump` by that name when it loads, unless it was built with AOT
thunks. apt.llvm.org packages keep those names in `/usr/lib/llvm-<N>/bin`.

OS-specific install hints (`check_environment()` prints the same ones):

```bash
# Ubuntu/Debian — what the Tier 2 Linux CI job installs
wget https://apt.llvm.org/llvm.sh && sudo bash llvm.sh 21
sudo apt install llvm-21-dev libmlir-21-dev mlir-21-tools
export PATH="/usr/lib/llvm-21/bin:$PATH"

# Fedora/RHEL
sudo dnf install llvm-devel mlir-devel clang-devel

# Arch
yay -S llvm mlir
```

On Windows, the [Windows](#Windows) steps below install all of it.

### The JLCS dialect

The dialect is a library, `libJLCS.so` (`libJLCS.dll` on Windows). It is not
shipped, because it links against your system MLIR. It is built into the
installed package, under `src/mlir/build/`. To find that folder:

```julia
joinpath(pkgdir(RepliBuild), "src", "mlir")
```

- **Linux:** `Pkg.add` builds it when `cmake`, `llvm-config` and `mlir-tblgen`
  are on `PATH`. Installed LLVM/MLIR afterwards? Run `Pkg.build("RepliBuild")`.
  Pkg writes the build's output to a log file (it prints the path), not to the
  REPL, so check the result with `check_environment()`.
- **Windows:** `Pkg.add` builds it too, once the [Windows](#Windows) toolchain
  is installed. It runs `build.sh` in MSYS2's bash as a CLANG64 login shell,
  from `C:\msys64` (or `MSYS2_ROOT`), so Julia does not have to be started from
  MSYS2. Installed the toolchain afterwards? Run `Pkg.build("RepliBuild")`, and
  check the result with `check_environment()`. To run it by hand instead, this
  line works from PowerShell; `check_environment()` prints it with your path
  filled in:

  ```bash
  C:/msys64/usr/bin/env.exe MSYSTEM=CLANG64 /usr/bin/bash -l 'C:/Users/<you>/.julia/packages/RepliBuild/<slug>/src/mlir/build.sh'
  ```

  Close the Julia sessions using RepliBuild before you rebuild: Windows will
  not overwrite a DLL a process has loaded.
- **From a clone of the repository:** `cd src/mlir && ./build.sh`.

Build it again:

- **after a system LLVM/MLIR upgrade that changes the minor version.** The library is
  pinned to the MLIR it was built against, and `check_environment()` reports a
  mismatch. `Pkg.build("RepliBuild")` rebuilds it.

Each RepliBuild version installs into a new folder, which has no dialect library
yet; Pkg builds it there when it installs the version.

A C++ wrapper needs the dialect when it **runs**, not only when it is
generated, with JIT and AOT thunks alike — see
[Ship a package](using-wrappers.md#Load-time).

## Windows

Use the **MSYS2 CLANG64** environment specifically — not MSYS, not MINGW64, not
UCRT64. It is the `x86_64-w64-windows-gnu` target that matches Julia's own mingw
build (so a wrapper `.dll` shares the process's C++ ABI), and it is the only
MSYS2 repo carrying MLIR.

1. Install MSYS2 from [msys2.org](https://www.msys2.org). The default location,
   `C:\msys64`, is the one RepliBuild looks in; set `MSYS2_ROOT` if you choose
   another.
2. Open **MSYS2 CLANG64** from the Start menu and run `pacman -Syu`. It closes
   the shell partway through; reopen it and repeat until there is nothing left
   to do.
3. Install the toolchain:

   ```bash
   pacman -S --needed mingw-w64-clang-x86_64-toolchain mingw-w64-clang-x86_64-mlir \
                      mingw-w64-clang-x86_64-cmake mingw-w64-clang-x86_64-ninja \
                      mingw-w64-x86_64-binutils
   ```

   `mingw-w64-x86_64-binutils` is the GNU `objdump` that DWARF extraction
   needs, from the MINGW64 tree on purpose: in CLANG64, "binutils" is
   llvm-objdump, whose dump format RepliBuild rejects. RepliBuild finds it at
   `C:\msys64\mingw64\bin\objdump.exe` from any shell; `REPLIBUILD_OBJDUMP`
   names a different program.
4. Add `C:\msys64\clang64\bin` to the Windows `PATH`, then restart Julia. Julia
   finds the toolchain through it, and `libJLCS.dll` imports `libLLVM-<N>.dll`
   and `libc++.dll` from it. Julia itself can run from any shell or editor; it
   does not have to be started from MSYS2.
5. For C++, build the dialect ([above](#The-JLCS-dialect)).

To work from a clone, run `git config --global core.longpaths true` before
cloning. Line endings take care of themselves: the repository's
`.gitattributes` checks every text file out with LF, whatever `core.autocrlf`
says. Without it, CRLF corrupted `src/mlir/build.sh`, and MSYS2 bash then
failed with `$'\r': command not found`.

**Smart App Control.** On Windows 11, Smart App Control (or another App Control
policy) can refuse to load the libraries RepliBuild builds, because they are new
and unsigned. [Troubleshooting](troubleshooting.md#Build-and-environment) says
how to recognise it.

[WINDOWS_PORT.md](https://github.com/obsidianjulua/RepliBuild.jl/blob/main/docs/updates/WINDOWS_PORT.md)
holds the port's engineering notes: what it found, and how.

## After install

```julia
RepliBuild.check_environment()          # what works
```

Then [wrap a library](guide.md). The registry is empty until `discover` or
`register` puts something in it — see [Registry](use.md).
