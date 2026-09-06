# Install

RepliBuild runs on **Linux** (ELF `.so`) and **Windows** (PE `.dll`, under
`x86_64-w64-windows-gnu` — mingw via MSYS2 CLANG64, not MSVC). Both read DWARF
through GNU binutils. macOS is refused at load: the AAPCS64 ABI classifier is
not built, so arm64 Mach-O has no struct-passing rules to apply.

```julia
using Pkg
Pkg.add("RepliBuild")
using RepliBuild
RepliBuild.check_environment()
```

`check_environment()` prints what this machine can do and how to install anything
missing. `status.ready` means C builds will work. `status.tier2_ready` means C++
and the hard ABI cases will work too.

## C libraries

On Linux, nothing beyond Julia 1.10+. Compilation uses the Clang JLL; linking
runs in-process on Julia's own libLLVM. If `using RepliBuild` loads, you can
wrap C.

On Windows the Clang JLL needs a sysroot. It is a bare compiler artifact — no
libc headers, no CRT — so it borrows MSYS2 CLANG64's. That is found from the
on-`PATH` clang automatically; `REPLIBUILD_C_SYSROOT` overrides it, and a
candidate is accepted only if it really holds `include/math.h`.

## C++ libraries

C++ needs a system LLVM/MLIR **21+** toolchain (22.x is what this tree is
verified against), CMake 3.20+, `clang++`, `mlir-tblgen`, and the JLCS dialect
built once:

```bash
cd src/mlir && ./build.sh
```

That writes `src/mlir/build/libJLCS.so` (gitignored). A fresh clone has no
dialect `.so` — C++ wrapping fails in a way that looks like a regression until
you build it.

That writes `src/mlir/build/libJLCS.dll` on Windows — CMake names the target by
host convention, so the extension follows the host and `MLIRNative` looks for it
with `Libdl.dlext`.

OS-specific install hints (also printed by `check_environment()`):

```bash
# Ubuntu/Debian
wget https://apt.llvm.org/llvm.sh && sudo bash llvm.sh 21

# Fedora/RHEL
sudo dnf install llvm21-devel mlir21-devel clang21-devel

# Windows — MSYS2 CLANG64 shell
pacman -S --needed mingw-w64-clang-x86_64-toolchain mingw-w64-clang-x86_64-mlir \
                   mingw-w64-clang-x86_64-cmake mingw-w64-clang-x86_64-ninja
```

Put `llvm-config` and `clang++` on `PATH`. Versioned names (`llvm-config-21`,
`clang++-21`) are found automatically, and so is a `.exe` suffix.

## Windows

Use the **MSYS2 CLANG64** environment specifically — not MSYS, not MINGW64, not
UCRT64. It is the `x86_64-w64-windows-gnu` target that matches Julia's own mingw
build (so a wrapper `.dll` shares the process's C++ ABI), and it is the only
MSYS2 repo carrying MLIR.

Before cloning:

```powershell
git config --global core.autocrlf false
git config --global core.longpaths true
```

`core.autocrlf false` is required rather than preferred: CRLF corrupts
`src/mlir/build.sh`, and MSYS2 bash then fails with `$'\r': command not found`,
which reads as a broken script rather than a line-ending problem.

`src/mlir/build.sh` is bash and runs as-is under MSYS2.

[WINDOWS_PORT.md](https://github.com/obsidianjulua/RepliBuild.jl/blob/main/WINDOWS_PORT.md)
has the rest — what the port found, and what is still open.

## After install

```julia
RepliBuild.check_environment()          # what works
```

Then [wrap a library](guide.md). The registry is empty until `discover` or
`register` puts something in it — see [Registry](use.md).
