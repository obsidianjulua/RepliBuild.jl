# Install

RepliBuild is Linux-only. It assumes ELF shared objects, DWARF, and GNU `nm`.

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

Nothing beyond Julia 1.10+. Compilation uses the Clang JLL; linking runs
in-process on Julia's own libLLVM. If `using RepliBuild` loads, you can wrap C.

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

OS-specific install hints (also printed by `check_environment()`):

```bash
# Ubuntu/Debian
wget https://apt.llvm.org/llvm.sh && sudo bash llvm.sh 21

# Fedora/RHEL
sudo dnf install llvm21-devel mlir21-devel clang21-devel
```

Put `llvm-config` and `clang++` on `PATH`. Versioned names (`llvm-config-21`,
`clang++-21`) are found automatically.

## After install

```julia
RepliBuild.check_environment()          # what works
```

Then [wrap a library](guide.md). The registry is empty until `discover` or
`register` puts something in it — see [Registry](use.md).
