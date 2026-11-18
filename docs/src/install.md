# Set up

Install the package, run the check, and install whatever the check says is
missing. That is the whole setup.

```julia
using Pkg
Pkg.add("RepliBuild")

using RepliBuild
RepliBuild.check_environment()
```

`check_environment()` tries every program a build runs and prints, under each
one that failed, the command that installs it. Run it again after each change
until the summary says what you need:

| Summary line | Means |
|:-------------|:------|
| `Tier 3 (ccall) — C wrappers will build` | You can wrap **C** libraries. |
| `C++ builds — clang++ and LLVM 21+ found` | C++ sources will compile. |
| `Tier 2 (MLIR thunks) — C++ methods, …` | You can wrap **C++** libraries. |

Wrapping C needs only the first line. Wrapping C++ needs all three.

## What you need

RepliBuild runs on **Linux** (x86-64 and AArch64) and **Windows** (x86-64, with
MSYS2). macOS is not supported: `using RepliBuild` refuses to load there.

**Julia 1.12 or newer.**

A build runs a compiler, a linker and a handful of other programs. Some of
them RepliBuild brings with it, and some it takes from your system:

| Program | Used for | Linux | Windows |
|:--------|:---------|:------|:--------|
| C compiler (`clang`) | Compiling C libraries | Comes with RepliBuild | Comes with RepliBuild, using MSYS2's headers and libraries |
| C library headers and startup files | Every build | **Your system** (`libc6-dev`, `glibc-devel`) | **Your system** (MSYS2 CLANG64) |
| C++ compiler (`clang++`) and LLVM tools | Compiling and running C++ libraries | **Your system**, version 21 or newer | **Your system** (MSYS2 CLANG64) |
| Linker | Linking C libraries | Comes with RepliBuild | Your system |
| Linker | Linking C++ libraries | Your system | Your system |
| `readelf`, `nm`, `objdump` | Reading the built library | Comes with RepliBuild | Your system (MSYS2) |
| `git` | Git dependencies | Comes with RepliBuild | Your system |
| `cmake`, `ninja` | [CMake projects](graph.md) | Comes with RepliBuild | Your system (MSYS2) |
| `pkg-config` | `system` dependencies only | Your system | Your system |
| `libJLCS` | RepliBuild's own C++ support library | Downloaded with RepliBuild | Built on your machine |

So on Linux a C library needs almost nothing from the system, and a C++
library needs `clang++` and LLVM. On Windows everything comes from MSYS2.

**The programs that "come with RepliBuild" are a recent addition, and the
system's are still used.** On Linux RepliBuild tries its own copy of a program
first and uses the one on `PATH` when its own copy cannot start. On Windows it
is the other way round: the system's first. `REPLIBUILD_TOOLS=system` makes
Linux prefer the system's programs too, and `REPLIBUILD_TOOL_GIT=/path/to/git`
(and the like) names one program exactly ([Settings](#Settings)). When a
build behaves differently on two machines, which copy of a program ran is the
first thing to compare; the check report prints the path of each.

## Linux

### C libraries

Nothing to install on most machines. What RepliBuild takes from the system is
the C library's headers and startup files, which a machine with `gcc` already
has. If the check reports
`sysroot — cannot compile a TU that includes <stdio.h>`, install them:

```bash
sudo apt install libc6-dev        # Debian, Ubuntu
```

```bash
sudo dnf install glibc-devel      # Fedora, RHEL
```

Arch has them as part of `glibc`.

### C++ libraries

Install `clang++` and the LLVM tools, version **21 or newer**, and put them on
`PATH`. Versions 22 and 23 are the ones tested.

```bash
# Debian, Ubuntu: from apt.llvm.org (the distribution's own LLVM is usually too old)
wget https://apt.llvm.org/llvm.sh && sudo bash llvm.sh 23
sudo apt install clang-23 llvm-23
export PATH="/usr/lib/llvm-23/bin:$PATH"     # add this line to your shell profile
```

```bash
sudo dnf install clang llvm llvm-devel       # Fedora, RHEL
```

```bash
sudo pacman -S clang llvm                    # Arch
```

RepliBuild looks for these programs by name: `clang++`, `llvm-config`,
`llvm-link`, `opt` and `llvm-dwarfdump`. On Debian and Ubuntu they are installed
as `clang++-23` and so on, with the plain names in `/usr/lib/llvm-23/bin`, which
is why that directory goes on `PATH`. Check with `clang++ --version`.

The last piece is RepliBuild's own C++ support library, `libJLCS` (messages
also call it "the JLCS dialect"). On x86-64 and AArch64 Linux `Pkg.add`
downloads it ready-made, and you install nothing for it. The check shows it as

```
OK   libJLCS        prebuilt (JLCS artifact, LLVM and MLIR linked in)
```

If the download was skipped (no network during `Pkg.add`), it is fetched the
first time a C++ wrapper needs it, or by `Pkg.build("RepliBuild")`.

!!! note "Releases up to 4.5.0 use the system's programs for everything"
    `Pkg.add` installs the newest registered release, and this manual describes
    the development version. Releases up to 4.5.0 bring only the C compiler:
    they use the system's `readelf`, `nm`, `git`, `cmake`, `ninja` and linker,
    and do not download `libJLCS`. On those, install `binutils` and `git` as
    well, and build `libJLCS` as described under
    [Building libJLCS yourself](#Building-libJLCS-yourself).
    `pkgversion(RepliBuild)` says which version you have.

## Windows

RepliBuild on Windows builds with **MSYS2**, in its **CLANG64** environment.
Not MSVC, and not the MINGW64 or UCRT64 environments: CLANG64 is the one whose
libraries are compatible with Julia's own.

1. Install MSYS2 from [msys2.org](https://www.msys2.org) into the default
   location, `C:\msys64`. If you choose another, set the environment variable
   `MSYS2_ROOT` to it.
2. Open **MSYS2 CLANG64** from the Start menu and run `pacman -Syu`. It closes
   the window partway through; open it again and repeat until there is nothing
   left to update.
3. In the same window, install the toolchain:

   ```bash
   pacman -S --needed mingw-w64-clang-x86_64-toolchain mingw-w64-x86_64-binutils
   ```

   That is enough for C libraries. For C++ libraries and CMake projects, also:

   ```bash
   pacman -S --needed mingw-w64-clang-x86_64-mlir mingw-w64-clang-x86_64-cmake mingw-w64-clang-x86_64-ninja
   ```

   The second package of the first command is spelled differently from the
   others on purpose (`mingw-w64-x86_64-`, not `mingw-w64-clang-x86_64-`): it is
   the GNU `objdump` RepliBuild reads libraries with, and the CLANG64 one is a
   different program.
4. Add `C:\msys64\clang64\bin` to the Windows `PATH` (Settings → System →
   About → Advanced system settings → Environment Variables), then restart
   Julia. Julia can run from any terminal or editor; it does not have to be
   started from MSYS2.
5. For sources fetched with git, have a `git` on the Windows `PATH`, such as
   [Git for Windows](https://git-scm.com/download/win).
6. For C++, build the support library. With the packages above installed:

   ```julia
   using Pkg; Pkg.build("RepliBuild")
   ```

   Then run `check_environment()` again. Pkg writes the build's output to a log
   file and prints its path.

**Each new RepliBuild version needs step 6 again.** Pkg does it when it
installs the version, as long as the toolchain is there. Do it yourself after
MSYS2 upgrades LLVM (`pacman -Syu`): the check then reports `libJLCS — built
against LLVM X, but llvm-config is Y`. Close every Julia session that is using
RepliBuild first, because Windows will not replace a library a running program
has open.

**Smart App Control.** Windows 11 can refuse to load the libraries RepliBuild
builds, because they are new and unsigned. The error is `An Application Control
policy has blocked this file`. RepliBuild cannot sign what it builds; whether to
change the policy is your decision. [Troubleshooting](troubleshooting.md#Setup)
says how to confirm that this is the cause.

**Working from a git clone of RepliBuild** on Windows: run
`git config --global core.longpaths true` before cloning.

## Reading the report

```
C bucket — Tier 3 (ccall) ~
  OK   clang (JLL)    clang 20.1.8
  OK   sysroot        system headers reachable

DWARF extraction — required by every tier ~
  OK   DWARF dumper   GNU readelf 2.47
  OK   symbol reader  GNU nm 2.47

C++ — Tier 2 (MLIR thunks) ~
  OK   llvm-config    LLVM 23.1.5
  OK   clang++        Clang 23.1.5
  …
  OK   libJLCS        prebuilt
```

Each line is one program or library, with `OK`, `WARN` or `FAIL`, and the fix
printed under anything that is not `OK`. You do not need to know what the
section names mean. What each group is for:

| Group | Needed for | If it fails |
|:------|:-----------|:------------|
| `C bucket` | Every build | Nothing can be wrapped. On Linux, install the C library's development files. On Windows, the CLANG64 toolchain is missing or not found. |
| `DWARF extraction` | Every build | Nothing can be wrapped. On Windows, install `mingw-w64-x86_64-binutils`. |
| `C++` | C++ libraries only | C libraries still work. Install `clang++` and LLVM, and check `PATH`. |
| `MSYS2` (Windows) | C++ libraries | The `clang++` found is not CLANG64's. Put `C:\msys64\clang64\bin` first on `PATH`. |

`mlir-tblgen` and `cmake` under `C++` are needed only to build `libJLCS`
yourself. With the prebuilt library they may show `FAIL` while the summary
still says C++ works.

From code:

```julia
status = RepliBuild.check_environment(verbose = false)
status.ready          # C libraries can be wrapped
status.tier2_ready    # C++ libraries can be wrapped
```

## Building libJLCS yourself

You need this only on Windows (step 6 above), on a Linux machine that has no
prebuilt library (the check says `no prebuilt libJLCS for this platform`), or
when you want it built against your own LLVM.

It needs the MLIR development packages and CMake in addition to `clang++`:

```bash
sudo apt install llvm-23-dev libmlir-23-dev mlir-23-tools cmake    # Debian, Ubuntu (apt.llvm.org)
```

```bash
sudo dnf install llvm-devel mlir-devel clang-devel cmake           # Fedora, RHEL
```

```bash
yay -S mlir && sudo pacman -S cmake                                # Arch
```

Then, with `llvm-config`, `mlir-tblgen` and `cmake` on `PATH`:

```julia
ENV["REPLIBUILD_JLCS"] = "local"
using Pkg; Pkg.build("RepliBuild")
```

A library you built is used in preference to the prebuilt one. It is tied to
the LLVM it was built against: build it again after a system LLVM upgrade that
changes the version's first or second number. The check tells you when.

## Settings

Environment variables that change where RepliBuild finds things. None is needed
for a normal setup.

| Variable | Effect |
|:---------|:-------|
| `REPLIBUILD_TOOLS` | Which copy of `readelf`, `nm`, `objdump`, `git`, `cmake`, `ninja` and the C linker is tried first. `jll`: RepliBuild's own (the default on Linux). `system`: the one on `PATH` (the default on Windows). The other is used when the first is missing or cannot start. Any other value is an error. |
| `REPLIBUILD_TOOL_<NAME>` | The path of one of those programs, used as given: `REPLIBUILD_TOOL_GIT`, `_CMAKE`, `_NINJA`, `_READELF`, `_NM`, `_OBJDUMP`. |
| `LLVM_CONFIG` | The path of the `llvm-config` whose LLVM compiles C++, when it is not on `PATH`. |
| `REPLIBUILD_JLCS` | `artifact` for the prebuilt `libJLCS`, `local` for the one you built, or the path of a library file. |
| `MSYS2_ROOT` | Windows: where MSYS2 is installed, when not `C:\msys64`. |
| `REPLIBUILD_C_SYSROOT` | Windows: the directory holding the C headers and libraries, when not MSYS2's `clang64`. |
| `REPLIBUILD_OBJDUMP` | Windows: the path of a GNU `objdump`, when not MSYS2's. |
| `REPLIBUILD_HOME` | Where registered builds are kept. Default `~/.replibuild`. See [Load it anywhere](use.md). |

## Machines that only run a wrapper

A machine that loads a wrapper someone else generated needs less than one that
builds: see [Build a package on it](using-wrappers.md#Deployment).

## Next

[Wrap your first library](guide.md).
