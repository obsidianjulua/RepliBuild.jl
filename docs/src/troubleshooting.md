# Troubleshooting

Find the message or the symptom in the left column.

Most problems with a wrapper are something missing from `replibuild.toml`, not
a bug: a build that succeeds is not a complete wrapper.
[What goes in the TOML](toml.md) is the checklist.

## Setup

| What you see | What to do |
|:-------------|:-----------|
| `check_environment()` shows `FAIL` or `WARN` | The lines printed under it are the fix. [Set up](install.md) has the whole procedure. |
| `RepliBuild supports Linux (x86-64, AArch64) and Windows (x86-64)` | You are on macOS or another system. It is not supported. |
| `sysroot — cannot compile a TU that includes <stdio.h>` | Linux: install the C library's development files (`libc6-dev`, `glibc-devel`). Windows: install the MSYS2 CLANG64 toolchain. |
| `LLVM Toolchain not found` when building C++ | No `clang++` and LLVM 21 or newer on `PATH`. Install them, or set `LLVM_CONFIG` to the path of `llvm-config`. |
| `LLVM 18 — need 21+` | The system's LLVM is too old. On Debian and Ubuntu install a newer one from apt.llvm.org and put its `bin` folder first on `PATH`. |
| `JLCS dialect library not found` | RepliBuild's C++ support library is missing. Linux: `Pkg.build("RepliBuild")` downloads it. Windows: [Set up, step 6](install.md#Windows). |
| `libJLCS — built against LLVM X, but llvm-config is Y` | The system's LLVM was upgraded. `Pkg.build("RepliBuild")` again. |
| `libJLCS — built for another dialect ABI` | The support library was built by a different RepliBuild version. `Pkg.build("RepliBuild")` again; for one you built yourself, rebuild it ([Set up](install.md#Building-libJLCS-yourself)). |
| `Failed to initialize JIT for lib….so` when a C++ wrapper loads | The wrapper's C++ functions will not work. The reason is printed under the message; usually the support library is missing or `llvm-dwarfdump` is not on `PATH`. Run `check_environment()`. |
| C works, C++ does not | Run `check_environment()` and read the `C++` section. |
| Windows: every C file fails on its first `#include` | The MSYS2 CLANG64 toolchain is missing or was not found. Install it, or set `MSYS2_ROOT`. |
| Windows: `libJLCS.dll` will not load, `The specified module could not be found` | `C:\msys64\clang64\bin` is not on the Windows `PATH`. Add it and restart Julia. If it is there, MSYS2 upgraded LLVM: `Pkg.build("RepliBuild")`. |
| Windows: `The specified module could not be found` opening a `…_thunks.dll` | The same: the file it cannot find is `libJLCS.dll`. |
| Windows: C++ stopped working after upgrading RepliBuild | The new version needs its support library built: `Pkg.build("RepliBuild")`. |
| Windows: `An Application Control policy has blocked this file` | Smart App Control refused to load a library RepliBuild just built, because it is new and unsigned. Event Viewer → Applications and Services Logs → Microsoft → Windows → CodeIntegrity → Operational, event 3077, names the file. RepliBuild cannot sign what it builds; changing the policy is your decision. |

## Building

| What you see | What to do |
|:-------------|:-----------|
| `Configuration file not found` | Run `discover` first, or pass the path of the TOML. |
| `No source files to compile`, then `Compilation produced no library` | No sources were found. If they come from a dependency, look further up for `Failed to clone dependency`. If the library is header-only: [Header-only libraries](header-only.md). |
| `Failed to clone dependency …` (a warning) | The source was not fetched; the build goes on and fails later. Check the `url`, the `tag` and the network. |
| A dependency with a `url` is never fetched | It has no `type`. Add `type = "git"`. |
| The dependency's commit does not match `commit` | The tag was moved, or you changed `tag` without `commit`. Update `commit` to the full 40-character commit of the tag. |
| A file fails on a missing header | Add the header's folder to `[compile] include_dirs`. Check the spelling: a folder that does not exist is passed on without complaint. |
| `config.h: No such file or directory` | The library's build system generates that header: [Headers the build system generates](toml.md#Headers-the-build-system-generates). |
| A file fails on an undefined name that the library's own build defines | Add the define: [Flags, defines and include folders](toml.md#Flags,-defines-and-include-folders). |
| Duplicate `main`, or errors in test and example files | Exclude them: [Files that are not the library](toml.md#Files-that-are-not-the-library). |
| A cloned repository has almost no files (Boost) | It is made of submodules: [Git submodules](dependencies.md#Git-submodules). |
| `In-process libLLVM link failed for C source` | Usually the same function defined in two files. A common cause is an `…_IMPLEMENTATION` define in `[compile] flags`; move it to [`[wrap] shim_defines`](header-only.md#An-implementation-section). |
| The `shim_headers` include resolved outside the project | A header of the same name is installed on the system and was found first. Give a longer path (`"c/blake3.h"`) or add the right folder to `include_dirs`. |
| `wrappable surface … is EMPTY`, or functions `NOT reachable` | `[compile] visibility = "hidden"` on a library that does not mark its public functions. Remove it, or add the define that switches the marking on: [Leaving things out](toml.md#Leaving-things-out). |
| A pattern `matched nothing` | An `exclude_symbols`, `surface_types_extra` or `[wrap.rename]` entry matches no name. Fix the pattern or delete it. |
| `wrap` fails with `The wrapper module would be named …` | The project's name, in CamelCase, is a name the wrapper uses itself (`String`, `Ptr`). Set `[wrap] module_name` to something else. |
| `Library not found: …/julia/lib….so` | Run `build` before `wrap`. |
| `No compilation_metadata.json next to …` | The same: run `build`. |
| An edit to a source file is not picked up | The file's timestamp is older than the cached result, which happens when the clock was set back. `build(toml; clean = true)`. |
| A setting has no effect | The key is misspelled or in the wrong section. Misspelled keys are ignored silently: check [Every key](config.md). |
| Windows: `failed to write output '….dll': Permission denied` | A Julia session has the library loaded. Close it and build again. |
| Windows: `clean` fails because a file is in use | The same. |
| Wrapper generation is refused with another message | These name something RepliBuild cannot wrap yet. Leave the named function out with [`[wrap] exclude_symbols`](toml.md#Leaving-things-out) so the rest is written, and please report it. |

## The wrapper is missing something

| What you see | What to add |
|:-------------|:------------|
| A macro is not in the wrapper | [`[wrap.macros]`](toml.md#Macros) |
| A macro has the wrong value | `shim_headers` found a different copy of the header. Give a longer path. |
| An inline function or method is missing | [Header-only libraries](header-only.md) |
| A struct or class has no fields | [`-fstandalone-debug`](toml.md#Types-declared-only-in-headers) |
| A C++ class has no constructor | The same flag. If the class is still missing, [`[types] records`](toml.md#Types-declared-only-in-headers). |
| A template type or a container is missing | [`[types] templates`](toml.md#C-templates-and-containers) |
| A `printf`-style function takes only its fixed arguments | [`[wrap.varargs]`](toml.md#Variadic-functions) |
| Memory grows with every call that returns a string | [`[wrap.cstring_owned]`](toml.md#Strings-the-caller-must-free) |
| A struct became raw bytes, with a warning | Julia cannot reproduce its layout. Use the accessor functions the wrapper defines for it. |
| `UndefVarError` for a type the library defines | Its name is one Julia already uses, so it is not exported. Write `MyModule.Number`. |
| A function or struct is missing under its own name | The wrapper needed that name itself and gave the library's a different one. Read the `names:` lines `wrap` printed: `String` is `String_`, `Integer` is `c_Integer` ([Find your way around](calling.md#Find-your-way-around)). |
| The wrapper is full of internal functions | [Leaving things out](toml.md#Leaving-things-out) |
| The module fails to load: a redefined constant, duplicate fields | Internal types are colliding: [`surface_types_only`](toml.md#Leaving-things-out) |
| A function's name is hundreds of characters long | [`[wrap.rename]`](toml.md#Names) |

## Messages from the TOML

| What you see | What it means |
|:-------------|:--------------|
| `invalid type '…' in [wrap.varargs.f]` | `[wrap.varargs]` takes Julia's type names (`Cint`, `Cstring`), not C's. |
| `must be a list of type lists` | Write `f = [["Cint"]]`, not `f = ["Cint"]`. |
| `[wrap.varargs] … is unversioned`, on every `wrap` | Add `proven_at` with the dependency's commit ([Variadic functions](toml.md#Variadic-functions)). A project without a git dependency cannot, and keeps the warning. |
| `Varargs function 'f' has no overloads configured` | `f` takes `...` and has no `[wrap.varargs]` entry. Only its fixed arguments can be passed. |

## Calling the wrapper

| What you see | What to do |
|:-------------|:-----------|
| `ABI Safety Trap: cannot call 'f'` | RepliBuild found no safe way to call `f`, and defined a function that raises instead of one that might corrupt memory. The message gives the reason: often a `long double`, or a type passed by value that is only declared in a header (add [`-fstandalone-debug`](toml.md#Types-declared-only-in-headers)). The rest of the module works. |
| `FFI Safety Trap: Cannot call function 'f'` | A type in `f`'s signature has no Julia equivalent. Give it one under `[types.custom]`: `MyHandle = "Ptr{Cvoid}"`. |
| `no method matching unsafe_convert(::Type{Ptr{T}}, ::T)` | A pointer parameter was given a struct. Pass `Ref(x)`. |
| `MethodError` passing a `Ref` to a struct parameter | A parameter declared as a struct takes the struct itself. |
| `MethodError` passing `2` where the signature says `Cdouble` or `Int32` | That parameter takes exactly one type. Write `2.0` or `Int32(2)`. |
| `MethodError` calling a C++ overload | The position where the overloads differ needs the exact type: a `String` for `char*`, the enum value for an enum, `Ref` or `Ptr` for a struct pointer ([Overloads](calling.md#Overloads-and-default-arguments)). |
| A variadic function rejects its extra arguments | Each combination is its own function: `f_Cint(…)`, not `f(…)`. |
| `no method matching unsafe_convert(::Type{Ptr{Nothing}}, ::Cstring)` | A `<name>_ptr` result was passed to a `void*` parameter. Wrap it: `Ptr{Cvoid}(p)`. |
| `C++ exception: …` | The C++ code threw. Catch `RepliBuild.JITManager.CxxException`. |
| A crash calling a C++ method on `T()` | `T()` is a zeroed value, not a constructed object ([Objects](calling.md#Objects)). |
| A function returning `std::string` returns raw bytes | The comment beside its call in the wrapper says why. If it mentions `string_layout`, rebuild: `build(toml; clean = true)`, then `wrap`. |
| ``thunk `…` not found, but `…_inplace` is`` | The wrapper was written by an older RepliBuild. Run `wrap(toml)` again. |
| `package MyPkg does not have Libdl in its dependencies` | A package that contains a wrapper must list `Libdl` as well as `RepliBuild`: `Pkg.add("Libdl")` in the package's environment. |
| `AOT thunk pointer is null` while building a system image | The wrapper's start-up code has not run in the workload. Call its `__init__` first ([Make it fast](performance.md#Compile-it-into-a-system-image)). |
| A wrong value, or a crash inside the library | [Debug it](debugging.md): the crash report names the library's source line, and gdb can stop in the library and in the C++ call stubs. |
| `method too new to be called from this world context` | A wrapper was loaded and called inside one function. Load it at the top level, or call through `Base.@invokelatest M.f(x)`. |
| A crash when freeing memory on Windows | Memory the library allocated was freed with `Libc.free`. Use the library's own function, or `M.libc_free`. |

## Registering and loading

| What you see | What to do |
|:-------------|:-----------|
| `Package '…' not in registry` | Nothing is registered under that name. `register` the recipe first ([Load it anywhere](use.md)). |
| `using MyProject` loads an old build | It loads the build registered last. Call `use("myproject")` once, then start a new session. |
| A wrapper warns that an older RepliBuild generated it | Run `wrap(toml)` again. If a changelog entry says the fix needs a rebuild, `build(toml; clean = true)` first. |
| `register` warns that it could not add the package to an environment | The message says why. `use` still loads the wrapper. Pass `env = …` with another environment, or `env = nothing`. |
| Windows: `unregister` could not delete a folder | A session still has that build loaded. Delete the folder after the session ends. |

## After upgrading RepliBuild

A project's build is reused as long as its TOML, sources and compiler have not
changed, so an upgrade alone does not rebuild it. `wrap(toml)` regenerates the
module with the new version. To also redo the build:

```julia
RepliBuild.build(toml; clean = true)      # from nothing; fetches sources again
RepliBuild.wrap(toml)
```

For a registered project, `RepliBuild.use(name; force_rebuild = true)` does
both.
