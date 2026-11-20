# Every key

The lookup table for `replibuild.toml`. To decide *which* keys your library
needs, read [What goes in the TOML](toml.md) first.

"Default" is the value used when the key is absent. Paths are relative to the
folder holding `replibuild.toml` unless they are absolute.

!!! warning "A misspelled key is ignored without a warning"
    The file is read for the keys below and everything else is dropped. When a
    setting seems to do nothing, check its spelling and its section here.

## `[project]`

| Key | Default | Meaning |
|:----|:--------|:--------|
| `name` | the folder's name | Names the library file (`lib<name>.so`), the Julia module (CamelCase of the name) and the name you `use` it by. |
| `root` | the TOML's folder | The project folder. |

`version` and `uuid` may be present (the Hub's recipes carry a `version`, and
`discover` writes a `uuid`). Neither is used.

## `[compile]`

| Key | Default | Meaning |
|:----|:--------|:--------|
| `flags` | `["-std=c++17", "-fPIC"]` | Compiler flags for every file. `discover` writes `["-fPIC"]` for C. Add `-O2`; neither default optimizes. |
| `defines` | none | A table: `NAME = "value"` becomes `-DNAME=value`. For a define with no value, put `-DNAME` in `flags`. |
| `include_dirs` | written by `discover` | Folders to search for headers. Added automatically: a git or archive dependency's top folder, `include/` and `src/`; a local dependency's folder and `include/`. |
| `source_files` | written by `discover` | The project's own files to compile, one by one. `build` does not search the project folder: a file not listed is not compiled. A dependency's files are **added** to these. Each listed file must exist. |
| `visibility` | `"default"` | `"hidden"` wraps only the functions the library marks as public. Check with the probe first: [Leaving things out](toml.md#Leaving-things-out). Any other value is an error. |
| `parallel` | `true` | Compile files in parallel. |
| `aot_thunks` | `false` | C++: have `wrap` compile the wrapper's C++ call stubs into a second library, `lib<name>_thunks.so`, instead of compiling them each time the wrapper loads. Faster loading; one more file to ship. |

## `[link]`

| Key | Default | Meaning |
|:----|:--------|:--------|
| `optimization_level` | `"0"` | `"0"`, `"1"`, `"2"`, `"3"`, `"s"` or `"z"`. Use `"2"`. |
| `link_libraries` | none | Libraries to link: `["m", "dl"]` is `-lm -ldl`. |
| `link_dirs` | none | Folders to search for those libraries. The library found there is also the one loaded at run time. |
| `enable_lto` | `true` for C, `false` for C++ | **Set to `false`.** `true` is an experimental calling mode ([Experimental keys](#Experimental-keys)). |

## `[wrap]`

| Key | Default | Meaning |
|:----|:--------|:--------|
| `language` | `"cpp"` | `"c"` or `"cpp"`, in lower case. Chooses the compiler. Anything other than `"c"` is treated as C++. A project with a `[rust]` section is `"rust"` and needs no `language`. |
| `module_name` | CamelCase of the project name | The Julia module's name, and the wrapper file's (`julia/<module_name>.jl`). |
| `shim_headers` | none | Headers that define the macros in `[wrap.macros]`, written as in an `#include`. Used only together with `[wrap.macros]` or `shim_defines`. |
| `shim_defines` | none | `#define`s placed before those headers, in that one generated file: `"NAME"` or `"NAME=VALUE"`. Compiles a single-header library's implementation. Needs `shim_headers`. |
| `exclude_symbols` | none | Patterns for functions and globals to leave out of the wrapper. One that matches nothing is an error. |
| `surface_types_only` | `false` | Emit only the types that wrapped functions and globals use. |
| `surface_types_extra` | none | Patterns for types to keep when `surface_types_only` is on. One that matches nothing is an error. |
| `headers` | every header at the top of each include folder | The headers read for enums and callback types that the compiled library does not record. An entry that is not found is an error. |
| `use_clang_jl` | `true` | `false` skips that header reading. |

Patterns use `*` for any text and `?` for one character, and must match the
whole name.

### `[wrap.macros.<NAME>]`

One table per macro, or per `static inline` function.

| Key | Meaning |
|:----|:--------|
| `ret` | The result's **C type**: `"int"`, `"const char*"`, `"lua_State*"`. |
| `args` | The arguments' C types. Present: a function-like macro, called as `NAME(a, b)`. Absent: a constant, called as `NAME()`. `args = []` is a function-like macro with no arguments. |

```toml
[wrap.macros.deflateInit]
ret  = "int"
args = ["z_stream*", "int"]

[wrap.macros.ZLIB_VERSION]
ret = "const char*"
```

### `[wrap.varargs]`

`function = [[types…], [types…]]`: one inner list per combination of extra
arguments, in **Julia's** names for C types. Fixed arguments are not listed.

| Key | Meaning |
|:----|:--------|
| `<function name>` | The list of combinations. Not a list of lists: error. |
| `proven_at` | The dependency commit the combinations were checked against. Absent: `wrap` warns. Different from the dependency's `commit`: error. |

Allowed type names: `Cstring`, `Cwstring`, `Cchar`, `Cuchar`, `Cshort`,
`Cushort`, `Cint`, `Cuint`, `Clong`, `Culong`, `Clonglong`, `Culonglong`,
`Cintmax_t`, `Cuintmax_t`, `Csize_t`, `Cssize_t`, `Cptrdiff_t`, `Cwchar_t`,
`Cfloat`, `Cdouble`, `Cintptr_t`, `Cuintptr_t`, `Bool`, `Any`, the sized
`Int8`…`Int64`, `UInt8`…`UInt64`, `Float32`, `Float64`, and `Ptr{Name}`.
Anything else is an error naming the entry.

```toml
[wrap.varargs]
proven_at       = "1ab3208a1fceb12fca8f24ba57d6e13c5bff15e3"
lua_pushfstring = [["Cstring"], ["Cint"], ["Cstring", "Cint"]]
```

### `[wrap.cstring_owned]`

`function = "release function"`. The wrapper copies the returned `char*` into
a `String`, then frees the buffer with the named function. `"free"` means the
C library's `free`.

### `[wrap.rename]`

`"pattern" = "julia_name"`. The pattern names a function by the name the
wrapper would give it, or by its symbol name in the library. The value must be
a valid Julia name. A pattern that matches nothing, or a name already in use,
is an error.

A function that `wrap` reports under a changed name (`String` as `String_`) is
keyed by the name the library gave it (`String`). The name you choose must not
be a Julia type.

## `[types]`

| Key | Default | Meaning |
|:----|:--------|:--------|
| `templates` | none | C++ template instantiations to compile in: `"std::vector<double>"`. |
| `template_headers` | none | Headers those need, beyond the standard ones found automatically. `"<fmt/format.h>"` for a system-style include, `"mylib.h"` for a quoted one. |
| `records` | none | C++ classes that no source file of the library uses, to make them appear in the wrapper. Spelled as in C++ (`ns::Class`). Not for templates. |
| `custom` | none | A table mapping a C type name to a Julia type, for a type the wrapper cannot map: `MyHandle = "Ptr{Cvoid}"`. |
| `strictness` | `"warn"` | What to do with a type that has no exact Julia equivalent: `"strict"` fails, `"warn"` reports and continues, `"permissive"` continues silently. |
| `allow_unknown_structs` | `true` | Treat an unknown struct as an opaque pointer instead of failing. |
| `allow_unknown_enums` | `false` | Treat an unknown enum as a 32-bit integer. |
| `allow_function_pointers` | `true` | Wrap function-pointer arguments as `Ptr{Cvoid}`. |

## `[dependencies.<name>]`

Source that is compiled into the library. [Where the source comes from](dependencies.md)
explains each type.

| Key | Default | Meaning |
|:----|:--------|:--------|
| `type` | `"local"` | `"git"`, `"archive"`, `"local"` or `"system"`. **Not `"git"` by default**: an entry with a `url` and no `type` fetches nothing. |
| `url` | | `git`: the repository. `archive`: the tarball's URL, or a path to one on disk. |
| `tag` | the default branch | `git`: the tag or branch to check out. |
| `commit` | | `git`: the full 40-character commit the tag must be at. A shorter value is an error; a checkout at another commit stops the build. |
| `sha256` | | `archive`: the tarball's SHA-256. Required. |
| `subdir` | | `archive`: the folder inside the tarball that holds the source. Not needed when the tarball has one top folder. |
| `path` | | `local`: the folder. Used in place, never copied, searched on every build. A `.a` or `.so` found inside it is linked. |
| `pkg_config` | | `system`: a package name. Its include and link flags are taken from `pkg-config`. Nothing is compiled. |
| `exclude` | none | Files and folders not to compile: an entry matches a file's name or any part of its path inside the dependency. Skipped without an entry: for `git` and `archive`, folders named `test`, `tests`, `testes`, `example`, `examples`, `fuzzing`, `build`, `doc`, `docs`; for `local`, only `build`, `.git`, `.cache`. |
| `submodules` | none | `git`: the submodule paths to fetch. Only these, at the commits the repository records. |

## `[platform.<system>]`

`<system>` is `linux`, `windows` or `macos`. Any section of the file may
appear under it, and is laid over the rest on that system:
`[platform.windows.link]`, `[platform.windows.dependencies.glfw]`. A value
replaces the one outside the platform table; a list is replaced whole. Any
other system name is an error.

## `[graph]`

Present: the project is a [CMake project](graph.md) and the build is taken
from CMake.

| Key | Default | Meaning |
|:----|:--------|:--------|
| `source` | `"."` | The folder holding `CMakeLists.txt`. With `dependency`, a folder inside that dependency. |
| `dependency` | | The `[dependencies]` entry whose source is the CMake project. |
| `targets` | every library | The CMake library targets to build and wrap. The libraries they link are included. Naming a program is an error. |
| `options` | none | Arguments for CMake's configure step: `["-DBUILD_TESTING=OFF"]`. |
| `build_type` | `"Release"` | `CMAKE_BUILD_TYPE`. |

An unknown key or a wrong type in `[graph]` is an error. In a graph project
`[compile]`, `[types]` and the `[wrap]` keys other than `module_name` do not
apply: flags come from CMake.

## `[rust]`

Present: the project is a Rust crate, and cargo builds it.
[Rust crates](rust.md) is the guide; `discover` writes this section for a
folder that holds a `Cargo.toml`.

`build`, `wrap` and `use` work on a Rust crate's public functions,
`#[inline]` ones included, on Linux (x86-64 and AArch64; Windows is not run
yet):

- numbers, `bool`, `char`, strings and slices in; records, tuples, enums and
  `Option` out;
- a `String` is a Julia `String` and a `Vec` a Julia `Vector`, both ways;
- anything else that owns something (a `Box`, a record holding a `String`)
  is a Julia object that frees it when Julia collects the object. Passing it
  to a Rust function by value moves it: the object is empty afterwards. The
  module's `OWNED` lists those types;
- a `Result` returned is its value, or a thrown
  `RepliBuild.RustRuntime.RustError` holding the error;
- a Rust panic arrives as a `RepliBuild.RustRuntime.RustPanic`;
- where a function takes `&dyn Trait`, pass a value of any type that
  implements the trait, or a `Box` of the trait object; a trait's methods are
  functions too (`Area_area(x)`);
- where a function takes a function (`fn(i32) -> i32`), pass a Julia
  function;
- a `static NAME` is `NAME()` for its value and `NAME_ptr()` for where it is.

A generic function is wrapped for the types `instantiate` names. A function
that is not wrapped is in the module as a stub that says why, and is listed
in the module's `UNSUPPORTED`.

| Key | Default | Meaning |
|:----|:--------|:--------|
| `manifest` | `"Cargo.toml"` | The crate's `Cargo.toml`, relative to the project root. With `dependency`, relative to that dependency's source. |
| `dependency` | | The `[dependencies]` entry whose source holds the crate. |
| `package` | the manifest's own package | The package to wrap, by its Cargo name. Needed when the manifest is a workspace. |
| `opt_level` | `0` | The optimization level the crate is compiled at, `0` to `3`. |
| `features` | none | Cargo features to turn on: `["simd"]`. |
| `default_features` | `true` | `false` turns the crate's default features off. |
| `instantiate` | none | Generic functions to compile for particular types, each written as Rust names it: `["geo::largest::<f64>", "geo::Pair::<i32>::sum"]`. |

A generic function such as `fn largest<T>(xs: &[T]) -> T` is not in a
library until something uses it with a type, so `instantiate` says which
types you want. Each entry becomes a function of the module named after it
(`largest_f64`, `Pair_i32_sum`). An entry that is not a function of the
crate is a build error that shows the compiler's message. A function that
takes `impl Trait` is named by the type it should have:
`"(geo::scale as fn(f32) -> f64)"`. Nothing else about a crate needs an
entry: its `#[inline]` functions are found by the build.

The crate is built as it is: it needs no `extern "C"`, no `#[repr(C)]` and no
other change. `cargo` and `rustc` must be installed; `REPLIBUILD_TOOL_CARGO`
and `REPLIBUILD_TOOL_RUSTC` name particular ones.

`wrap` compiles the calls a plain Julia call cannot make into
`lib<name>_thunks.so` beside the library, whenever it can; there is no key
for it. Without that file the module compiles them at first use.

An unknown key or a wrong type in `[rust]` is an error. In a Rust project
`[compile]`, `[link]` and `[types]` do not apply, and `[rust]` cannot be
combined with `[graph]` or `[ingest]`.

## `[paths]`, `[binary]`, `[cache]`

| Key | Default | Meaning |
|:----|:--------|:--------|
| `[paths] output` | `"julia"` | Where the library and the wrapper are written. |
| `[paths] build` | `"build"` | Where intermediate files are written. |
| `[binary] type` | `"shared"` | Leave as is. A wrapper needs a shared library. |
| `[binary] output_name` | `lib<name>.so` | The library's file name. |
| `[cache] enabled` | `true` | Reuse compiled files between builds. |
| `[cache] directory` | `".replibuild_cache"` | Where they, and fetched dependencies, are kept. |

## Keys that do nothing

`discover` writes these and nothing reads them. Leave them or delete them:

`[discovery]` (the whole section), `[workflow] stages`, `[wrap] enabled`,
`[wrap] style`, `[binary] strip_symbols`, `[paths] source`, `[paths] include`,
`[paths] cache`, `[project] uuid`, `[llvm]`.

In particular `[discovery] ignore_patterns` does not keep files out of a
build. Use `exclude` on the dependency.

## Experimental keys

These belong to features that are not part of the supported workflow. A
recipe does not need them.

| Key | Leave it at | What it is |
|:----|:------------|:-----------|
| `[link] enable_lto` | `false` | Calls C functions through the library's compiled code embedded in the wrapper, instead of through the library file. Fails on large libraries. |
| `[wrap.tier1]` | absent | A per-function version of the same idea. Described under [Tier 1](tier1.md). |
| `[link] promote_statics`, `[link] fallback` | absent | Support for the two keys above. |
| `[ingest]` | absent | Wraps a C library you compiled yourself, without its source. Written by [`RepliBuild.ingest`](api.md#Ingest). |

## What a forced discover keeps

`discover` on a project that already has a `replibuild.toml` leaves the file
alone. `discover(path; force = true)` scans again and rewrites it, keeping
these keys from the old file:

- `[dependencies]`, `[graph]`, `[rust]` and `[platform.*]`, whole;
- `[types] templates`, `template_headers`, `records`;
- `[wrap] macros`, `varargs`, `cstring_owned`, `rename`, `shim_headers`,
  `shim_defines`, `exclude_symbols`, `surface_types_only`,
  `surface_types_extra`, `headers`, `tier1`;
- `[compile] visibility`, `[link] promote_statics`.

**Everything else is rewritten from the scan**, including `[compile] flags`,
`include_dirs` and `[link] link_libraries`. Do not use `force = true` on a
recipe you have tuned by hand; keep the file in version control.

## What stops, what warns

**Errors when the file is read**

- `commit` that is not 40 hexadecimal characters.
- `[wrap.varargs]` entry that is not a list of lists.
- `[compile] visibility` other than `"default"` or `"hidden"`, or `"hidden"`
  together with a different `-fvisibility=` in `flags`.
- `shim_defines` without `shim_headers`, or an entry that is not `NAME` or
  `NAME=VALUE`.
- `submodules` on a dependency that is not `git`, or an entry that is not a
  plain relative path.
- An unknown `[platform.<system>]` name.
- An empty `[project] name`, a `root` that does not exist, a `source_files`
  entry that does not exist, an `optimization_level` outside the list.

**Errors during `build` or `wrap`**

- The checked-out dependency is not at `commit`, or a tarball does not match
  `sha256`.
- A `shim_headers` entry resolved to a file outside the project and its
  dependencies.
- A `[wrap.varargs]` type name outside the allowed list, or a `proven_at` that
  differs from the dependency's `commit`.
- An `exclude_symbols`, `surface_types_extra` or `[wrap.rename]` pattern that
  matches nothing.
- A `[wrap] headers` entry that is not found.
- The built library exports none of its functions: `visibility = "hidden"` on
  a library that does not mark its public functions.

**Warnings**

- A variadic function with no `[wrap.varargs]` entry, and a `[wrap.varargs]`
  table with no `proven_at`.
- A `[types] templates` entry that produced nothing usable.
- A struct that could not be given named fields and is wrapped as raw bytes
  with accessor functions.
- A git dependency that could not be fetched. The build continues and then
  fails on the missing files, so look for `Failed to clone dependency` above
  the first compile error.
