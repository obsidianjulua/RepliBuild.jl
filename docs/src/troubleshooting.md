# Troubleshooting

The build succeeding is not the same as the wrapper being complete. Most "bugs"
are a missing TOML entry. This page is symptom → what to edit.

## Missing from the wrapper, or wrong at runtime

| What you see | Look at |
|--------------|---------|
| A C++ template type is missing, or members are `Any` | `[types] templates` / `template_headers` |
| A struct from a header has **no fields** | `-fstandalone-debug` in `[compile] flags` |
| A struct became an opaque blob with a layout warning | Unmodellable member; read the warning's named type |
| A macro-based API is absent (no error) | `[wrap.macros]` + `shim_headers` |
| Macro values are wrong but everything compiled | Shim header resolved to a **system** copy of the same name — use a more specific path or an explicit `-I` |
| `printf`-style function only takes its fixed arguments | `[wrap.varargs]` |
| `invalid type '…' in [wrap.varargs.f]` | You wrote C types; varargs want **Julia** types (`Cint`, `Cstring`, …) |
| `must be a list of type lists` | You wrote `f = ["Cint"]`; write `f = [["Cint"]]` |
| Memory grows on every string-returning call | `[wrap.cstring_owned]` |
| Unrelated symbols from tests/examples in the wrapper | `exclude` on the dependency |
| A TU fails to compile with a missing header | `[compile] include_dirs` |
| A TU fails on an identifier upstream's build defines | `[compile] flags` / `[compile.defines]` |
| A setting seems to have no effect | Typos are silent (no unknown-key warning). Also check the ignored-keys list in [Edit the TOML](config.md#5.-What-fails-loudly-what-warns-what-is-ignored) |
| `use()` serves a stale wrapper | Cache keys on RepliBuild's version too; `use(name; force_rebuild=true)` once |

Details and examples for each key: [Edit the TOML](config.md).

## Build and environment

| What you see | What to do |
|--------------|------------|
| `RepliBuild supports Linux only` | It does. ELF / DWARF / `nm`. |
| C++ wrap fails, C works | [Install](install.md): system LLVM/MLIR 21+, then `cd src/mlir && ./build.sh` |
| Fresh clone, C++ path looks "regressed" | No `src/mlir/build/libJLCS.so` yet. Build the dialect. |
| `Configuration file not found` | Run `discover` first, or pass the path to the TOML. |
| `Library not found: …/julia/lib….so` | Run `build` before `wrap`. |
| Dependency HEAD disagrees with `commit` | The tag moved. Update `commit` (full 40 hex) or pin a different tag. |
| Shim `#include` resolved outside the project | `shim_headers` hit a system header. More specific path, or `-I`. |
| Wrapper generation refused (undeclared type in a `ccall`) | A type did not make it into the module. Usually a missing template, or a header-only struct without `-fstandalone-debug`. |

## C vs C++ mix-ups

`[wrap] language` selects the toolchain, not just the generator. A C project
forced to `"cpp"` needs a system LLVM you may not have. A C++ project forced to
`"c"` will wrap the `extern "C"` surface and drop classes, methods, and virtual
dispatch.

Ingest is C only. Ingesting a C++ `.so` will not give you a usable class API —
build that library from source.

## Cache

Per-file IR is keyed on source plus compile fingerprint (flags, defines,
includes, LLVM version). Changing flags rebuilds without a manual clean.

`~/.replibuild/builds/<hash>/` is the `use()` cache. The hash includes the
generator version. `clean(toml)` removes project-local `build/`, `julia/`, and
`.replibuild_cache/`. It does not wipe the global `use()` cache; `force_rebuild=true`
does that for one package.

## Loud vs silent

A typo'd TOML key is **accepted and ignored**. If a setting "does nothing",
check spelling against the [section reference](config.md#3.-Section-reference)
before anything else.

Hard errors (the useful kind): bad `commit` shape, bad `[wrap.varargs]` shape,
a shim include that escaped the tree, a wrapper that would name an undeclared
type. Those messages name the fix.
