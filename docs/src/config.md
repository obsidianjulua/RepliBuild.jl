# The `replibuild.toml` Reference

`replibuild.toml` is the interface. Every piece of library-specific knowledge
lives in this file and nowhere else — not in a generator fork, not in a patch, not
in a hand-written shim. `discover()` writes what it can see by scanning; the rest
is what **you** have to tell it, and §2 is the list of exactly what that is.

The file is read in one place ([`ConfigurationManager.load_config`](internals.md#Configuration-Manager))
and frozen into an immutable `RepliBuildConfig`. Everything below is what that
parser actually does — including the keys it validates loudly, the keys it warns
about, and the handful it currently accepts and ignores.

---

## 1. How the file is created and maintained

```julia
toml = RepliBuild.discover("path/to/project")      # writes replibuild.toml
RepliBuild.discover("path/to/project", force=true) # rescan, preserving intent keys
```

Discovery scans sources, walks the `#include` graph, picks `wrap.language` from
the file extensions it found, and writes a config that compiles. It cannot infer
anything that is not visible in the source tree's *shape* — see §2.

**Forced re-discovery preserves hand-curated keys.** `discover(force=true)`
regenerates the file, and without carry-over it would emit these empty, silently
destroying your intent. The preserved set (`Discovery.PRESERVED_TOML_KEYS`) is:

| Section | Key |
|---|---|
| `[types]` | `templates` |
| `[types]` | `template_headers` |
| `[wrap]` | `varargs` |
| `[wrap]` | `macros` |
| `[wrap]` | `shim_headers` |
| `[wrap]` | `cstring_owned` |
| `[wrap]` | `tier1` |
| `[link]` | `promote_statics` |

A regenerated *non-empty* value wins; an empty or absent one gets the preserved
value. **If you add a new user-intent key to the schema, add it to that list too**
— otherwise the next forced re-discovery eats it. (This is not hypothetical: the
`stl_test` fixture was red for six weeks because `[types].templates` vanished on
every run.)

Anything **not** in that table is regenerated from the scan. Notably `[compile]
flags`, `include_dirs`, and `[dependencies]` are not preserved — keep a
hand-written config under version control rather than relying on `force=true` to
round-trip it.

`save_config` only writes keys with non-default or non-empty values, so a
round-tripped file is smaller than the full schema. Absent keys take the
parse-time defaults in §3, which are **not always the same as what `discover()`
writes** — the divergences are flagged in the tables.

---

## 2. What discovery cannot know

This is the important section. Everything here is information that exists only in
a human's head or in the upstream build system, and that RepliBuild will not
guess. Omit it and you get a build that *succeeds* and a wrapper that is missing
things or is wrong.

| If your library… | You must declare | Symptom if you don't |
|---|---|---|
| is C++ and you want STL/template types | `[types] templates` + `template_headers` | The type never appears in DWARF; members typed `Any`, methods missing |
| exposes API through preprocessor macros | `[wrap.macros.*]` + `[wrap] shim_headers` | The macro simply does not exist in the wrapper — no error |
| has variadic functions worth calling with arguments | `[wrap.varargs]` | Only the zero-variadic base wrapper is generated |
| returns malloc'd `char*` | `[wrap.cstring_owned]` | Every call leaks the C buffer |
| defines its public "def" structs header-only | `-fstandalone-debug` in `[compile] flags` | Structs wrap to **empty** Julia structs |
| ships tests/examples/backends in the same repo | `exclude` on the dependency | Unrelated sources compile in and pollute the DWARF |
| keeps headers somewhere other than `<root>`, `<root>/include`, `<root>/src` | `[compile] include_dirs` | Compile failure, or a TU silently missing declarations |
| needs `-lm`, `-lpthread`, `-ldl` | `[link] link_libraries` | Loads fine; Tier-1 slices demote, or a symbol is missing at first use |
| is configured by its build system through `-D` | `[compile] flags` / `[compile] defines` | One TU fails to compile, or a feature is silently off |
| builds with `-fvisibility=hidden` | Nothing extra for macros (handled); mind `[compile] flags` parity with upstream | Exports missing from `nm -g` |
| is C and you want cross-language inlining | `[wrap.tier1] enable = true` | Everything stays Tier 3 `ccall` (correct, just not inlined) |
| must be reproducible | `commit` on the dependency | A moved tag silently changes what you compiled |

### 2.1 C++ templates and the STL — "STL needs STL in the TOML"

**A template that is never instantiated does not exist in the binary.** The
compiler emits DWARF for code it generated; an uninstantiated `std::vector<int>`
generated nothing, so there is nothing to read. No amount of header scanning
changes this — the information is not missing, it was never created.

So you name the instantiations you want, and RepliBuild writes a stub `.cpp` that
forces them:

```toml
[types]
templates        = ["std::vector<int>", "std::string", "std::map<int, int>"]
template_headers = ["<vector>", "<string>", "<map>"]
```

- `templates` entries are **C++ type spellings**, exactly as you would write them
  in `template class X;`.
- Common STL headers are auto-detected from the type name (`std::vector` →
  `<vector>`, `std::map` → `<map>`, and the same for `unordered_map`, `set`,
  `unordered_set`, `deque`, `list`, `basic_string`/`string`). `template_headers`
  is for everything else — your own headers, or a library's:
  ```toml
  templates        = ["fmt::basic_memory_buffer<char>"]
  template_headers = ["<fmt/format.h>"]
  ```
- Typedef spellings are expanded for you where an explicit instantiation would
  otherwise be invalid C++: `std::string` → `std::basic_string<char>`, and the
  same for `wstring` / `u16string` / `u32string`.
- Entries starting with `<` are emitted as `#include <…>`; anything else becomes
  `#include "…"`.

Once instantiated, the STL method symbols are extracted
(`extract_stl_method_symbols`) and become container accessor thunks. A C++
library whose *public* API is STL-free needs none of this — box2d 2.4 reports
`stl_methods = 0` and declares no templates.

!!! warning "This section is preserved but not derived"
    `[types].templates` is one of the preserved keys precisely because a rescan
    would otherwise wipe it. If your C++ types come back as `Any` or your struct
    collapses to an opaque blob, this is the first key to check.

### 2.2 Preprocessor macros

Macros do not survive compilation, so they are absent from both the symbol table
and DWARF. RepliBuild generates a typed C/C++ shim function per declared macro,
compiles it into the library, and wraps the resulting real symbol.

```toml
[wrap]
shim_headers = ["lua.h", "lauxlib.h"]     # what the shim TU #includes

# Function-like macro: `args` present → emits MACRO(arg0, arg1)
[wrap.macros.lua_pop]
ret  = "void"
args = ["lua_State*", "int"]

# Value macro: `args` ABSENT → emits the bare MACRO expression
[wrap.macros.LUA_REGISTRYINDEX]
ret = "int"
```

- `ret` and `args` are **C type strings**, pasted verbatim into generated C. They
  are not Julia types. (Contrast §2.3, which *is* Julia types — this asymmetry is
  the single easiest mistake to make in this file.)
- **The presence of the `args` key is the switch**, not its contents:
  `args = []` means a zero-argument function-like macro (`MACRO()`), while
  omitting `args` entirely means a value macro (`MACRO`).
- Shims are emitted with `__attribute__((used, visibility("default")))` so they
  survive `-fvisibility=hidden` builds and LTO internalization — without that,
  projects like box2d3 would silently lose every macro.
- In Julia the `replibuild_shim_` prefix is stripped: you call `lua_pop(L, 1)`.

!!! warning "The shim's `#include` must resolve inside your tree"
    The shim TU is generated under the build cache, not inside the source tree,
    so a bare `#include "blake3.h"` can fall through the `-I` path to a
    **system-installed** copy of the header at a different version — and bake
    wrong values into every macro, with no other symptom. A guard checks each
    direct shim include and hard-errors if it resolved outside the
    project/dependency tree. The fix is either a more specific subpath
    (`shim_headers = ["c/blake3.h"]`) or an explicit `-I<dir>` in `[compile] flags`.

### 2.3 Variadic functions

DWARF records that a function is variadic. It cannot record what you intend to
pass. Declare one entry per concrete overload:

```toml
[wrap.varargs]
lua_pushfstring = [
    ["Cstring"],
    ["Cint"],
    ["Cstring", "Cint"],
]
```

Three rules, each of which has bitten someone:

1. **List only the variadic arguments.** The fixed parameters (`lua_State*`,
   the `const char* fmt`) come from DWARF and are prepended automatically.
2. **The names are Julia types, not C types.** Allowed: `Any`, `Bool`, `Cstring`,
   `Cwstring`, `Cchar`, `Cuchar`, `Cshort`, `Cushort`, `Cint`, `Cuint`, `Clong`,
   `Culong`, `Clonglong`, `Culonglong`, `Cintmax_t`, `Cuintmax_t`, `Csize_t`,
   `Cssize_t`, `Cptrdiff_t`, `Cwchar_t`, `Cfloat`, `Cdouble`, `Cintptr_t`,
   `Cuintptr_t`, the sized `Int*`/`UInt*`/`Float*` types, and `Ptr{Name}`.
   Anything else is a hard error naming the entry.
3. **It is a list of lists.** `f = ["Cint"]` is one overload taking… nothing
   coherent; the parser rejects it and tells you to write `f = [["Cint"]]`.

Vararg wrappers lower as **true variadic calls** (the `@ccall` semicolon form), so
the x86-64 SysV variadic protocol — including the `AL` register-count setup that
gates the callee's `va_start` — is formally correct, float varargs included. A
`Ptr{X}` overload sanitizes to `_PtrX` in the generated function name.

### 2.4 `char*` return ownership

Whether a returned `char*` must be freed, and by what, is not in DWARF and not
reliably in the name. Declare it:

```toml
[wrap.cstring_owned]
cJSON_Print     = "cJSON_free"
sqlite3_mprintf = "sqlite3_free"
```

The wrapper copies the string into Julia and then frees the C buffer through the
named symbol. Undeclared functions get the default policy —
`Union{String,Nothing}`, where NULL is a value rather than an exception, and the
buffer is left alone. Every `Cstring`-returning function additionally gets a raw
`<name>_ptr` variant that returns the pointer unchanged for lifetime-sensitive
callers. The value must be a non-empty string; anything else is warned about and
ignored.

!!! note "The policy does not depend on the dispatch tier"
    A `char*` return is presented the same way whether the function is called
    through `ccall`, a Tier-1 slice, or a Tier-2 MLIR thunk — the tier decides
    how the call is *made*, never how the result is presented. That was not
    always true: until 2026-08-12 the C++ generator's Tier-2 path skipped the
    policy entirely, so 75 functions across five Hub packages returned a bare
    `Cstring` and any `cstring_owned` declaration on one of them was silently
    discarded. Both halves now come from one derivation, and a guard on the
    write path refuses any wrapper in which a `char*` return escapes as a bare
    `Cstring` without a `_ptr` sibling.

### 2.5 Header-only aggregates need `-fstandalone-debug`

Clang's default "limited" debug info emits a header-only aggregate that no TU
instantiates as a `DW_AT_declaration`-only stub. Those are exactly the "def"
structs a caller is supposed to *fill in* — `b2BodyDef`, `b2FixtureDef`,
`llama_model_params`, `llama_context_params` — so they wrap to **empty Julia
structs** and the API becomes uncallable.

```toml
[compile]
flags = ["-O2", "-fPIC", "-std=c++17", "-fstandalone-debug"]
```

DWARF is RepliBuild's input format, so this flag is load-bearing, not bloat. Reach
for it whenever a struct you can see in the header comes out with no fields.

### 2.6 Excluding what is not the library

A repository is usually more than the library: tests, examples, GUI testbeds,
vendored third-party trees, alternative backends, other CPU architectures. The
dependency resolver walks the whole tree, so anything compilable gets compiled and
lands in the DWARF the wrapper is generated from.

```toml
[dependencies.box2d]
type = "git"
url  = "https://github.com/erincatto/box2d.git"
tag  = "v2.4.1"
exclude = ["testbed", "unit-test", "extern", "docs"]
```

`exclude` entries are matched against the path **relative to the dependency
root**, and the match is deliberately generous — an entry hits if it equals the
filename, or is a prefix, suffix, **or any substring** of the relative path.
That is why `"ggml-cuda/"` and `"tools/"` work as directory filters and
`"xmltest.cpp"` works as a file filter, and also why a very short entry can
over-match: prefer `"tests"` over `"t"`, and keep a trailing `/` on directory
entries. Excluding aggressively is normal — llama.cpp's config excludes roughly
thirty paths, each with a comment explaining why.

The walker additionally drops directories named `test`, `tests`, `testes`,
`example`, `examples`, `fuzzing`, `build`, `.git`, `doc`, and `docs` on its own,
so those need no entry (several Hub configs list them anyway, harmlessly).

### 2.7 Include roots

The resolver adds `<clone>`, `<clone>/include` and `<clone>/src` automatically.
Anything at a different depth must be named:

```toml
[compile]
include_dirs = [
    ".replibuild_cache/deps/llamacpp/ggml/include",
    ".replibuild_cache/deps/llamacpp/ggml/src",
]
```

Entries are passed through verbatim as `-I<dir>`. Relative paths resolve against
the directory containing `replibuild.toml`, because `build()` runs there.

### 2.8 Link libraries

```toml
[link]
link_libraries = ["m", "pthread", "dl"]   # → -lm -lpthread -ldl
link_dirs      = ["/opt/thing/lib"]       # → -L/opt/thing/lib
```

A shared library links happily with `sin` left undefined, so omitting `-lm` is not
a build error — it is a `.so` whose `DT_NEEDED` chain is short. That matters
beyond the obvious runtime failure: the Tier-1 slice pre-flight scopes symbol
lookup to the library **plus its `DT_NEEDED` chain**, so on miniaudio a missing
`-lm` demoted all 51 slices that declare `sin`/`pow`/`log` back to `ccall` with no
other sign.

!!! warning "`link_libraries` vs `extra_link_libs`"
    `[link] link_libraries` is the source-build key. `[ingest] extra_link_libs` is
    the ingest-mode key. Putting `extra_link_libs` in a source-build config is
    **silently ignored** — it is not part of `[link]` and nothing reads it there.

### 2.9 Defines the build system would have supplied

Upstream's CMake or Makefile often injects defines that no header declares. Without
them a TU fails to compile, or a subsystem is silently absent:

```toml
[compile]
flags = [
    "-O2", "-fPIC",
    "-DGGML_USE_CPU",                 # else the backend registry is empty
    '-DGGML_VERSION="0.18.1"',        # else ggml.c is the one TU that won't build
]

[compile.defines]                     # equivalent, emitted as -Dkey=value
SQLITE_THREADSAFE = "1"
```

`[compile.defines]` always emits `-Dkey=value`; for a bare `-DFOO` with no value,
use `[compile] flags`. Both feed the compile fingerprint, so changing either
correctly invalidates the per-file IR cache.

### 2.10 Tier 1 (C only)

Opt in per package. See [Zero-cost LTO dispatch](guide.md#Zero-Cost-LTO-Dispatch)
for what a slice is:

```toml
[wrap.tier1]
enable = true
```

!!! warning "Experimental — off by default, and not a supported tier"

    Tier 1 is a side project inside RepliBuild. `enable` defaults to `false`,
    no Hub package sets it, and its tests are deliberately unwired from
    `devtests.jl`. Leaving it off costs you nothing but inlining: every
    function stays on `ccall`, which is correct. The [guide](guide.md) has the
    mechanism and the caveats.

### 2.11 Pinning content, not just a name

A git tag is a mutable ref. `commit` is the only check that survives a cache wipe:

```toml
[dependencies.lua]
type   = "git"
url    = "https://github.com/lua/lua.git"
tag    = "v5.4.7"
commit = "1ab3208a1fceb12fca8f24ba57d6e13c5bff15e3"
```

It must be a full 40-hex object name — abbreviated SHAs are rejected at config
load, because they are ambiguous against future history. A mismatch after
clone/checkout is a **hard error**, not a warning.

---

## 3. Section reference

Types are TOML types. "Default" is what the parser uses when the key is
**absent**; where `discover()` writes something different, that is called out.

### `[project]`

| Key | Type | Default | Notes |
|---|---|---|---|
| `name` | String | directory basename | Drives `lib<name>.so` and the module name |
| `root` | String | directory containing the TOML | Absolute or relative |
| `uuid` | String | freshly generated | An unparseable UUID warns and is regenerated. Keep the one `discover()` wrote — an absent `uuid` means a new one on every load |
| `version` | String | — | **Not part of the schema.** Every Hub config carries one and it is useful documentation (it should match the pinned `tag`), but RepliBuild parses and drops it; the Hub's own `index.toml` is what publishes a version |

### `[paths]`

| Key | Type | Default | Notes |
|---|---|---|---|
| `output` | String | `"julia"` | Wrapper + `.so` + `compilation_metadata.json` land here |
| `build` | String | `"build"` | Intermediate IR and objects |
| `source` | String | `"src"` | **Parsed, not consulted** — sources come from discovery or `[compile] source_files` |
| `include` | String | `"include"` | **Parsed, not consulted** — include roots come from discovery or `[compile] include_dirs` |
| `cache` | String | `".replibuild_cache"` | **Parsed, not consulted** — the cache directory is `[cache] directory` |

### `[discovery]`

| Key | Type | Default |
|---|---|---|
| `enabled` | Bool | `true` |
| `walk_dependencies` | Bool | `true` |
| `max_depth` | Int | `10` |
| `ignore_patterns` | Vector{String} | `["build", ".git", ".cache"]` |
| `parse_ast` | Bool | `true` |

!!! note
    This whole section is currently **parsed and stored but not consulted** by the
    scanner. It round-trips faithfully; it does not change behaviour. Do not
    reach for `ignore_patterns` to trim a dependency — use `exclude` on the
    dependency (§2.6).

### `[compile]`

| Key | Type | Default | Notes |
|---|---|---|---|
| `flags` | Vector{String} | `["-std=c++17", "-fPIC"]` | `discover()` writes `["-fPIC"]` for a C-only scan |
| `defines` | Table{String→String} | `{}` | Emitted as `-Dkey=value` |
| `parallel` | Bool | `true` | Multi-threaded per-file compilation |
| `source_files` | Vector{String} | `[]` (auto-discovered) | Explicit list; validated to exist at load |
| `include_dirs` | Vector{String} | `[]` (auto-discovered) | Passed as `-I`; relative to the TOML's directory |
| `aot_thunks` | Bool | `false` | Pre-compile Tier-2 thunks into `<name>_thunks.so`; needs `libJLCS.so` |

Changing `flags`, `defines`, or `include_dirs` correctly invalidates the per-file
IR cache — it is keyed on a compile fingerprint as well as source `mtime`.

### `[link]`

| Key | Type | Default | Notes |
|---|---|---|---|
| `optimization_level` | String | `"0"` | `"0"`–`"3"`, `"s"`, `"z"`; a leading `O`/`o` is stripped, so `"O2"` works. `discover()` writes `"0"`; Hub configs use `"2"` |
| `enable_lto` | Bool | `true` for `language = "c"`, else `false` | Whole-module `llvmcall` payload — see the warning below |
| `link_libraries` | Vector{String} | `[]` | `-l` entries |
| `link_dirs` | Vector{String} | `[]` | `-L` entries |
| `fallback` | Bool | `false` | **C only.** `false` = link/optimize/assemble in-process on Julia's resident libLLVM, and a failure is a hard error. `true` = external `llvm-link`/`opt`. C++ always uses the external pipeline |
| `promote_statics` | Bool | `true` | **C in-process bucket only.** Rename anything a slice may bind by `declare` but that cannot reach the `.so` dynamic symbol table to an exported `__rb_<lib>_<name>`. Required by `[wrap.tier1]`, harmless otherwise |

!!! warning "`enable_lto` embeds the whole module — prefer `[wrap.tier1]`"
    `enable_lto` embeds the **entire linked module** at every `llvmcall` site. At
    whole-library scale that can crash Julia's JIT, and it duplicates file-local
    `static` state between the embedded bitcode and the `.so` — two calls to the
    same library can observe different copies of its internal state. Every Hub
    config sets `enable_lto = false`.

    This is not a statement about Tier 1. The per-function slicing path under
    `[wrap.tier1]` is immune to both problems and is the supported way to get
    Tier 1. The knobs are independent, and `enable_lto = false` together with
    `[wrap.tier1] enable = true` is the intended combination.

### `[binary]`

| Key | Type | Default | Notes |
|---|---|---|---|
| `type` | String | `"shared"` | `"shared"`, `"static"`, `"executable"`; anything else warns and falls back to `"shared"` |
| `output_name` | String | `""` → `lib<project>.so` (`.a` when static) | In ingest mode this is forced to the ingested `.so`'s basename |
| `strip_symbols` | Bool | `false` | **Parsed, not consulted** |

### `[wrap]`

| Key | Type | Default | Notes |
|---|---|---|---|
| `language` | String | `"cpp"` | `"c"`, `"cpp"`, `"rust"` (experimental). Selects generator, toolchain, and the `enable_lto` default. `discover()` sets it from the scanned extensions |
| `enabled` | Bool | `true` | |
| `module_name` | String | `""` → CamelCase of the project name | |
| `use_clang_jl` | Bool | `true` | `false` skips AST extraction (DWARF-only path) |
| `shim_headers` | Vector{String} | `[]` | Headers the macro-shim TU includes *(preserved)* |
| `dag` | Bool | `false` | Export DAG type-graph diffs to `<project>/dag/` |
| `style` | String | `"clang"` | `"clang"`, `"basic"`, `"none"`; validated, but **not currently dispatched on** — the basic symbol-only generator is selected by *missing* `compilation_metadata.json`, not by this key |

### `[wrap.varargs]` *(preserved)*

Typed overloads for variadic functions. Julia type names, variadic arguments
only, list-of-lists. See §2.3 for the full rules and the allowed type names.

```toml
[wrap.varargs]
lua_pushfstring = [["Cstring"], ["Cint"], ["Cstring", "Cint"]]
```

### `[wrap.macros]` *(preserved)*

One sub-table per macro. `ret` and `args` are **C type strings**; the presence of
`args` selects function-like vs value macro. See §2.2.

```toml
[wrap.macros.deflateInit]
ret  = "int"
args = ["z_stream*", "int"]

[wrap.macros.ZLIB_VERSION]
ret = "const char*"
```

### `[wrap.cstring_owned]` *(preserved)*

`function name = "deallocator symbol"`. See §2.4.

### `[wrap.tier1]` *(preserved)*

**C projects only.** Routes eligible functions through `Base.llvmcall` on a
per-function bitcode slice instead of `ccall`.

```toml
[wrap.tier1]
enable       = true
max_slice_kb = 64
allow_setjmp = false
exclude      = ["lua_error"]
```

| Key | Type | Default | Notes |
|---|---|---|---|
| `enable` | Bool | `false` | Run the slicer over every Tier-1 candidate and emit `Base.llvmcall` for the accepted ones |
| `max_slice_kb` | Int | `64` | A tripwire, not a tuning knob — declarations-only slices are kilobyte-sized regardless of function size, so a hit means something unexpected got embedded |
| `allow_setjmp` | Bool | `false` | Allow slices whose **own body** calls `setjmp`/`longjmp` or a `returns_twice` callee. Direct-reach only: `setjmp` buried in a `.so`-side callee never trips the gate, and does not need to — `longjmp` *across* the calling Julia frame is identical exposure for `ccall` and `llvmcall` |
| `exclude` | Vector{String} | `[]` | Names (mangled or plain) that stay on `ccall` unconditionally |

Requires `[link] promote_statics = true` (the default) and the in-process C
pipeline. On a `fallback` or ingest build there is no promoted module, so Tier 1
disables itself for the whole wrap with a warning rather than shipping unverified
slices. Every non-accepted function emits exactly the `ccall` it would have
emitted with Tier 1 off, and the module exports `TIER1_FUNCTIONS::Set{String}`.

The wrap prints two counts — `N slices accepted` then `N slices emitted
(M functions)`. A gap between them is the ABI shape gate refusing a `Cstring` or
struct crossing, not a failure.

### `[types]`

| Key | Type | Default | Notes |
|---|---|---|---|
| `strictness` | String | `"warn"` | `"strict"`, `"warn"`, `"permissive"`; any unrecognised value silently becomes `"warn"` |
| `allow_unknown_structs` | Bool | `true` | Opaque pointers for unknown structs instead of failing |
| `allow_unknown_enums` | Bool | `false` | Map unknown enums to `Int32` |
| `allow_function_pointers` | Bool | `true` | `Ptr{Cvoid}` for function pointers |
| `custom` | Table{String→String} | `{}` | Custom type mappings, e.g. `MyHandle = "Ptr{Cvoid}"`; merged into the type registry |
| `templates` | Vector{String} | `[]` | C++ instantiations to force-emit *(preserved)* — §2.1 |
| `template_headers` | Vector{String} | `[]` | Headers for the instantiation stub *(preserved)* — §2.1 |

Strictness modes: **`strict`** fails on any type that cannot be mapped exactly;
**`warn`** reports the imperfect mapping and continues (`Ptr{Cvoid}` for complex
pointers); **`permissive`** falls back silently. This governs *type mapping*; it
does not relax the exact-or-opaque rule for struct layouts, which is
unconditional.

### `[dependencies]`

Each dependency is a named sub-table. Both spellings work:

```toml
[dependencies.cjson]
type    = "git"
url     = "https://github.com/DaveGamble/cJSON.git"
tag     = "v1.7.18"
commit  = "acc76239bee01d8e9c858ae2cab296704e52d916"
exclude = ["test", "fuzzing"]

[dependencies]
pugixml = { type = "git", url = "https://github.com/zeux/pugixml.git", tag = "v1.15" }
```

| Key | Type | Default | Notes |
|---|---|---|---|
| `type` | String | `"local"` | `"git"`, `"local"`, `"system"` — **note the default is `local`, not `git`**; a `[dependencies.x]` with a `url` but no `type` will not clone |
| `url` | String | `""` | Required for `type = "git"` |
| `tag` | String | `""` (default branch) | Tag or branch; a tag makes the clone shallow |
| `commit` | String | `""` | Expected 40-hex object name. Validated at load; mismatch after checkout is a hard error — §2.11 |
| `path` | String | `""` | Required for `type = "local"`; scanned in place, never copied |
| `pkg_config` | String | `""` | For `type = "system"`; runs `pkg-config --cflags` |
| `exclude` | Vector{String} | `[]` | Paths to skip during source injection — §2.6 |

Git dependencies clone into `.replibuild_cache/deps/<name>/` and re-checkout on
tag change. Resolved sources merge into the compilation graph before `[compile]`
runs, so they are compiled with the same toolchain and flags as your own files —
which is exactly why the resulting DWARF is consistent across the whole project.

### `[ingest]`

**Experimental, C only.** The *presence* of this section is the mode switch: no
compilation happens, only DWARF extraction and wrapper generation over a `.so`
you built yourself (with `-g`). Ingested libraries dispatch through Tier 3
(`ccall`) exclusively — no LTO bitcode, no Tier-1 slices, no Tier-2 thunks, and
the C++ API surface of an ingested binary is unsupported.

| Key | Type | Default | Notes |
|---|---|---|---|
| `library` | String | — (required) | Path to the pre-built `.so`; relative paths resolve against the TOML's directory. Missing → the section is **ignored with a warning** and you silently get source-build mode |
| `headers` | Vector{String} | `[]` | Header search dirs for type extraction and the `noexcept` scan; merged into the include dirs |
| `extra_link_libs` | Vector{String} | `[]` | Additional `-l` libraries the wrapper loads at runtime. **Ingest-only** — in a source build use `[link] link_libraries` |

Generate it with `RepliBuild.ingest("/path/to/libfoo.so", headers=[...])` rather
than by hand.

### `[llvm]`

| Key | Type | Default | Notes |
|---|---|---|---|
| `toolchain` | String | `"auto"` | `"auto"`, `"system"`, `"jll"`; anything else warns and falls back to `"auto"` |
| `version` | String | `""` (auto-detect) | Relevant to the C++ bucket; the C bucket uses the bundled JLL clang and Julia's own libLLVM |

### `[cache]`

| Key | Type | Default | Notes |
|---|---|---|---|
| `enabled` | Bool | `true` | Skip recompiling unchanged files |
| `directory` | String | `".replibuild_cache"` | The real cache path — this key, not `[paths] cache` |

### `[workflow]`

| Key | Type | Default |
|---|---|---|
| `stages` | Vector{String} | `["discover", "compile", "link", "binary", "wrap"]` |

Validated against the five known stage names (an unknown one is a config
validation error), but **not currently consulted** — the pipeline runs its fixed
order. Do not use it to skip a stage; call the individual verbs instead.

---

## 4. Interactions and precedence

- **`wrap.language` decides more than the generator.** It selects the toolchain
  (JLL `clang` + in-process libLLVM for `c`; system `clang++` + the external
  pipeline for `cpp`) and the default value of `[link] enable_lto`. Setting it
  wrong produces a build that works and a wrapper that is missing the whole
  reason you chose the other one.
- **`[wrap.tier1]` requires `[link] promote_statics` and the in-process C path.**
  `fallback = true`, ingest mode, or a C++ project all disable Tier 1 for the
  whole wrap, loudly.
- **`enable_lto` and `[wrap.tier1]` are independent knobs** and both can emit.
  Slicing did not replace the whole-module call-site path.
- **`[ingest]` wins over source-build keys.** A config with both gets warnings
  naming `[compile] source_files` and any `git`/`local` dependency, and proceeds
  in ingest mode.
- **`[binary] output_name` is overridden in ingest mode** when left empty, so
  `wrap()` finds the library where it expects it.
- **`[cache] directory` is the cache path; `[paths] cache` is inert.** Likewise
  `[paths] source` and `[paths] include`.

---

## 5. What fails loudly, what warns, what is ignored

Knowing which is which is the difference between a five-minute fix and an
afternoon.

**Hard error at config load**

- `[dependencies.<n>] commit` that is not a full 40-hex object name.
- `[wrap.varargs.<f>]` that is not a list, or whose entries are not lists — with a
  message showing the correct shape.
- `validate_config!`: empty `project.name`; non-existent `project.root`; a
  `[compile] source_files` entry that does not exist on disk; an
  `optimization_level` outside `0/1/2/3/s/z`; an unknown `[workflow] stages`
  entry; an invalid `binary.type` / `wrap.style` / `llvm.toolchain` /
  `types.strictness` value reaching validation.

**Hard error later in the pipeline**

- A resolved dependency HEAD that disagrees with the declared `commit`.
- A macro shim `#include` that resolved outside the project/dependency tree.
- A vararg overload naming a type outside the allowed set (§2.3).
- A wrapper whose `ccall` signatures would name an undeclared type — refused
  before it is written.
- A generated wrapper containing an unqualified `error("…")` call — refused,
  because a library that defines its own `error` symbol would rebind it and take
  out every failure path in the module.

**Warning, then proceeds**

- Missing `[project]` section or `project.name`.
- An unparseable `uuid` (a new one is generated).
- `[ingest]` without `library` → the section is dropped and you are in
  source-build mode.
- `[ingest]` alongside `source_files` or fetchable dependencies.
- Invalid `binary.type` / `wrap.style` / `llvm.toolchain` → default substituted.
- A `[wrap.cstring_owned]` value that is not a non-empty string → entry ignored.
- A struct whose layout cannot be modelled consistently → degrades to a
  correctly-sized opaque region (capped at ten warnings plus a summary).

**Accepted and ignored**

- `[discovery]` (whole section), `[workflow] stages`, `[wrap] style`,
  `[binary] strip_symbols`, `[paths] source` / `include` / `cache`.
- `extra_link_libs` outside `[ingest]`.
- Any key not in the schema — TOML parses it, the config drops it. **There is no
  unknown-key warning**, so a typo'd key name is silent. Check your spelling
  against §3 when a setting appears to have no effect.

---

## 6. Worked configurations

### A C library from git, with Tier 1

```toml
[project]
name    = "lua"
version = "5.4.7"
root    = "."

[dependencies.lua]
type    = "git"
url     = "https://github.com/lua/lua.git"
tag     = "v5.4.7"
commit  = "1ab3208a1fceb12fca8f24ba57d6e13c5bff15e3"
exclude = ["onelua.c", "lua.c", "luac.c", "ltests.c"]   # the interpreter, not the library

[compile]
flags = ["-O1", "-fPIC", "-DLUA_USE_LINUX"]

[link]
enable_lto         = false      # whole-module payload stays off
optimization_level = "2"

[binary]
type = "shared"

[wrap]
language     = "c"
shim_headers = ["lua.h", "lauxlib.h", "lualib.h"]

[wrap.tier1]
enable = true

[wrap.varargs]
lua_pushfstring = [["Cstring"], ["Cint"], ["Cdouble"], ["Cstring", "Cint"]]

[wrap.macros.lua_pop]
ret  = "void"
args = ["lua_State*", "int"]

[wrap.macros.LUA_REGISTRYINDEX]
ret = "int"

[types]
strictness              = "warn"
allow_unknown_structs   = true
allow_function_pointers = true

[cache]
enabled = true
```

### A C++ library with templates

```toml
[project]
name    = "fmt"
version = "12.1.0"
root    = "."

[dependencies.fmt]
type    = "git"
url     = "https://github.com/fmtlib/fmt.git"
tag     = "12.1.0"
commit  = "407c905e45ad75fc29bf0f9bb7c5c2fd3475976f"
exclude = ["test", "doc", "support", "benchmark"]

[compile]
flags = ["-O2", "-fPIC", "-std=c++20"]

[link]
enable_lto         = false      # C++ default; Tier 2/3
optimization_level = "2"

[wrap]
language = "cpp"

[types]
strictness       = "warn"
templates        = ["fmt::basic_memory_buffer<char>"]
template_headers = ["<fmt/format.h>"]

[cache]
enabled = true
```

### A pre-built C binary (ingest)

```toml
[project]
name = "foo"

[ingest]
library         = "/path/to/libfoo.so"      # must have been built with -g
headers         = ["/path/to/include"]
extra_link_libs = ["m", "pthread"]

[wrap]
language = "c"
```

---

## 7. Troubleshooting index

| What you see | Look at |
|---|---|
| A C++ template type is missing, or members are `Any` | `[types] templates` / `template_headers` — §2.1 |
| A struct from a header has **no fields** | `-fstandalone-debug` — §2.5 |
| A struct became an opaque blob with a layout warning | Unmodellable member; check the warning's named type |
| A macro-based API is absent from the wrapper | `[wrap.macros]` + `shim_headers` — §2.2 |
| Macro values are wrong but everything compiled | A shim header resolved to a system copy — §2.2 |
| `printf`-style function only takes its fixed arguments | `[wrap.varargs]` — §2.3 |
| `invalid type '…' in [wrap.varargs.f]` | You wrote C types; use Julia types — §2.3 |
| `must be a list of type lists` | You wrote `f = ["Cint"]`; write `f = [["Cint"]]` |
| Memory grows on every string-returning call | `[wrap.cstring_owned]` — §2.4 |
| Unrelated symbols from tests/examples in the wrapper | `exclude` — §2.6 |
| A TU fails to compile with a missing header | `[compile] include_dirs` — §2.7 |
| A TU fails on an undeclared identifier upstream's build defines | `[compile] flags` / `defines` — §2.9 |
| Tier 1 accepted slices but most functions demoted | Missing `-l` in `link_libraries` — §2.8 |
| "Tier 1 disabled for this wrap" | `fallback = true`, ingest mode, C++, or `promote_statics = false` |
| A setting seems to have no effect at all | Check it is not in the "accepted and ignored" list — §5 |
| `use()` serves a stale wrapper | The build cache keys on RepliBuild's own version too; see [the registry](internals.md#PackageRegistry) |
