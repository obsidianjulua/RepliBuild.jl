# User Guide

This guide covers the common workflows for using `RepliBuild.jl`.

## Basic Workflow

The standard workflow involves three steps: discovery, building, and wrapping.

### 1. Discovery

The `discover` function scans your directory for C/C++ files and generates a `replibuild.toml` configuration file.

```julia
RepliBuild.discover()               # current directory
RepliBuild.discover("path/to/project")
```

Re-running discovery with `force=true` regenerates the config but **preserves hand-curated keys** that cannot be derived from source: `[types].templates`/`template_headers` and `[wrap].varargs`/`macros`/`shim_headers`/`cstring_owned`. A `preserved: …` line in the output reports what carried over.

### 2. Building

Once configured, the `build` function compiles your code into a shared library.

```julia
RepliBuild.build()
```

This step performs:
- Compilation to LLVM IR (`.c` files use the bundled JLL `clang`; `.cpp` files use system `clang++`).
- Linking and optimization — for C projects this runs in-process on Julia's own libLLVM (version-matched by construction); C++ uses the external LLVM pipeline.
- Generation of the shared library.
- Extraction of DWARF metadata for wrapping.

### 3. Wrapping

Finally, generate the Julia wrapper module.

```julia
RepliBuild.wrap()
```

This creates a Julia module in the `julia/` directory. Load it with the standard pattern:

```julia
include("julia/MyProject.jl")
using .MyProject
```

## Automated Workflow

You can chain these steps together using the flags in `discover`:

```julia
# Discover, Build, and Wrap in one go
RepliBuild.discover(build=true, wrap=true)
```

## Package Registry and the Hub

RepliBuild includes a local package registry (`~/.replibuild/registry/`) that caches build artifacts so repeated loads are instant, and a community registry — [RepliBuild-Hub](https://github.com/obsidianjulua/RepliBuild-Hub) — holding ready-made `replibuild.toml` configs for popular C/C++ libraries.

```julia
# Register a project locally (discover() does this automatically)
RepliBuild.register("replibuild.toml")

# Load a registered package — builds on first call, cached thereafter.
# On a local-registry miss, the config is fetched from the Hub.
Lua = RepliBuild.use("lua")
Lua.luaL_newstate()

# Search the Hub by name, description, tags, or language
RepliBuild.search()          # list all Hub packages
RepliBuild.search("xml")

# List / remove local registrations
RepliBuild.list_registry()
RepliBuild.unregister("lua")

# Scaffold a distributable Julia package from a registered project
RepliBuild.scaffold_package("LuaWrapper")
```

The `use()` function handles the full lifecycle: resolve dependencies, build (or load from cache), wrap, and return a loaded Julia module. Cached builds live in `~/.replibuild/builds/<hash>/`; the cache key includes RepliBuild's own version and git revision, so upgrading RepliBuild rebuilds each package once with current codegen instead of serving stale wrappers. The `REPLIBUILD_HOME` environment variable overrides the registry location; `REPLIBUILD_HUB_URL` points `search()`/`use()` at a private Hub mirror.

## C vs C++ Projects

RepliBuild uses `wrap.language` as an extensible dispatch key to select the generator, compiler toolchain, and build defaults for a project:

```toml
[wrap]
language = "c"   # pure-C project: JLL clang, in-process libLLVM link/opt
language = "cpp" # C++ project: system clang++, external LLVM pipeline
```

`discover()` sets this automatically from the scanned source file extensions.

The toolchain requirements differ by bucket: **C projects need no external LLVM at all**, while **C++ projects need a system LLVM/MLIR 21+ install** for the JLCS dialect and Tier 2 thunks. An experimental Rust generator (`language = "rust"`) exists for `extern "C"` + `#[repr(C)]` surfaces.

For C projects `enable_lto` defaults to `true`, which emits the LTO bitcode artifact alongside the library — see [Zero-cost LTO dispatch](@ref "Zero-Cost LTO Dispatch") below for why production configurations disable it anyway, and use `[wrap.tier1]` instead.

## Ingest Mode (pre-built binaries) — experimental, C only

For **C** libraries with elaborate build systems (autotools, CMake with code generators) that RepliBuild's source pipeline cannot reproduce, build upstream yourself with `-g` and ingest the resulting binary. RepliBuild skips compilation — only DWARF extraction and wrapper generation run.

```julia
# Scaffold an ingest config
toml = RepliBuild.ingest("/path/to/libfoo.so",
                         headers=["/path/to/include"],
                         name="foo",
                         language=:c)

# Or run the whole pipeline immediately
toml = RepliBuild.ingest("/path/to/libfoo.so",
                         headers=["/path/to/include"],
                         build=true, wrap=true)
```

The generated `replibuild.toml` carries an `[ingest]` section whose presence flips RepliBuild into ingest mode:

```toml
[ingest]
library = "/path/to/libfoo.so"
headers = ["/path/to/include"]
extra_link_libs = ["m", "pthread"]   # optional — additional -l libs at load time
```

**Constraints to understand before reaching for ingest:**

- Ingest is a **fallback, not the flagship path**. Extraction quality depends on upstream's compiler and debug-info settings, which RepliBuild does not control.
- Ingested libraries dispatch through **Tier 3 (`ccall`) only** — no LTO bitcode, no Tier 2 thunks (both require the source build).
- **C++ API surfaces are unsupported.** Classes, methods, templates, and virtual dispatch need the MLIR dialect thunks only the source build generates; at best the `extern "C"` surface of a C++ binary works. Both entry points warn accordingly.

## Idiomatic Julia Class Wrappers

When your C++ library exposes a class through factory/destructor pairs, RepliBuild automatically generates an idiomatic `mutable struct` wrapper on top of the raw FFI bindings.

**Detection**: the wrapper generator scans for:
- Factory functions whose name matches `create_X`, `new_X`, `make_X`, `alloc_X`, `init_X`, or whose return type is `X*`.
- Destructor/deleter functions whose name matches `delete_X`, `destroy_X`, `free_X`, `dealloc_X`, or `X_destroy`.
- Instance methods associated with the same class via the DWARF `class` field.

**Generated output** for a `Circle` class:

```julia
# Raw bindings (always generated)
function create_circle(radius::Cdouble)::Ptr{Cvoid} ... end
function get_area(this::Ptr{Cvoid})::Cdouble ... end
function delete_shape(this::Ptr{Cvoid}) ... end

# Idiomatic wrapper (generated automatically on top)
mutable struct Circle
    handle::Ptr{Cvoid}

    function Circle(radius::Cdouble)
        handle = create_circle(radius)
        obj = new(handle)
        finalizer(obj) do o
            delete_shape(o.handle)
        end
        return obj
    end
end

# Method proxies via multiple dispatch
get_area(c::Circle) = get_area(c.handle)
```

For C++ classes without factory functions, the generator emits in-place constructor/destructor thunks and `Managed` handle types whose GC finalizers call the DWARF-resolved destructor directly. Classes with multiple inheritance additionally get `<Derived>_as_<Base>` upcast helpers (static offset adjustment), and virtual bases get dynamic `<Derived>_as_<VBase>` helpers that read the correct offset through the object's vtable at runtime — the same helper is correct for every dynamic type.

## `char*` Returns: Ownership Policy

C APIs use `char*` returns three different ways, and the wrapper handles each explicitly:

- **Default:** the wrapper returns `Union{String,Nothing}` — NULL becomes `nothing` (a NULL `char*` is a value in C APIs, not an exception), anything else is copied into a Julia `String`.
- **Owned returns:** for functions returning malloc'd buffers, declare the deallocator in the TOML and the wrapper frees the C buffer after copying:

  ```toml
  [wrap.cstring_owned]
  cJSON_Print = "cJSON_free"
  ```

- **Raw access:** every `Cstring`-returning function also gets an exported `<name>_ptr` variant that returns the pointer unchanged — no copy, no NULL check, never freed — for lifetime-sensitive callers.

Ownership of a returned `char*` is not recoverable from DWARF, so it is declared per-library in the TOML rather than guessed from function names.

## Replacing Manual Shims

When wrapping C/C++ libraries, developers often have to write manual C wrappers ("shims") for things that aren't native functions: templates, varargs, and preprocessor macros. RepliBuild handles all of these declaratively via `replibuild.toml`.

### Template Instantiation

C++ templates are only emitted into DWARF if the compiler actually instantiates them. To force instantiation for types you want to wrap, add them to `[types]`:

```toml
[types]
templates        = ["std::vector<int>", "std::vector<double>", "std::pair<int,float>"]
template_headers = ["<vector>", "<utility>"]
```

RepliBuild auto-generates a stub `.cpp` file that explicitly instantiates each requested type, ensuring it appears in the DWARF metadata and the generated Julia module.

### Varargs

Vararg wrappers lower as **true variadic calls** (the `@ccall` semicolon form), so the callee is declared variadic in LLVM IR and the x86-64 SysV register-count protocol (the `AL` setup that gates the callee's `va_start`) is emitted correctly — including for float varargs. Declare typed overloads in `[wrap.varargs]`:

```toml
[wrap.varargs]
printf = [
    ["const char*", "int"],
    ["const char*", "double", "int"]
]
```

RepliBuild generates a concrete Julia binding for each signature.

### Macro Expansion

C/C++ preprocessor macros don't exist in compiled binaries or DWARF metadata. To expose them to Julia, configure `[wrap.macros]`:

```toml
[wrap]
shim_headers = ["<stdio.h>"]

[wrap.macros.MY_MATH_MACRO]
ret = "int"
args = ["int", "float"]
```

RepliBuild generates a C/C++ source file that wraps `MY_MATH_MACRO` inside a typed function and compiles it alongside your project. Shims are emitted with default symbol visibility (they survive `-fvisibility=hidden` builds), and a header-collision guard verifies each shim `#include` resolves inside your project/dependency tree rather than to a system-installed copy of the same header at a different version.

## Zero-Cost LTO Dispatch

Tier 1 hands the C function's LLVM IR to `Base.llvmcall`, so Julia's JIT merges
the C body directly into the calling Julia function — full cross-language
inlining, no call instruction. There are two payloads that can carry that IR,
and they are configured independently.

### Per-function slices (`[wrap.tier1]`) — the supported path

A **slice** is a declarations-only module: one function's body, and every callee
and global it reaches left as a bare `declare`, resolved at JIT time against the
`.so` the wrapper already `dlopen`'d `RTLD_GLOBAL`. Size stops tracking library
size and starts tracking *function* size — in Lua, `lua_gettop` is a 2.8 KB
slice cut from a 15.8 MB module, and even `luaL_openlibs` comes out at 6 KB.

```toml
[link]
promote_statics = true    # default

[wrap.tier1]
enable = true
```

Each accepted function gets a `@generated` kernel that decides ccall vs
llvmcall once, at generation time, plus a public wrapper that routes through
it:

```julia
const _SLICE_lua_gettop = joinpath(@__DIR__, "slices", "lua_gettop.ll")
isfile(_SLICE_lua_gettop) && include_dependency(_SLICE_lua_gettop)

@generated function _TIER1_lua_gettop(__ptr_L::Ptr{lua_State})
    # llvmcall is opportunistic: any doubt resolves to the ccall body.
    if ccall(:jl_generating_output, Cint, ()) == 1 || !isfile(_SLICE_lua_gettop)
        return :(ccall((:lua_gettop, LIBRARY_PATH), Cint, (Ptr{lua_State},), __ptr_L))
    end
    ir = read(_SLICE_lua_gettop, String)
    return :(Base.llvmcall(($ir, "lua_gettop"), Cint, Tuple{Ptr{lua_State}}, __ptr_L))
end

function lua_gettop(L::Any)::Cint
    __cc_L = Base.cconvert(Ptr{lua_State}, L)
    __ptr_L = Base.unsafe_convert(Ptr{lua_State}, __cc_L)
    GC.@preserve __cc_L begin
        return _TIER1_lua_gettop(__ptr_L)
    end
end
```

The slice is read at generation time — the first call — and spliced into the
returned expression as a literal, which satisfies `Base.llvmcall`'s
statically-evaluable IR requirement while keeping module load free of slice
I/O. The generation-time `jl_generating_output` check exists because emitting
a sliced llvmcall inside a precompile worker deadlocks the JIT engine lock
whenever a `declare` binds a dlopened library's symbol (and an untaken
top-level branch reaches emission through inference alone): inside a
precompile worker the kernel splices the plain ccall body instead, and a
runtime first call regenerates to the slice. The `isfile` guard means a
wrapper shipped or relocated without its `slices/` directory demotes to a
plain ccall wrapper instead of failing.

There is no `if !isempty(...)` branch: a function is decided at *generation*
time, not call time. Anything not accepted emits the same `ccall` it always did,
and the module exports `TIER1_FUNCTIONS::Set{String}` naming the functions that
dispatch through a slice. Slicing a function successfully is necessary but not
sufficient — the ABI shape gate still refuses Cstring and struct crossings — so
that set can be smaller than the number of slices the wrap accepted.

Three guarantees make this safe where whole-module embedding was not:

1. **One copy of internal state.** Static promotion (`[link] promote_statics`)
   renames anything a slice might bind by `declare` but that cannot reach the
   `.so`'s dynamic symbol table — file-local statics, and external-linkage
   symbols marked `hidden` — to an exported `__rb_<lib>_<name>`, on the exact
   module that becomes both the `.so` and the slice source. Tier-1 and Tier-3
   calls provably see the same state. Promoted names are filtered out of the
   wrappable API.
2. **Slices are refused, never guessed.** Variadic targets, `blockaddress`,
   alias/ifunc, and an unpromoted module come back as refusals; `:weak`,
   `:inline_asm` and `:module_asm` demote through the hazard gate. Every slice
   is verified before it is written.
3. **Symbol pre-flight.** An unresolved `declare` does not raise — ORC prints
   `Symbols not found: [...]` and then blocks forever on the first call. So
   before any slice reaches disk, the generator `dlopen`s the `.so` and
   `dlsym`-checks every name the slice declares, which is the exact lookup ORC
   will perform. A miss demotes that one function to `ccall` with a warning
   naming the symbol; a `.so` that will not `dlopen` disables Tier 1 for the
   whole wrap.

Each slice const is paired with an `include_dependency` on the same path, so the
`.ll` files are real precompilation dependencies: a wrapper vendored into a
package recompiles when its slices change. On Julia 1.11+ that tracking is by
**content**, so restoring a slice with a different mtime but identical bytes
correctly does not force a rebuild.

### Whole-module bitcode (`[link] enable_lto`) — scale-limited

The older payload emits `<name>_lto.bc` and embeds the **whole linked module**
at each call site, behind a runtime branch:

```julia
if !isempty(LTO_IR)
    return Base.llvmcall((LTO_IR, "vector_dot"), Cdouble, Tuple{Ptr{Cvoid}, Ptr{Cvoid}, Cint}, a, b, n)
else
    return ccall((:vector_dot, LIBRARY_PATH), Cdouble, (Ptr{Cvoid}, Ptr{Cvoid}, Cint), a, b, n)
end
```

!!! warning "Why production configs set `enable_lto = false`"
    Whole-module embedding has two verified consequences at real-library scale:

    1. **JIT scale limit** — embedding a whole library's IR (hundreds of functions) per call can crash Julia's JIT. Small benchmark modules work; whole libraries do not reliably.
    2. **Duplicated internal state** — file-local `static` definitions stay private to the embedded bitcode, so Tier-1 and Tier-3 calls can observe *different copies* of the library's internal state (observed live on a JSON parser's error-reporting path).

    Both are properties of the *whole-module payload*, not of Tier 1. Slices fix
    the first by construction and the second through static promotion. Treat
    `enable_lto` as an experimentation feature for small stateless kernels, and
    use `[wrap.tier1]` for real libraries. The two are independent knobs.

## AOT Thunks

Tier 2 functions normally compile their MLIR thunks through the in-process JIT at module load. Setting `aot_thunks = true` pre-compiles them at build time instead:

```toml
[compile]
aot_thunks = true
```

During `RepliBuild.build()`, the JLCS dialect thunks are generated and compiled into a companion `<name>_thunks.so` placed alongside the main library. The generated wrapper emits static `ccall` bindings against `THUNKS_LIBRARY_PATH` — no JIT startup, no MLIR runtime dependency after build. Requires `src/mlir/build/libJLCS.so` (`cd src/mlir && ./build.sh`).

## Running Tests

The CI suite (no C++ toolchain required):

```bash
julia --project=. test/runtests.jl
```

The full developer integration suite (requires the C++ bucket: system LLVM/MLIR + Clang + CMake):

```bash
julia --project=. test/devtests.jl
```

Sections cover the real-world fixtures (Lua, SQLite, cJSON, Duktape, pugixml), the dialect suites (`test_mlir_templates.jl`, `test_jlcs_invariants.jl`, `test_jlcs_producers.jl`, `test_struct_abi.jl`), inheritance verification (`test/mi_test/`, `test/vi_test/`), and the ABI trace fixtures. External sources are downloaded on first run.
