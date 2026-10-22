# Ship a package

The generated wrapper is enough for a script (`include` + `using .Module`).
This page is the next step: a real Julia package that precompiles, owns C++
lifetimes, and does not fork the generated file.

The working reference is
[`examples/BoxWorld`](https://github.com/obsidianjulua/RepliBuild-Hub/tree/main/examples/BoxWorld)
in RepliBuild-Hub — a physics sandbox on the Box2D wrapper.

How to *call* the generated API is [Call a wrapper](calling.md). This page is
layout, precompilation, and the ergonomic layer you write once.

## Two layers

- **ABI layer** — RepliBuild output (`<Name>.jl`, the `.so`,
  `compilation_metadata.json`, `thunk_manifest.json`). Vendor it unmodified.
  Regenerate with `wrap()`. Never edit it.
- **Ergonomic layer** — your package. Julia-idiomatic types, lifecycles,
  defaults, naming. Every C++-ism the wrapper cannot hide (ctor-only classes,
  header-inline defaults, abstract-class vtables) is wrapped **once**, here.

Library-specific niceties belong in your package, not in hand-edits to the
generated file. That is what keeps the wrapper regenerable.

## Layout: vendor the wrapper

```
MyPkg/
├── Project.toml            # [deps] must include RepliBuild
├── lib/                    # verbatim from julia/
│   ├── Box2d.jl
│   ├── libbox2d.so
│   ├── compilation_metadata.json
│   └── thunk_manifest.json
├── src/MyPkg.jl
└── test/runtests.jl
```

```julia
module MyPkg

using RepliBuild        # C++ wrappers dispatch thunks through RepliBuild

include(joinpath(@__DIR__, "..", "lib", "Box2d.jl"))
using .Box2d

# … your ergonomic layer …
end
```

The copy in `lib/` is self-contained:

1. The wrapper resolves its `.so` next to itself first (build-time absolute path
   is the fallback), so the vendored pair stays bound wherever the package lives.
2. Metadata is read from the wrapper's own directory at load.
3. The only Julia dependency is `RepliBuild`: every wrapper imports it for its
   build-identity check, and C++ wrappers dispatch through it. A C wrapper needs
   nothing more. A wrapper with Tier 2 functions — most C++ wrappers — also
   needs the JLCS dialect library loadable on the machine that runs it, in that
   machine's RepliBuild install. That means a system LLVM/MLIR 21+ there, and the
   dialect built against it ([Install](install.md#The-JLCS-dialect)).

Alternatives: `RepliBuild.use("box2d")` at runtime if that name is already in
your local registry (scripts, not precompiled packages), or
`RepliBuild.scaffold_package` for a skeleton that *builds* the library on the
consumer's machine instead of shipping a binary.

## Precompilation

The generated module is precompilation-safe:

- `dlopen`, JIT setup, and C stdout unbuffering live in the wrapper's
  `__init__`, which Julia runs at **load**, never during precompilation.
- Duplicate method signatures are deduplicated at generation time. Method
  overwriting is a hard error during package precompilation.

!!! warning "Wrappers generated before v3.0.2 do not precompile"
    Older files fail with `Method overwriting is not permitted`. Regenerate
    with a current `RepliBuild.wrap(toml)`. `use()`-managed packages rebuild
    automatically — the cache fingerprints the generator.

Your layer must follow the same two rules:

- **Never bake process addresses at top level.** A `const` from `dlsym`,
  `cglobal`, or a JIT lookup is evaluated during precompilation and baked stale.
  Use a `Ref` filled in your package's `__init__`.
- **Baking pure data is good.** Parsing `compilation_metadata.json` into `const`
  offset tables at top level is safe — the metadata ships inside the package
  and cannot drift from the vendored `.so`.

`dispatch_tier` is not a read-only lookup; do not call it at module scope. Ask
at runtime. See [Call a wrapper](calling.md#Introspection).

## Load time

Loading a C++ wrapper stands up a per-binary engine and compiles the thunks
that wrapper dispatches to. That costs on the order of a second for a mid-sized
library, once per process, in `__init__`.

Multiple wrapped libraries compose: one engine per `.so`. One library's
initialization failure disables the hard ABI path for *that* library only —
its `ccall` wrappers and every other library keep working. The failure does not
throw: the wrapper logs `Failed to initialize JIT for lib….so` with the reason,
`using MyPkg` succeeds, and calling one of that library's Tier 2 functions then
raises. To degrade features instead, check the engine in your `__init__`:
`RepliBuild.JITManager.GLOBAL_JIT.engines` holds one entry per library, with
`binary_path` and `init_error` (`nothing` when it worked).

With `[compile] aot_thunks = true`, `wrap` compiles the thunks the wrapper binds
into a companion `_thunks.so` instead of leaving them to be compiled at load:
faster startup, a second binary, and no JIT compilation in the consumer. It
does not remove the dialect library: the AOT
thunks still need `libJLCS` when they run. On Linux they import
`jlcs_catch_current_exception` from it, and on Windows every thunk call goes
through its `jlcs_guard_*` frame, which is what catches a C++ exception there.
An AOT wrapper whose thunks library will not open fails in `__init__`, naming
the state of `libJLCS`.

## C++-isms to wrap once

The wrapper gives you the raw material. Your layer hides each of these behind
ordinary Julia functions:

**Ctor-only classes** (no factory, e.g. `b2World`): allocate `struct_size`
bytes, call the constructor symbol on that storage, run the destructor at
teardown.

```julia
mem = zeros(UInt8, struct_size("b2World"))
ccall((:_ZN7b2WorldC2ERK6b2Vec2, Box2d.LIBRARY_PATH), Cvoid,
      (Ptr{Cvoid}, Ptr{b2Vec2}), pointer(mem), gravity_ref)
```

**Header-inline default constructors** (`b2BodyDef()`): inline functions have
no symbol. Write fields at their DWARF offsets — `member_offset`, never a
literal — so layout drift fails loudly.

**Header-inline accessors** (`GetPosition()`): same, read the member at its
metadata offset.

**Abstract-class instances the library virtual-calls** (a `b2CircleShape` it
`Clone()`s): plant the real compiler vtable — `dlsym` the `_ZTV…` symbol, skip
16 bytes (offset-to-top + RTTI) to the address point, store it at offset 0.
Resolve that address in `__init__` (it is a process address).

**Finalizers that call a C++ destructor:** warm the thunk once in `__init__` so
the first GC pass is not the first compile. Offer `destroy!(x)` for
deterministic teardown; keep the finalizer as the safety net. Make destruction
idempotent.

Every layout fact your layer uses should come from the wrapper's own
`struct_size` / `member_offset` (or `compilation_metadata.json`).

## Testing

Treat the wrapper boundary as RepliBuild's job. Test *your* claims:
create/use/destroy/idempotent-destroy, library behavior through your API, the
finalizer path (`GC.gc()` after dropping a live object), and that the C++
engine registered if you depend on it:

```julia
@test any(e -> occursin("libbox2d", e.binary_path) && e.init_error === nothing,
          RepliBuild.JITManager.GLOBAL_JIT.engines)
```

That holds for a JIT wrapper. An AOT wrapper creates no engine; for it, call
one of its Tier 2 functions in the test instead.

## Deployment

- The vendored `.so` is platform-specific. Multi-platform: per-platform
  artifacts, or `scaffold_package` so the consumer builds locally. The `.jl` is
  platform-independent apart from the binary it binds.
- Strip nothing. The DWARF in the `.so` is read at load.
- Linux and Windows, same as RepliBuild. The vendored binary is `.so` or
  `.dll`; the `.jl` is the same either way.
- On Windows, a C wrapper's `.dll` imports only the Windows C runtime and
  whatever `[link] link_libraries` adds. A C++ wrapper also needs
  `C:\msys64\clang64\bin` on the consumer's `PATH`: `libJLCS.dll` imports
  `libLLVM-<N>.dll` and `libc++.dll` from there, and a library that uses the
  standard library imports `libc++.dll` itself.
- A JIT wrapper also runs `llvm-dwarfdump` at load, so the consumer needs it on
  `PATH` too.
