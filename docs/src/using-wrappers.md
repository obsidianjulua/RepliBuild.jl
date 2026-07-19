# Using a Wrapper in Your Package

The generated wrapper is a complete Julia module, and the README's
`include` + `using .Module` pattern is enough for scripts. This page covers
the next step: **building a real Julia package on top of a generated
wrapper** — one that precompiles, drives the Tier-2 MLIR JIT, manages C++
object lifetimes, and composes with other wrapped libraries.

Everything here is demonstrated by a working reference package:
[`examples/BoxWorld`](https://github.com/obsidianjulua/RepliBuild-Hub/tree/main/examples/BoxWorld)
in the RepliBuild-Hub repository — a physics sandbox driving the Box2D (C++)
wrapper through Tier-2 thunks from a precompiled package, with a full test
suite.

## The two-layer discipline

A package built on a wrapper has two layers with a hard boundary:

- **The ABI layer** — the RepliBuild output (`<Name>.jl`, the `.so`,
  `compilation_metadata.json`, `thunk_manifest.json`), vendored into your
  package unmodified. Regenerate it with `wrap()`; never edit it.
- **The ergonomic layer** — your package's own code: Julia-idiomatic types,
  lifecycles, defaults, and naming. Every C++-ism the wrapper cannot hide
  (ctor-only classes, header-inline defaults, abstract-class vtables) is
  encapsulated here **once**, behind ordinary Julia functions.

Keeping library-specific ergonomics in your package — not in hand-edits to
the generated file — is what keeps the wrapper regenerable.

## Package layout: vendor the wrapper

```
MyPkg/
├── Project.toml            # [deps] must include RepliBuild
├── lib/                    # vendored wrapper output, verbatim from julia/
│   ├── Box2d.jl
│   ├── libbox2d.so
│   ├── compilation_metadata.json
│   └── thunk_manifest.json
├── src/MyPkg.jl
└── test/runtests.jl
```

```julia
module MyPkg

using RepliBuild        # the wrapper dispatches Tier 2 through RepliBuild.JITManager

include(joinpath(@__DIR__, "..", "lib", "Box2d.jl"))
using .Box2d

# ... your ergonomic layer ...
end
```

Three facts make the vendored copy self-contained:

1. **Sibling-first library resolution.** The wrapper resolves its `.so` next
   to its own file first, with the build-time absolute path as fallback — so
   the copy in `lib/` binds to `lib/libbox2d.so` wherever the package lives.
2. **Metadata rides along.** `compilation_metadata.json` and
   `thunk_manifest.json` are read from the wrapper's own directory at load
   time (JIT initialization) and are available to *your* code for layout
   facts (see below).
3. **The dependency is `RepliBuild`, nothing else.** Tier-2 dispatch calls
   `RepliBuild.JITManager.invoke`; Tier-3 wrappers are plain `ccall`. There
   is no MLIR/LLVM Julia dependency in the consuming package — `libJLCS.so`
   is found through the installed RepliBuild.

Alternatives to vendoring: `RepliBuild.use("box2d")` loads a registry/Hub
package at runtime (good for scripts and exploration, not for precompiled
packages), and `RepliBuild.scaffold_package` generates a package skeleton
that *builds* the library at the consumer's machine instead of shipping a
binary.

## Precompilation: what the wrapper guarantees, what you must uphold

A package that `include`s a generated wrapper precompiles normally. The
generated module is precompilation-safe by construction:

- All process-level state — `dlopen`, JIT engine creation, C stdout
  unbuffering — lives in the wrapper's `__init__`, which Julia runs at **load
  time**, never during precompilation.
- Every emitted method has a unique dispatch signature. Distinct C++ symbols
  that collapse to the same Julia signature (destructor D1/D2 pairs,
  overloads whose parameter types all map to `::Any`) are deduplicated at
  generation time, keeping the last definition — method overwriting is a
  hard **error** during package precompilation, not the warning you see from
  a script `include`.

!!! warning "Wrappers generated before v3.0.2 do not precompile"
    Older generated wrappers contain those duplicate definitions and will
    fail package precompilation with `Method overwriting is not permitted`.
    Regenerate with a current RepliBuild (`RepliBuild.wrap(toml)`) — the
    registry cache fingerprints the generator, so `use()`-managed packages
    rebuild automatically.

Your ergonomic layer must follow the same two rules:

- **Never bake process addresses at top level.** A `const` computed from
  `dlsym`, `cglobal`, or a JIT lookup is evaluated during precompilation and
  baked stale into the image. Use a `Ref` filled in your package's
  `__init__` (BoxWorld does this for the shape vtable address points).
- **Baking pure data is good.** Parsing `compilation_metadata.json` into
  `const` offset tables at top level is safe and fast — the metadata ships
  inside your package and cannot drift from the vendored `.so`.

## The JIT lifecycle from a package's point of view

When your package loads, the wrapper's `__init__` calls
`RepliBuild.JITManager.initialize_global_jit(LIBRARY_PATH)`, which:

1. registers **one JIT engine for that binary** (parsing its DWARF vtables,
   reading its thunk manifest, generating + lowering + compiling exactly the
   thunks the wrapper dispatches to);
2. leaves per-symbol pointers to be resolved lazily — the first call to each
   thunk takes a locked slow path, every later call is a lock-free
   dictionary read.

**Multiple wrapped libraries compose.** Engines are per-binary: loading a
second wrapper registers a second engine, and thunk lookups search all of
them (thunk names derive from mangled symbols, so they are unique across
libraries). One library's initialization failure disables Tier 2 for that
library only — its `ccall` wrappers and every other library keep working.

**Degradation is catchable, not fatal.** If JIT initialization fails
(missing `libJLCS.so`, an untranslatable type refused by the pre-flight
guard), the wrapper module still loads; Tier-2 calls raise a descriptive
error naming the root cause. Design your package so Tier-3-only operation
degrades features rather than crashing imports.

**Load-time cost is real.** Engine initialization parses DWARF and compiles
thunks — around a second for a mid-sized C++ library. It happens once per
process, in `__init__`, not per call.

## Calling conventions your layer will use

The generated docstrings state each function's argument types. The patterns
that come up when driving a C++ library:

**By-value handle objects as `this`.** Small C++ handle structs (an 8-byte
`xml_node`, a `b2Vec2`) are returned by value. To call a method *on* such a
value, pass a pointer to it:

```julia
withptr(f, h) = (r = Ref(h); GC.@preserve r f(Base.unsafe_convert(Ptr{typeof(h)}, r)))
name = withptr(node -> pugi_xml_node_name(node), root)
```

**`Managed*` types own heap pointers.** For pointer-returning
factory/creator functions, the wrapper emits `Managed<Class>` types whose
finalizer calls the C++ destructor. Use the `_safe` variants
(`make_thing_safe() -> ManagedThing`) when the returned object is yours to
free; use raw pointers when the library retains ownership (a `b2Body*` is
owned by its world — never wrap it in a finalizer).

**C++ exceptions arrive as `CxxException`.** May-throw functions route
through landing-pad thunks; a C++ exception escaping the callee surfaces as
`RepliBuild.JITManager.CxxException` carrying the original `what()` string.
Catch it like any Julia exception.

**Finalizers and the JIT.** A finalizer that calls a Tier-2 destructor thunk
should be **warmed** at `__init__` (one `_lookup_cached` call), so the
finalizer always takes the lock-free path — first-call compilation inside a
GC finalizer means taking a lock at GC time. And make destruction
idempotent + explicit-first: offer `destroy!(x)` for deterministic teardown
with the finalizer as the safety net (see `BoxWorld.World`).

## The C++-isms your layer will encapsulate

Real C++ libraries have constructs no binding generator can fully hide. The
wrapper gives you the raw material; your layer wraps each **once**:

- **Ctor-only classes** (no factory function, e.g. `b2World`): allocate
  `byte_size` bytes (from the metadata), call the constructor symbol on that
  storage, run the Tier-2 destructor thunk at teardown:

  ```julia
  mem = zeros(UInt8, struct_size("b2World"))
  ccall((:_ZN7b2WorldC2ERK6b2Vec2, Box2d.LIBRARY_PATH), Cvoid,
        (Ptr{Cvoid}, Ptr{b2Vec2}), pointer(mem), gravity_ref)
  ```

- **Header-inline default constructors** (`b2BodyDef()`): inline functions
  never exist as symbols, so their defaults are unreachable through any
  binding. Replicate them by writing fields at their DWARF offsets — read
  the offsets from `compilation_metadata.json`, never hardcode them, so
  layout drift fails loudly.

- **Header-inline accessors** (`GetPosition()`): same story — read the
  member directly at its metadata offset.

- **Abstract-class instances that C++ code virtual-calls** (a
  `b2CircleShape` the library `Clone()`s): plant the real compiler vtable —
  `dlsym` the `_ZTV…` symbol, `+16` past offset-to-top and RTTI to the
  address point, store it at offset 0. Resolve the vtable address in
  `__init__` (it is a process address).

Each of these is a few lines, written once, tested by your package's suite.
The metadata-offset habit is the load-bearing one: every layout fact your
layer uses should come from the wrapper's own `compilation_metadata.json`.

## Testing your package

Treat the wrapper boundary as tested (RepliBuild's suites own it) and test
**your** claims: lifecycles (create/use/destroy/idempotent-destroy),
behavior of the underlying library through your API (BoxWorld asserts
free-fall against the semi-implicit Euler closed form), the finalizer path
(drop a live object, `GC.gc()`, assert no crash), and — worth pinning in any
consumer — that the JIT engine registered:

```julia
@test any(e -> occursin("libbox2d", e.binary_path) && e.init_error === nothing,
          RepliBuild.JITManager.GLOBAL_JIT.engines)
```

## Deployment notes

- The vendored `.so` is platform-specific. For multi-platform distribution,
  build per-platform artifacts or scaffold with `deps/`-driven builds
  (`scaffold_package`); the wrapper file itself is platform-independent
  apart from the binary it binds.
- `aot_thunks = true` at wrap time replaces the load-time JIT with a
  pre-compiled `_thunks.so` — no MLIR at runtime, faster load, at the cost
  of shipping a second binary.
- Strip nothing: the DWARF in the `.so` is what the JIT initialization
  parses at load time.
