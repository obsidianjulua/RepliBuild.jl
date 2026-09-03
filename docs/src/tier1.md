# Tier 1 (experimental)

!!! note "Developer documentation"
    This page is conserved from the old user guide. Tier 1 is an experiment,
    off by default, and not a supported calling path. Production and Hub
    configs leave it off. The user-facing rule is: `[link] enable_lto = false`,
    do not set `[wrap.tier1] enable` unless you are working on this.

Tier 1 hands a C function's LLVM IR to `Base.llvmcall`, so Julia's JIT can
merge the C body into the calling Julia function. Two payloads can carry that
IR. They are configured independently.

## Per-function slices (`[wrap.tier1]`) — off by default

!!! warning "Not a supported tier"

    It ships, it works on the libraries it has been driven over, and it is
    **not** something to reach for in production. `enable` defaults to
    `false`, no Hub package should depend on it, and its tests are unwired
    from `devtests.jl` — run `test/test_static_promotion.jl`,
    `test/test_slicer.jl`, and `test/test_tier1_dispatch.jl` by hand.

    llvmcall is a passenger path and never the driver — any doubt resolves to
    `ccall`. The machinery is mostly refusals, demotions, and guards.

A **slice** is a declarations-only module: one function's body, and every
callee and global it reaches left as a bare `declare`, resolved at JIT time
against the `.so` the wrapper already `dlopen`'d `RTLD_GLOBAL`. Size tracks
the *function*, not the library — in Lua, `lua_gettop` is a 2.8 KB slice cut
from a 15.8 MB module.

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
```

The slice is read at generation time — the first call — and spliced into the
returned expression as a literal, which satisfies `Base.llvmcall`'s
statically-evaluable IR requirement while keeping module load free of slice
I/O. Inside a precompile worker the kernel splices the plain ccall body
instead (emitting llvmcall there deadlocks the JIT engine lock whenever a
`declare` binds a dlopened symbol); a runtime first call regenerates to the
slice. The `isfile` guard means a wrapper shipped without its `slices/`
directory demotes to ccall instead of failing.

A function is decided at *generation* time, not call time. Anything not
accepted emits the same `ccall` it always did. The module exports
`TIER1_FUNCTIONS::Set{String}` naming the functions that dispatch through a
slice. Slicing successfully is necessary but not sufficient — the ABI shape
gate still refuses Cstring and struct crossings.

Three guarantees make this safe where whole-module embedding was not:

1. **One copy of internal state.** Static promotion (`[link] promote_statics`)
   renames anything a slice might bind by `declare` but that cannot reach the
   `.so`'s dynamic symbol table — file-local statics, and external-linkage
   symbols marked `hidden` — to an exported `__rb_<lib>_<name>`, on the exact
   module that becomes both the `.so` and the slice source. Promoted names are
   filtered out of the wrappable API.
2. **Slices are refused, never guessed.** Variadic targets, `blockaddress`,
   alias/ifunc, and an unpromoted module come back as refusals; `:weak`,
   `:inline_asm` and `:module_asm` demote through the hazard gate. Every slice
   is verified before it is written.
3. **Symbol pre-flight.** An unresolved `declare` does not raise — ORC prints
   `Symbols not found: […]` and then blocks forever on the first call. Before
   any slice reaches disk, the generator `dlopen`s the `.so` and `dlsym`-checks
   every name the slice declares. A miss demotes that one function to `ccall`
   with a warning naming the symbol; a `.so` that will not `dlopen` disables
   Tier 1 for the whole wrap.

Each slice const is paired with an `include_dependency` on the same path, so
the `.ll` files are real precompilation dependencies. On Julia 1.11+ that
tracking is by **content**.

## Whole-module bitcode (`[link] enable_lto`) — leave off

The older payload emits `<name>_lto.bc` and embeds the **whole linked module**
at each call site. Two verified consequences at library scale:

1. **JIT scale limit** — embedding a whole library's IR per call can crash
   Julia's JIT. Small benchmark modules work; whole libraries do not reliably.
2. **Duplicated internal state** — file-local `static` definitions stay private
   to the embedded bitcode, so two calls into the same library can observe
   *different copies* of its internals (observed live on a JSON parser's
   error-reporting path).

Both are properties of the *whole-module payload*, not of llvmcall. Slices fix
the first by construction and the second through static promotion. Treat
`enable_lto` as an experimentation feature for small stateless kernels. The two
knobs are independent; Hub configs set `enable_lto = false`.

Pipeline placement and the rest of the calling paths: [Internals](internals.md).
