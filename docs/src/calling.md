# Call a wrapper

A generated wrapper is an ordinary Julia module sitting next to its `.so`.
Load it, call the C API, do not edit the file.

```julia
include("path/to/project/julia/MyProject.jl")   # module name = CamelCase of [project].name
using .MyProject

MyProject.some_function(...)     # everything exported; docs on each symbol
```

If the project is in your local registry (`discover` registers it),
`RepliBuild.use("myproject")` returns the loaded module and you can skip
`include`. [Registry](use.md).

Never hand-edit `julia/MyProject.jl`. Regenerate with `wrap()`. Put Julia-shaped
API in your own module — [Ship a package](using-wrappers.md).

## Types

**Structs.** Named fields when Julia's layout reproduces every DWARF offset and
the total size. Otherwise an opaque byte blob with typed getters/setters. Padding
fields are explicit. Circular references are ordered so the module loads.

**Enums.** `@enum` with the underlying type from the headers.

**Unions.** `NTuple{N,UInt8}` backing plus typed accessors for each arm.

**Bitfields.** Accessors that read and write only the field's byte span.

**Function pointers.** `@cfunction`-compatible types, signature in the docstring.

**Globals.** A value accessor plus a `_ptr` accessor. Unprovable types get the
pointer only.

**`char*` returns.** Default is `Union{String,Nothing}` — NULL is `nothing`,
anything else is copied into a Julia `String`. Declare malloc'd returns in
`[wrap.cstring_owned]` so the wrapper frees the C buffer through the library's
own deallocator after the copy. Every such function also has a raw `<name>_ptr`
variant: no copy, no NULL check, never freed.

```toml
[wrap.cstring_owned]
cJSON_Print = "cJSON_free"
```

**Pointers and references.** `T*` → `Ptr{T}`, `T**` → `Ptr{Ptr{T}}`, `T&` → `Ref{T}`.

Pass a pointer or `Ref` for by-value structs. Convenience overloads that took a
struct by value were removed — they were UB-adjacent.

## C++

**Methods** are `Class_method(this, …)` and expect `this` to already point at
the **`Class` subobject**.

**Upcasts.** Multiple inheritance emits `<Derived>_as_<Base>(p)` (a constant
offset). Virtual inheritance emits `<Derived>_as_<VBase>(p)` (offset read from
the object's vtable). To call a `Base` method on a `Derived`, pass
`Derived_as_Base(obj)` as the handle. That is the only rule.

MI base *members* are flattened onto the derived type. Virtual-base members are
not — upcast, then use the base's accessors.

**`Managed*` types** own a heap pointer; the finalizer runs the destructor. Use
the `_safe` factory (`make_thing_safe() -> ManagedThing`) when the object is
yours to free. Use a raw pointer when the library retains ownership (a `b2Body*`
owned by its world).

**Factory/destructor pairs** (`create_X` / `destroy_X`) also become a `mutable
struct` with a GC finalizer and method proxies.

**By-value handles as `this`.** Small handle structs returned by value (an
8-byte `xml_node`) need a pointer to call a method on them:

```julia
withptr(f, h) = (r = Ref(h); GC.@preserve r f(Base.unsafe_convert(Ptr{typeof(h)}, r)))
name = withptr(node -> pugi_xml_node_name(node), root)
```

**Exceptions.** A C++ exception that escapes the callee arrives as
`RepliBuild.JITManager.CxxException` with the original `what()` string. Catch it
like any Julia exception.

## Introspection

None of these are exported. Reach for them qualified:

| Name | What it answers |
|------|-----------------|
| `M.dispatch_tier(f)` | Which calling path `f` **actually** uses right now |
| `M.DISPATCH_TIER` | What the generator **emitted** for each function |
| `M.struct_size(:Type)` | Byte size, from DWARF |
| `M.member_offset(:Type, :field)` | Byte offset of a member |
| `M.BUILD_GENERATOR` | RepliBuild version that wrote this wrapper |

```julia
Lua.dispatch_tier(Lua.lua_gettop)                 # :tier1 | :tier2 | :tier3 | …
buf = zeros(UInt8, Box2d.struct_size(:b2BodyDef)) # in-place constructor space
off = Imgui.member_offset(:ImGuiIO, :MousePos)
```

`dispatch_tier` and `DISPATCH_TIER` disagree when a generated kernel demoted at
load (missing `slices/` directory, unresolvable symbol). Ask `dispatch_tier` in
tests; it is what will run. Do not call it during precompilation — it forces
code generation and will refuse with `:deferred`.

Sizes and offsets throw with named alternatives rather than returning zero.

## Layout facts in your own code

When you must poke a field the type system cannot name (header-inline
accessors, in-place construction), read the offset from the wrapper, never
hardcode it:

```julia
off = M.member_offset(:ImGuiIO, :MousePos)
```

Those numbers come from the same DWARF the wrapper was generated from, so a
layout change fails loudly on the next wrap instead of silently writing the
wrong byte.
