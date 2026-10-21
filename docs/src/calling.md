# Call a wrapper

A generated wrapper is an ordinary Julia module sitting next to its `.so`.
Load it, call the C API, do not edit the file.

```julia
include("path/to/project/julia/MyProject.jl")   # module name = CamelCase of [project].name
using .MyProject

MyProject.some_function(...)     # everything exported; docs on each symbol
```

Once the project is registered (`register(toml)`), `RepliBuild.use("myproject")`
returns the loaded module, and `using MyProject` loads it from your Julia
environment, so you can skip `include`. [Registry](use.md).

Load a wrapper at top level. `include` and `use` define the module while your
code runs, so a function that loads one and then calls into it in the same call
fails with `method too new to be called from this world context`. If you must
do both in one function, call through `Base.@invokelatest M.f(x)`.

Never hand-edit `julia/MyProject.jl`. Regenerate with `wrap()`. Put Julia-shaped
API in your own module — [Ship a package](using-wrappers.md).

## Types

**Structs.** Named fields when Julia's layout reproduces every DWARF offset and
the total size. Otherwise an opaque byte blob with typed getters/setters. Padding
fields are explicit. Circular references are ordered so the module loads.
Structs are values, because they cross the ABI by value: `setproperty(x, :f, v)`
returns a copy with the field changed, and `x.f = v` is an error. `T()` returns
one with every byte zero.

**Enums.** `@enum` with the underlying type from the headers. A C enum whose
members are bit flags is an integer alias plus named constants instead, so
`A | B` works.

**Unions.** A `mutable struct` holding the union's bytes
(`data::NTuple{N,UInt8}`), with `U()` for a zeroed one and typed `get_…` /
`set_…!` accessors for each member.

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

## Arguments

A parameter declared as a struct, `f(Vec2 v)`, takes the struct value:
`f(Vec2(1.0, 2.0))`. A parameter declared as a pointer, `g(const Point *p)`,
takes a `Ref{Point}` or a `Ptr{Point}`, never the bare value. Before v3.0 an
overload accepted the value and passed the callee a pointer to a temporary
copy, so whatever it wrote or freed went to the copy; it was removed.

Most parameters accept any value Julia can convert: an `int` takes any
`Integer`, checked for range, and in a C wrapper a `double` takes any `Real`.
Some take exactly one Julia type — fixed-width integers such as `int32_t`
(`Int32`), and floating point in a C++ wrapper (`Cdouble`): write `2.0`, not
`2`. Each function's signature says which.

**Memory the library allocated** is freed by the library: through its own free
function, or `M.libc_free(ptr)` for memory it got from `malloc`. Never
`Libc.free` it. On Windows Julia's `Libc` is a different C runtime from the
library's, and freeing across them corrupts the heap. `Libc.free` is for
memory you allocated with `Libc.malloc`.

## C++

**Methods** are `Class_method(this, …)` and expect `this` to already point at
the **`Class` subobject**.

### Objects

A class whose constructor is in the library gets a constructor wrapper, named
`Class_Class(this, args…)`, and a destructor wrapper, `Class_destroy_Class(this)`.
They work on storage you provide, the way placement `new` and an explicit
destructor call do in C++:

```julia
p = Ptr{M.Rect}(Libc.malloc(M.struct_size(:Rect)))
try
    M.Rect_Rect(p, 1, 2.0, 3.0)      # Rect::Rect(int, double, double), in place
    M.Rect_area(p)                    # methods take the pointer
    M.Rect_destroy_Rect(p)            # Rect::~Rect(); does not free the storage
finally
    Libc.free(p)
end
```

`Rect()` is not a constructor. It is the zeroed Julia value every struct gets:
no C++ constructor has run, so a class with virtual methods has a null vtable
pointer, and calling a method on it crashes.

**`Managed*` types** own a heap pointer the library allocated with `new`. Their
finalizer runs the class's deleting destructor, which is C++ `delete`. Give
`ManagedThing(ptr)` only such a pointer — never storage from `Libc.malloc` or a
Julia array. When a function the wrapper calls through `ccall` (in C++, a
`noexcept` one) returns `Thing*` and there is a `ManagedThing`, the wrapper also
emits `<function>_safe(…) -> ManagedThing`. Keep a raw pointer when the library
retains ownership (a `b2Body*` owned by its world).

**Factory/destructor pairs** (`create_X` / `destroy_X`) also become a `mutable
struct` with a GC finalizer and method proxies.

**Upcasts.** Multiple inheritance emits `<Derived>_as_<Base>(p)` (a constant
offset). Virtual inheritance emits `<Derived>_as_<VBase>(p)` (offset read from
the object's vtable). To call a `Base` method on a `Derived`, pass
`Derived_as_Base(obj)` as the handle. That is the only rule.

MI base *members* are flattened onto the derived type. Virtual-base members are
not — upcast, then use the base's accessors.

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
| `M.struct_size(:Type)` | Byte size, from DWARF |
| `M.member_offset(:Type, :field)` | Byte offset of a member |
| `M.BUILD_GENERATOR` | RepliBuild version that wrote this wrapper |
| `M.LIBRARY_SHA` | SHA-256 of the library the wrapper was generated against |
| `M.dispatch_tier(f)` | Which calling path `f` **actually** uses right now |
| `M.DISPATCH_TIER` | What the generator **emitted** for each function |

`dispatch_tier` and `DISPATCH_TIER` exist only in a wrapper that mixes tiers, or
that has Tier 1 kernels. When every function takes the same path, the wrapper
names that tier in a comment instead, and the two names are `UndefVarError`.

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
