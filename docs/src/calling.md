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

A name that would shadow one of Base's or Core's is defined but not exported: a
C union called `Number` is `MyProject.Number`, never plain `Number`. `wrap`
prints those names, and the wrapper lists them in a "Withheld from `export`"
comment.

## Types

**Structs.** Named fields when Julia's layout reproduces every DWARF offset and
the total size. Otherwise an opaque byte blob with typed getters/setters. Padding
fields are explicit. Circular references are ordered so the module loads.
Structs are values, because they cross the ABI by value: `x.f = v` is an error,
and `T()` returns one with every byte zero. To change a field, make a new value:
`Vec2(9.0, v.y)`, or `setproperty(x, :f, v)` / `setproperties(x; f = v)` where
the wrapper defines them, which return a copy. A C++ wrapper defines them for
every struct, a C wrapper for its byte-blob structs.

**Enums.** `@enum` with the underlying type from the headers. A C enum whose
members are bit flags is an integer alias plus named constants instead, so
`A | B` works.

**Unions.** A `mutable struct` holding the union's bytes
(`data::NTuple{N,UInt8}`), with `U()` for a zeroed one and, for each member, a
getter and a setter that writes into the union: `get_U_m(u)` / `set_U_m!(u, v)`
in a C wrapper, `get_m(u)` / `set_m!(u, v)` in a C++ wrapper.

**Bitfields.** A getter and a setter for each field, which read and write only
its bits: `get_f(s)`, and in a C wrapper `set_f(s, v)`, which returns a new
struct; in a C++ wrapper the record is a `mutable struct` and `set_f!(s, v)`
writes into it.

**Macros** declared in `[wrap.macros]` are functions: `lua_pop(L, 1)`. A value
macro takes no arguments: `LUA_REGISTRYINDEX()`.

**Function pointers.** `Ptr{Cvoid}`, which a `@cfunction` converts to. The
docstring shows the `@cfunction` call for the callback's signature: the header's
when the parameter is declared with a typedef the headers spell
(`typedef int (*progress_cb)(int done, void *user);`), and otherwise the one DWARF
records, with each type spelled as the wrapper spells its own parameters. A
variadic callback, or a pointer to one, has no example; its docstring shows the
DWARF signature instead, such as `function_ptr(int; const char*, ...)*`.

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

A `<name>_ptr` result is a `Cstring`. To free it yourself, pass the library's
free function a `Ptr{Cvoid}` (`M.cJSON_free(Ptr{Cvoid}(p))`; the `Cstring` itself
is a `MethodError`), or give a buffer the library got from `malloc` to
`M.libc_free(p)`.

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
`2`. Each function's signature says which. C++ overloads that differ only in
an integer parameter's width keep that width on the parameter they differ
on, so they stay distinct methods: `to_chars(…, int, int)` takes an `Int32`
value and `to_chars(…, long long, int)` an `Int64`, while the `base` they
agree on takes any `Integer`. Two C types that are one width on the host
(`int` and `long` on Windows, `long` and `long long` on Linux) are one
method, taking any `Integer`.

**Variadic functions** have one function per overload declared in
`[wrap.varargs]`, named after its variadic types: `lua_pushfstring_Cstring(L,
fmt, s)`. The plain name takes the fixed parameters only
([Edit the TOML](config.md#2.3-Variadic-functions)).

**Memory the library allocated** is freed by the library: through its own free
function, or `M.libc_free(ptr)` for memory it got from `malloc`. Never
`Libc.free` it. On Windows Julia's `Libc` is a different C runtime from the
library's, and freeing across them corrupts the heap. `Libc.free` is for
memory you allocated with `Libc.malloc`.

## C++

**Methods** are `Class_method(this, …)` and expect `this` to already point at
the **`Class` subobject**. A namespace is part of the name:
`tinyxml2::XMLComment::ParseDeep` is `tinyxml2_XMLComment_ParseDeep`, and a
template's arguments are too (`hol::Box<double>::get` is `hol_Box_double_get`).

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
pointer, and calling a method on it crashes. A class a factory makes is the
exception (below).

**`Managed*` types** own a heap pointer the library allocated. Their finalizer
calls the library's deleter function for the class if it has one (a free or
static function taking `Thing*` whose name says delete, destroy or free), and
otherwise the class's deleting destructor, which is C++ `delete`. A class with
neither, which is any class without a virtual destructor and without a deleter
function, gets no `ManagedThing`. Give `ManagedThing(ptr)` only a pointer that
deleter may free: never storage from `Libc.malloc` or a Julia array. The
deleting destructor is called directly, so the object must be a `Thing`, not an
object of a subclass. When a function the wrapper calls through `ccall` (in
C++, a `noexcept` one) returns `Thing*` and there is a `ManagedThing`, the
wrapper also emits `<function>_safe(…) -> ManagedThing`. Keep a raw pointer when
the library retains ownership (a `b2Body*` owned by its world).

**Factories.** A free function named `create_`, `new_`, `make_`, `alloc_`,
`init_` or `build_` that returns a pointer to a class makes that class a
`mutable struct` holding the pointer, in place of its layout struct:
`M.Rect(2.0, 3.0)` calls `make_rect`, and each method has a proxy
(`M.area(r)`). The class is the one the returned pointer names, or one the
function's name names that derives from it (`create_circle()` returning
`Shape*`). Its GC finalizer is the library's deleter function for the class or
for the returned type, or else the class's deleting destructor when nothing in
the library derives from the class. A class with neither keeps its raw
bindings.

**Upcasts.** When a base subobject does not start where the object does, the
wrapper emits `<Derived>_as_<Base>(p)`: a constant offset for a non-virtual base
(multiple inheritance), an offset read from the object's vtable for a virtual
base. To call a `Base` method on a `Derived`, pass `Derived_as_Base(obj)` as the
handle. Where there is no such helper, as in plain single inheritance, the base
starts at the object: pass the derived pointer itself (`Shape_id(p)` for a
`p::Ptr{Rect}`).

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
| `M.struct_size(:Type)` | Byte size, from DWARF (a `String` works too) |
| `M.member_offset(:Type, :field)` | Byte offset of a member |
| `M.STRUCT_SIZES`, `M.STRUCT_OFFSETS` | Every size and member offset those two read, as `Dict`s |
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
