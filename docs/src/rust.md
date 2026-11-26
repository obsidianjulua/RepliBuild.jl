# Rust crates

A Rust crate is wrapped **as it is**. It needs no `extern "C"`, no
`#[repr(C)]` and no other change: cargo builds the crate, and RepliBuild
writes a Julia module for its public functions.

The recipe for a crate is short, because almost nothing has to be stated: the
crate's name, and the few things only you can decide (how optimized, which
features, which types a generic function is wanted for).

!!! note "Where this runs"
    Rust crates are wrapped on **Linux**, x86-64 and AArch64. Windows has not
    been run yet.

## What you need

`cargo` and `rustc`, from [rustup](https://rustup.rs) or your distribution's
`rust` package. RepliBuild brings no Rust of its own, and a crate needs
nothing of the C or C++ setup.

```julia
using RepliBuild
RepliBuild.check_environment()
```

The report has a section for Rust, and the summary says:

| Summary line | Means |
|:-------------|:------|
| `Rust crates — cargo and rustc found` | You can wrap Rust crates. |

`REPLIBUILD_TOOL_CARGO` and `REPLIBUILD_TOOL_RUSTC` name particular programs.
A `rust-toolchain.toml` in the crate is respected: cargo is started from the
crate's folder.

## The recipe

In the crate's folder, the one that holds `Cargo.toml`:

```julia
RepliBuild.discover("path/to/geo")
```

This asks cargo what the crate is and writes `replibuild.toml` beside
`Cargo.toml`. Nothing is compiled yet, and nothing else is written into the
crate.

```toml
[project]
name = "geo"                 # the module: julia/Geo.jl

[rust]
manifest = "Cargo.toml"
opt_level = 0
```

That file is enough to build. What you may want to add, all under `[rust]`:

| Key | When to write it |
|:----|:-----------------|
| `opt_level` | `0` to `3`. `0` compiles fastest and is what `discover` writes. `3` is what `cargo build --release` gives: use it for a library you will call in earnest. |
| `features` | The crate has Cargo features you want on: `features = ["simd"]`. `default_features = false` turns the default ones off. |
| `instantiate` | The crate has **generic** functions you want to call. See [Generic functions](#Generic-functions). |
| `package` | `Cargo.toml` is a workspace with several library crates: name the one to wrap. `discover` writes it for you when the workspace has only one, and lists them when it has more. |
| `dependency` | The crate is fetched, not in the project folder. See below. |

Every key is in [Every key](config.md#[rust]).

`discover` keeps a `[rust]` recipe that is already there, with `force = true`
too: there is nothing to scan, since cargo knows the crate's files.

### A crate from a git repository

A crate can come from a `[dependencies]` entry, pinned like any other source
([Where the source comes from](dependencies.md)). This recipe wraps the
`semver` crate from its repository:

```toml
[project]
name = "rustsemver"

[dependencies.semver]
type   = "git"
url    = "https://github.com/dtolnay/semver.git"
tag    = "1.0.28"
commit = "7625c7aa3f0e8ba21e099d1765bcebcb72aa8816"

[rust]
dependency = "semver"        # the crate is that dependency
opt_level  = 3
```

`manifest` is then a path inside the fetched source (`"Cargo.toml"` by
default). Nothing is written into the checkout.

## Build and wrap

```julia
RepliBuild.build("path/to/geo/replibuild.toml")    # cargo builds the crate
RepliBuild.wrap("path/to/geo/replibuild.toml")     # writes julia/Geo.jl
```

or, in one call from nothing:

```julia
RepliBuild.discover("path/to/geo", build = true, wrap = true)
Geo = RepliBuild.use("geo")
```

The output folder (`julia/` by default) then holds:

| File | What it is |
|:-----|:-----------|
| `Geo.jl` | The Julia module. |
| `libgeo.so` | The crate, as a shared library. |
| `libgeo_thunks.so` | The calls a plain Julia call cannot make, compiled when the module was written. Keep it beside the other two. |
| `compilation_metadata.json` | What the build found. The module is generated from it. |

`wrap` reports how many functions it wrapped and names each one it did not,
with the reason.

If `wrap` says `thunks: not compiled ahead of time`, the module still works:
it compiles those calls the first time one is used. The reason is printed
after the message; on a machine with no LLVM it is a missing `llc`, which
the system's LLVM provides ([Set up](install.md#What-you-need)). That needs RepliBuild's
support library (`libJLCS` in the check's report) on the machine that runs
the module, where the compiled file needs nothing.

## Calling it

The examples use this crate:

```rust
pub struct Point { pub x: f64, pub y: f64 }

impl Point {
    pub fn new(x: f64, y: f64) -> Point { Point { x, y } }
    pub fn norm(&self) -> f64 { (self.x * self.x + self.y * self.y).sqrt() }
}

pub fn midpoint(a: &Point, b: &Point) -> Point { /* ... */ }
pub fn greet(name: &str) -> String { format!("hello, {name}") }
pub fn squares(n: u32) -> Vec<f64> { (0..n).map(|i| (i * i) as f64).collect() }
pub fn first_even(xs: &[i64]) -> Option<i64> { /* ... */ }
pub fn parse_age(text: &str) -> Result<u32, String> { /* ... */ }
pub fn checked_div(a: i32, b: i32) -> i32 { a / b }
pub fn apply(f: fn(f64) -> f64, x: f64) -> f64 { f(x) }
```

A function keeps its name. A method is `Type_method`, and takes `self` as its
first argument:

```julia
julia> Geo.greet("Ada")
"hello, Ada"

julia> Geo.squares(4)
4-element Vector{Float64}:
 0.0
 1.0
 4.0
 9.0

julia> p = Geo.Point_new(3.0, 4.0);

julia> (p.x, p.y)
(3.0, 4.0)

julia> Geo.Point_norm(p)
5.0

julia> m = Geo.midpoint(p, Geo.Point_new(1.0, 0.0)); (m.x, m.y)
(2.0, 2.0)
```

What you pass and what you get back:

| Rust | Julia |
|:-----|:------|
| integers, floats, `bool`, `char` | the same kind of number, `Bool`, `Char`. An integer argument is converted, and one that does not fit is an error. |
| `&str`, `String` | a `String`, both ways |
| `&[T]`, `Vec<T>` | a `Vector`, both ways |
| `[T; N]` | a tuple out; a tuple or a vector of exactly `N` in |
| a tuple | a tuple |
| a `struct` | a Julia value with the same field names (`p.x`) |
| an `enum` without data | a Julia enum; the variant `Level::Low` is `Geo.Level_Low` |
| an `enum` with data | a value; `RepliBuild.RustRuntime.variant(x)` names its variant, and the variant's fields are properties |
| `Option<T>` | the value, or `nothing` |
| `Result<T, E>` | the value, or a thrown error (below) |
| `&T` | the value, or a `Ref` of it |
| `&mut T` | a `Ref` of the value: Rust changes it in place, and `r[]` reads it back |
| `fn(A) -> B` | any Julia function |
| `&dyn Trait` | a value of any type that implements the trait |

A `&mut String` or a `&mut Vec<f64>` is changed in place too, so it needs an
object Rust can grow: `s = Geo.RustString("hey")` or `v = Geo.Vec_f64([1.0])`,
passed as it is and read back with `String(s)` or `collect(v)`.

```julia
julia> Geo.first_even([3, 5, 8, 9])
8

julia> Geo.first_even([1, 3]) === nothing
true

julia> Geo.apply(x -> x^2 + 1, 3.0)
10.0
```

### Errors and panics

A function that returns `Result` gives the value of an `Ok` and throws for an
`Err`. The error's own value is in the exception:

```julia
julia> Geo.parse_age(" 42 ")
0x0000002a

julia> Geo.parse_age("old")
ERROR: RustError: not an age: invalid digit found in string
```

Catch it as `RepliBuild.RustRuntime.RustError`; `e.value` is what the `Err`
held.

A Rust panic does not end Julia. It arrives as a
`RepliBuild.RustRuntime.RustPanic` carrying the panic's message and where it
happened:

```julia
julia> Geo.checked_div(1, 0)
ERROR: RustPanic: panicked at /path/to/geo/src/lib.rs:36:5:
attempt to divide by zero
```

### Values Rust owns

Some values hold memory that only Rust can free: a `Box`, or a struct that
keeps a `String` or a `Vec` inside it. Such a value is a Julia object, and
Rust frees it when Julia collects the object. You do nothing.

```rust
pub struct Inventory { items: Vec<String> }

impl Inventory {
    pub fn new() -> Inventory { /* ... */ }
    pub fn add(&mut self, item: &str) { /* ... */ }
    pub fn len(&self) -> usize { /* ... */ }
    pub fn names(&self) -> Vec<String> { /* ... */ }
}
```

```julia
julia> inv = Geo.Inventory_new();

julia> Geo.Inventory_add(inv, "rope"); Geo.Inventory_add(inv, "lamp");

julia> Geo.Inventory_names(inv)
2-element Vector{String}:
 "rope"
 "lamp"
```

Two things to know:

- **Passing such an object to a function that takes it by value moves it**,
  as in Rust. The object is empty afterwards, and using it again is an error.
  `RepliBuild.RustRuntime.isowned(x)` says whether it still holds its value.
- `RepliBuild.RustRuntime.drop!(x)` frees it now instead of when Julia gets
  to it.

A `String` and a `Vec` of plain values are not objects like that: they are
copied into a Julia `String` or `Vector`, and Rust's copy is freed at once.
The module's `OWNED` lists the types that are.

### Generic functions

A generic function is not in a library until something uses it with a type:

```rust
pub fn largest<T: PartialOrd + Copy>(xs: &[T]) -> T { /* ... */ }
```

Say which types you want, in the recipe, written as Rust names them:

```toml
[rust]
instantiate = ["geo::largest::<f64>"]
```

Each entry becomes a function of the module named after it:

```julia
julia> Geo.largest_f64([2.5, 9.0, 1.0])
9.0
```

A method of a generic type is `"geo::Pair::<i32>::sum"`. A function that
takes `impl Trait` is named by the type it should have:
`"(geo::scale as fn(f32) -> f64)"`. An entry that is not a function of the
crate is a build error that shows the compiler's message.

Nothing else about a crate needs an entry. Its `#[inline]` functions, its
statics (`NAME()` reads one) and its traits are found by the build.

### What was not wrapped

A function RepliBuild cannot call yet is still in the module, as a function
that raises an error saying why. The module lists them:

```julia
julia> Geo.UNSUPPORTED          # name => reason; empty for this crate
Dict{Symbol, String}()

julia> Geo.FUNCTIONS            # every wrapped function
```

Not wrapped yet:

- a callback the library keeps and calls later (one called during the call
  works), and a callback that takes a string, a slice or a struct by value;
- a function that returns `&dyn Trait`, and a Julia type standing in for a
  Rust trait;
- building a struct that owns memory from its fields in Julia (get one from
  a Rust function instead);
- a generic function for a type the recipe does not name.

Two things Rust checks at compile time are yours to keep in Julia: a value
that borrows from another must not outlive it, and a value should be freed on
the thread that made it.

## Sharing it

`RepliBuild.register` and `RepliBuild.use` work as for any library
([Load it anywhere](use.md)). The crate is built where it lives, and the
finished build is kept whole, the thunks file with it, so a later
`use("geo")` only loads it.
