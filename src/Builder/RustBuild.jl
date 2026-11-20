# RustBuild.jl — `RepliBuild.build` for a `[rust]` project.
#
# clang gives RepliBuild one IR file per translation unit and leaves the link
# to RepliBuild. rustc's unit is the crate, and its link reads files rustc
# writes during the link (the export list, the allocator shim, the metadata
# object, the standard library's archives), so RepliBuild could link a Rust
# library only by re-implementing rustc's linker driver. It does not: cargo
# and rustc compile AND link, and RepliBuild sets what it needs on their
# command line and reads back what it reads from a clang build.
#
# **The library is a generated root crate, and the crate is its dependency.**
# `<[paths] build>/rust/root` holds a crate of one line, `extern crate <name>;`,
# built as a `dylib`, with the wrapped crate as a path dependency. A Rust
# `dylib` carries its dependencies and exports their public functions, so the
# library is the crate's, and three things follow that building the crate
# itself as the `dylib` does not give:
#
# - a `#![no_std]` crate (libm) builds: a library that is the end of a link
#   needs a panic handler, which the root's `std` supplies;
# - nothing is written into the crate's source tree, which is often a pinned
#   checkout: the lock file is the root's;
# - there is a place for the Rust RepliBuild has to supply itself (the frame
#   that catches a panic, drop functions, generic instantiations), when that
#   is built.
#
#   crate type `dylib`     every non-generic `pub fn` is exported; a `cdylib`
#                          exports only `extern "C"` items
#   debug info 2           every type and parameter is in the DWARF
#   one codegen unit       one IR file per crate, `…/deps/<crate>-<hash>.ll`
#   --emit=llvm-ir,link    the `define` lines, which say how each call is made
#   v0 symbol names        the names say which crate an item belongs to
#   --cap-lints=warn       the crate's own `#![deny(warnings)]` is its
#                          author's, and a newer rustc has newer warnings;
#                          cargo caps a registry dependency's the same way
#   -Wl,-Bsymbolic (ELF)   the library binds its own symbols to itself, as
#                          every library RepliBuild links does
#   panic = unwind         `abort` would end the process on any panic
#   -C llvm-args (two)     a function the optimizer merged into another keeps
#                          its own `define` line and debug info
#                          (`MERGED_FUNCTION_FLAGS`)
#
# The flags (the first four and the last) reach every crate of the build, through
# `CARGO_ENCODED_RUSTFLAGS`: the wrapped crate is a dependency, and what
# follows `--` on a `cargo rustc` line reaches the root alone.
#
# rustc's IR is read as text and never loaded into an LLVM: rustc carries its
# own LLVM, which is not Julia's and need not be the system's.
#
# **Optimized builds and the symbol table** (measured 2026-11-19, rustc
# 1.97.1). Above opt-level 0 rustc marks a small `pub fn` as inlinable across
# crates and gives it no symbol in its own crate: of a 17-function crate, 5
# kept a symbol at opt-level 1 and 4 at opt-level 3. rustc does not make that
# inference when incremental compilation is on, and all 17 kept their symbol
# at opt-level 3 with it. So a build above opt-level 0 keeps incremental on.
# That is how rustc behaves, not something it promises: the surface is read
# from the library that was built, never assumed.
#
# What this writes, beside the library in `[paths] output`:
# `compilation_metadata.json` with `"language": "rust"`: one row per function
# of the crate (`RustABI.join_function`) and the types those functions reach
# (`RustDWARF`). Wrapper generation reads that file; it is not built yet.
# Design and measurements: docs/updates/RUST_FRONT_END.md.

module RustBuild

using Dates
using JSON: JSON
using TOML
import SHA
import Libdl

import ..ConfigurationManager
import ..ConfigurationManager: RepliBuildConfig, RustConfig, get_output_path, get_library_name
import ..BuildBridge
import ..DependencyResolver
import ..Compiler
import ..LibDW
import ..Tools
import ..RustDWARF
import ..RustABI
import ..RustGlue
import .._parse_json
import .._read_json

"""
    manifest_path(config) -> String

The crate's `Cargo.toml`, absolute and `/`-separated: `[rust] manifest`
itself, or, with `[rust] dependency`, `manifest` inside that dependency's
checkout (`DependencyResolver.checkout_root`, which answers whether or not
the checkout has been fetched). The one place the manifest is read from the
config.
"""
function manifest_path(config::RepliBuildConfig)::String
    rc = config.rust
    isempty(rc.dependency) && return rc.manifest
    root = DependencyResolver.checkout_root(config, rc.dependency)
    return ConfigurationManager._graph_path(joinpath(root, rc.manifest))
end

"""
    source_root(config) -> String

The directory whose files are the crate's source: the checkout of `[rust]
dependency`, whole (a workspace's other members and its lock file are in
it), or the manifest's directory when the crate is in the project.
"""
function source_root(config::RepliBuildConfig)::String
    rc = config.rust
    isempty(rc.dependency) && return dirname(rc.manifest)
    return ConfigurationManager._graph_path(DependencyResolver.checkout_root(config, rc.dependency))
end

"""
    input_files(config) -> Vector{Pair{String,String}}

Every file under `source_root` that a build of the crate may read, as
`relative path => absolute path`, sorted: what the registry hashes to decide
whether a vendored build is still this crate's. Left out: `.git`, cargo's
own `target` directories, and RepliBuild's cache, build and output
directories when the crate is in the project.
"""
function input_files(config::RepliBuildConfig)::Vector{Pair{String,String}}
    root = source_root(config)
    isdir(root) || return Pair{String,String}[]
    own = [ConfigurationManager._graph_path(p) for p in (ConfigurationManager.get_build_path(config),
                                                        ConfigurationManager.get_output_path(config))]
    out = Pair{String,String}[]
    for (dir, dirs, files) in walkdir(root)
        filter!(d -> !(d in (".git", "target", ".replibuild_cache")) &&
                     !(ConfigurationManager._graph_path(joinpath(dir, d)) in own), dirs)
        sort!(dirs)
        for f in sort(files)
            path = joinpath(dir, f)
            isfile(path) || continue
            push!(out, replace(relpath(path, root), '\\' => '/') => path)
        end
    end
    return out
end

"""
    source_identity(config) -> String

What names the crate's source apart from its files: `""` for a crate in the
project, and for a `[rust] dependency` the entry's name, type, pin and
location with `[rust] manifest`, as `GraphMode.source_identity` does for a
graph. A moved pin is another build even before the checkout follows it.
"""
function source_identity(config::RepliBuildConfig)::String
    rc = config.rust
    isempty(rc.dependency) && return ""
    dep = config.dependencies.items[rc.dependency]
    return join(["replibuild-rust-dependency", rc.dependency, dep.type, dep.url, dep.tag,
                 dep.commit, dep.sha256, dep.subdir, dep.path, rc.manifest], "\n")
end

"""
    toolchain_identity(config) -> String

`rustc -vV` and `cargo -V` as the crate's directory sees them, or
`"unavailable"` when they cannot be run: a build made by one rustc is not a
build by another, and the registry must not serve it as one.
"""
function toolchain_identity(config::RepliBuildConfig)::String
    dir = source_root(config)
    return try
        toolchain(isdir(dir) ? dir : config.project.root).identity
    catch
        "unavailable"
    end
end

# `[rust]` with its manifest where it is on this machine.
_located(config::RepliBuildConfig) =
    RustConfig(manifest_path(config), "", config.rust.package, config.rust.opt_level,
               config.rust.features, config.rust.default_features, config.rust.target_dir,
               config.rust.instantiate)

"The shape of a Rust project's `compilation_metadata.json`. Bump it with any change to what a row means."
const METADATA_VERSION = "rust-4"

# =============================================================================
# The toolchain
# =============================================================================

"""
    tool(name) -> String

`cargo` or `rustc`: `REPLIBUILD_TOOL_CARGO` / `REPLIBUILD_TOOL_RUSTC` when
set, else the one on PATH, else the bare name, so a machine with neither
fails by naming the tool. There is no bundled Rust: the toolchain is the
user's, like the system clang a C++ project compiles with.
"""
function tool(name::AbstractString)::String
    name in ("cargo", "rustc") || error("RustBuild.tool: $(repr(name)) is not `cargo` or `rustc`")
    override = get(ENV, "REPLIBUILD_TOOL_" * uppercase(name), "")
    isempty(override) || return override
    w = Sys.which(name)
    return w === nothing ? String(name) : String(w)
end

# Run `cmd` to its end; `(stdout, stderr, exit code)`. A program that never
# started is `BuildBridge.SPAWN_FAILED`, which callers report as the tool
# missing and not as the tool failing.
function _capture(cmd::Cmd)::Tuple{String,String,Int}
    out = tempname()
    err = tempname()
    try
        proc = try
            run(pipeline(ignorestatus(cmd); stdin = devnull, stdout = out, stderr = err))
        catch e
            e isa Base.IOError || rethrow()
            return ("", sprint(showerror, e), BuildBridge.SPAWN_FAILED)
        end
        code = proc.exitcode != 0 ? Int(proc.exitcode) : (proc.termsignal != 0 ? 128 + Int(proc.termsignal) : 0)
        return (read(out, String), read(err, String), code)
    finally
        rm(out; force = true)
        rm(err; force = true)
    end
end

"""
The Rust toolchain a crate is built with: the programs, and what they say
they are. `identity` is `rustc -vV` and `cargo -V` whole; it goes into the
metadata key, so another rustc extracts again.
"""
struct Toolchain
    cargo::String
    rustc::String
    release::String
    host::String
    llvm::String
    identity::String
end

"""
    toolchain(dir) -> Toolchain

Ask `rustc` and `cargo` what they are, from `dir`: with rustup, the
directory decides which toolchain a `rust-toolchain.toml` selects.
"""
function toolchain(dir::AbstractString)::Toolchain
    cargo = tool("cargo")
    rustc = tool("rustc")
    (vv, err, code) = _capture(Cmd(`$rustc -vV`; dir = String(dir)))
    code == 0 || error(_missing_tool_message("rustc", rustc, code, err))
    (cv, cerr, ccode) = _capture(Cmd(`$cargo -V`; dir = String(dir)))
    ccode == 0 || error(_missing_tool_message("cargo", cargo, ccode, cerr))
    field(k) = (m = match(Regex("^" * k * ":\\s*(.+)\$", "m"), vv); m === nothing ? "" : String(strip(m.captures[1])))
    return Toolchain(cargo, rustc, field("release"), field("host"), field("LLVM version"),
                     String(strip(vv)) * "\n" * String(strip(cv)))
end

_missing_tool_message(name, path, code, err) =
    code == BuildBridge.SPAWN_FAILED ?
        "`$name` could not be started ($path). A [rust] project is built by the Rust toolchain: " *
        "install it (rustup, or the distribution's `rust` package) or set REPLIBUILD_TOOL_$(uppercase(name)) " *
        "to the program." :
        "`$path` exited with $code:\n$(strip(err))"

# cargo, with the rustc this module resolved when one was named explicitly.
# No toolchain overlay and no library path: cargo starts rustc, which starts
# the system linker, and neither wants Julia's libraries first.
function _cargo(tc::Toolchain, args::Vector{String}, dir::AbstractString; keep_merged::Bool = true)::Cmd
    cmd = Cmd(`$(tc.cargo) $args`; dir = String(dir))
    env = Dict{String,String}(String(k) => String(v) for (k, v) in ENV)
    env["CARGO_TERM_COLOR"] = "never"
    isempty(get(ENV, "REPLIBUILD_TOOL_RUSTC", "")) || (env["RUSTC"] = tc.rustc)
    # One source of rustc flags, read before any configuration file (`rustflags`).
    env["CARGO_ENCODED_RUSTFLAGS"] = join(rustflags(; keep_merged), '\x1f')
    delete!(env, "RUSTFLAGS")
    return setenv(cmd, env)
end

# =============================================================================
# The crate
# =============================================================================

"""
The crate a `[rust]` section names, as cargo resolved it.

- `package`: the Cargo package name (`rb-fix`).
- `name`: the crate name, which is the library target's with `-` as `_`
  (`rb_fix`): what the symbols and the file names carry.
- `manifest`: the package's own `Cargo.toml`.
- `workspace_root`: where `Cargo.lock` lives.
"""
struct Crate
    package::String
    name::String
    manifest::String
    workspace_root::String
end

# What another crate can link: the root crate depends on the wrapped one.
const _LIBRARY_KINDS = ("lib", "rlib", "dylib")
const _ANY_LIBRARY_KINDS = ("lib", "rlib", "dylib", "cdylib", "staticlib")

_same_file(a::AbstractString, b::AbstractString) =
    normpath(abspath(String(a))) == normpath(abspath(String(b))) ||
    (ispath(a) && ispath(b) && Base.samefile(String(a), String(b)))

"""
    select_crate(metadata, rc) -> Crate

Pick the crate out of `cargo metadata`'s answer (parsed JSON): the package
`[rust] package` names, else the package whose manifest is `[rust] manifest`.
A workspace manifest with no package of its own, a name the manifest does
not have, and a package with no library target are errors that say what the
manifest does have.
"""
function select_crate(metadata::AbstractDict, rc::RustConfig)::Crate
    pkgs = metadata["packages"]
    names = join(sort!(String[String(p["name"]) for p in pkgs]), ", ")
    pkg = nothing
    if !isempty(rc.package)
        i = findfirst(p -> p["name"] == rc.package, pkgs)
        i === nothing && error("[rust] package = $(repr(rc.package)): $(rc.manifest) has no such package. It has: $names")
        pkg = pkgs[i]
    else
        i = findfirst(p -> _same_file(p["manifest_path"], rc.manifest), pkgs)
        i === nothing && error("[rust] manifest = $(rc.manifest) is a workspace with no package of its own. " *
                               "Name the package to wrap with [rust] package. It has: $names")
        pkg = pkgs[i]
    end
    j = findfirst(t -> any(k -> k in _LIBRARY_KINDS, t["kind"]), pkg["targets"])
    if j === nothing
        targets = join((String(t["name"]) * " " * string(t["kind"]) for t in pkg["targets"]), ", ")
        any(t -> any(k -> k in _ANY_LIBRARY_KINDS, t["kind"]), pkg["targets"]) &&
            error("Cargo package $(repr(String(pkg["name"]))) builds its library only as a C library " *
                  "(its targets: $targets), which another Rust crate cannot depend on. RepliBuild builds " *
                  "the crate as a dependency of a generated root crate, so it needs `lib` or `rlib` among " *
                  "its `crate-type`s.")
        error("Cargo package $(repr(String(pkg["name"]))) has no library target " *
              "(its targets: $targets). A binary crate has no API to wrap.")
    end
    return Crate(String(pkg["name"]), replace(String(pkg["targets"][j]["name"]), '-' => '_'),
                 String(pkg["manifest_path"]), String(metadata["workspace_root"]))
end

"`cargo metadata` for the manifest, resolved to the crate to wrap (`select_crate`)."
function crate_info(rc::RustConfig, tc::Toolchain)::Crate
    isfile(rc.manifest) ||
        error("[rust] manifest: $(rc.manifest) does not exist. It names the crate's Cargo.toml, relative to " *
              "the project root, or to the checkout of [rust] dependency.")
    args = ["metadata", "--format-version", "1", "--no-deps", "--manifest-path", rc.manifest]
    (out, err, code) = _capture(_cargo(tc, args, dirname(rc.manifest)))
    code == 0 || error("cargo metadata failed for $(rc.manifest) (exit $code):\n$(strip(err))")
    return select_crate(_parse_json(out), rc)
end

# =============================================================================
# The build
# =============================================================================

# The Rust RepliBuild supplies itself, in the root crate: the frame that ends
# a panic. A Rust function that panics unwinds through the thunk that called
# it, and only Rust may end a Rust panic: C++ `catch (...)` aborts when it
# destroys one, and a Julia frame cannot take one at all. So a call that may
# panic is made as `__rb_catch(thunk, args, ret)`, which is where the unwind
# stops; the panic's own text (where it happened and what it said) is kept
# for the caller, per thread, and printed nowhere.
#
# These two functions are RepliBuild's own, so they are `extern "C"` and
# unmangled: Julia calls them with plain `ccall`. Nothing of the wrapped
# crate is touched. The hook passes every panic that is not inside
# `__rb_catch` on to the hook that was there before.
const _CATCH_FRAME = raw"""

use std::cell::{Cell, RefCell};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Once;

thread_local! {
    static RB_PANIC: RefCell<String> = RefCell::new(String::new());
    static RB_DEPTH: Cell<u32> = Cell::new(0);
}
static RB_HOOK: Once = Once::new();

/// Call `thunk(args, ret)`. A panic that unwinds out of it ends here:
/// 1 for a normal return, 0 for a panic, whose text `__rb_panic_message` gives.
#[no_mangle]
pub extern "C" fn __rb_catch(
    thunk: extern "C-unwind" fn(*mut u8, *mut u8),
    args: *mut u8,
    ret: *mut u8,
) -> u8 {
    RB_HOOK.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if RB_DEPTH.with(|d| d.get()) > 0 {
                let text = info.to_string();
                RB_PANIC.with(|m| *m.borrow_mut() = text);
            } else {
                previous(info);
            }
        }));
    });
    RB_PANIC.with(|m| m.borrow_mut().clear());
    RB_DEPTH.with(|d| d.set(d.get() + 1));
    let outcome = catch_unwind(AssertUnwindSafe(|| thunk(args, ret)));
    RB_DEPTH.with(|d| d.set(d.get() - 1));
    match outcome {
        Ok(()) => 1,
        Err(_) => 0,
    }
}

/// Copy this thread's last panic text into `buf` (at most `cap` bytes); its whole length.
#[no_mangle]
pub extern "C" fn __rb_panic_message(buf: *mut u8, cap: usize) -> usize {
    RB_PANIC.with(|m| {
        let text = m.borrow();
        let n = text.len().min(cap);
        if n > 0 {
            unsafe { std::ptr::copy_nonoverlapping(text.as_ptr(), buf, n) };
        }
        text.len()
    })
}
"""

# The second thing RepliBuild supplies: what ends a value Julia holds. A Rust
# value that owns memory is freed by Rust's own drop code for its type, so
# Julia holds the value's bytes and calls back here to end it. `__rb_drop` is
# that call for any type; `rb_glue.rs` has one exported line per type that
# names the type for it (`RustGlue`). A `String` and a `Vec` are also read and
# built here, by the standard library's own functions: their layout is the
# standard library's business, and is never spelled in Julia.
const _OWNERSHIP = raw"""

/// The address of a function this crate compiled because a line of `rb_glue.rs` named it.
#[repr(transparent)]
pub struct __RbAddr(*const ());
unsafe impl Sync for __RbAddr {}

/// Run the drop code of the value at `p`. 0: done. The largest integer: that code panicked.
#[allow(dead_code)]
unsafe fn __rb_drop<T>(p: *mut T) -> usize {
    match catch_unwind(AssertUnwindSafe(|| core::ptr::drop_in_place(p))) {
        Ok(()) => 0,
        Err(_) => usize::MAX,
    }
}

/// The drop function of a type the compiler chose, not this file: a function
/// that RETURNS the type is named (`__rb_ret1(dep::parse)`), and its return
/// type becomes `T` here. Which of the two the compiler picks is its answer
/// to "is this type dropped"; both do the same thing.
type __RbDrop<T> = unsafe extern "C" fn(*mut T, usize) -> usize;
#[allow(dead_code)]
unsafe extern "C" fn __rb_drop_owned<T>(p: *mut T, _unused: usize) -> usize {
    if p.is_null() { 0 } else { __rb_drop(p) }
}
#[allow(dead_code)]
unsafe extern "C" fn __rb_drop_plain<T>(p: *mut T, _unused: usize) -> usize {
    if p.is_null() { 0 } else { __rb_drop(p) }
}
#[allow(dead_code)]
const fn __rb_pick<T>() -> __RbDrop<T> {
    if core::mem::needs_drop::<T>() { __rb_drop_owned::<T> } else { __rb_drop_plain::<T> }
}
// One set per number of parameters: what the function returns, and the
// payloads of a `Result` or an `Option` it returns.
macro_rules! __rb_by_return {
    ($ret:ident $ok:ident $err:ident $some:ident : $($a:ident),*) => {
        #[allow(dead_code)] const fn $ret<$($a,)* R>(_f: fn($($a),*) -> R) -> __RbDrop<R> { __rb_pick::<R>() }
        #[allow(dead_code)] const fn $ok<$($a,)* T, E>(_f: fn($($a),*) -> Result<T, E>) -> __RbDrop<T> { __rb_pick::<T>() }
        #[allow(dead_code)] const fn $err<$($a,)* T, E>(_f: fn($($a),*) -> Result<T, E>) -> __RbDrop<E> { __rb_pick::<E>() }
        #[allow(dead_code)] const fn $some<$($a,)* T>(_f: fn($($a),*) -> Option<T>) -> __RbDrop<T> { __rb_pick::<T>() }
    };
}
__rb_by_return!(__rb_ret0 __rb_ok0 __rb_err0 __rb_some0 : );
__rb_by_return!(__rb_ret1 __rb_ok1 __rb_err1 __rb_some1 : A1);
__rb_by_return!(__rb_ret2 __rb_ok2 __rb_err2 __rb_some2 : A1, A2);
__rb_by_return!(__rb_ret3 __rb_ok3 __rb_err3 __rb_some3 : A1, A2, A3);
__rb_by_return!(__rb_ret4 __rb_ok4 __rb_err4 __rb_some4 : A1, A2, A3, A4);
__rb_by_return!(__rb_ret5 __rb_ok5 __rb_err5 __rb_some5 : A1, A2, A3, A4, A5);
__rb_by_return!(__rb_ret6 __rb_ok6 __rb_err6 __rb_some6 : A1, A2, A3, A4, A5, A6);
__rb_by_return!(__rb_ret7 __rb_ok7 __rb_err7 __rb_some7 : A1, A2, A3, A4, A5, A6, A7);
__rb_by_return!(__rb_ret8 __rb_ok8 __rb_err8 __rb_some8 : A1, A2, A3, A4, A5, A6, A7, A8);

/// Build a `String` at `out` from `len` bytes of text.
/// 0: built. 1: the bytes are not UTF-8, and nothing was built.
#[no_mangle]
pub unsafe extern "C" fn __rb_string_new(data: *const u8, len: usize, out: *mut String) -> u8 {
    let bytes: &[u8] = if len == 0 { &[] } else { core::slice::from_raw_parts(data, len) };
    match core::str::from_utf8(bytes) {
        Ok(text) => {
            out.write(String::from(text));
            0
        }
        Err(_) => 1,
    }
}

/// Where the text of the `String` at `s` is, and its length in bytes.
#[no_mangle]
pub unsafe extern "C" fn __rb_string_view(s: *const String, data: *mut *const u8, len: *mut usize) {
    *data = (*s).as_ptr();
    *len = (*s).len();
}

/// Where the elements of the `Vec` at `v` are, and how many there are.
#[allow(dead_code)]
unsafe fn __rb_vec_view<T>(v: *const Vec<T>, data: *mut *const u8, len: *mut usize) {
    *data = (*v).as_ptr() as *const u8;
    *len = (*v).len();
}

/// Build a `Vec` at `out` whose `len` elements are the bytes at `data`, moved in.
#[allow(dead_code)]
unsafe fn __rb_vec_new<T>(data: *const u8, len: usize, out: *mut Vec<T>) {
    let mut v = Vec::<T>::with_capacity(len);
    if len > 0 {
        core::ptr::copy_nonoverlapping(data, v.as_mut_ptr() as *mut u8, len * core::mem::size_of::<T>());
    }
    v.set_len(len);
    out.write(v);
}
"""

"The generated root crate's name: the library cargo builds is `lib<this>.<ext>`."
root_name(crate::Crate) = crate.name * "_rb"

"Where the generated root crate is: beside cargo's target directory, under `[paths] build`."
root_dir(rc::RustConfig) = dirname(rc.target_dir) * "/root"

"""
    root_files(rc, crate) -> Dict{String,String}

The generated root crate, file name to text: a `Cargo.toml` that builds a
`dylib` and depends on the wrapped crate by path, with the features `[rust]`
asks for, and a `lib.rs` whose `extern crate` line makes the crate part of
the library, followed by the frame that catches a panic (`_CATCH_FRAME`),
what drops, reads and builds an owned value (`_OWNERSHIP`), and the line
that includes `rb_glue.rs`, the per-type file the build writes
(`RustGlue`; not one of these files, since it is written from what a build
found). The empty `[workspace]` table keeps a workspace in a directory above
from claiming it.
"""
function root_files(rc::RustConfig, crate::Crate)::Dict{String,String}
    dep = Dict{String,Any}("path" => dirname(crate.manifest))
    rc.default_features || (dep["default-features"] = false)
    isempty(rc.features) || (dep["features"] = rc.features)
    manifest = Dict{String,Any}(
        "package" => Dict{String,Any}("name" => root_name(crate), "version" => "0.0.0",
                                      "edition" => "2021", "publish" => false),
        "lib" => Dict{String,Any}("crate-type" => ["dylib"], "path" => "lib.rs"),
        "dependencies" => Dict{String,Any}(crate.package => dep),
        "workspace" => Dict{String,Any}())
    toml = "# Generated by RepliBuild: the root of the library built from the crate `$(crate.name)`.\n" *
           "# Rewritten by every build; an edit here is lost.\n" *
           sprint(io -> TOML.print(io, manifest; sorted = true))
    lib = "// Generated by RepliBuild. The library is this crate; `$(crate.name)` is its\n" *
          "// dependency, and a `dylib` carries its dependencies and exports their functions.\n" *
          "#![allow(improper_ctypes_definitions)]\n" *
          "extern crate $(crate.name);\n" *
          "extern crate alloc;\n" * _CATCH_FRAME * _OWNERSHIP *
          "\ninclude!(\"$(RustGlue.FILE)\");\n"
    return Dict("Cargo.toml" => toml, "lib.rs" => lib)
end

# Write the root crate, touching only a file whose text changed: cargo
# rebuilds what is newer than its last build.
function _write_root(rc::RustConfig, crate::Crate)::String
    dir = root_dir(rc)
    mkpath(dir)
    for (name, text) in root_files(rc, crate)
        path = joinpath(dir, name)
        (isfile(path) && read(path, String) == text) || write(path, text)
    end
    # `lib.rs` includes it; a build that has found nothing yet has an empty one.
    isfile(joinpath(dir, RustGlue.FILE)) || _write_glue(rc, RustGlue.Plan())
    # The crate's own lock file pins its dependencies. A path dependency's
    # lock is not read by cargo, so the root starts from a copy of it, and is
    # given a new copy when it changes.
    theirs = joinpath(crate.workspace_root, "Cargo.lock")
    ours = joinpath(dir, "Cargo.lock")
    from = joinpath(dir, "Cargo.lock.from")
    if isfile(theirs)
        sha = _file_sha(theirs)
        if !(isfile(ours) && isfile(from) && strip(read(from, String)) == sha)
            cp(theirs, ours; force = true)
            chmod(ours, 0o644)
            write(from, sha)
        end
    end
    return dir
end

# -----------------------------------------------------------------------------
# rb_glue.rs: one line per type (`RustGlue`), and what it was written from
# -----------------------------------------------------------------------------

_glue_path(rc::RustConfig) = joinpath(root_dir(rc), RustGlue.FILE)
_plan_path(rc::RustConfig) = joinpath(root_dir(rc), "rb_glue.json")

# What a plan belongs to: a path that resolves for one compiler, crate and
# set of features need not for another.
function _glue_identity(rc::RustConfig, crate::Crate, tc::Toolchain, generator::AbstractString)::String
    ctx = SHA.SHA256_CTX()
    for part in (METADATA_VERSION, generator, tc.identity, crate.name, join(rc.features, ","), string(rc.default_features))
        Compiler._key_feed!(ctx, part)
    end
    return bytes2hex(SHA.digest!(ctx))
end

# Write the file for `plan`, and beside it what the file was written from.
function _write_glue(rc::RustConfig, plan::RustGlue.Plan; identity::AbstractString = "")
    (text, _, _) = RustGlue.glue_text(plan.rows, plan.fns)
    path = _glue_path(rc)
    mkpath(dirname(path))
    (isfile(path) && read(path, String) == text) || write(path, text)
    open(_plan_path(rc), "w") do io
        JSON.print(io, Dict{String,Any}(
            "identity" => String(identity),
            "file_sha" => bytes2hex(SHA.sha256(text)),
            "wanted" => plan.wanted,
            "rows" => [Dict{String,Any}("n" => r.n, "key" => r.key, "spelling" => r.spelling, "vec" => r.vec, "call" => r.call)
                       for r in plan.rows],
            "refused" => plan.refused,
            "wanted_fns" => plan.wanted_fns,
            "fns" => [Dict{String,Any}("n" => f.n, "expr" => f.expr, "source" => f.source, "symbol" => f.symbol)
                      for f in plan.fns],
            "fn_refused" => plan.fn_refused), 2)
    end
    return path
end

# The plan the file on disk was written from. When the file and the record
# beside it do not belong together, or belong to another compiler or crate,
# both are replaced by empty ones: stale lines must not reach a compile.
function _read_plan(rc::RustConfig, identity::AbstractString)::RustGlue.Plan
    path = _glue_path(rc)
    side = _plan_path(rc)
    if isfile(path) && isfile(side)
        plan = try
            d = _read_json(side)
            (d["identity"] == identity && d["file_sha"] == _file_sha(path)) ?
                RustGlue.Plan(String[String(k) for k in d["wanted"]],
                              RustGlue.Row[RustGlue.Row(Int(r["n"]), String(r["key"]), String(r["spelling"]), r["vec"] === true,
                                                        String(get(r, "call", "")))
                                           for r in d["rows"]],
                              Dict{String,String}(String(k) => String(v) for (k, v) in d["refused"]),
                              String[String(k) for k in d["wanted_fns"]],
                              RustGlue.FnRow[RustGlue.FnRow(Int(f["n"]), String(f["expr"]), String(f["source"]), String(f["symbol"]))
                                             for f in d["fns"]],
                              Dict{String,String}(String(k) => String(v) for (k, v) in d["fn_refused"])) : nothing
        catch
            nothing
        end
        plan === nothing || return plan
    end
    _write_glue(rc, RustGlue.Plan(); identity)
    return RustGlue.Plan()
end

"""
    _plan_glue!(rc, crate, need, fn_need, types, identity, build) -> (plan, messages)

Write `rb_glue.rs` for the types `need` and the functions `fn_need`, and
build the library with it. `build()` runs cargo on the root crate and returns
`(messages, stderr, exit code)`. Three steps, each a compile of the root
crate alone (the wrapped crate is already built):

1. the probe: every candidate path of every type and every found function,
   one `use` line each. The compiler names the lines that do not resolve
   from outside the crate;
2. the lines, each spelled with the first candidate that resolved;
3. again without any line the compiler refused, until it builds.

`fn_need` is a list of `(source, key, path)`: a function the build found in
the crate (`"inline"`, its symbol, its path as the demangler prints it), or
an entry of `[rust] instantiate` (`"instantiate"`, the expression, twice).
A found function with no public path is not the crate's API and is left out
without a word; an `instantiate` entry that does not compile is the user's
own line, and is an error with the compiler's message.

A type that ends with no line is in the plan's `refused`, with the reason.
Returns the plan the library was built with and that build's messages.
"""
function _plan_glue!(rc::RustConfig, crate::Crate, need::Vector{String}, fn_need::Vector, types::AbstractDict,
                     identity::AbstractString, build; functions = Any[])
    refused = Dict{String,String}()
    fn_refused = Dict{String,String}()
    names = Dict{String,String}()
    for key in need
        why = RustGlue.unnameable(key)
        why === nothing ? (names[key] = RustGlue.source_name(key)) : (refused[key] = why)
    end
    # For a type no path may spell: the functions a value of it comes out of.
    # Naming one of those has the compiler pick the type's drop function.
    from = Dict{String,Vector{Any}}()
    for key in need
        found = Any[]
        for pos in RustGlue.positions(functions, types, key)
            e = RustGlue.function_expression(pos.path)
            e === nothing || push!(found, (expr = e, helper = pos.helper, what = pos.what))
        end
        isempty(found) || (from[key] = found)
    end
    # A found function's path, as an expression could write it once its paths are chosen.
    exprs = Dict{String,String}()
    for f in fn_need
        f.source in ("inline", "dyn") || continue
        e = RustGlue.function_expression(f.path)
        e === nothing ? (fn_refused[f.key] = "it has no name a Rust expression can write") : (exprs[f.key] = e)
    end
    # A type and a trait, for a reference to one made into a trait object.
    coerce = [(f.key, RustGlue.source_name(f.path), String(f.trait)) for f in fn_need
              if f.source == "coerce" && RustGlue.unnameable(f.path) === nothing]
    # A function pointer's parameter and return types, each spelled on its own.
    parts = Dict{String,Any}()
    for f in fn_need
        f.source == "callback" || continue
        sig = get(types, something(get(get(types, f.key, Dict()), "target", nothing), ""), nothing)
        (sig === nothing || get(sig, "kind", "") != "subroutine") && continue
        ps = Any[p for p in get(sig, "parameters", Any[])]
        ret = get(sig, "return", nothing)
        (all(p -> p isa AbstractString && RustGlue.unnameable(p) === nothing, ps) &&
         (ret === nothing || RustGlue.unnameable(ret) === nothing)) || continue
        parts[f.key] = (params = String[RustGlue.source_name(p) for p in ps], ret = ret === nothing ? "" : RustGlue.source_name(ret))
    end
    texts = vcat([names[k] for k in sort!(collect(keys(names)))], [exprs[k] for k in sort!(collect(keys(exprs)))],
                 [p.expr for k in sort!(collect(keys(from))) for p in from[k]],
                 [c[2] for c in coerce], [c[3] for c in coerce],
                 [x for k in sort!(collect(keys(parts))) for x in vcat(parts[k].params, parts[k].ret)])
    all_paths = sort!(unique!(reduce(vcat, [RustGlue.paths(t) for t in texts]; init = String[])))
    chosen = Dict{String,String}()
    unresolved = Dict{String,String}()
    fail(what, messages, diagnostics) = begin
        _write_glue(rc, RustGlue.Plan(); identity)
        error("cargo could not build the root crate of `$(crate.name)` $what:\n" * _failure_text(messages, diagnostics))
    end
    if !isempty(all_paths)
        (text, lines) = RustGlue.probe_text(Dict(p => RustGlue.candidates(p, crate.name) for p in all_paths))
        write(_glue_path(rc), text)
        (messages, diagnostics, code) = build()
        (bad, _) = code == 0 ? (Dict{Int,String}(), String[]) : RustGlue.error_lines(messages)
        (code != 0 && isempty(bad)) && fail("with the lines that ask which paths are public", messages, diagnostics)
        for n in sort!(collect(keys(lines)))
            (path, candidate) = lines[n]
            if haskey(bad, n)
                get!(unresolved, path, bad[n])       # the first is the path itself: its error is the one to show
            else
                get!(chosen, path, candidate)
            end
        end
    end
    rows = RustGlue.Row[]
    for (i, key) in enumerate(need)
        spelling = haskey(names, key) ? RustGlue.spell(key, chosen) : nothing
        if spelling !== nothing
            push!(rows, RustGlue.Row(i, key, spelling, RustGlue.vec_element(types, key) !== nothing))
            continue
        end
        if haskey(names, key)
            p = first(p for p in RustGlue.paths(names[key]) if !haskey(chosen, p))
            refused[key] = "no path names `$p` from outside the crate ($(get(unresolved, p, "it did not resolve")))"
        end
        # Not spelled: taken from the first function that returns one and has a public path.
        for pos in get(from, key, Any[])
            e = RustGlue.spell(pos.expr, chosen)
            e === nothing && continue
            push!(rows, RustGlue.Row(i, key, pos.what, false, "$(pos.helper)($e)"))
            delete!(refused, key)
            break
        end
        haskey(refused, key) && (refused[key] *= ", and no function with a public path returns one")
    end
    fns = RustGlue.FnRow[]
    for (i, f) in enumerate(fn_need)
        if f.source in ("inline", "dyn")
            haskey(exprs, f.key) || continue
            e = RustGlue.spell(exprs[f.key], chosen)
            e === nothing ? (fn_refused[f.key] = "no public path names it") :
                            push!(fns, RustGlue.FnRow(i, e, String(f.source), f.source == "inline" ? f.key : ""))
        elseif f.source == "callback"
            haskey(parts, f.key) || continue
            ps = [RustGlue.spell(p, chosen) for p in parts[f.key].params]
            ret = isempty(parts[f.key].ret) ? "" : RustGlue.spell(parts[f.key].ret, chosen)
            (any(isnothing, ps) || ret === nothing) ? (fn_refused[f.key] = "no public path names one of its types") :
                                                      push!(fns, RustGlue.FnRow(i, join(ps, '\t'), "callback", ret))
        elseif f.source == "coerce"
            RustGlue.unnameable(f.path) === nothing || continue
            t = RustGlue.spell(RustGlue.source_name(f.path), chosen)
            d = RustGlue.spell(String(f.trait), chosen)
            (t === nothing || d === nothing) ? (fn_refused[f.key] = "no public path names it") :
                                               push!(fns, RustGlue.FnRow(i, t, "coerce", d))
        else
            push!(fns, RustGlue.FnRow(i, f.key, "instantiate", ""))
        end
    end
    wanted_fns = String[f.source * "|" * f.key for f in fn_need]
    for round in 1:8
        plan = RustGlue.Plan(need, rows, refused, wanted_fns, fns, fn_refused)
        _write_glue(rc, plan; identity)
        (messages, diagnostics, code) = build()
        code == 0 && return (plan, messages)
        (bad, _) = RustGlue.error_lines(messages)
        (_, line_of, fn_line_of) = RustGlue.glue_text(rows, fns)
        dropped = RustGlue.Row[r for r in rows if haskey(bad, line_of[r.n])]
        dropped_fns = RustGlue.FnRow[f for f in fns if haskey(bad, fn_line_of[f.n])]
        asked = [f for f in dropped_fns if f.source == "instantiate"]
        if !isempty(asked)
            _write_glue(rc, RustGlue.Plan(); identity)
            error("[rust] instantiate: " * join(("`$(f.expr)` does not compile: $(bad[fn_line_of[f.n]])" for f in asked), "; ") *
                  ". Each entry is a Rust expression that names one function of the crate, with its types: " *
                  "\"$(crate.name)::largest::<f64>\".")
        end
        ((isempty(dropped) && isempty(dropped_fns)) || round == 8) && fail("with its generated lines", messages, diagnostics)
        for r in dropped
            refused[r.key] = "the line that drops it does not compile: $(bad[line_of[r.n]])"
        end
        for f in dropped_fns
            fn_refused[f.source == "inline" ? f.symbol : "$(f.source)|$(f.expr)|$(f.symbol)"] =
                "the line that names it does not compile: $(bad[fn_line_of[f.n]])"
        end
        rows = RustGlue.Row[r for r in rows if !haskey(bad, line_of[r.n])]
        fns = RustGlue.FnRow[f for f in fns if !haskey(bad, fn_line_of[f.n])]
    end
    error("unreachable")
end

"""
    discovery_args(rc, crate) -> Vector{String}

The command line of the compile that LISTS the crate's functions: the wrapped
crate alone (`-p`), with the build's own profile and into the build's own
target directory, so its dependencies are the ones already compiled, and
`-C link-dead-code` for that one crate. With it rustc compiles every
function that is not generic, used or not, `#[inline]` ones included; a
normal compile leaves out what nothing uses and what is only ever inlined.
The output is a second IR file of the crate beside the build's own, read for
its `define` lines and linked into nothing (measured 2026-11-19: the next
normal build is still up to date).
"""
function discovery_args(rc::RustConfig, crate::Crate)::Vector{String}
    args = cargo_args(rc, crate)
    cut = findfirst(==("--"), args)
    return vcat(cut === nothing ? args : args[1:cut-1], ["-p", crate.package, "--", "-C", "link-dead-code"])
end

# The IR file of the wrapped crate that a cargo run reported building, or `nothing`.
function _crate_ir(messages::AbstractString, crate::Crate)::Union{String,Nothing}
    for line in eachsplit(messages, '\n')
        startswith(line, "{") || continue
        msg = try
            _parse_json(line)
        catch
            continue
        end
        get(msg, "reason", "") == "compiler-artifact" || continue
        target = get(msg, "target", Dict())
        (replace(String(get(target, "name", "")), '-' => '_') == crate.name &&
         _same_file(get(msg, "manifest_path", ""), crate.manifest)) || continue
        k = findfirst(f -> endswith(f, ".rlib"), String[String(f) for f in get(msg, "filenames", String[])])
        k === nothing && continue
        rlib = String(msg["filenames"][k])
        if basename(dirname(rlib)) != "deps"
            # The crate cargo was asked for by name is reported where cargo
            # puts a copy for the user (`target/debug/lib<crate>.rlib`), with
            # no hash in its name. The IR is beside the file it is a copy of.
            deps = joinpath(dirname(rlib), "deps")
            isdir(deps) || return nothing
            twins = [joinpath(deps, f) for f in readdir(deps) if startswith(f, "lib$(crate.name)-") && endswith(f, ".rlib")]
            i = findfirst(t -> Base.samefile(t, rlib), twins)
            i === nothing && (sha = _file_sha(rlib); i = findfirst(t -> _file_sha(t) == sha, twins))
            i === nothing && return nothing
            rlib = twins[i]
        end
        return joinpath(dirname(rlib), replace(replace(basename(rlib), r"^lib" => ""; count = 1), r"\.rlib$" => ".ll"))
    end
    return nothing
end

"""
    unexported_functions(rc, crate, tc; keep_merged, exported) -> Union{Vector{String},Nothing}

The symbols of the crate's own functions that the library does not export:
what the listing compile (`discovery_args`) defines, rooted in the crate,
less `exported`. `nothing` when that compile could not be made, which the
build reports and goes on without: the library is as complete as it was.
"""
function unexported_functions(rc::RustConfig, crate::Crate, tc::Toolchain; keep_merged::Bool = true,
                              exported = Set{String}())::Union{Vector{String},Nothing}
    (messages, _, code) = _capture(_cargo(tc, discovery_args(rc, crate), dirname(crate.manifest); keep_merged))
    code == 0 || return nothing
    ir = _crate_ir(messages, crate)
    (ir !== nothing && isfile(ir)) || return nothing
    out = String[]
    for sym in sort!(collect(keys(RustABI.definitions([ir]))))
        sym in exported && continue
        root = RustABI.v0_root(sym)
        (root !== nothing && root.crate == crate.name && root.namespace == 'v') && push!(out, sym)
    end
    return out
end

# What a failed cargo run said: the compiler's errors, then cargo's own lines.
function _failure_text(messages::AbstractString, diagnostics::AbstractString)::String
    rendered = RustGlue.rendered_errors(messages)
    return String(strip(isempty(rendered) ? String(diagnostics) : rendered * "\n" * strip(diagnostics)))
end

"""
    rustflags(; base = ENV) -> Vector{String}

What every rustc of the build is given: the user's own `RUSTFLAGS` (or
`CARGO_ENCODED_RUSTFLAGS`), then the flags this file's header lists. They go
to cargo as `CARGO_ENCODED_RUSTFLAGS`, which cargo reads before anything in a
configuration file, so what is built does not depend on the machine's cargo
configuration.
"""
function rustflags(; base = ENV, keep_merged::Bool = true)::Vector{String}
    user = if !isempty(get(base, "CARGO_ENCODED_RUSTFLAGS", ""))
        String[String(x) for x in split(base["CARGO_ENCODED_RUSTFLAGS"], '\x1f'; keepempty = false)]
    else
        String[String(x) for x in split(get(base, "RUSTFLAGS", ""))]
    end
    return vcat(user, ["--emit=llvm-ir,link", "-Csymbol-mangling-version=v0", "--cap-lints=warn"],
                keep_merged ? MERGED_FUNCTION_FLAGS : String[])
end

"""
What keeps a function the optimizer merged into another. Above opt-level 1
LLVM finds functions with the same code (`flip(Side) -> Side` and
`flag(bool) -> bool` are one function once compiled) and keeps one. rustc
asks it to leave the other as an ALIAS: a second name with no `define` line
and no debug info, so nothing in the build says what its Rust types were.
With these two arguments the other stays a function of its own that jumps to
the kept one, with its own `define` line and its own debug info (measured
2026-11-19, rustc 1.97.1: the fixture crate's three aliases became three
functions, each with its own types).

They are LLVM's own options, passed through rustc's stable `-C llvm-args`.
An LLVM that does not know them refuses to start, which `build_rust_project`
answers by building without them; a merged function is then described
through its alias (`RustABI.aliases`, `own_debug_info` false).
"""
const MERGED_FUNCTION_FLAGS = ["-Cllvm-args=-mergefunc-use-aliases=false", "-Cllvm-args=-mergefunc-preserve-debug-info"]

# What LLVM says of an option it does not have.
_unknown_llvm_argument(text::AbstractString) =
    occursin("mergefunc", text) && occursin(r"[Uu]nknown command line argument", text)

"""
    cargo_args(rc, crate; windows = Sys.iswindows()) -> Vector{String}

The `cargo rustc` command line that builds the generated root crate (this
file's header says why each part is there). The profile is `dev` with every
setting that matters stated, so the crate's own `[profile]` tables and the
user's cargo configuration do not decide what is built: at opt-level 0 it is
a debug build, and above it a release build (no debug assertions, no overflow
checks) that keeps its debug info and incremental compilation. The profile
is the whole build's, so the wrapped crate is compiled under it.

What follows `--` reaches rustc for the root crate alone, which is the one
that is linked: the link arguments.
"""
function cargo_args(rc::RustConfig, crate::Crate; windows::Bool = Sys.iswindows())::Vector{String}
    opt = rc.opt_level
    profile = ["opt-level=$opt",
               "debug=2",
               "codegen-units=1",
               "incremental=$(opt > 0)",
               "panic=\"unwind\"",
               "lto=\"off\"",
               "strip=\"none\"",
               "split-debuginfo=\"off\"",
               "debug-assertions=$(opt == 0)",
               "overflow-checks=$(opt == 0)"]
    args = ["rustc", "--manifest-path", root_dir(rc) * "/Cargo.toml", "--lib",
            "--target-dir", rc.target_dir, "--message-format=json"]
    for p in profile
        append!(args, ["--config", "profile.dev." * p])
    end
    link = vcat(Compiler._self_binding_link_flags(; windows), Compiler._reproducible_link_flags(; windows))
    isempty(link) || push!(args, "--")
    for flag in link
        append!(args, ["-C", "link-arg=" * flag])
    end
    return args
end

"""
    find_artifacts(messages, crate) -> (library, fresh, ir)

What cargo built, from its JSON messages (one object per line of
`--message-format=json`): the root crate's dynamic library, cargo's word for
"nothing was recompiled" (`fresh`: the root's and the wrapped crate's both),
and the wrapped crate's IR file, which rustc writes beside the crate's
`.rlib` under the same name (`…/deps/lib<crate>-<hash>.rlib`,
`…/deps/<crate>-<hash>.ll`).

The wrapped crate is told from another of its name (a second version among
the dependencies) by its manifest. An error when either is missing, which
names what cargo did report.
"""
function find_artifacts(messages::AbstractString, crate::Crate)
    seen = String[]
    library = nothing
    rlib = nothing
    fresh = true
    for line in eachsplit(messages, '\n')
        startswith(line, "{") || continue
        msg = try
            _parse_json(line)
        catch
            continue
        end
        get(msg, "reason", "") == "compiler-artifact" || continue
        target = get(msg, "target", Dict())
        tname = replace(String(get(target, "name", "")), '-' => '_')
        types = get(target, "crate_types", String[])
        push!(seen, tname * " (" * join(types, ", ") * ")")
        files = String[String(f) for f in get(msg, "filenames", String[])]
        if tname == root_name(crate) && "dylib" in types
            k = findfirst(f -> endswith(f, "." * Libdl.dlext), files)
            k === nothing || (library = files[k]; fresh &= get(msg, "fresh", false) === true)
        elseif tname == crate.name && _same_file(get(msg, "manifest_path", ""), crate.manifest)
            k = findfirst(f -> endswith(f, ".rlib"), files)
            k === nothing || (rlib = files[k]; fresh &= get(msg, "fresh", false) === true)
        end
    end
    reported = isempty(seen) ? "It reported no artifact at all." : "It reported: " * join(seen, "; ")
    library === nothing &&
        error("cargo reported no dynamic library for the root crate `$(root_name(crate))`. " * reported)
    rlib === nothing &&
        error("cargo reported no library for the crate `$(crate.name)` ($(crate.manifest)). " * reported)
    stem = replace(basename(rlib), r"^lib" => ""; count = 1)
    ir = joinpath(dirname(rlib), replace(stem, r"\.rlib$" => ".ll"))
    return (library = library, fresh = fresh, ir = ir)
end

function _file_sha(path::AbstractString)::String
    return open(io -> bytes2hex(SHA.sha256(io)), path)
end

# What the metadata was extracted from, and by what: the library and the IR
# by content, the toolchain, the code that reads them, and what the project
# file asks to have compiled in (`asked`: `[rust] instantiate`, which changes
# the library only once a build acts on it).
function _metadata_key(library::AbstractString, ir_files::Vector{String}, tc::Toolchain, crate::Crate,
                       generator::AbstractString, asked::Vector{String} = String[])::String
    ctx = SHA.SHA256_CTX()
    Compiler._key_feed!(ctx, METADATA_VERSION)
    Compiler._key_feed!(ctx, generator)
    foreach(a -> Compiler._key_feed!(ctx, "instantiate:" * a), asked)
    Compiler._key_feed!(ctx, tc.identity)
    Compiler._key_feed!(ctx, crate.name)
    Compiler._key_feed!(ctx, _file_sha(library))
    for f in ir_files
        Compiler._key_feed!(ctx, _file_sha(f))
    end
    return bytes2hex(SHA.digest!(ctx))
end

_metadata_path(config::RepliBuildConfig) = joinpath(get_output_path(config), "compilation_metadata.json")
_key_path(config::RepliBuildConfig) = joinpath(get_output_path(config), "rust_metadata.key")

"""
    build_rust_project(config; generator = "") -> String

`RepliBuild.build` for a `[rust]` project: write the root crate, build it
with cargo, copy its dynamic library to `[paths] output`, and write
`compilation_metadata.json` beside it. Returns the library's path.

cargo decides what is recompiled; it is always asked. The metadata is
extracted again only when the library, the IR, the toolchain or RepliBuild
itself changed (`rust_metadata.key`). `generator` names RepliBuild for that
key; `RepliBuild.build` passes the registry's fingerprint of the checkout,
which this module cannot ask for itself (the registry hashes Rust projects
through it, so the registry comes after). The key is removed before an
extraction, so one that fails leaves nothing claiming success.
"""
function build_rust_project(config::RepliBuildConfig; generator::AbstractString = "")::String
    config.rust === nothing && error("build_rust_project: the project has no [rust] section")
    rc = _located(config)
    println("RepliBuild | $(config.project.name) [rust]")
    start = time()

    tc = toolchain(isfile(rc.manifest) ? dirname(rc.manifest) : config.project.root)
    crate = crate_info(rc, tc)
    mkpath(dirname(rc.target_dir))
    out_dir = get_output_path(config)
    mkpath(out_dir)

    _write_root(rc, crate)
    identity = _glue_identity(rc, crate, tc, generator)
    plan = _read_plan(rc, identity)
    # From the crate's own directory, like the toolchain question above: with
    # rustup, a `rust-toolchain.toml` there chooses the compiler.
    keep_merged = Ref(true)
    build() = _capture(_cargo(tc, cargo_args(rc, crate), dirname(crate.manifest); keep_merged = keep_merged[]))
    (messages, diagnostics, code) = build()
    code == BuildBridge.SPAWN_FAILED && error(_missing_tool_message("cargo", tc.cargo, code, diagnostics))
    if code != 0 && _unknown_llvm_argument(diagnostics * messages)
        # This rustc's LLVM does not have the options that keep a merged
        # function (`MERGED_FUNCTION_FLAGS`): build without them.
        println("  note: this rustc's LLVM refuses the options that keep merged functions apart; building without them")
        keep_merged[] = false
        (messages, diagnostics, code) = build()
    end
    if code != 0 && !(isempty(plan.rows) && isempty(plan.fns))
        # The generated lines were written for the crate as it was. It changed
        # under them: build without, and write them again from what is found.
        plan = RustGlue.Plan()
        _write_glue(rc, plan; identity)
        (messages, diagnostics, code) = build()
    end
    code == 0 || error("cargo could not build the crate `$(crate.name)` (exit $code):\n" * _failure_text(messages, diagnostics))
    artifact = find_artifacts(messages, crate)
    isfile(artifact.library) || error("cargo named $(artifact.library) as the library and it does not exist")
    ir = artifact.ir
    if !isfile(ir)
        deps = dirname(ir)
        found = isdir(deps) ? filter(f -> endswith(f, ".ll"), readdir(deps)) : String[]
        error("rustc wrote no IR file $ir for the crate `$(crate.name)`. " *
              (isempty(found) ? "No .ll file is in $deps." : "The .ll files there: $(join(found, ", "))."))
    end
    println("  cargo: $(crate.name) $(artifact.fresh ? "up to date" : "built") " *
            "(opt-level $(rc.opt_level), rustc $(tc.release), LLVM $(tc.llvm))")

    library = joinpath(out_dir, get_library_name(config))
    key = _metadata_key(artifact.library, [ir], tc, crate, generator, rc.instantiate)
    keyfile = _key_path(config)
    if isfile(library) && isfile(_metadata_path(config)) && isfile(keyfile) &&
       strip(read(keyfile, String)) == key && _file_sha(library) == _file_sha(artifact.library)
        println("  metadata: cached")
        println("  done: $(round(time() - start, digits = 2))s")
        return library
    end
    # Gone before the extraction, so a failed one leaves nothing claiming success.
    rm(keyfile; force = true)
    # A new file, not new bytes in the old one: a session that has the
    # previous library open keeps the file it mapped.
    rm(library; force = true)
    cp(artifact.library, library)

    metadata = extract_metadata(config, crate, tc, library, [ir]; plan, allow_empty = true)
    need = RustGlue.wanted(metadata["functions"], metadata["types"])
    found = _functions_wanted(rc, crate, tc, metadata; keep_merged = keep_merged[])
    fn_need = vcat(found, _trait_objects_wanted(metadata))
    if need != plan.wanted || String[f.source * "|" * f.key for f in fn_need] != plan.wanted_fns
        # What this build's functions move, and the functions the library
        # has no symbol for, are not what `rb_glue.rs` was written for (on a
        # first build, nothing): write it, build the root crate again, and
        # read the library that has it. The functions compiled in move types
        # of their own, which are asked about in a second round.
        for round in 1:3
            had = Set{String}(String(f["symbol"]) for f in metadata["functions"])
            (plan, messages) = _plan_glue!(rc, crate, need, fn_need, metadata["types"], identity, build;
                                           functions = metadata["functions"])
            artifact = find_artifacts(messages, crate)
            rm(library; force = true)
            cp(artifact.library, library)
            metadata = extract_metadata(config, crate, tc, library, [artifact.ir]; plan, allow_empty = true)
            again = RustGlue.wanted(metadata["functions"], metadata["types"])
            fns_again = vcat(found, _trait_objects_wanted(metadata))
            # Again when the functions compiled in move types that were not
            # asked about, name trait objects that were not known of, or are
            # functions a type with no spelling could be taken from.
            more = Set{String}(String(f["symbol"]) for f in metadata["functions"]) != had
            (again == need && length(fns_again) == length(fn_need) && !(more && !isempty(metadata["glue_refused"]))) && break
            need = again
            fn_need = fns_again
        end
        key = _metadata_key(artifact.library, [artifact.ir], tc, crate, generator, rc.instantiate)
    end
    isempty(metadata["functions"]) && error(_no_surface_message(crate, metadata))
    open(_metadata_path(config), "w") do io
        JSON.print(io, metadata, 2)
    end
    report(stdout, metadata)
    write(keyfile, key)
    println("  done: $(round(time() - start, digits = 2))s")
    return library
end

# The functions the library has no symbol for and should have: the crate's
# own that the listing compile finds, and the entries of `[rust] instantiate`.
function _functions_wanted(rc::RustConfig, crate::Crate, tc::Toolchain, metadata::AbstractDict; keep_merged::Bool = true)
    exported = Set{String}(String(f["symbol"]) for f in metadata["functions"] if !haskey(f, "address"))
    foreach(r -> push!(exported, String(r["symbol"])), metadata["skipped"])
    found = unexported_functions(rc, crate, tc; keep_merged, exported)
    if found === nothing
        println("  note: the crate's unexported functions could not be listed (the listing compile failed); " *
                "its `#[inline]` functions are not wrapped")
        found = String[]
    end
    paths = _demangle(found)
    out = Any[(source = "inline", key = s, path = get(paths, s, s), trait = "") for s in found]
    append!(out, [(source = "instantiate", key = e, path = e, trait = "") for e in rc.instantiate])
    return out
end

# What the functions' trait objects need: each trait's methods, called
# through a trait object, and for each type that implements it the function
# that makes a reference to one into a reference to a trait object.
function _trait_objects_wanted(metadata::AbstractDict)
    out = Any[]
    for t in RustGlue.trait_objects(metadata["functions"])
        for m in t.methods
            push!(out, (source = "dyn", key = "$(t.trait)|$m", path = "<dyn $(t.trait) as $(t.trait)>::$m", trait = t.trait))
        end
        for ty in t.types
            push!(out, (source = "coerce", key = "$ty|$(t.trait)", path = ty, trait = t.trait))
        end
    end
    # Each kind of function pointer a function takes: a Rust function of that
    # signature for the library to be given, which calls on to Julia's.
    for k in RustGlue.callbacks(metadata["functions"], metadata["types"])
        push!(out, (source = "callback", key = k, path = k, trait = ""))
    end
    return out
end

_no_surface_message(crate::Crate, metadata::AbstractDict) =
    "The crate `$(crate.name)` has no function to wrap: its library exports none of the crate's own, and " *
    "none of its unexported functions has a public path. A crate's API is its `pub fn`s; a generic " *
    "function exists only for the types it is used with, which `[rust] instantiate` names." *
    (isempty(metadata["skipped"]) ? "" : " $(length(metadata["skipped"])) exported definitions were not read " *
                                         "(first: $(metadata["skipped"][1]["symbol"]): $(metadata["skipped"][1]["reason"])).")

# =============================================================================
# The metadata
# =============================================================================

# The library's DWARF as binutils prints it, one line at a time, handed to
# `f`. Streamed from the dumper: a Rust library carries the standard
# library's debug info, and the whole dump is never needed at once.
function _with_dump_lines(f, library::AbstractString)
    if Compiler._dwarf_reader() === :libdw
        return f(LibDW.DumpLines(String(library)))
    end
    dumper = Compiler._dwarf_dumper()
    dumper === nothing && error(Compiler._no_dwarf_dumper_error())
    args = [dumper.dwarf * "info", String(library)]
    name = Tools._tool_of(dumper.tool)
    cmd = name === nothing ? `$(dumper.tool) $args` : Tools.with_env(name, `$(Tools.tool_path(name)) $args`)
    try
        return open(io -> f(eachline(io)), pipeline(cmd; stderr = devnull))
    catch e
        e isa ProcessFailedException || rethrow()
        error("$(dumper.tool) $(dumper.dwarf)info failed on $library")
    end
end

# rustc's v0 names, demangled in one run of the binutils demangler. A crate's
# name is printed with its disambiguator (`probe[721ce43cb31eafab]::add`),
# which is a hash of the build and is cut: the symbol carries it already.
function _demangle(symbols::Vector{String})::Dict{String,String}
    isempty(symbols) && return Dict{String,String}()
    out = try
        read(pipeline(Tools.tool_cmd(`c++filt`); stdin = IOBuffer(join(symbols, "\n") * "\n"), stderr = devnull), String)
    catch
        ""
    end
    lines = split(chomp(out), '\n')
    length(lines) == length(symbols) || return Dict(s => s for s in symbols)
    return Dict(s => String(replace(strip(l), r"(?<=[A-Za-z0-9_])\[[0-9a-f]+\]" => ""))
                for (s, l) in zip(symbols, lines))
end

# `a::b::name` is (["a", "b"], "name"); a path that is not a plain one
# (`<T as Trait>::name`) keeps its name and has no scope to give.
function _path_scope_name(path::AbstractString)
    p = String(path)
    if startswith(p, "<")
        i = findlast("::", p)
        return (String[], i === nothing ? p : p[last(i)+1:end])
    end
    parts = String[String(x) for x in split(replace(p, r"::<.*$" => ""), "::")]
    return (parts[1:end-1], parts[end])
end

"""
    extract_metadata(config, crate, tc, library, ir_files) -> Dict

Everything `compilation_metadata.json` holds for a Rust library.

- `functions`: one row per function of the crate's surface
  (`RustABI.surface`), as `RustABI.join_function` writes it, plus `path`, its
  Rust path as the demangler prints it. Sorted by path, then symbol. A symbol
  that is an alias of another function (LLVM merged two with the same code)
  has `alias_of`, that function's symbol: its call is that function's. When
  it has no DIE of its own (`own_debug_info` false) its types are that
  function's too, which is right for the call and may name other Rust types
  than the merged function was written with.
- `types`: every type those functions reach (`RustDWARF.describe_functions`).
- `glue`: per type the root crate has a line for (`RustGlue.verify`): the
  function that drops a value of it, and whether dropping one does anything.
  `glue_refused`: the types that were asked about and have none, with why.
  `glue_helpers`: the root crate's `String` functions the library exports.
- A row with `address` is a function the library has no symbol for
  (`instance`: `"inline"`, found in the crate, or `"instantiate"`, asked for
  in the TOML): the root crate compiled it, and the exported static named
  by `address` holds where it is. `instances` counts them.
- `dyn`: per trait object (`dyn rb_fix::Area`), `implementors`: for each type
  that implements the trait, the function that makes a reference to a value
  of it into a reference to the trait object.
- `callbacks`: per kind of Rust function pointer a function takes
  (`fn(i32) -> i32`), `at`: the static holding the address of the Rust
  function of that signature the library is to be given, and `set`: the
  function that sets the C function it calls on to.
- `statics`: the crate's `static` items the library exports, each with its
  symbol, path, type, and whether it may be written (`mutable`).
- `skipped`: exported definitions that were left out, each with its reason.

An empty surface is an error, as it is for a C library
(`verify_wrap_surface` R1): a `cdylib`, a crate with no `pub fn`, or symbol
names this cannot read all look like that, and a wrapper of nothing is not a
result.
"""
function extract_metadata(config::RepliBuildConfig, crate::Crate, tc::Toolchain,
                          library::AbstractString, ir_files::Vector{String};
                          plan::RustGlue.Plan = RustGlue.Plan(), allow_empty::Bool = false)::Dict{String,Any}
    exported = Compiler._runtime_symbol_names(String(library))
    exported === nothing && error("the dynamic symbol table of $library could not be read")
    defs = RustABI.definitions(ir_files)
    aliases = RustABI.aliases(ir_files)
    (symbols, skipped, foreign) = RustABI.surface(defs, exported, crate.name; aliases)
    # Functions the library has no symbol for, compiled into the root crate
    # because a line of `rb_glue.rs` named them. The root crate's IR says
    # which function each line's static points at; its `define` line is
    # there too (or in the crate's own IR, when the crate had built it).
    instances = NamedTuple[]
    fn_refused = Dict{String,String}(plan.fn_refused)
    table = Dict{String,String}()
    picked = String[]                    # the drop functions the compiler chose for types that were not spelled
    if !isempty(plan.fns) || any(r -> !isempty(r.call), plan.rows)      # (callbacks are among `plan.fns`)
        root_ir = joinpath(dirname(ir_files[1]), root_name(crate) * ".ll")
        isfile(root_ir) || error("rustc wrote no IR file $root_ir for the root crate `$(root_name(crate))`")
        table = RustGlue.address_table([root_ir])
        picked = String[table[RustGlue.dropfn_symbol(r.n)] for r in plan.rows
                        if !isempty(r.call) && haskey(table, RustGlue.dropfn_symbol(r.n))]
    end
    if !isempty(plan.fns)
        root_ir = joinpath(dirname(ir_files[1]), root_name(crate) * ".ll")
        root_defs = RustABI.definitions([root_ir])
        named = _demangle(unique!(vcat(collect(values(table)), String[f.symbol for f in plan.fns if !isempty(f.symbol)])))
        taken = Set{String}(symbols)
        for f in plan.fns
            f.source in ("coerce", "callback") && continue
            label = f.source == "inline" ? f.symbol : f.expr
            address = RustGlue.fn_symbol(f.n)
            sym = get(table, address, nothing)
            if sym === nothing || !(address in exported)
                fn_refused[label] = "its address is not in the library"
                continue
            end
            sym in taken && continue                        # exported already, or named twice
            if f.source == "inline" && get(named, sym, sym) != get(named, f.symbol, f.symbol)
                fn_refused[label] = "its path names another function (`$(get(named, sym, sym))`)"
                continue
            end
            def = get(root_defs, sym, get(defs, sym, nothing))
            if def === nothing
                fn_refused[label] = "the build has no `define` line for it"
                continue
            end
            push!(taken, sym)
            push!(instances, (symbol = sym, address = address, source = f.source, def = def))
        end
    end
    (isempty(symbols) && isempty(instances) && !allow_empty) &&
        error(_no_surface_message(crate, Dict{String,Any}("skipped" => skipped)))
    # An alias has its target's code, and often no DIE of its own.
    target = Dict(sym => RustABI.alias_target(sym, defs, aliases) for sym in symbols)
    # The crate's statics: global variables of its IR that the library exports.
    globals = RustABI.statics(ir_files)
    static_symbols = String[]
    for sym in sort!(collect(keys(globals)))
        sym in exported || continue
        r = RustABI.v0_root(sym)
        ((r !== nothing && r.crate == crate.name && r.namespace == 'v') || !startswith(sym, "_R")) && push!(static_symbols, sym)
    end
    wanted = union(Set{String}(symbols), Set{String}(values(target)),
                   Set{String}(RustGlue.drop_symbol(r.n) for r in plan.rows),
                   Set{String}(String(i.symbol) for i in instances), Set{String}(picked),
                   Set{String}(RustGlue.coerce_symbol(f.n) for f in plan.fns if f.source == "coerce"),
                   Set{String}(table[RustGlue.callback_symbol(f.n)] for f in plan.fns
                               if f.source == "callback" && haskey(table, RustGlue.callback_symbol(f.n))))
    (dwarf_functions, types, dwarf_statics) = _with_dump_lines(library) do lines
        units = RustDWARF.read_units(lines, union(wanted, Set{String}(static_symbols)))
        RustDWARF.describe(units, wanted, Set{String}(static_symbols))
    end
    paths = _demangle(vcat(symbols, String[String(r["symbol"]) for r in skipped], String[String(i.symbol) for i in instances],
                           static_symbols))
    # A static the DWARF does not describe has no type to read it by, and is left out.
    statics = Dict{String,Any}[]
    for sym in static_symbols
        v = get(dwarf_statics, sym, nothing)
        (v === nothing || v["type"] === nothing) && continue
        push!(statics, Dict{String,Any}("symbol" => sym, "path" => get(paths, sym, sym), "name" => v["name"],
                                        "scope" => v["scope"], "type" => v["type"], "mutable" => globals[sym]))
    end
    sort!(statics; by = r -> (r["path"], r["symbol"]))
    for r in skipped
        r["path"] = get(paths, r["symbol"], r["symbol"])
    end
    functions = Dict{String,Any}[]
    for sym in symbols
        tsym = target[sym]
        own = get(dwarf_functions, sym, nothing)
        row = RustABI.join_function(sym, defs[tsym], own === nothing ? get(dwarf_functions, tsym, nothing) : own, types)
        row["path"] = get(paths, sym, sym)
        row["own_debug_info"] = own !== nothing
        if tsym != sym
            row["alias_of"] = tsym
            if own === nothing
                # The types are the TARGET's, which has the same code and so
                # the same call, and not necessarily the same Rust types:
                # `flip(Side) -> Side` and `flag(bool) -> bool` are one
                # function once optimized, and nothing in the library says
                # which types the merged one had. `own_debug_info` is false.
                # The name is this symbol's own, read off its path.
                (row["scope"], row["name"]) = _path_scope_name(row["path"])
                row["owner"] = nothing
            end
        end
        push!(functions, row)
    end
    for inst in instances
        own = get(dwarf_functions, inst.symbol, nothing)
        row = RustABI.join_function(inst.symbol, inst.def, own, types)
        # A method called through a trait object is a shim the compiler wrote for the trait's own method.
        row["path"] = replace(get(paths, inst.symbol, inst.symbol), r"::\{shim:[^}]*\}$" => "")
        row["own_debug_info"] = own !== nothing
        # Not a name the loader binds: the function is reached through the
        # exported static that holds its address.
        row["address"] = inst.address
        row["instance"] = inst.source
        push!(functions, row)
    end
    sort!(functions; by = r -> (r["path"], r["symbol"]))
    # Trait objects: per `dyn Trait`, the function that makes a reference to
    # each implementing type into one. The function's own debug info says
    # which type and which trait object its line named.
    dyn = Dict{String,Any}()
    pointee(key) = (t = get(types, something(key, ""), nothing);
                    (t !== nothing && get(t, "kind", "") == "pointer") ? get(t, "target", nothing) : nothing)
    for f in plan.fns
        f.source == "coerce" || continue
        sym = RustGlue.coerce_symbol(f.n)
        d = get(dwarf_functions, sym, nothing)
        (sym in exported && d !== nothing && length(d["parameters"]) == 2) || continue
        of = pointee(d["parameters"][1]["type"])
        fat = pointee(d["parameters"][2]["type"])
        (of isa AbstractString && fat isa AbstractString && startswith(fat, "&dyn ")) || continue
        entry = get!(dyn, String(fat[2:end]), Dict{String,Any}("implementors" => Dict{String,Any}()))
        entry["implementors"][String(of)] = sym
    end
    # Callbacks: per kind of function pointer, the Rust function of that
    # signature the library is given, and the function that sets what it
    # calls on to. Its debug info says the signature its line was given.
    callbacks = Dict{String,Any}()
    wanted_callbacks = RustGlue.callbacks(functions, types)
    for f in plan.fns
        f.source == "callback" || continue
        at = RustGlue.callback_symbol(f.n)
        set = RustGlue.callback_setter(f.n)
        d = get(dwarf_functions, get(table, at, ""), nothing)
        (at in exported && set in exported && d !== nothing) || continue
        for key in wanted_callbacks
            sig = get(types, something(get(types[key], "target", nothing), ""), nothing)
            sig === nothing && continue
            (Any[p["type"] for p in d["parameters"]] == Any[p for p in sig["parameters"]] && d["return"] == sig["return"]) || continue
            haskey(callbacks, key) || (callbacks[key] = Dict{String,Any}("at" => at, "set" => set))
        end
    end
    (glue, glue_refused) = RustGlue.verify(plan, dwarf_functions, types, exported; table, demangled = _demangle(picked))
    first_symbol = isempty(symbols) ? (isempty(instances) ? "" : String(instances[1].symbol)) : symbols[1]
    root = RustABI.v0_root(first_symbol)
    rc = config.rust
    return Dict{String,Any}(
        "metadata_version" => METADATA_VERSION,
        "language" => "rust",
        "crate" => Dict{String,Any}("name" => crate.name, "package" => crate.package,
                                    "root" => root_name(crate),
                                    "disambiguator" => root === nothing ? "" : root.disambiguator),
        "library" => basename(String(library)),
        "library_sha" => _file_sha(library),
        "functions" => functions,
        "types" => types,
        "glue" => glue,
        "glue_refused" => glue_refused,
        "glue_helpers" => String[h for h in RustGlue.STRING_HELPERS if h in exported],
        "statics" => statics,
        "dyn" => dyn,
        "callbacks" => callbacks,
        "instances" => Dict{String,Any}(
            "inline" => count(i -> i.source == "inline", instances),
            "instantiate" => count(i -> i.source == "instantiate", instances),
            "not_public" => count(f -> startswith(f, "inline|") , plan.wanted_fns) - count(f -> f.source == "inline", plan.fns)),
        "instances_refused" => Dict{String,Any}(k => v for (k, v) in fn_refused if any(f -> f.expr == k && f.source == "instantiate", plan.fns)),
        "skipped" => skipped,
        "foreign_definitions" => foreign,
        "toolchain" => Dict{String,Any}("rustc" => tc.release, "host" => tc.host, "llvm" => tc.llvm,
                                        "identity" => tc.identity),
        "build" => Dict{String,Any}("opt_level" => rc.opt_level, "features" => rc.features,
                                    "default_features" => rc.default_features,
                                    "target_triple" => tc.host),
        "timestamp" => string(now()),
    )
end

"""
    report(io, metadata)

What a build found, in the build's own lines: how many functions, how many
can be called as they are, and each refused or skipped one's reason, grouped
so a reason shared by many functions is one line.
"""
function report(io::IO, metadata::AbstractDict)
    functions = metadata["functions"]
    refused = [f for f in functions if f["refusal"] !== nothing]
    nstatics = length(get(metadata, "statics", Any[]))
    println(io, "  rust: $(length(functions)) functions, $(length(functions) - length(refused)) callable, " *
                "$(length(metadata["types"])) types", nstatics == 0 ? "" : ", $nstatics statics")
    inst = get(metadata, "instances", Dict{String,Any}())
    if get(inst, "inline", 0) + get(inst, "instantiate", 0) + get(inst, "not_public", 0) > 0
        println(io, "  no symbol of their own: $(inst["inline"]) inline functions and $(inst["instantiate"]) asked for in ",
                "[rust] instantiate, compiled into the library; $(inst["not_public"]) more are not public")
        _report_group(io, "not instantiated", Tuple{String,String}[(String(k), String(v)) for (k, v) in get(metadata, "instances_refused", Dict())])
    end
    glue = get(metadata, "glue", Dict{String,Any}())
    if !isempty(glue) || !isempty(get(metadata, "glue_refused", Dict{String,Any}()))
        owned = count(g -> g["owns"] === true, values(glue))
        println(io, "  ownership: $owned types Rust must drop, $(length(glue) - owned) that are plain data")
        _report_group(io, "no drop function", Tuple{String,String}[(String(k), String(v)) for (k, v) in metadata["glue_refused"]])
    end
    _report_group(io, "refused", Tuple{String,String}[(String(f["path"]), String(f["refusal"])) for f in refused])
    _report_group(io, "skipped", Tuple{String,String}[(String(get(s, "path", s["symbol"])), String(s["reason"]))
                                                      for s in metadata["skipped"]])
    return nothing
end

function _report_group(io::IO, label::String, rows::Vector{Tuple{String,String}})
    isempty(rows) && return
    by_reason = Dict{String,Vector{String}}()
    for (name, reason) in rows
        push!(get!(by_reason, reason, String[]), name)
    end
    for reason in sort!(collect(keys(by_reason)))
        names = sort!(by_reason[reason])
        shown = join(names[1:min(end, 3)], ", ") * (length(names) > 3 ? ", and $(length(names) - 3) more" : "")
        println(io, "  $label ($(length(names))): $reason: $shown")
    end
end

end # module RustBuild
