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
               config.rust.features, config.rust.default_features, config.rust.target_dir)

"The shape of a Rust project's `compilation_metadata.json`. Bump it with any change to what a row means."
const METADATA_VERSION = "rust-2"

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

/// Run the drop code of the value at `p`. 0: done. The largest integer: that code panicked.
#[allow(dead_code)]
unsafe fn __rb_drop<T>(p: *mut T) -> usize {
    match catch_unwind(AssertUnwindSafe(|| core::ptr::drop_in_place(p))) {
        Ok(()) => 0,
        Err(_) => usize::MAX,
    }
}

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
    (text, _) = RustGlue.glue_text(plan.rows)
    path = _glue_path(rc)
    mkpath(dirname(path))
    (isfile(path) && read(path, String) == text) || write(path, text)
    open(_plan_path(rc), "w") do io
        JSON.print(io, Dict{String,Any}(
            "identity" => String(identity),
            "file_sha" => bytes2hex(SHA.sha256(text)),
            "wanted" => plan.wanted,
            "rows" => [Dict{String,Any}("n" => r.n, "key" => r.key, "spelling" => r.spelling, "vec" => r.vec)
                       for r in plan.rows],
            "refused" => plan.refused), 2)
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
                              RustGlue.Row[RustGlue.Row(Int(r["n"]), String(r["key"]), String(r["spelling"]), r["vec"] === true)
                                           for r in d["rows"]],
                              Dict{String,String}(String(k) => String(v) for (k, v) in d["refused"])) : nothing
        catch
            nothing
        end
        plan === nothing || return plan
    end
    _write_glue(rc, RustGlue.Plan(); identity)
    return RustGlue.Plan()
end

"""
    _plan_glue!(rc, crate, need, types, identity, build) -> (plan, messages)

Write `rb_glue.rs` for the types `need` and build the library with it.
`build()` runs cargo on the root crate and returns `(messages, stderr, exit
code)`. Three steps, each a compile of the root crate alone (the wrapped
crate is already built):

1. the probe: every candidate path of every type, one `use` line each. The
   compiler names the lines that do not resolve from outside the crate;
2. the lines, each type spelled with the first candidate that resolved;
3. again without any line the compiler refused, until it builds.

A type that ends with no line is in the plan's `refused`, with the reason.
Returns the plan the library was built with and that build's messages.
"""
function _plan_glue!(rc::RustConfig, crate::Crate, need::Vector{String}, types::AbstractDict,
                     identity::AbstractString, build)
    refused = Dict{String,String}()
    names = Dict{String,String}()
    for key in need
        why = RustGlue.unnameable(key)
        why === nothing ? (names[key] = RustGlue.source_name(key)) : (refused[key] = why)
    end
    all_paths = sort!(unique!(reduce(vcat, [RustGlue.paths(names[k]) for k in sort!(collect(keys(names)))]; init = String[])))
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
        haskey(names, key) || continue
        spelling = RustGlue.spell(key, chosen)
        if spelling === nothing
            p = first(p for p in RustGlue.paths(names[key]) if !haskey(chosen, p))
            refused[key] = "no path names `$p` from outside the crate ($(get(unresolved, p, "it did not resolve")))"
            continue
        end
        push!(rows, RustGlue.Row(i, key, spelling, RustGlue.vec_element(types, key) !== nothing))
    end
    for round in 1:8
        plan = RustGlue.Plan(need, rows, refused)
        _write_glue(rc, plan; identity)
        (messages, diagnostics, code) = build()
        code == 0 && return (plan, messages)
        (bad, _) = RustGlue.error_lines(messages)
        (_, line_of) = RustGlue.glue_text(rows)
        dropped = RustGlue.Row[r for r in rows if haskey(bad, line_of[r.n])]
        (isempty(dropped) || round == 8) && fail("with its per-type lines", messages, diagnostics)
        for r in dropped
            refused[r.key] = "the line that drops it does not compile: $(bad[line_of[r.n]])"
        end
        rows = RustGlue.Row[r for r in rows if !haskey(bad, line_of[r.n])]
    end
    error("unreachable")
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
# by content, the toolchain, and the code that reads them.
function _metadata_key(library::AbstractString, ir_files::Vector{String}, tc::Toolchain, crate::Crate,
                       generator::AbstractString)::String
    ctx = SHA.SHA256_CTX()
    Compiler._key_feed!(ctx, METADATA_VERSION)
    Compiler._key_feed!(ctx, generator)
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
    if code != 0 && !isempty(plan.rows)
        # The per-type lines were written for the crate as it was. It changed
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
    key = _metadata_key(artifact.library, [ir], tc, crate, generator)
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

    metadata = extract_metadata(config, crate, tc, library, [ir]; plan)
    need = RustGlue.wanted(metadata["functions"], metadata["types"])
    if need != plan.wanted
        # What this build's functions move is not what `rb_glue.rs` was
        # written for (on a first build, nothing): write it, build the root
        # crate again, and read the library that has it.
        (plan, messages) = _plan_glue!(rc, crate, need, metadata["types"], identity, build)
        artifact = find_artifacts(messages, crate)
        rm(library; force = true)
        cp(artifact.library, library)
        metadata = extract_metadata(config, crate, tc, library, [artifact.ir]; plan)
        key = _metadata_key(artifact.library, [artifact.ir], tc, crate, generator)
    end
    open(_metadata_path(config), "w") do io
        JSON.print(io, metadata, 2)
    end
    report(stdout, metadata)
    write(keyfile, key)
    println("  done: $(round(time() - start, digits = 2))s")
    return library
end

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
- `skipped`: exported definitions that were left out, each with its reason.

An empty surface is an error, as it is for a C library
(`verify_wrap_surface` R1): a `cdylib`, a crate with no `pub fn`, or symbol
names this cannot read all look like that, and a wrapper of nothing is not a
result.
"""
function extract_metadata(config::RepliBuildConfig, crate::Crate, tc::Toolchain,
                          library::AbstractString, ir_files::Vector{String};
                          plan::RustGlue.Plan = RustGlue.Plan())::Dict{String,Any}
    exported = Compiler._runtime_symbol_names(String(library))
    exported === nothing && error("the dynamic symbol table of $library could not be read")
    defs = RustABI.definitions(ir_files)
    aliases = RustABI.aliases(ir_files)
    (symbols, skipped, foreign) = RustABI.surface(defs, exported, crate.name; aliases)
    if isempty(symbols)
        error("The crate `$(crate.name)` has no function to wrap: its library exports $(length(exported)) " *
              "symbols and its IR defines $(length(defs)) functions, and none is an exported function of " *
              "the crate. A crate's API is its `pub fn`s outside generic code; generic functions exist " *
              "only where they are instantiated." *
              (isempty(skipped) ? "" : " $(length(skipped)) exported definitions were not read " *
                                       "(first: $(skipped[1]["symbol"]): $(skipped[1]["reason"]))."))
    end
    # An alias has its target's code, and often no DIE of its own.
    target = Dict(sym => RustABI.alias_target(sym, defs, aliases) for sym in symbols)
    wanted = union(Set(symbols), Set(values(target)), Set(RustGlue.drop_symbol(r.n) for r in plan.rows))
    (dwarf_functions, types) = _with_dump_lines(library) do lines
        units = RustDWARF.read_units(lines, wanted)
        RustDWARF.describe_functions(units, wanted)
    end
    paths = _demangle(vcat(symbols, String[String(r["symbol"]) for r in skipped]))
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
    sort!(functions; by = r -> (r["path"], r["symbol"]))
    (glue, glue_refused) = RustGlue.verify(plan, dwarf_functions, types, exported)
    root = RustABI.v0_root(symbols[1])
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
    println(io, "  rust: $(length(functions)) functions, $(length(functions) - length(refused)) callable, " *
                "$(length(metadata["types"])) types")
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
