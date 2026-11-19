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
#
# The first four reach every crate of the build, through
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
import ..PackageRegistry
import .._parse_json

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

# `[rust]` with its manifest where it is on this machine.
_located(config::RepliBuildConfig) =
    RustConfig(manifest_path(config), "", config.rust.package, config.rust.opt_level,
               config.rust.features, config.rust.default_features, config.rust.target_dir)

"The shape of a Rust project's `compilation_metadata.json`. Bump it with any change to what a row means."
const METADATA_VERSION = "rust-1"

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
function _cargo(tc::Toolchain, args::Vector{String}, dir::AbstractString)::Cmd
    cmd = Cmd(`$(tc.cargo) $args`; dir = String(dir))
    env = Dict{String,String}(String(k) => String(v) for (k, v) in ENV)
    env["CARGO_TERM_COLOR"] = "never"
    isempty(get(ENV, "REPLIBUILD_TOOL_RUSTC", "")) || (env["RUSTC"] = tc.rustc)
    # One source of rustc flags, read before any configuration file (`rustflags`).
    env["CARGO_ENCODED_RUSTFLAGS"] = join(rustflags(), '\x1f')
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

"The generated root crate's name: the library cargo builds is `lib<this>.<ext>`."
root_name(crate::Crate) = crate.name * "_rb"

"Where the generated root crate is: beside cargo's target directory, under `[paths] build`."
root_dir(rc::RustConfig) = dirname(rc.target_dir) * "/root"

"""
    root_files(rc, crate) -> Dict{String,String}

The generated root crate, file name to text: a `Cargo.toml` that builds a
`dylib` and depends on the wrapped crate by path, with the features `[rust]`
asks for, and a `lib.rs` whose one line makes the crate part of the library.
The empty `[workspace]` table keeps a workspace in a directory above from
claiming it.
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
          "extern crate $(crate.name);\n"
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

"""
    rustflags(; base = ENV) -> Vector{String}

What every rustc of the build is given: the user's own `RUSTFLAGS` (or
`CARGO_ENCODED_RUSTFLAGS`), then the flags this file's header lists. They go
to cargo as `CARGO_ENCODED_RUSTFLAGS`, which cargo reads before anything in a
configuration file, so what is built does not depend on the machine's cargo
configuration.
"""
function rustflags(; base = ENV)::Vector{String}
    user = if !isempty(get(base, "CARGO_ENCODED_RUSTFLAGS", ""))
        String[String(x) for x in split(base["CARGO_ENCODED_RUSTFLAGS"], '\x1f'; keepempty = false)]
    else
        String[String(x) for x in split(get(base, "RUSTFLAGS", ""))]
    end
    return vcat(user, ["--emit=llvm-ir,link", "-Csymbol-mangling-version=v0", "--cap-lints=warn"])
end

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
            "--target-dir", rc.target_dir, "--message-format=json-render-diagnostics"]
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
function _metadata_key(library::AbstractString, ir_files::Vector{String}, tc::Toolchain, crate::Crate)::String
    ctx = SHA.SHA256_CTX()
    Compiler._key_feed!(ctx, METADATA_VERSION)
    Compiler._key_feed!(ctx, PackageRegistry._generator_fingerprint())
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
    build_rust_project(config) -> String

`RepliBuild.build` for a `[rust]` project: write the root crate, build it
with cargo, copy its dynamic library to `[paths] output`, and write
`compilation_metadata.json` beside it. Returns the library's path.

cargo decides what is recompiled; it is always asked. The metadata is
extracted again only when the library, the IR, the toolchain or RepliBuild
itself changed (`rust_metadata.key`). The key is removed before an
extraction, so one that fails leaves nothing claiming success.
"""
function build_rust_project(config::RepliBuildConfig)::String
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
    # From the crate's own directory, like the toolchain question above: with
    # rustup, a `rust-toolchain.toml` there chooses the compiler.
    (messages, diagnostics, code) = _capture(_cargo(tc, cargo_args(rc, crate), dirname(crate.manifest)))
    code == BuildBridge.SPAWN_FAILED && error(_missing_tool_message("cargo", tc.cargo, code, diagnostics))
    code == 0 || error("cargo could not build the crate `$(crate.name)` (exit $code):\n$(strip(diagnostics))")
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
    key = _metadata_key(artifact.library, [ir], tc, crate)
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

    metadata = extract_metadata(config, crate, tc, library, [ir])
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
- `skipped`: exported definitions that were left out, each with its reason.

An empty surface is an error, as it is for a C library
(`verify_wrap_surface` R1): a `cdylib`, a crate with no `pub fn`, or symbol
names this cannot read all look like that, and a wrapper of nothing is not a
result.
"""
function extract_metadata(config::RepliBuildConfig, crate::Crate, tc::Toolchain,
                          library::AbstractString, ir_files::Vector{String})::Dict{String,Any}
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
    wanted = union(Set(symbols), Set(values(target)))
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
