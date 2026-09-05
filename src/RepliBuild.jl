#!/usr/bin/env julia

# Focus: ABI Compiler/Generator

module RepliBuild

using TOML
using JSON
# Re-exposed as `RepliBuild.SHA` for generated wrappers: their build-identity
# check hashes the library beside them, and a wrapper vendored into a consumer
# package cannot `using SHA` unless that package declares it as a dep. Routing
# through RepliBuild — which the wrapper already imports — keeps the consumer's
# Project.toml untouched.
import SHA

# Version — DERIVED from Project.toml, never a second literal.
#
# These were two independent literals that nothing reconciled, so a release
# could ship with `Project.toml` saying one version and `RepliBuild.VERSION`
# saying another. That is not cosmetic: `VERSION` feeds `_generator_fingerprint`
# (which gates the registry build cache, so a stale-codegen wrapper would be
# served rather than rebuilt) and the `BUILD_GENERATOR` stamped into every
# emitted wrapper. Both would have been confidently wrong.
#
# `include_dependency` makes an edit to Project.toml invalidate this module's
# precompile cache — without it the old version stays baked into the `.ji`.
const _PROJECT_TOML = joinpath(dirname(@__DIR__), "Project.toml")
include_dependency(_PROJECT_TOML)
const VERSION = VersionNumber(TOML.parsefile(_PROJECT_TOML)["version"])

# Stable path constants — modules use these instead of @__DIR__ so file moves don't break paths
const PROJECT_ROOT = dirname(@__DIR__)
const SRC_DIR = @__DIR__

"""
    INTERNAL_TYPE_BLOCKLIST

Compiler and libc internals that leak through DWARF and must never reach a
generated artifact — not a wrapper's type declarations, not its export list,
and not the thunks the JIT compiles.

Defined at package level rather than inside `Wrapper` because `IRGen` loads
FIRST and needs the same screen. It was Wrapper-local when only the wrapper
screened types, and the array-view thunk producer — added later, in IRGen —
consequently emitted accessors for `_IO_FILE` members that no wrapper would ever
declare a type for, let alone call.
"""
const INTERNAL_TYPE_BLOCKLIST = Set([
    "__va_list_tag", "__mbstate_t", "__loadu_pd", "__storeu_pd",
    "__loadu_ps", "__storeu_ps", "__loadu_si128", "__storeu_si128",
    "_va_list_tag", "_mbstate_t", "_loadu_pd", "_storeu_pd",
    "_loadu_ps", "_storeu_ps",
    "ldiv_t", "lldiv_t", "div_t", "max_align_t", "imaxdiv_t",
    "_IO_FILE", "_IO_marker", "_IO_codecvt", "_IO_wide_data",
])

# RepliBuild targets Linux (ELF/.so) and Windows (PE/.dll, x86_64-w64-windows-gnu).
#
# The gate admits a kernel only once the pipeline's format assumptions hold for
# it. What made Linux special was never the kernel — it was four assumptions:
# a shared library the loader accepts, DWARF debug info, a GNU-dialect DWARF
# dumper, and an LLVM layout to compile against. Windows/mingw satisfies all
# four: clang emits DWARF into PE, and GNU `objdump` reads PE while printing
# byte-identically to `readelf` (both print from binutils' dwarf.c), so the
# readelf-format parser in Compiler.jl needs no second dialect.
#
# macOS stays unsupported for a real reason, not an untested one: the AAPCS64
# classifier is unbuilt (JLCSPasses.cpp carries SysV and Win64 only, and
# `#error`s on non-x86-64), so an arm64 Mach-O host has no ABI rules to run.
#
# The gate stays because the alternative is four confusing failures deep in the
# pipeline instead of one clear message at load.
function __init__()
    if !(Sys.islinux() || Sys.iswindows())
        error("RepliBuild supports Linux and Windows (detected $(Sys.KERNEL)). " *
              "macOS is unsupported: the AAPCS64 ABI classifier is not built, " *
              "so Mach-O/arm64 has no struct-passing rules to apply.")
    end
end

# ============================================================================
# LOAD MODULES
# ============================================================================

# Builder: config, environment, compilation, metadata extraction
include("Builder.jl")

# IRGen: MLIR bindings, IR generation, JIT execution
include("IRGen.jl")

# Wrapper: Julia binding generation
include("Wrapper.jl")
include("Wrapper/Cpp/STLWrappers.jl")

# ThunkBuilder: bridge between Builder and IRGen (needs Wrapper.is_c_lto_safe)
include("Builder/ThunkBuilder.jl")

# Debug: static inspection of what the JIT emitted. Depends on nothing above —
# it reads artifacts off disk — so it loads last and can be used against a
# package this process never built.
include("Debug/Debug.jl")

# Import submodules for internal use
using .LLVMEnvironment
using .ConfigurationManager
using .BuildBridge
using .DependencyResolver
using .ASTWalker
using .Discovery
using .ClangJLBridge
using .Compiler
using .DWARFParser
using .MLIRNative
using .JLCSIRGenerator
using .DAGDiff
using .JITManager
using .Wrapper
using .STLWrappers
using .ThunkBuilder
using .EnvironmentDoctor
using .PackageRegistry
# NOT `using .Debug` — nothing in core calls it, and it exports generic names
# (`walk`, `disassemble`, `thunks`, `dwarf`) that would sit in this namespace
# waiting to collide with a future addition. Reached as `RepliBuild.Debug.x`.

# ============================================================================
# EXPORTS
# ============================================================================

# --- Core Build Orchestration ---
export build, wrap, info, discover, clean, ingest

# --- Environment & Registry ---
export check_environment
export use, register, unregister, list_registry, search, scaffold_package

# --- Submodules (direct access) ---
export Compiler, Wrapper, Discovery, ConfigurationManager, DWARFParser,
       JLCSIRGenerator, DAGDiff, MLIRNative, STLWrappers,
       LLVMEnvironment, BuildBridge, ASTWalker, JITManager, ClangJLBridge,
       DependencyResolver, EnvironmentDoctor, PackageRegistry, Debug

# --- Everything else is INTERNAL and reachable qualified ---
#
# The public API is the three blocks above: the six pipeline verbs, the registry,
# and the submodules. Everything under those submodules stays reachable as
# `RepliBuild.Compiler.compile_to_ir`, `RepliBuild.parse_vtables`, and so on --
# removing an `export` does not remove a binding, it only stops `using RepliBuild`
# from dumping the name into the caller's namespace.
#
# This block used to publish 221 names, ~195 of them implementation detail:
# every DWARF parser entry point, the whole MLIR lifecycle, the IR-gen thunk
# emitters, BuildBridge's command runners, the config accessors. A generated
# wrapper is a namespace the LIBRARY populates, and the same logic applies one
# level up -- `execute`, `capture`, `lookup`, `search`, `info`, `build` are
# plausible names in a consumer's own code, and exporting them means
# `using RepliBuild` silently competes for them.
#
# If something here turns out to be genuinely public, export it deliberately and
# document it. The default is qualified access.

"""
    check_environment(; verbose=true, throw_on_error=false) -> ToolchainStatus

Run environment diagnostics to verify LLVM 21+, MLIR, CMake, and other toolchain requirements.

Prints a colorful report showing which tools are found, their versions, and installation
instructions for anything missing. Use `throw_on_error=true` to abort on missing requirements.

# Example
```julia
status = RepliBuild.check_environment()
status.ready          # true if Tier 3 (ccall) builds will work
status.tier2_ready    # true if MLIR JIT tier is also available
```
"""
function check_environment(; verbose::Bool=true, throw_on_error::Bool=false)
    return EnvironmentDoctor.check_environment(verbose=verbose, throw_on_error=throw_on_error)
end

"""
    scaffold_package(name::String; path::String=".") -> String

Generate a standardized Julia package for distributing RepliBuild wrappers.

Creates a complete package with Project.toml, replibuild.toml, source stub,
deps/build.jl hook, and test skeleton. Edit the replibuild.toml to point at
your C/C++ source, then `Pkg.build()` compiles and wraps automatically.

# Example
```julia
RepliBuild.scaffold_package("MyEigenWrapper")
```
"""
function scaffold_package(name::String; path::String=".", from_registry::Bool=true)
    return PackageRegistry.scaffold_package(name; path=path, from_registry=from_registry)
end

"""
    use(name::String; force_rebuild=false, verbose=true) -> Module

Load a wrapper by **local registry** name. Resolves dependencies, checks
environment, builds if needed, and returns the loaded Julia module.

The name must already be registered (`discover` does this; otherwise
`register(toml)`). There is no Hub fallback — a missing name errors.

# Example
```julia
RepliBuild.register("path/to/replibuild.toml")   # or discover(), which registers
M = RepliBuild.use("myproject")                  # [project].name
```
"""
function use(name::String; force_rebuild::Bool=false, verbose::Bool=true)
    return PackageRegistry.use(name; force_rebuild=force_rebuild, verbose=verbose)
end

"""
    register(toml_path::String; name="", verified=false) -> RegistryEntry

Hash and store a replibuild.toml in the global registry (~/.replibuild/registry/).
Name is inferred from [project].name if not provided. Called automatically by `discover()`.
"""
function register(toml_path::String; name::String="", verified::Bool=false)
    return PackageRegistry.register(toml_path; name=name, verified=verified)
end

"""
    unregister(name::String)

Remove a package from the global registry.
"""
function unregister(name::String)
    PackageRegistry.unregister(name)
end

"""
    list_registry()

Print all registered packages in the global RepliBuild registry.
"""
function list_registry()
    PackageRegistry.list_registry()
end

"""
    search(query::String="")

Browse the RepliBuild Hub catalog by name, description, tags, or language.
Does not register or install anything — `use` only sees the local registry.

```julia
RepliBuild.search()           # list catalog names
RepliBuild.search("json")     # filter by keyword
```
"""
function search(query::String="")
    PackageRegistry.search(query)
end

# ============================================================================
# PUBLIC API - Build Orchestration
# ============================================================================

"""
    discover(target_dir="."; force=false, build=false, wrap=false) -> String

Scan C++ project and generate replibuild.toml configuration file.

**This is the entry point for new projects.** Run this first to set up RepliBuild.

# Arguments
- `target_dir`: Project directory to scan (default: current directory)
- `force`: Force rediscovery even if replibuild.toml exists (default: false)
- `build`: Automatically run build() after discovery (default: false)
- `wrap`: Automatically run wrap() after build (requires build=true, default: false)

# Returns
Path to generated `replibuild.toml` file

# Workflow

## Basic workflow (step-by-step):
```julia
# 1. Discover and create config
toml_path = RepliBuild.discover()

# 2. Build the library
RepliBuild.build(toml_path)

# 3. Generate Julia wrappers
RepliBuild.wrap(toml_path)
```

## Chained workflow (automated):
```julia
# Discover → Build → Wrap (all at once)
toml_path = RepliBuild.discover(build=true, wrap=true)

# Or just discover and build
toml_path = RepliBuild.discover(build=true)
```

# Examples
```julia
# Discover current directory
RepliBuild.discover()

# Discover another directory
RepliBuild.discover("path/to/cpp/project")

# Force regenerate config
RepliBuild.discover(force=true)

# Full automated pipeline
RepliBuild.discover(build=true, wrap=true)
```
"""
function discover(path::String="."; force::Bool=false, build::Bool=false, wrap::Bool=false)
    result = Discovery.discover(path, force=force, build=build, wrap=wrap)
    return result
end


"""
    ingest(library_path; headers=String[], extra_link_libs=String[],
                         name="", project_dir=".", language=:c, build=false, wrap=false) -> String

**EXPERIMENTAL.** Scaffold a `replibuild.toml` for **ingest mode**: wrap a pre-built
`.so` (built by upstream's own build system) without recompiling. RepliBuild only runs
DWARF metadata extraction + wrapper generation; the library must be built with `-g`.

Support matrix — the maintained, flagship path is the **source-build pipeline**
(`discover`/`build`/`wrap`), where RepliBuild's own version-matched compilation
guarantees the DWARF it consumes:

- `language = :c` — works for plain-C ABIs (Tier-3 `ccall` only, no bitcode/thunks),
  but is best-effort: upstream's compiler and debug-info settings are outside
  RepliBuild's control, so extraction quality varies. Prefer the source build when
  the sources compile under one flag set.
- `language = :cpp` — **NOT supported.** The C++ ABI surface (classes, methods,
  templates, virtual dispatch) requires the MLIR dialect to marshal calls and
  generate thunks, which only the source-build pipeline produces. Ingesting a C++
  library can at best expose its `extern "C"` surface; the generated wrapper for
  the C++ API proper is unusable. If the library ships a C API variant, ingest
  that with `language = :c` instead.

This is the fallback for C libraries with elaborate build systems (autotools, CMake
with code generators, configure scripts) that RepliBuild's source-build pipeline can't
reproduce.

# Arguments
- `library_path`: path to the pre-built `.so` / `.dylib` / `.dll`
- `headers`: header search dirs for type extraction (recommended)
- `extra_link_libs`: additional `-l` libraries the wrapper needs at load time
- `name`: project name (default: derived from library basename)
- `project_dir`: where to write replibuild.toml (default: cwd)
- `language`: `:c` or `:cpp` — drives wrapper generator selection
- `build`: also run `build()` after scaffolding
- `wrap`: also run `wrap()` (requires `build=true`)

# Returns
Path to the generated `replibuild.toml`.

# Example
```julia
toml = RepliBuild.ingest("/usr/lib/libsqlite3.so",
                         headers=["/usr/include"],
                         name="sqlite_ingest",
                         build=true, wrap=true)
```
"""
function ingest(library_path::String;
                headers::Vector{String}=String[],
                extra_link_libs::Vector{String}=String[],
                name::String="",
                project_dir::String=".",
                language::Symbol=:c,
                build::Bool=false,
                wrap::Bool=false,
                register::Bool=true)

    language in (:c, :cpp) ||
        error("ingest: language must be :c or :cpp, got :$(language)")

    # Ingest is experimental and C-focused; the C++ ABI surface needs the
    # source-build pipeline (dialect marshalling + thunks). Warn loudly here
    # AND in ingest_library (hand-written [ingest] TOMLs bypass this function).
    if language == :cpp
        @warn """
        ingest(language=:cpp): the C++ API surface of an ingested binary is NOT supported.
        Classes, methods, templates, and virtual dispatch need the MLIR dialect thunks that
        only the source-build pipeline (discover/build/wrap) generates. At best the library's
        extern \"C\" surface will be usable; wrappers for the C++ API proper will not be.
        If the library ships a C API variant, ingest that with language=:c instead — or build
        from source. (See the Ingest section of the docs; tracking: issue #4.)"""
    else
        @info "ingest is experimental: best-effort C wrapping of a binary RepliBuild didn't build (Tier-3 ccall only). The maintained path is the source-build pipeline (discover/build/wrap)."
    end

    isfile(library_path) || error("Library not found: $library_path")
    library_abs = abspath(library_path)

    project_dir = abspath(project_dir)
    mkpath(project_dir)

    # Derive a project name from the library if the user didn't pick one.
    if isempty(name)
        base = basename(library_abs)
        # Strip lib prefix and shared-object suffixes.
        base = replace(base, r"\.so(\.\d+)*$" => "", r"\.dylib$" => "", r"\.dll$" => "")
        base = startswith(base, "lib") ? base[4:end] : base
        # Sanitize for use as a project/module name.
        name = isempty(base) ? "ingested_lib" : replace(base, r"[^A-Za-z0-9_]" => "_")
    end

    # Auto-derive a CamelCase module name (mirrors get_module_name's logic).
    parts = split(replace(name, r"[^A-Za-z0-9]" => "_"), "_")
    module_name = join([uppercasefirst(p) for p in parts if !isempty(p)], "")
    isempty(module_name) && (module_name = "IngestedLib")

    toml_path = joinpath(project_dir, "replibuild.toml")

    # Serialize through the stdlib printer, one section at a time so the emitted
    # order stays project → ingest → wrap (a bare `TOML.print` of one dict would
    # order by hash, and this file is meant to be read and hand-edited).
    #
    # This was hand-rolled interpolation — `println(io, "library = \"$library_abs\"")`
    # and friends — which is a SECOND, disagreeing derivation of what
    # `save_config` already does correctly, and it emitted invalid TOML for any
    # path containing a backslash. On Windows that is every absolute path:
    # `C:\Users\...` reaches the parser as the escape `\U`, which fails as
    # "invalid unicode scalar" rather than as anything mentioning quoting. The
    # library path, the header paths and the project name are all user-supplied
    # strings, so none of them can be assumed escape-free on any platform.
    open(toml_path, "w") do io
        println(io, "# Generated by RepliBuild.ingest — BYOB mode")
        TOML.print(io, Dict("project" => Dict("name" => name)))
        println(io)
        TOML.print(io, Dict("ingest" => Dict(
            "library"         => library_abs,
            "headers"         => collect(String, headers),
            "extra_link_libs" => collect(String, extra_link_libs),
        )))
        println(io)
        TOML.print(io, Dict("wrap" => Dict(
            "language"    => string(language),
            "module_name" => module_name,
        )))
    end

    # Auto-register, mirroring discover()'s behavior so `RepliBuild.use(name)` works.
    if register
        try
            PackageRegistry.register(toml_path)
        catch e
            @debug "Auto-registration skipped" exception=e
        end
    end

    if build
        build_func = getfield(@__MODULE__, :build)
        build_func(toml_path)
        if wrap
            wrap_func = getfield(@__MODULE__, :wrap)
            wrap_func(toml_path)
        end
    end

    return toml_path
end


"""
    build(toml_path="replibuild.toml"; clean=false)

Compile C++ project → library (.so/.dylib/.dll)

**What it does:**
1. Compiles your C++ code to LLVM IR
2. Links and optimizes IR
3. Generates library file
4. Extracts metadata (DWARF + symbols) for wrapping

**What it does NOT do:**
- Does NOT generate Julia wrappers (use `wrap()` for that)

# Arguments
- `toml_path`: Path to replibuild.toml configuration file (default: "replibuild.toml")
- `clean`: Clean before building (default: false)

# Returns
Library path (String)

# Examples
```julia
# Build using replibuild.toml in current directory
RepliBuild.build()

# Build with specific config file
RepliBuild.build("path/to/replibuild.toml")

# Clean build
RepliBuild.build(clean=true)

# Then generate Julia wrappers:
RepliBuild.wrap("replibuild.toml")
```
"""
function build(toml_path::String="replibuild.toml"; clean::Bool=false)

    # Validate environment before attempting build
    env_status = EnvironmentDoctor.check_environment(verbose=false)
    if !env_status.ready
        EnvironmentDoctor.check_environment(verbose=true, throw_on_error=true)
    end

    # Resolve absolute path to TOML file
    toml_path = abspath(toml_path)

    if !isfile(toml_path)
        error("Configuration file not found: $toml_path\nRun RepliBuild.Discovery.discover() first!")
    end

    project_dir = dirname(toml_path)
    original_dir = pwd()

    try
        cd(project_dir)

        if clean
            clean_internal(project_dir)
        end

        # Load config
        config = ConfigurationManager.load_config(toml_path)
        config = DependencyResolver.resolve_dependencies(config)

        # Ingest mode: skip the whole compile pipeline, run metadata extraction over a
        # pre-built .so. Tier 3 (ccall) only — no LTO bitcode, no AOT thunks.
        if config.ingest !== nothing
            return abspath(Compiler.ingest_library(config))
        end

        # Compile the project (C++ → IR → library + metadata)
        library_path = Compiler.compile_project(config)

        if library_path === nothing
            error("Compilation produced no library. Check that source files exist and are listed in the config.")
        end

        # Build MLIR AOT thunks for C++ when enabled (C goes through ccall+LTO only,
        # no thunks — packed/union returns use explicit-sret ccall).
        if config.compile.aot_thunks && config.binary.type != :executable && config.wrap.language != :c
            ThunkBuilder.build_aot_thunks(config, library_path)
        end

        return abspath(library_path)

    finally
        cd(original_dir)
    end
end

"""
    wrap(toml_path="replibuild.toml"; headers=String[])

Generate Julia wrapper from compiled library

**What it does:**
1. Loads metadata from build (DWARF + symbols)
2. Generates Julia module with ccall wrappers
3. Creates type definitions from C++ structs
4. Saves to julia/ directory

**Requirements:**
- Must run `build()` first
- Metadata must exist in julia/compilation_metadata.json

# Arguments
- `toml_path`: Path to replibuild.toml configuration file (default: "replibuild.toml")
- `headers`: C++ headers for advanced wrapping (optional)

# Returns
Path to generated Julia wrapper file

# Examples
```julia
# Generate wrapper using replibuild.toml in current directory
RepliBuild.wrap()

# Generate wrapper with specific config file
RepliBuild.wrap("path/to/replibuild.toml")

# With headers for better type info
RepliBuild.wrap("replibuild.toml", headers=["mylib.h"])
```
"""
function wrap(toml_path::String="replibuild.toml"; headers::Vector{String}=String[])

    # Resolve absolute path to TOML file
    toml_path = abspath(toml_path)

    if !isfile(toml_path)
        error("Configuration file not found: $toml_path\nRun RepliBuild.Discovery.discover() first!")
    end

    project_dir = dirname(toml_path)
    original_dir = pwd()

    try
        cd(project_dir)

        # Load config
        config = ConfigurationManager.load_config(toml_path)
        config = DependencyResolver.resolve_dependencies(config)

        # Find library
        output_dir = ConfigurationManager.get_output_path(config)
        lib_name = ConfigurationManager.get_library_name(config)
        library_path = joinpath(output_dir, lib_name)

        if !isfile(library_path)
            error("Library not found: $library_path\nRun RepliBuild.build(\"$toml_path\") first!")
        end

        # Check for metadata
        metadata_path = joinpath(output_dir, "compilation_metadata.json")
        if !isfile(metadata_path)
            @warn "No metadata found. Wrapper quality may be limited."
        end


        # Generate wrapper
        wrapper_path = Wrapper.wrap_library(
            config,
            library_path,
            headers=headers,
            generate_tests=false,
            generate_docs=true
        )


        return abspath(wrapper_path)

    finally
        cd(original_dir)
    end
end

"""
    clean(toml_path="replibuild.toml")

Remove build artifacts (build/, julia/, caches)

# Arguments
- `toml_path`: Path to replibuild.toml configuration file (default: "replibuild.toml")

# Examples
```julia
# Clean using replibuild.toml in current directory
RepliBuild.clean()

# Clean specific project
RepliBuild.clean("path/to/replibuild.toml")
```
"""
function clean(toml_path::String="replibuild.toml")
    # Resolve absolute path to TOML file
    toml_path = abspath(toml_path)

    if !isfile(toml_path)
        error("Configuration file not found: $toml_path")
    end

    project_dir = dirname(toml_path)
    clean_internal(project_dir)
end

# Internal clean function
function clean_internal(path::String)
    # .debug holds the generated MLIR the JIT'd thunks carry in their DWARF, so
    # gdb can show source when you break in one. Regenerated at the next JIT
    # init from the module text, so removing it costs nothing but a rebuild.
    dirs_to_remove = ["build", "julia", ".replibuild_cache", ".debug"]

    removed = String[]
    for dir in dirs_to_remove
        dir_path = joinpath(path, dir)
        if isdir(dir_path)
            rm(dir_path, recursive=true, force=true)
            push!(removed, dir)
        end
    end
    if !isempty(removed)
        println("  clean: $(join(removed, ", "))")
    end
end

"""
    info(toml_path="replibuild.toml")

Show project status (config, library, wrapper)

# Arguments
- `toml_path`: Path to replibuild.toml configuration file (default: "replibuild.toml")

# Examples
```julia
# Show info for current directory
RepliBuild.info()

# Show info for specific project
RepliBuild.info("path/to/replibuild.toml")
```
"""
function info(toml_path::String="replibuild.toml")
    toml_path = abspath(toml_path)

    if !isfile(toml_path)
        println("No replibuild.toml at: $toml_path")
        return
    end

    project_dir = dirname(toml_path)
    data = TOML.parsefile(toml_path)
    project = get(data, "project", Dict())

    println("RepliBuild | $(get(project, "name", "unnamed"))")

    julia_dir = joinpath(project_dir, "julia")
    if isdir(julia_dir)
        lib_files = filter(f -> endswith(f, ".so") || endswith(f, ".dylib") || endswith(f, ".dll"),
                          readdir(julia_dir))
        if !isempty(lib_files)
            println("  library: $(lib_files[1])")
        else
            println("  library: not built")
        end

        jl_files = filter(f -> endswith(f, ".jl"), readdir(julia_dir))
        if !isempty(jl_files)
            println("  wrapper: $(jl_files[1])")
        else
            println("  wrapper: not generated")
        end

        lto_bc_files = filter(f -> endswith(f, "_lto.bc") && !contains(f, "thunks"), readdir(julia_dir))
        if !isempty(lto_bc_files)
            println("  lto_ir:  $(lto_bc_files[1])")
        end

        aot_bc_files = filter(f -> endswith(f, "_thunks_lto.bc"), readdir(julia_dir))
        if !isempty(aot_bc_files)
            println("  aot_ir:  $(aot_bc_files[1])")
        end

        aot_lib_files = filter(f -> contains(f, "_thunks") && (endswith(f, ".so") || endswith(f, ".dylib") || endswith(f, ".dll")), readdir(julia_dir))
        if !isempty(aot_lib_files)
            println("  aot_lib: $(aot_lib_files[1])")
        end
    else
        println("  not built yet")
    end

end

# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

end # module RepliBuild
