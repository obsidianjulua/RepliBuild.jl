#!/usr/bin/env julia

# Focus: ABI Compiler/Generator

module RepliBuild

using TOML
using JSON

# Version
const VERSION = v"2.5.3"

# ============================================================================
# LOAD CORE MODULES
# ============================================================================

# Internal utilities (not exported)
include("LLVMEnvironment.jl")
include("ConfigurationManager.jl")
include("BuildBridge.jl")
include("DependencyResolver.jl")

# Core build system modules
include("ASTWalker.jl")
include("Discovery.jl")
include("ClangJLBridge.jl")
include("Compiler.jl")
include("DWARFParser.jl")
include("MLIRNative.jl")
include("JLCSIRGenerator.jl")
include("JITManager.jl")
include("Wrapper.jl")
include("STLWrappers.jl")

# Introspection module
include("Introspect.jl")

# Polish modules
include("EnvironmentDoctor.jl")
include("PackageRegistry.jl")

# Import submodules for internal use
using .LLVMEnvironment
using .ConfigurationManager
using .BuildBridge
using .DependencyResolver
using .ASTWalker
using .Discovery
using .ClangJLBridge
using .Compiler
using .Wrapper
using .DWARFParser
using .JLCSIRGenerator
using .MLIRNative
using .JITManager
using .STLWrappers
using .Introspect
using .EnvironmentDoctor
using .PackageRegistry

# ============================================================================
# EXPORTS
# ============================================================================

# --- Core Build Orchestration ---
export build, wrap, info, discover, clean

# --- Environment & Registry ---
export check_environment
export use, register, unregister, list_registry, search, scaffold_package

# --- Submodules (direct access) ---
export Compiler, Wrapper, Discovery, ConfigurationManager, DWARFParser,
       JLCSIRGenerator, MLIRNative, STLWrappers, Introspect,
       LLVMEnvironment, BuildBridge, ASTWalker, JITManager, ClangJLBridge,
       DependencyResolver, EnvironmentDoctor, PackageRegistry

# --- Configuration Types & Functions ---
export RepliBuildConfig,
       ProjectConfig, PathsConfig, DiscoveryConfig, CompileConfig,
       LinkConfig, BinaryConfig, WrapConfig, LLVMConfig,
       WorkflowConfig, CacheConfig, DependenciesConfig, DependencyItem, TypesConfig,
       load_config, save_config, create_default_config,
       merge_compile_flags, with_source_files, with_include_dirs, with_discovery_results,
       get_output_path, get_build_path, get_cache_path,
       get_library_name, get_module_name,
       is_stage_enabled, get_source_files, get_include_dirs, get_compile_flags,
       is_parallel_enabled, is_cache_enabled,
       validate_config, validate_config!,
       print_config

# --- Compiler Tooling ---
export compile_to_ir, link_optimize_ir, create_library, create_executable, compile_project,
       extract_compilation_metadata, save_compilation_metadata,
       compute_project_hash, is_project_cache_valid, save_project_hash,
       assemble_bitcode, sanitize_ir_for_julia,
       # Previously internal compiler utilities
       needs_recompile, compile_single_to_ir,
       generate_template_instantiations, generate_macro_shims,
       extract_stl_method_symbols, extract_symbols_from_binary,
       extract_mangled_name, extract_dwarf_return_types,
       extract_function_name, extract_class_name,
       dwarf_type_to_julia, get_type_size,
       cpp_to_julia_type, build_type_registry,
       infer_return_type, parse_parameters, parse_function_signatures

# --- DWARF & Binary Analysis ---
export parse_vtables, export_vtable_json,
       VirtualMethod, MemberInfo, ClassInfo, VtableInfo,
       # Previously internal DWARF utilities
       parse_dwarf_output, parse_dwarf_output_robust,
       parse_symbol_table, read_vtable_data

# --- LLVM Environment & Toolchain ---
export LLVMToolchain,
       get_toolchain, init_toolchain, with_llvm_env,
       get_tool, has_tool, get_library,
       get_include_flags, get_link_flags,
       run_tool, print_toolchain_info, verify_toolchain,
       get_jll_llvm_root, get_llvm_root, LLVM_JLL_AVAILABLE

# --- Build Bridge (command execution & tool discovery) ---
export run_command, execute, capture,
       find_executable, command_exists,
       analyze_compiler_error, compile_with_analysis,
       throw_compilation_error, execute_with_retry,
       get_llvm_version, get_compiler_info

# --- MLIR Native (full MLIR lifecycle) ---
export create_context, create_module, destroy_context, parse_module, clone_module,
       get_module_operation, print_module,
       get_function_op, get_function_type,
       get_num_inputs, get_input_type,
       is_integer, get_integer_width, is_f32, is_f64,
       lower_to_llvm,
       create_jit, destroy_jit, register_symbol, register_symbol_global, lookup,
       jit_invoke, invoke_safe,
       emit_llvmir, emit_object,
       check_library, test_dialect

# --- JLCS IR Generation ---
export generate_jlcs_ir, generate_mlir_module

# --- JIT Manager ---
export get_jit_thunk, ensure_jit_initialized, JITContext

# --- AST Walker & Dependency Analysis ---
export FileDependencies, DependencyGraph,
       build_dependency_graph,
       extract_includes_simple, extract_includes_clang,
       parse_source_structure,
       resolve_include_path,
       export_dependency_graph_json, load_dependency_graph_json,
       print_dependency_summary

# --- Wrapper & Type System ---
export wrap_library, wrap_basic, extract_symbols,
       TypeRegistry, SymbolInfo, ParamInfo,
       TypeStrictness, STRICT, WARN, PERMISSIVE,
       is_struct_like, is_enum_like, is_function_pointer_like

# --- Clang.jl Bridge ---
export generate_bindings_clangjl, extract_header_types,
       sanitize_module_name, add_replibuild_metadata,
       discover_headers, generate_from_config

# --- STL Wrappers ---
export CppVector, CppString, CppMap, CppUnorderedMap

# --- Introspection: Project ---
export project_artifacts, lto_ir, aot_ir, aot_symbols

# --- Introspection: Binary ---
export symbols, dwarf_info, disassemble, headers, dwarf_dump

# --- Introspection: Julia Code Analysis ---
export code_lowered, code_typed, code_llvm, code_native, code_warntype,
       analyze_type_stability, analyze_simd, analyze_allocations, analyze_inlining,
       compilation_pipeline

# --- Introspection: LLVM IR Tooling ---
export llvm_ir, optimize_ir, compare_optimization, run_passes, compile_to_asm,
       analyze_ir_structure, extract_function_names, compare_ir_files

# --- Introspection: Benchmarking ---
export benchmark, benchmark_suite, track_allocations,
       compare_benchmarks, fastest, slowest, is_significant, speedup

# --- Introspection: Data Export ---
export export_json, export_csv, export_dataset, to_json_dict, to_dataframe

# --- Introspection Types ---
export BenchmarkResult, TypeStabilityAnalysis, SIMDAnalysis, AllocationAnalysis,
       CompilationPipelineResult, OptimizationResult,
       CodeLoweredInfo, CodeTypedInfo, LLVMIRInfo, AssemblyInfo,
       DWARFInfo, HeaderInfo, FunctionInfo, StructInfo

# --- IR Gen (struct/function/STL thunk generation) ---
export generate_function_thunks, generate_stl_thunks,
       generate_struct_definitions, get_struct_type_string, get_struct_definition_string,
       is_struct_packed, get_julia_offsets,
       get_llvm_equivalent_type_string, get_llvm_aligned_type_string,
       map_cpp_type, get_llvm_signature

# --- Dependency Resolution ---
export resolve_dependencies

# --- Environment Doctor ---
export ToolchainStatus, ToolStatus

# --- Package Registry Types ---
export RegistryEntry, RegistryIndex

"""
    check_environment(; verbose=true, throw_on_error=false) -> ToolchainStatus

Run environment diagnostics to verify LLVM 21+, MLIR, CMake, and other toolchain requirements.

Prints a colorful report showing which tools are found, their versions, and installation
instructions for anything missing. Use `throw_on_error=true` to abort on missing requirements.

# Example
```julia
status = RepliBuild.check_environment()
status.ready          # true if Tier 1 (ccall) builds will work
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

Load a wrapper by registry name. Resolves dependencies, checks environment,
builds if needed, and returns the loaded Julia module.

# Example
```julia
Lua = RepliBuild.use("lua")
Lua.luaL_newstate()
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

Search the RepliBuild Hub for available packages. Matches against names,
descriptions, tags, and language. Call with no arguments to list everything.

```julia
RepliBuild.search()           # list all hub packages
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
    _build_c_thunks(config, library_path)

Generate and compile C sret thunk wrappers for functions with packed struct
or union return types.  Uses the same Clang that compiled the library, so
the resulting IR is version-compatible with Julia's internal LLVM — no
sanitizing or MLIR involvement needed.
"""
function _build_c_thunks(config, library_path)
    output_dir = ConfigurationManager.get_output_path(config)
    metadata_path = joinpath(output_dir, "compilation_metadata.json")

    if !isfile(metadata_path)
        @warn "Cannot build C thunks: metadata not found."
        return
    end

    println("  c-thunks: Scanning for unsafe returns...")
    start_time = time()

    metadata = JSON.parsefile(metadata_path)
    functions = get(metadata, "functions", [])
    dwarf_structs = get(metadata, "struct_definitions", Dict())

    # Find functions that need sret thunks
    unsafe_funcs = []
    for func in functions
        if !Wrapper.is_c_lto_safe(func, dwarf_structs)
            push!(unsafe_funcs, func)
        end
    end

    if isempty(unsafe_funcs)
        println("  c-thunks: All functions are ccall-safe, no thunks needed.")
        return
    end

    # Collect original headers for #include
    include_dirs = ConfigurationManager.get_include_dirs(config)
    source_files = ConfigurationManager.get_source_files(config)

    # Find header files to include (from include dirs)
    header_includes = String[]
    for inc_dir in include_dirs
        if isdir(inc_dir)
            for f in readdir(inc_dir)
                if endswith(f, ".h") || endswith(f, ".H")
                    push!(header_includes, "#include \"$f\"")
                end
            end
        end
    end

    # Generate C thunk source
    thunk_lines = String[]
    push!(thunk_lines, "/* Auto-generated sret thunks for packed/union returns */")
    push!(thunk_lines, "/* Compiled by the same Clang as the library — version-matched */")
    push!(thunk_lines, "")
    for inc in header_includes
        push!(thunk_lines, inc)
    end
    push!(thunk_lines, "")

    for func in unsafe_funcs
        mangled = func["mangled"]
        c_ret = get(func["return_type"], "c_type", "void")
        params = get(func, "parameters", [])

        # Build C parameter list
        c_params = String[]
        c_args = String[]
        push!(c_params, "$c_ret *__sret")  # leading sret pointer
        for (i, p) in enumerate(params)
            p_type = get(p, "c_type", "int")
            p_name = get(p, "name", "arg$i")
            push!(c_params, "$p_type $p_name")
            push!(c_args, p_name)
        end

        thunk_name = "_c_sret_$mangled"
        push!(thunk_lines, "void $thunk_name($(join(c_params, ", "))) {")
        push!(thunk_lines, "    *__sret = $mangled($(join(c_args, ", ")));")
        push!(thunk_lines, "}")
        push!(thunk_lines, "")
    end

    thunk_c_path = joinpath(output_dir, "_c_thunks.c")
    write(thunk_c_path, join(thunk_lines, "\n"))

    println("  c-thunks: Generated $(length(unsafe_funcs)) sret wrappers")

    # Compile to object
    thunks_obj = joinpath(output_dir, "_c_thunks.o")
    compile_args = ["-fPIC", "-c", "-O2", "-o", thunks_obj, thunk_c_path]
    for dir in include_dirs
        push!(compile_args, "-I$dir")
    end
    for (k, v) in config.compile.defines
        push!(compile_args, "-D$k=$v")
    end

    (output, exitcode) = BuildBridge.execute("clang", compile_args)
    if exitcode != 0
        @warn "Failed to compile C thunks" output
        return
    end

    # Link into companion shared library
    lib_name = basename(library_path)
    thunks_name = replace(lib_name, ".so" => "_thunks.so", ".dylib" => "_thunks.dylib", ".dll" => "_thunks.dll")
    thunks_so = joinpath(output_dir, thunks_name)
    lib_dir = dirname(abspath(library_path))

    link_args = ["-shared", "-fPIC", "-o", thunks_so, thunks_obj,
                 "-L", lib_dir, "-l:$lib_name", "-Wl,-rpath,$lib_dir"]
    (output, exitcode) = BuildBridge.execute("clang", link_args)
    if exitcode != 0
        @warn "Failed to link C thunks" output
        return
    end

    # If LTO enabled, also emit LLVM IR → bitcode (version-matched)
    if config.link.enable_lto
        thunks_lto_ll = joinpath(output_dir, replace(lib_name, ".so" => "_thunks_lto.ll", ".dylib" => "_thunks_lto.ll"))
        ir_args = ["-S", "-emit-llvm", "-fPIC", "-O2", "-o", thunks_lto_ll, thunk_c_path]
        for dir in include_dirs
            push!(ir_args, "-I$dir")
        end
        for (k, v) in config.compile.defines
            push!(ir_args, "-D$k=$v")
        end

        (output, exitcode) = BuildBridge.execute("clang", ir_args)
        if exitcode != 0
            @warn "Failed to emit C thunks LLVM IR" output
        else
            # Sanitize and assemble to bitcode
            lto_ir_text = read(thunks_lto_ll, String)
            lto_ir_text = Compiler.sanitize_ir_for_julia(lto_ir_text)
            write(thunks_lto_ll, lto_ir_text)

            thunks_bc_path = replace(thunks_lto_ll, ".ll" => ".bc")
            Compiler.assemble_bitcode(thunks_lto_ll, thunks_bc_path)
        end
    end

    elapsed = round(time() - start_time, digits=2)
    size_kb = round(filesize(thunks_so) / 1024, digits=1)
    println("  c-thunks: $thunks_name ($size_kb KB) in $(elapsed)s")

    # Cleanup
    rm(thunks_obj, force=true)
end

function _build_aot_thunks(config, library_path)
    output_dir = ConfigurationManager.get_output_path(config)
    metadata_path = joinpath(output_dir, "compilation_metadata.json")
    
    if !isfile(metadata_path)
        @warn "Cannot AOT compile thunks: metadata not found."
        return
    end
    
    println("  aot: Generating MLIR thunks...")
    start_time = time()
    
    vtable_info = DWARFParser.parse_vtables(library_path)
    metadata = JSON.parsefile(metadata_path)
    ir_source = JLCSIRGenerator.generate_jlcs_ir(vtable_info, metadata)
    
    ctx = MLIRNative.create_context()
    try
        mod = MLIRNative.parse_module(ctx, ir_source)
        if mod == C_NULL
            error("Failed to parse generated MLIR for AOT.")
        end
        
        if !MLIRNative.lower_to_llvm(mod)
            error("Failed to lower MLIR to LLVM for AOT.")
        end
        
        thunks_obj = joinpath(output_dir, "thunks.o")
        if !MLIRNative.emit_object(mod, thunks_obj)
            error("Failed to emit object file for AOT thunks.")
        end
        
        # Link into a companion shared library
        lib_name = basename(library_path)
        thunks_name = replace(lib_name, ".so" => "_thunks.so", ".dylib" => "_thunks.dylib", ".dll" => "_thunks.dll")
        thunks_so = joinpath(output_dir, thunks_name)

        # Link thunks against the main library so C function symbols resolve
        lib_dir = dirname(abspath(library_path))
        linker = config.wrap.language == :c ? "clang" : "clang++"
        link_args = ["-shared", "-fPIC", "-o", thunks_so, thunks_obj,
                     "-L", lib_dir, "-l:$lib_name", "-Wl,-rpath,$lib_dir"]
        (output, exitcode) = BuildBridge.execute(linker, link_args)
        if exitcode != 0
            error("Failed to link thunks.o: $output")
        end
        
        # Emit LTO text IR for AOT thunks if LTO is enabled
        if config.link.enable_lto
            thunks_lto_name = replace(lib_name, ".so" => "_thunks_lto.ll", ".dylib" => "_thunks_lto.ll", ".dll" => "_thunks_lto.ll")
            thunks_lto_path = joinpath(output_dir, thunks_lto_name)
            if MLIRNative.emit_llvmir(mod, thunks_lto_path)
                lto_ir_text = read(thunks_lto_path, String)
                lto_ir_text = Compiler.sanitize_ir_for_julia(lto_ir_text)
                write(thunks_lto_path, lto_ir_text)

                # Assemble to bitcode via Julia's libLLVM for version-matched bc
                thunks_bc_path = replace(thunks_lto_path, ".ll" => ".bc")
                Compiler.assemble_bitcode(thunks_lto_path, thunks_bc_path)
            else
                @warn "Failed to emit LLVM IR for AOT thunks LTO."
            end
        end
        
        elapsed = round(time() - start_time, digits=2)
        size_kb = round(filesize(thunks_so) / 1024, digits=1)
        println("  aot: $thunks_name ($size_kb KB) in $(elapsed)s")
        
        # Cleanup
        rm(thunks_obj, force=true)
    catch e
        @warn "AOT MLIR compilation failed." exception=e
    finally
        MLIRNative.destroy_context(ctx)
    end
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

        # Compile the project (C++ → IR → library + metadata)
        library_path = Compiler.compile_project(config)

        # Build thunks if enabled (C uses Clang-compiled sret, C++ uses MLIR)
        if config.compile.aot_thunks && config.binary.type != :executable
            if config.wrap.language == :c
                _build_c_thunks(config, library_path)
            else
                _build_aot_thunks(config, library_path)
            end
        end

        return library_path

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


        return wrapper_path

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
    dirs_to_remove = ["build", "julia", ".replibuild_cache"]

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
