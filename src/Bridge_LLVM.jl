#!/usr/bin/env julia
# Bridge_LLVM.jl - BuildBridge-powered LLVM compiler orchestrator
# Complete integration: BuildBridge + ClangJLBridge for binding generation
# NOTE: This file is meant to be included from RepliBuild.jl which handles module loading

using Pkg
using TOML
using JSON
using Dates

# Use already-loaded modules from parent RepliBuild module
# (BuildBridge, CMakeParser, ClangJLBridge, ASTWalker are loaded by RepliBuild.jl)

"""
Enhanced compiler configuration with BuildBridge integration
"""
mutable struct BridgeCompilerConfig
    # Project settings
    project_name::String
    project_root::String

    # Paths
    source_dir::String
    output_dir::String
    build_dir::String
    include_dirs::Vector{String}

    # Bridge settings
    auto_discover::Bool
    enable_learning::Bool
    cache_tools::Bool

    # Discovered tools (populated by BuildBridge)
    tools::Dict{String,String}

    # Toolchain cache (for performance)
    toolchain_cached::Bool

    # Compilation settings
    compile_flags::Vector{String}
    defines::Dict{String,String}
    walk_dependencies::Bool
    max_depth::Int

    # Target settings
    target_triple::String
    target_cpu::String
    opt_level::String
    enable_lto::Bool

    # Workflow stages
    stages::Vector{String}
    parallel::Bool

    # Cache settings
    cache_enabled::Bool
    cache_dir::String

    # Binding generation settings
    binding_style::String  # "auto", "clangjl", "basic"

    function BridgeCompilerConfig(config_file::String="replibuild.toml")
        if !isfile(config_file)
            error("Config file not found: $config_file")
        end

        data = TOML.parsefile(config_file)

        # Parse configuration
        project = get(data, "project", Dict())
        paths = get(data, "paths", Dict())
        bridge = get(data, "bridge", Dict())
        compile = get(data, "compile", Dict())
        target = get(data, "target", Dict())
        workflow = get(data, "workflow", Dict())
        cache = get(data, "cache", Dict())
        binding = get(data, "binding", Dict())

        # Load include_dirs from compile section or cache
        include_dirs = get(compile, "include_dirs", String[])
        if isempty(include_dirs) && get(cache, "enabled", true)
            project_root = get(project, "root", ".")
            cache_dir = get(cache, "directory", ".replibuild_cache")
            cache_file = joinpath(project_root, cache_dir, "build_cache.toml")
            if isfile(cache_file)
                cache_data = TOML.parsefile(cache_file)
                discovery_results = get(cache_data, "discovery_results", Dict())
                include_dirs = get(discovery_results, "include_dirs", String[])
            end
        end

        config = new(
            get(project, "name", "MyProject"),
            get(project, "root", "."),
            get(paths, "source", "src"),
            get(paths, "output", "julia"),
            get(paths, "build", "build"),
            include_dirs,  # Loaded from compile section or cache
            get(bridge, "auto_discover", true),
            get(bridge, "enable_learning", true),
            get(bridge, "cache_tools", true),
            Dict{String,String}(),  # tools - populated later
            false,  # toolchain_cached
            get(compile, "flags", String[]),
            get(compile, "defines", Dict{String,String}()),
            get(compile, "walk_dependencies", true),
            get(compile, "max_depth", 10),
            get(target, "triple", ""),
            get(target, "cpu", "generic"),
            get(target, "opt_level", "O2"),
            get(target, "lto", false),
            get(workflow, "stages", String[]),
            get(workflow, "parallel", true),
            get(cache, "enabled", true),
            get(cache, "directory", ".bridge_cache"),
            get(binding, "style", "auto")  # "auto", "clangjl", "basic"
        )

        return config
    end
end

"""
Discover LLVM toolchain using BuildBridge (with caching)
"""
function discover_tools!(config::BridgeCompilerConfig)
    # Check if already cached
    if config.toolchain_cached && !isempty(config.tools)
        println("⚡ Using cached toolchain ($(length(config.tools)) tools)")
        return
    end

    println("🔍 Discovering LLVM tools via BuildBridge...")

    required_tools = [
        "clang", "clang++", "llvm-config", "llvm-link",
        "opt", "nm", "objdump", "llvm-ar"
    ]

    # Use BuildBridge tool discovery
    tools = BuildBridge.discover_llvm_tools(required_tools)

    for (tool, path) in tools
        config.tools[tool] = path
        println("  ✅ $tool → $path")
    end

    # Check for missing tools
    for tool in required_tools
        if !haskey(config.tools, tool)
            println("  ⚠️  $tool not found")
        end
    end

    if !haskey(config.tools, "clang++")
        error("❌ clang++ is required but not found")
    end

    println("  📊 Found $(length(config.tools)) tools")

    # Mark as cached
    config.toolchain_cached = true
end

"""
Check if source file needs recompilation (incremental build support)
"""
function needs_recompile(source_file::String, ir_file::String, cache_enabled::Bool)::Bool
    # IR file doesn't exist - must compile
    if !isfile(ir_file)
        return true
    end

    # Cache disabled - always compile
    if !cache_enabled
        return true
    end

    # Check modification time
    source_mtime = mtime(source_file)
    ir_mtime = mtime(ir_file)

    return source_mtime > ir_mtime
end

"""
Walk dependency tree using clang -M via BuildBridge
"""
function walk_dependencies(config::BridgeCompilerConfig, entry_file::String)
    println("📂 Walking dependencies from: $entry_file")

    deps = Set{String}([entry_file])
    to_process = [entry_file]
    processed = Set{String}()
    depth = 0

    while !isempty(to_process) && depth < config.max_depth
        current_batch = copy(to_process)
        empty!(to_process)
        depth += 1

        for current in current_batch
            if current in processed
                continue
            end
            push!(processed, current)

            # Build include flags
            includes = ["-I$dir" for dir in config.include_dirs]

            # Execute clang -M via BuildBridge
            cmd_args = vcat(["-M", "-MF", "/dev/null"], includes, [current])

            (output, exitcode) = BuildBridge.execute("clang++", cmd_args)

            if exitcode == 0
                # Parse Make-style dependency output
                for line in split(output, "\n")
                    # Extract header files
                    for m in eachmatch(r"([^\s:]+\.h(?:pp)?)", line)
                        dep = String(m.captures[1])
                        if isfile(dep) && !(dep in processed)
                            push!(deps, dep)
                            push!(to_process, dep)
                        end
                    end
                end
            end
        end
    end

    println("  📊 Found $(length(deps)) dependencies (depth: $depth)")
    return collect(deps)
end

"""
Parse C++ AST using clang via BuildBridge
"""
function parse_ast_bridge(config::BridgeCompilerConfig, cpp_file::String)
    println("🔍 Parsing AST: $(basename(cpp_file))")

    # Build command
    includes = ["-I$dir" for dir in config.include_dirs]
    flags = config.compile_flags

    cmd_args = vcat(
        ["-Xclang", "-ast-dump=json", "-fsyntax-only"],
        flags,
        includes,
        [cpp_file]
    )

    (output, exitcode) = BuildBridge.execute("clang++", cmd_args)

    if exitcode != 0
        @warn "  ❌ AST parsing failed: $output"
        return nothing
    end

    try
        ast = JSON.parse(output)
        functions = extract_functions_from_ast(ast)
        println("  ✅ Found $(length(functions)) functions")
        return functions
    catch e
        @warn "  ❌ Failed to parse AST JSON: $e"
        return nothing
    end
end

"""
Extract function declarations from AST
"""
function extract_functions_from_ast(ast::Dict)
    functions = []

    function visit_node(node::Dict)
        if get(node, "kind", "") == "FunctionDecl"
            if get(node, "isImplicit", false)
                return
            end

            func_info = Dict{String,Any}(
                "name" => get(node, "name", ""),
                "return_type" => get(get(node, "type", Dict()), "qualType", "void"),
                "params" => []
            )

            # Extract parameters
            if haskey(node, "inner")
                for inner in node["inner"]
                    if get(inner, "kind", "") == "ParmVarDecl"
                        param = Dict(
                            "name" => get(inner, "name", ""),
                            "type" => get(get(inner, "type", Dict()), "qualType", "")
                        )
                        push!(func_info["params"], param)
                    end
                end
            end

            push!(functions, func_info)
        end

        if haskey(node, "inner")
            for child in node["inner"]
                if isa(child, Dict)
                    visit_node(child)
                end
            end
        end
    end

    if haskey(ast, "inner")
        for node in ast["inner"]
            if isa(node, Dict)
                visit_node(node)
            end
        end
    end

    return functions
end

"""
Compile single C++ file to LLVM IR
"""
function compile_single_to_ir(config::BridgeCompilerConfig, cpp_file::String)
    base = basename(cpp_file)
    ir_file = joinpath(config.build_dir, "$base.ll")

    # Build command
    includes = ["-I$dir" for dir in config.include_dirs]
    defines = ["-D$k=$v" for (k, v) in config.defines]

    cmd_args = vcat(
        ["-S", "-emit-llvm"],
        config.compile_flags,
        includes,
        defines,
        ["-o", ir_file, cpp_file]
    )

    # Use simple compilation (ErrorLearning removed)
    (output, exitcode) = BuildBridge.execute("clang++", cmd_args)

    if !isempty(output) && exitcode != 0
        println("  ❌ $(basename(cpp_file)): $output")
    end

    success = isfile(ir_file)
    return (ir_file, success, exitcode)
end

"""
Compile C++ to LLVM IR via BuildBridge (with incremental build + parallel compilation)
"""
function compile_to_ir(config::BridgeCompilerConfig, cpp_files::Vector{String})
    println("🔧 Compiling to LLVM IR...")

    mkpath(config.build_dir)
    ir_files = String[]
    compiled_count = 0
    cached_count = 0

    # Separate files into cached vs needs-compilation
    files_to_compile = Tuple{String,String}[]  # (cpp_file, ir_file)

    for cpp_file in cpp_files
        base = basename(cpp_file)
        ir_file = joinpath(config.build_dir, "$base.ll")

        # Check if recompilation needed (incremental build)
        if !needs_recompile(cpp_file, ir_file, config.cache_enabled)
            println("  ⚡ Cached: $(basename(cpp_file))")
            push!(ir_files, ir_file)
            cached_count += 1
        else
            push!(files_to_compile, (cpp_file, ir_file))
        end
    end

    # Compile needed files (parallel if enabled and multiple files)
    if !isempty(files_to_compile)
        if config.parallel && length(files_to_compile) > 1
            println("  ⚡ Parallel compilation: $(length(files_to_compile)) files on $(Threads.nthreads()) threads")

            # Parallel compilation using threads
            results = Vector{Tuple{String,Bool,Int}}(undef, length(files_to_compile))
            Threads.@threads for i in 1:length(files_to_compile)
                cpp_file, _ = files_to_compile[i]
                results[i] = compile_single_to_ir(config, cpp_file)
            end

            # Collect results
            for (ir_file, success, exitcode) in results
                if success
                    push!(ir_files, ir_file)
                    println("  ✅ $(basename(ir_file))")
                    compiled_count += 1
                else
                    @warn "  ❌ Failed: $(basename(ir_file)) (exit code: $exitcode)"
                end
            end
        else
            # Sequential compilation
            for (cpp_file, ir_file) in files_to_compile
                (ir_result, success, exitcode) = compile_single_to_ir(config, cpp_file)

                if success
                    push!(ir_files, ir_result)
                    println("  ✅ $(basename(cpp_file)) → $(basename(ir_result))")
                    compiled_count += 1
                else
                    @warn "  ❌ Failed: $cpp_file (exit code: $exitcode)"
                end
            end
        end
    end

    # Summary
    total = compiled_count + cached_count
    if total > 0
        if cached_count > 0
            cache_pct = round(100*cached_count/total, digits=1)
            println("  📊 Compiled: $compiled_count, Cached: $cached_count (⚡ $cache_pct% cache hit)")
        end
        if config.parallel && compiled_count > 1
            println("  ⚡ Used $(Threads.nthreads()) threads for parallel compilation")
        end
    end

    println("  📊 Generated $(length(ir_files)) IR files")
    return ir_files
end

"""
Link and optimize IR files via BuildBridge
"""
function link_optimize_ir(config::BridgeCompilerConfig, ir_files::Vector{String}, output_name::String)
    println("🔗 Linking and optimizing IR...")

    # Link
    linked_ir = joinpath(config.build_dir, "$output_name.linked.ll")
    cmd_args = vcat(["-S", "-o", linked_ir], ir_files)

    (output, exitcode) = BuildBridge.execute("llvm-link", cmd_args)

    if !isfile(linked_ir)
        @warn "  ❌ Linking failed\n$output"
        return nothing
    end

    println("  ✅ Linked $(length(ir_files)) files")

    # Optimize
    optimized_ir = joinpath(config.build_dir, "$output_name.opt.ll")
    opt_level = replace(config.opt_level, "O" => "")

    cmd_args = ["-S", "-O$opt_level", "-o", optimized_ir, linked_ir]

    (output, exitcode) = BuildBridge.execute("opt", cmd_args)

    if isfile(optimized_ir)
        println("  ✅ Optimized with -O$opt_level")
        return optimized_ir
    end

    return linked_ir
end

"""
Create shared library via BuildBridge
"""
function create_library(config::BridgeCompilerConfig, ir_file::String, lib_name::String)
    println("📦 Creating shared library...")

    mkpath(config.output_dir)
    lib_path = joinpath(config.output_dir, "lib$lib_name.so")

    cmd_args = ["-shared", "-o", lib_path, ir_file]

    if config.enable_lto
        push!(cmd_args, "-flto")
    end

    (output, exitcode) = BuildBridge.execute("clang++", cmd_args)

    if isfile(lib_path)
        println("  ✅ Created: $lib_path")
        return lib_path
    end

    @warn "  ❌ Library creation failed\n$output"
    return nothing
end

"""
Create executable via BuildBridge (for multi-stage builds)
"""
function create_executable(config::BridgeCompilerConfig, ir_file::String, exe_name::String, link_libraries::Vector{String}=String[], lib_dirs::Vector{String}=String[])
    println("🔨 Creating executable...")

    mkpath(config.output_dir)
    exe_path = joinpath(config.output_dir, exe_name)

    # Build linker flags
    link_flags = String[]

    # Add library search paths
    for lib_dir in lib_dirs
        push!(link_flags, "-L$lib_dir")
    end

    # Add libraries to link
    for lib in link_libraries
        push!(link_flags, "-l$lib")
    end

    # Add rpath so executable finds libraries at runtime
    if !isempty(lib_dirs)
        for lib_dir in lib_dirs
            push!(link_flags, "-Wl,-rpath,$lib_dir")
        end
    end

    cmd_args = vcat(["-o", exe_path, ir_file], link_flags)

    if config.enable_lto
        push!(cmd_args, "-flto")
    end

    (output, exitcode) = BuildBridge.execute("clang++", cmd_args)

    if exitcode != 0
        # Show helpful error messages for linking issues
        if contains(output, "undefined reference")
            println("\n💡 Linking Error - Undefined symbols:")
            println("  Check that all required libraries are specified in 'link_libraries = [...]'")
            println("  Check that library paths are in 'lib_dirs = [...]'")
        end

        if contains(output, "cannot find -l")
            println("\n💡 Missing Library:")
            # Extract library name from error
            m = match(r"cannot find -l(\w+)", output)
            if m !== nothing
                lib_name = m.captures[1]
                println("  Library: lib$lib_name")
                println("  1. Check if library is built")
                println("  2. Add path to lib_dirs in replibuild.toml")
            end
        end

        @warn "  ❌ Executable creation failed\n$output"
        return nothing
    end

    if isfile(exe_path)
        println("  ✅ Created: $exe_path")
        # Make executable
        chmod(exe_path, 0o755)
        return exe_path
    end

    @warn "  ❌ Executable creation failed\n$output"
    return nothing
end

"""
Discover header files in project for Clang.jl binding generation
"""
function discover_project_headers(config::BridgeCompilerConfig)
    headers = String[]

    # Search source directory
    if isdir(config.source_dir)
        for (root, dirs, files) in walkdir(config.source_dir)
            # Skip build directories
            filter!(d -> !in(d, ["build", ".git", ".cache", ".bridge_cache"]), dirs)

            for file in files
                ext = lowercase(splitext(file)[2])
                if ext in [".h", ".hpp", ".hxx", ".h++", ".hh"]
                    push!(headers, abspath(joinpath(root, file)))
                end
            end
        end
    end

    # Search include directories
    for inc_dir in config.include_dirs
        if isdir(inc_dir)
            for (root, dirs, files) in walkdir(inc_dir)
                filter!(d -> !in(d, ["build", ".git", ".cache", ".bridge_cache"]), dirs)

                for file in files
                    ext = lowercase(splitext(file)[2])
                    if ext in [".h", ".hpp", ".hxx", ".h++", ".hh"]
                        header_path = abspath(joinpath(root, file))
                        if !in(header_path, headers)
                            push!(headers, header_path)
                        end
                    end
                end
            end
        end
    end

    return headers
end

"""
Extract symbols from binary via BuildBridge
"""
function extract_symbols(config::BridgeCompilerConfig, binary_path::String)
    println("🔍 Extracting symbols...")

    # Try nm first
    if haskey(config.tools, "nm")
        (output, exitcode) = BuildBridge.execute("nm", ["-DC", binary_path])

        if exitcode == 0
            symbols = Dict{String,Any}[]

            for line in split(output, "\n")
                parts = split(strip(line))
                if length(parts) >= 3 && parts[2] in ["T", "t"]
                    push!(symbols, Dict(
                        "name" => parts[3],
                        "type" => "function",
                        "visibility" => parts[2] == "T" ? "global" : "local"
                    ))
                end
            end

            println("  ✅ Found $(length(symbols)) symbols")
            return symbols
        end
    end

    return Dict{String,Any}[]
end

"""
Generate Julia bindings from symbols and function info
"""
function generate_julia_bindings(config::BridgeCompilerConfig, lib_path::String,
                                  symbols::Vector{Dict{String,Any}},
                                  functions::Vector)::Union{String,Nothing}
    println("\n📝 Generating Julia bindings...")

    if isempty(symbols)
        println("  ⚠️  No symbols to wrap")
        return nothing
    end

    # Create output directory
    mkpath(config.output_dir)

    # Generate module name from project name
    module_name = replace(config.project_name, r"[^a-zA-Z0-9_]" => "_")
    module_name = uppercase(module_name[1:1]) * module_name[2:end]

    bindings_file = joinpath(config.output_dir, "$(module_name).jl")
    lib_name = "lib$(config.project_name)"

    open(bindings_file, "w") do f
        # Module header
        write(f, """
        # Auto-generated Julia bindings for $(config.project_name)
        # Generated: $(Dates.now())
        # Library: $(basename(lib_path))

        module $module_name

        # Library path
        const LIB_PATH = "$(abspath(lib_path))"

        # Verify library exists
        if !isfile(LIB_PATH)
            error("Library not found: \$LIB_PATH")
        end

        """)

        # Generate wrappers for each symbol
        for sym in symbols
            name = sym["name"]

            # Skip symbols starting with underscore (internal)
            if startswith(name, "_")
                continue
            end

            # Try to infer type from function info if available
            sig = nothing
            if !isempty(functions)
                for func in functions
                    if get(func, "name", "") == name
                        sig = func
                        break
                    end
                end
            end

            # Generate wrapper
            if !isnothing(sig) && haskey(sig, "return_type") && haskey(sig, "parameters")
                # We have type information!
                ret_type = julia_type_from_cpp(sig["return_type"])
                params = sig["parameters"]

                # Build parameter list
                param_names = String[]
                param_types = String[]
                ccall_types = String[]

                for (i, param) in enumerate(params)
                    param_name = get(param, "name", "arg$i")
                    cpp_type = get(param, "type", "void*")
                    jl_type = julia_type_from_cpp(cpp_type)

                    push!(param_names, param_name)
                    push!(param_types, "$param_name::$jl_type")
                    push!(ccall_types, jl_type)
                end

                # Generate function with types
                params_str = join(param_types, ", ")
                ccall_types_str = join(ccall_types, ", ")
                args_str = join(param_names, ", ")

                write(f, """
                \"\"\"
                    $name($params_str)

                Auto-generated wrapper for C++ function: $name
                \"\"\"
                function $name($params_str)::$ret_type
                    ccall((:$name, LIB_PATH), $ret_type, ($ccall_types_str), $args_str)
                end

                """)
            else
                # No type info - generate generic wrapper with comment
                write(f, """
                # $name - type information not available
                # Usage: ccall((:$name, $module_name.LIB_PATH), RetType, (ArgTypes...), args...)

                """)
            end
        end

        # Export only valid Julia identifiers (no special characters)
        # Valid Julia identifier: starts with letter/underscore, contains only alphanumeric/underscore
        is_valid_identifier(name) = !isempty(name) && match(r"^[a-zA-Z_][a-zA-Z0-9_!]*$", name) !== nothing

        exported_funcs = [sym["name"] for sym in symbols
                         if !startswith(sym["name"], "_") && is_valid_identifier(sym["name"])]

        if !isempty(exported_funcs)
            write(f, "\n# Exports (only valid Julia identifiers)\n")
            write(f, "export ")
            write(f, join(exported_funcs, ", "))
            write(f, "\n")
        end

        # Note about mangled names
        non_exported = count(s -> !startswith(s["name"], "_") && !is_valid_identifier(s["name"]), symbols)
        if non_exported > 0
            write(f, "\n# Note: $non_exported symbols with C++ mangled names are available via ccall\n")
            write(f, "# but cannot be exported as Julia identifiers.\n")
        end

        write(f, "\nend # module $module_name\n")
    end

    println("  ✅ Generated: $(bindings_file)")
    println("  📦 Module: $module_name")
    println("  🔧 Functions: $(count(s -> !startswith(s["name"], "_"), symbols))")

    return bindings_file
end

"""
Convert C++ type to Julia type
"""
function julia_type_from_cpp(cpp_type::String)::String
    cpp_type = strip(replace(cpp_type, r"\s+" => " "))

    # Basic types
    type_map = Dict(
        "void" => "Cvoid",
        "bool" => "Bool",
        "char" => "Cchar",
        "unsigned char" => "Cuchar",
        "short" => "Cshort",
        "unsigned short" => "Cushort",
        "int" => "Cint",
        "unsigned int" => "Cuint",
        "long" => "Clong",
        "unsigned long" => "Culong",
        "long long" => "Clonglong",
        "unsigned long long" => "Culonglong",
        "float" => "Cfloat",
        "double" => "Cdouble",
        "size_t" => "Csize_t",
        "int32_t" => "Int32",
        "int64_t" => "Int64",
        "uint32_t" => "UInt32",
        "uint64_t" => "UInt64"
    )

    # Handle pointers
    if contains(cpp_type, "*")
        base_type = replace(cpp_type, "*" => "")
        base_type = strip(base_type)

        if base_type == "char" || base_type == "const char"
            return "Cstring"
        else
            return "Ptr{Cvoid}"
        end
    end

    # Handle const
    cpp_type = replace(cpp_type, "const " => "")
    cpp_type = strip(cpp_type)

    return get(type_map, cpp_type, "Cvoid")
end

"""
Main compilation pipeline
"""
function compile_project(config::BridgeCompilerConfig)
    println("🚀 RepliBuild Bridge LLVM - Unified Build System")
    println("=" ^ 60)
    println("📁 Project: $(config.project_name)")
    println("📁 Source:  $(config.source_dir)")
    println("📁 Output:  $(config.output_dir)")
    println("=" ^ 60)

    # Stage 1: Discover tools
    if "discover_tools" in config.stages
        discover_tools!(config)
    end

    # Find C++ sources - read from TOML source_files if available
    data = TOML.parsefile(joinpath(config.project_root, "replibuild.toml"))
    compile_section = get(data, "compile", Dict())
    discovery_section = get(data, "discovery", Dict())
    source_files_from_toml = get(compile_section, "source_files", String[])

    # Also try loading from cache if not in main TOML
    if isempty(source_files_from_toml) && config.cache_enabled
        cache_file = joinpath(config.cache_dir, "build_cache.toml")
        if isfile(cache_file)
            cache_data = TOML.parsefile(cache_file)
            source_files_from_toml = get(cache_data, "compile_sources", String[])
            if !isempty(source_files_from_toml)
                println("   📦 Loaded $(length(source_files_from_toml)) source files from cache")
            end
        end
    end

    # Try to load dependency graph for intelligent compilation order
    dep_graph_file = get(discovery_section, "dependency_graph_file", "")

    # Also check cache for dependency_graph_file location
    if isempty(dep_graph_file) && config.cache_enabled
        cache_file = joinpath(config.cache_dir, "build_cache.toml")
        if isfile(cache_file)
            cache_data = TOML.parsefile(cache_file)
            discovery_results = get(cache_data, "discovery_results", Dict())
            dep_graph_file = get(discovery_results, "dependency_graph_file", "")
        end
    end

    dep_graph = nothing

    if !isempty(dep_graph_file)
        dep_graph_path = joinpath(config.project_root, dep_graph_file)
        if isfile(dep_graph_path)
            println("\n📊 Loading dependency graph: $dep_graph_file")
            dep_graph = ASTWalker.load_dependency_graph_json(dep_graph_path)

            if !isnothing(dep_graph)
                println("   ✓ Graph loaded: $(length(dep_graph.files)) files")
                println("   ✓ Compilation order computed: $(length(dep_graph.compilation_order)) files")
            end
        else
            @warn "Dependency graph file specified but not found: $dep_graph_path"
        end
    end

    cpp_files = if !isnothing(dep_graph) && !isempty(dep_graph.compilation_order)
        # Use compilation order from dependency graph (BEST - respects dependencies)
        println("   ℹ️  Using dependency-aware compilation order")
        # Filter to only .cpp and .cc files (exclude headers)
        filter(f -> endswith(f, ".cpp") || endswith(f, ".cc"), dep_graph.compilation_order)
    elseif !isempty(source_files_from_toml)
        # Use source files from TOML (from discovery, but no order info)
        println("   ℹ️  Using source files from discovery (no dependency order)")
        [joinpath(config.project_root, f) for f in source_files_from_toml]
    else
        # Fallback: walk source_dir (WORST - arbitrary order)
        println("   ⚠️  No dependency info, using arbitrary file order")
        files = String[]
        for (root, dirs, file_list) in walkdir(config.source_dir)
            for file in file_list
                if endswith(file, ".cpp") || endswith(file, ".cc")
                    push!(files, joinpath(root, file))
                end
            end
        end
        files
    end

    println("\n📊 Compiling $(length(cpp_files)) C++ files")

    if isempty(cpp_files)
        println("❌ No C++ files found")
        return nothing
    end

    # Stage 2: Walk dependencies
    all_deps = Set{String}()
    if "walk_deps" in config.stages && config.walk_dependencies
        for cpp in cpp_files
            deps = walk_dependencies(config, cpp)
            union!(all_deps, deps)
        end
        println("\n📦 Total dependencies: $(length(all_deps))")
    end

    # Stage 3: Parse AST
    all_functions = []
    if "parse_ast" in config.stages
        println("\n🔍 Parsing AST for all files...")
        for cpp in cpp_files
            functions = parse_ast_bridge(config, cpp)
            if !isnothing(functions)
                append!(all_functions, functions)
            end
        end
        println("  📊 Total functions: $(length(all_functions))")
    end

    # Stage 4: Compile to IR
    ir_files = String[]
    if "compile" in config.stages || "compile_to_ir" in config.stages
        ir_files = compile_to_ir(config, cpp_files)
    end

    if isempty(ir_files)
        println("❌ Compilation failed")
        return nothing
    end

    # Stage 5: Link and optimize
    optimized_ir = nothing
    if "link" in config.stages || "link_ir" in config.stages || "optimize_ir" in config.stages
        optimized_ir = link_optimize_ir(config, ir_files, config.project_name)
    end

    if isnothing(optimized_ir)
        println("❌ Optimization failed")
        return nothing
    end

    # Stage 6: Create library OR executable
    output_path = nothing

    if "create_executable" in config.stages
        # Build executable - read link configuration from TOML
        data = TOML.parsefile(joinpath(config.project_root, "replibuild.toml"))
        compile_section = get(data, "compile", Dict())
        link_libraries = get(compile_section, "link_libraries", String[])
        lib_dirs = get(compile_section, "lib_dirs", String[])

        output_path = create_executable(config, optimized_ir, config.project_name, link_libraries, lib_dirs)

        if isnothing(output_path)
            println("❌ Executable creation failed")
            return nothing
        end

        # For executables, skip symbol extraction and binding generation
        println("\n🎉 Build complete!")
        println("📦 Executable: $output_path")
        return output_path

    elseif "binary" in config.stages || "create_library" in config.stages
        output_path = create_library(config, optimized_ir, config.project_name)

        if isnothing(output_path)
            println("❌ Library creation failed")
            return nothing
        end
    end

    # Stage 7: Extract symbols (libraries only)
    symbols = []
    if "symbols" in config.stages || "extract_symbols" in config.stages
        symbols = extract_symbols(config, output_path)
    end

    # Stage 8: Generate bindings (libraries only)
    bindings_file = nothing
    if "wrap" in config.stages || "generate_bindings" in config.stages
        # Determine binding style
        style = config.binding_style

        # Auto-detect: prefer Clang.jl if available
        if style == "auto"
            style = "clangjl"  # Try Clang.jl first
        end

        if style == "clangjl"
            # Use Clang.jl for type-aware bindings
            println("\n📝 Using Clang.jl for binding generation...")

            try
                # Discover header files
                headers = discover_project_headers(config)

                if !isempty(headers)
                    # Load TOML data for ClangJLBridge
                    config_data = TOML.parsefile(joinpath(config.project_root, "replibuild_auto.toml"))

                    # Generate with Clang.jl
                    bindings_file = ClangJLBridge.generate_bindings_clangjl(
                        config_data, output_path, headers
                    )
                else
                    @warn "No headers found for Clang.jl, falling back to basic bindings"
                    bindings_file = generate_julia_bindings(config, output_path, symbols, all_functions)
                end
            catch e
                @warn "Clang.jl binding generation failed: $e\nFalling back to basic bindings"
                bindings_file = generate_julia_bindings(config, output_path, symbols, all_functions)
            end
        else
            # Use basic binding generation
            bindings_file = generate_julia_bindings(config, output_path, symbols, all_functions)
        end
    end

    # Show error learning statistics
    if config.enable_learning
        println("\n📊 Error Learning Statistics:")
        stats = BuildBridge.get_error_stats()
        println("  Total errors recorded: $(stats["total_errors"])")
        println("  Successful fixes: $(stats["successful_fixes"])")
        println("  Success rate: $(round(stats["success_rate"] * 100, digits=1))%")
    end

    println("\n🎉 Compilation complete!")
    println("📦 Library: $output_path")
    println("🔧 Symbols: $(length(symbols))")

    return output_path
end

"""
CLI interface
"""
function main()
    if length(ARGS) == 0
        println("""
        RepliBuild Bridge LLVM - UnifiedBridge + LLVM/Julia

        Usage:
            julia Bridge_LLVM.jl compile [config]
            julia Bridge_LLVM.jl discover [config]
            julia Bridge_LLVM.jl stats

        Examples:
            julia Bridge_LLVM.jl compile replibuild.toml
            julia Bridge_LLVM.jl discover
        """)
        return
    end

    command = ARGS[1]

    if command == "compile"
        config_file = length(ARGS) >= 2 ? ARGS[2] : "replibuild.toml"
        config = BridgeCompilerConfig(config_file)
        compile_project(config)

    elseif command == "discover"
        config_file = length(ARGS) >= 2 ? ARGS[2] : "replibuild.toml"
        config = BridgeCompilerConfig(config_file)
        discover_tools!(config)

    elseif command == "stats"
        stats = BuildBridge.get_error_stats()
        println("📊 Error Learning Statistics")
        println("=" ^ 50)
        println("Total Errors: $(stats["total_errors"])")
        println("Total Fixes Attempted: $(stats["total_fixes"])")
        println("Successful Fixes: $(stats["successful_fixes"])")
        println("Success Rate: $(round(stats["success_rate"] * 100, digits=1))%")
        println("\nCommon Error Patterns:")
        for row in eachrow(stats["common_patterns"])
            println("  - $(row.error_pattern): $(row.count) occurrences")
        end

    else
        println("Unknown command: $command")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
