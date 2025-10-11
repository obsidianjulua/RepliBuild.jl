#!/usr/bin/env julia
# LLVMake.jl - Enhanced LLVM/Clang to Julia compiler with project-based targeting
# Supports custom LLVM paths, target configurations, and project-specific settings
# Part of the RepliBuild build system

module LLVMake

using Pkg
using JSON
using TOML
using Dates

# Load BuildBridge for error learning and command execution
include("BuildBridge.jl")
using .BuildBridge

"""
Configuration for LLVM compilation targets and options
"""
struct TargetConfig
    triple::String              # Target triple (e.g., "x86_64-unknown-linux-gnu")
    cpu::String                 # Target CPU (e.g., "generic", "native", "haswell")
    features::Vector{String}    # CPU features to enable/disable
    opt_level::String           # Optimization level (O0, O1, O2, O3, Os, Oz)
    debug::Bool                 # Include debug symbols
    lto::Bool                   # Link-time optimization
    sanitizers::Vector{String}  # Address, thread, memory sanitizers

    function TargetConfig(;
        triple::String="",
        cpu::String="generic",
        features::Vector{String}=String[],
        opt_level::String="O2",
        debug::Bool=false,
        lto::Bool=false,
        sanitizers::Vector{String}=String[]
    )
        new(triple, cpu, features, opt_level, debug, lto, sanitizers)
    end
end

"""
Project-specific LLVM/Clang compiler configuration
"""
struct CompilerConfig
    # Paths
    project_root::String
    source_dir::String
    output_dir::String
    build_dir::String

    # LLVM toolchain (can be project-local)
    llvm_root::Union{String,Nothing}
    clang_path::String
    llvm_config_path::String
    llvm_link_path::String
    opt_path::String

    # Compilation settings
    target::TargetConfig
    include_dirs::Vector{String}
    lib_dirs::Vector{String}
    libraries::Vector{String}
    defines::Dict{String,String}
    extra_flags::Vector{String}

    # Binding generation
    binding_style::Symbol  # :simple, :advanced, :cxxwrap
    type_mappings::Dict{String,String}
    exclude_patterns::Vector{Regex}
    include_patterns::Vector{Regex}
end

"""
Enhanced LLVM-based C++ to Julia compiler
"""
struct LLVMJuliaCompiler
    config::CompilerConfig
    metadata::Dict{String,Any}

    function LLVMJuliaCompiler(config_file::String="replibuild.toml")
        config = load_config(config_file)
        metadata = Dict{String,Any}()
        new(config, metadata)
    end

    function LLVMJuliaCompiler(config::CompilerConfig)
        new(config, Dict{String,Any}())
    end
end

"""
Load configuration from TOML file
"""
function load_config(config_file::String)::CompilerConfig
    if !isfile(config_file)
        # Create default config file
        create_default_config(config_file)
        println("📝 Created default config: $config_file")
        println("   Please edit it to match your project settings.")
    end

    config_data = TOML.parsefile(config_file)

    # Parse paths
    project_root = get(config_data, "project_root", pwd())
    paths = get(config_data, "paths", Dict())
    source_dir = joinpath(project_root, get(paths, "source", "src"))
    output_dir = joinpath(project_root, get(paths, "output", "julia"))
    build_dir = joinpath(project_root, get(paths, "build", "build"))

    # Parse LLVM settings
    llvm = get(config_data, "llvm", Dict())
    llvm_root = get(llvm, "root", nothing)

    # Find LLVM tools - use LLVMEnvironment if available, fallback to manual discovery
    if !isnothing(llvm_root)
        # Try to use LLVMEnvironment for proper JLL vs in-tree detection
        try
            # Check if tools are already in TOML (preferred)
            if haskey(llvm, "tools") && !isempty(llvm["tools"])
                clang_path = get(llvm["tools"], "clang++", joinpath(llvm_root, "tools", "clang++"))
                llvm_config_path = get(llvm["tools"], "llvm-config", joinpath(llvm_root, "tools", "llvm-config"))
                llvm_link_path = get(llvm["tools"], "llvm-link", joinpath(llvm_root, "tools", "llvm-link"))
                opt_path = get(llvm["tools"], "opt", joinpath(llvm_root, "tools", "opt"))
            else
                # Fallback: Try both bin/ (JLL) and tools/ (in-tree) directories
                bin_dir = isdir(joinpath(llvm_root, "bin")) ? joinpath(llvm_root, "bin") : joinpath(llvm_root, "tools")
                clang_path = joinpath(bin_dir, "clang++")
                llvm_config_path = joinpath(bin_dir, "llvm-config")
                llvm_link_path = joinpath(bin_dir, "llvm-link")
                opt_path = joinpath(bin_dir, "opt")
            end
        catch
            # Ultimate fallback: assume tools/ directory
            clang_path = joinpath(llvm_root, "tools", "clang++")
            llvm_config_path = joinpath(llvm_root, "tools", "llvm-config")
            llvm_link_path = joinpath(llvm_root, "tools", "llvm-link")
            opt_path = joinpath(llvm_root, "tools", "opt")
        end
    else
        # Find system LLVM
        clang_path = find_tool("clang++", get(llvm, "clang", ""))
        llvm_config_path = find_tool("llvm-config", get(llvm, "config", ""))
        llvm_link_path = find_tool("llvm-link", get(llvm, "link", ""))
        opt_path = find_tool("opt", get(llvm, "opt", ""))
    end

    # Parse target configuration
    target_data = get(config_data, "target", Dict())
    target = TargetConfig(
        triple=get(target_data, "triple", ""),
        cpu=get(target_data, "cpu", "generic"),
        features=String[get(target_data, "features", String[])...],  # Convert to String[] to handle TOML empty arrays
        opt_level=get(target_data, "opt_level", "O2"),
        debug=get(target_data, "debug", false),
        lto=get(target_data, "lto", false),
        sanitizers=String[get(target_data, "sanitizers", String[])...]  # Convert to String[] to handle TOML empty arrays
    )

    # Parse compilation settings
    compile = get(config_data, "compile", Dict())
    include_dirs = [joinpath(project_root, dir) for dir in get(compile, "include_dirs", String[])]
    lib_dirs = [joinpath(project_root, dir) for dir in get(compile, "lib_dirs", String[])]
    libraries = get(compile, "libraries", String[])
    defines = Dict(String(k) => String(v) for (k, v) in get(compile, "defines", Dict()))
    extra_flags = get(compile, "extra_flags", String[])

    # Parse binding settings
    bindings = get(config_data, "bindings", Dict())
    binding_style = Symbol(get(bindings, "style", "simple"))
    type_mappings = Dict(String(k) => String(v) for (k, v) in get(bindings, "type_mappings", Dict()))
    exclude_patterns = [Regex(p) for p in get(bindings, "exclude_patterns", String[])]
    include_patterns = [Regex(p) for p in get(bindings, "include_patterns", String[])]

    return CompilerConfig(
        project_root, source_dir, output_dir, build_dir,
        llvm_root, clang_path, llvm_config_path, llvm_link_path, opt_path,
        target, include_dirs, lib_dirs, libraries, defines, extra_flags,
        binding_style, type_mappings, exclude_patterns, include_patterns
    )
end

"""
Create default configuration file
"""
function create_default_config(config_file::String)
    default_config = raw"""
    # RepliBuild LLVMake Configuration
    project_root = "."

    [paths]
    source = "src"           # C++ source directory
    output = "julia"         # Julia output directory
    build = "build"          # Build artifacts directory

    [llvm]
    # Leave empty to use system LLVM, or specify paths
    # root = "/path/to/llvm"  # Project-local LLVM installation
    # clang = "clang++-20"    # Specific clang version
    # config = "llvm-config-20"

    [target]
    # triple = "x86_64-unknown-linux-gnu"  # Target triple (empty = host)
    cpu = "generic"          # Target CPU: generic, native, haswell, etc.
    features = []            # CPU features: +avx2, +fma, -sse4.2, etc.
    opt_level = "O2"         # Optimization: O0, O1, O2, O3, Os, Oz
    debug = false            # Include debug symbols
    lto = false              # Link-time optimization
    sanitizers = []          # ["address", "thread", "memory"]

    [compile]
    include_dirs = ["include", "third_party/include"]
    lib_dirs = ["lib", "third_party/lib"]
    libraries = []           # External libraries to link
    extra_flags = []         # Additional compiler flags

    [compile.defines]
    # NDEBUG = "1"
    # CUSTOM_FLAG = "value"

    [bindings]
    style = "simple"         # simple, advanced, cxxwrap
    exclude_patterns = ["^test_", "^internal_", ".*_impl$"]
    include_patterns = []    # If specified, only include matching functions

    [bindings.type_mappings]
    # Map C++ types to Julia types
    "std::string" = "String"
    "std::vector<double>" = "Vector{Float64}"
    "std::vector<float>" = "Vector{Float32}"
    "std::vector<int>" = "Vector{Int32}"
    """

    open(config_file, "w") do f
        write(f, default_config)
    end
end

"""
Find a tool in system PATH or use provided path
"""
function find_tool(tool_name::String, provided_path::String="")
    if !isempty(provided_path) && isfile(provided_path)
        return provided_path
    end

    # Common version suffixes
    suffixes = ["", "-15", "-14", "-13", "-12", "-11"]

    for suffix in suffixes
        tool = tool_name * suffix
        try
            result = strip(read(`which $tool`, String))
            if !isempty(result) && isfile(result)
                return result
            end
        catch
            continue
        end
    end

    error("❌ Tool not found: $tool_name")
end

"""
Get compiler flags for the target configuration
"""
function get_compiler_flags(compiler::LLVMJuliaCompiler)
    config = compiler.config
    flags = String[]

    # Basic flags
    push!(flags, "-std=c++17")
    push!(flags, "-fPIC")

    # Target triple
    if !isempty(config.target.triple)
        push!(flags, "--target=$(config.target.triple)")
    end

    # CPU and features
    if config.target.cpu != "generic"
        push!(flags, "-mcpu=$(config.target.cpu)")
    end

    for feature in config.target.features
        push!(flags, "$feature")
    end

    # Optimization
    push!(flags, "-$(config.target.opt_level)")

    # Debug symbols
    if config.target.debug
        push!(flags, "-g")
        push!(flags, "-fno-omit-frame-pointer")
    end

    # LTO
    if config.target.lto
        push!(flags, "-flto")
    end

    # Sanitizers
    for sanitizer in config.target.sanitizers
        push!(flags, "-fsanitize=$sanitizer")
    end

    # Include directories
    for dir in config.include_dirs
        push!(flags, "-I$dir")
    end

    # Defines
    for (key, value) in config.defines
        if isempty(value)
            push!(flags, "-D$key")
        else
            push!(flags, "-D$key=$value")
        end
    end

    # Extra flags
    append!(flags, config.extra_flags)

    return flags
end

"""
Parse C++ file using Clang AST
"""
function parse_cpp_ast(compiler::LLVMJuliaCompiler, cpp_file::String)
    println("🔍 Parsing AST: $cpp_file")

    # Create temporary AST dump file
    ast_file = joinpath(compiler.config.build_dir, "$(basename(cpp_file)).ast.json")
    mkpath(dirname(ast_file))

    flags = get_compiler_flags(compiler)

    try
        # Generate AST dump
        cmd = `$(compiler.config.clang_path) -Xclang -ast-dump=json -fsyntax-only $flags $cpp_file`
        ast_json = read(cmd, String)

        # Parse JSON AST
        ast = JSON.parse(ast_json)

        # Extract function information
        functions = extract_functions_from_ast(ast)

        # Apply include/exclude patterns
        filtered_functions = filter_functions(functions, compiler.config)

        return filtered_functions
    catch e
        @warn "Failed to parse AST for $cpp_file: $e"
        # Fallback to simple parsing
        return parse_cpp_simple(compiler, cpp_file)
    end
end

"""
Extract function information from Clang AST
"""
function extract_functions_from_ast(ast::Dict)
    functions = []

    function visit_node(node::Dict)
        if get(node, "kind", "") == "FunctionDecl"
            # Skip if not a definition or is implicit
            if get(node, "isImplicit", false) || !haskey(node, "inner")
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

        # Recursively visit children
        if haskey(node, "inner")
            for child in node["inner"]
                if isa(child, Dict)
                    visit_node(child)
                end
            end
        end
    end

    # Start traversal from root
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
Simple C++ parsing fallback
"""
function parse_cpp_simple(compiler::LLVMJuliaCompiler, cpp_file::String)
    content = read(cpp_file, String)
    functions = []

    # Enhanced regex for function detection
    func_pattern = r"(?:(?:static|inline|extern|virtual|constexpr)\s+)*([a-zA-Z_][\w:*&\s]*?)\s+([a-zA-Z_]\w*)\s*\(([^)]*)\)\s*(?:const)?\s*(?:noexcept)?\s*[{;]"m

    for match in eachmatch(func_pattern, content)
        return_type = strip(match.captures[1])
        func_name = match.captures[2]
        params_str = match.captures[3]

        # Skip constructors/destructors and keywords
        if func_name in ["if", "while", "for", "switch", "return", "class", "struct", "namespace"] ||
           return_type in ["class", "struct", "enum", "namespace"]
            continue
        end

        # Parse parameters
        params = []
        if !isempty(strip(params_str)) && strip(params_str) != "void"
            param_parts = split(params_str, ",")
            for param in param_parts
                param = strip(param)
                # Simple parameter parsing
                parts = split(param)
                if length(parts) >= 2
                    param_type = join(parts[1:end-1], " ")
                    param_name = parts[end]
                    push!(params, Dict("type" => param_type, "name" => param_name))
                elseif length(parts) == 1 && parts[1] != "void"
                    push!(params, Dict("type" => parts[1], "name" => ""))
                end
            end
        end

        func_info = Dict{String,Any}(
            "name" => func_name,
            "return_type" => return_type,
            "params" => params
        )

        push!(functions, func_info)
    end

    return functions
end

"""
Filter functions based on include/exclude patterns
"""
function filter_functions(functions::Vector, config::CompilerConfig)
    filtered = []

    # Default exclusions for production
    default_excludes = [
        r"^std::",           # C++ standard library
        r"^__",              # Compiler internals
        r"^operator",        # C++ operators
        r"^_Z",              # Mangled names
        r"::operator",       # Member operators
        r"^decltype",        # Type deduction
        r"^typename",        # Template typename
    ]

    for func in functions
        func_name = func["name"]

        # Skip empty names
        if isempty(func_name)
            continue
        end

        # Check default exclusions
        excluded = any(pattern -> occursin(pattern, func_name), default_excludes)

        # Check user exclude patterns
        if !excluded
            excluded = any(pattern -> occursin(pattern, func_name), config.exclude_patterns)
        end

        # Check include patterns (if specified)
        if !isempty(config.include_patterns)
            included = any(pattern -> occursin(pattern, func_name), config.include_patterns)
            if !included
                excluded = true
            end
        end

        if !excluded
            push!(filtered, func)
        end
    end

    return filtered
end

"""
Compile C++ files to LLVM IR
"""
function compile_to_ir(compiler::LLVMJuliaCompiler, cpp_files::Vector{String})
    println("🔧 Compiling to LLVM IR...")

    ir_files = String[]
    flags = get_compiler_flags(compiler)
    db = BuildBridge.get_error_db(joinpath(compiler.config.build_dir, "replibuild_errors.db"))

    for cpp_file in cpp_files
        ir_file = joinpath(compiler.config.build_dir, "$(basename(cpp_file)).ll")
        mkpath(dirname(ir_file))

        # Build command args
        args = ["-S", "-emit-llvm", flags..., "-o", ir_file, cpp_file]

        # Execute with error learning
        output, exitcode = BuildBridge.execute(compiler.config.clang_path, args)

        if exitcode == 0
            push!(ir_files, ir_file)
            println("  ✓ $(basename(cpp_file)) → $(basename(ir_file))")
        else
            # Record error in database
            (error_id, pattern_name, description) = BuildBridge.ErrorLearning.record_error(
                db, "$(compiler.config.clang_path) $(join(args, " "))", output,
                project_path=compiler.config.build_dir, file_path=cpp_file)

            @error "Failed to compile $cpp_file: $pattern_name - $description"
            println("Error output:\n$output")

            # Get suggestions
            suggestions = BuildBridge.ErrorLearning.suggest_fixes(db, output,
                project_path=compiler.config.build_dir)

            if !isempty(suggestions)
                println("\n💡 Suggestions:")
                for (i, sug) in enumerate(suggestions[1:min(3, length(suggestions))])
                    println("  $i. $(sug["description"]) (confidence: $(round(sug["confidence"], digits=2)))")
                end
            end
        end
    end

    return ir_files
end

"""
Optimize and link LLVM IR files
"""
function optimize_and_link_ir(compiler::LLVMJuliaCompiler, ir_files::Vector{String}, output_name::String)
    println("⚡ Optimizing and linking IR...")
    db = BuildBridge.get_error_db(joinpath(compiler.config.build_dir, "replibuild_errors.db"))

    # Link all IR files
    linked_ir = joinpath(compiler.config.build_dir, "$output_name.linked.ll")
    link_args = ["-S", "-o", linked_ir, ir_files...]

    output, exitcode = BuildBridge.execute(compiler.config.llvm_link_path, link_args)

    if exitcode == 0
        println("  ✓ Linked $(length(ir_files)) files")
    else
        BuildBridge.ErrorLearning.record_error(
            db, "$(compiler.config.llvm_link_path) $(join(link_args, " "))", output,
            project_path=compiler.config.build_dir)
        @error "Failed to link IR files"
        println("Error output:\n$output")
        return nothing
    end

    # Optimize if requested
    if compiler.config.target.opt_level != "O0"
        optimized_ir = joinpath(compiler.config.build_dir, "$output_name.opt.ll")
        opt_level = replace(compiler.config.target.opt_level, "O" => "")
        opt_args = ["-S", "-O$opt_level", "-o", optimized_ir, linked_ir]

        output, exitcode = BuildBridge.execute(compiler.config.opt_path, opt_args)

        if exitcode == 0
            println("  ✓ Optimized with -O$opt_level")
            return optimized_ir
        else
            BuildBridge.ErrorLearning.record_error(
                db, "$(compiler.config.opt_path) $(join(opt_args, " "))", output,
                project_path=compiler.config.build_dir)
            @warn "Optimization failed, using unoptimized IR"
            println("Error output:\n$output")
        end
    end

    return linked_ir
end

"""
Compile IR to shared library
"""
function compile_ir_to_shared_lib(compiler::LLVMJuliaCompiler, ir_file::String, lib_name::String)
    println("📦 Creating shared library...")
    db = BuildBridge.get_error_db(joinpath(compiler.config.build_dir, "replibuild_errors.db"))

    output_lib = joinpath(compiler.config.output_dir, "lib$lib_name.so")
    mkpath(dirname(output_lib))

    flags = get_compiler_flags(compiler)

    # Add library directories and libraries
    link_flags = String[]
    for dir in compiler.config.lib_dirs
        push!(link_flags, "-L$dir")
    end
    for lib in compiler.config.libraries
        push!(link_flags, "-l$lib")
    end

    args = vcat(["-shared"], flags, link_flags, ["-o", output_lib, ir_file])

    output, exitcode = BuildBridge.execute(compiler.config.clang_path, args)

    if exitcode == 0
        println("  ✓ Created: $output_lib")
        return output_lib
    else
        BuildBridge.ErrorLearning.record_error(
            db, "$(compiler.config.clang_path) $(join(args, " "))", output,
            project_path=compiler.config.build_dir)
        @error "Failed to create shared library"
        println("Error output:\n$output")
        return nothing
    end
end

"""
Compile IR to executable (NEW - supports multi-stage builds)
"""
function compile_ir_to_executable(compiler::LLVMJuliaCompiler, ir_file::String, exe_name::String;
                                   link_libs::Vector{String}=String[])
    println("🔨 Creating executable...")
    db = BuildBridge.get_error_db(joinpath(compiler.config.build_dir, "replibuild_errors.db"))

    output_exe = joinpath(compiler.config.output_dir, exe_name)
    mkpath(dirname(output_exe))

    flags = get_compiler_flags(compiler)

    # Add library directories and libraries
    link_flags = String[]

    # Add output dir to lib search path (for linking against .so from previous stage)
    push!(link_flags, "-L$(compiler.config.output_dir)")

    for dir in compiler.config.lib_dirs
        push!(link_flags, "-L$dir")
    end

    # Add specified libraries (for multi-stage: link against previously built .so)
    for lib in link_libs
        push!(link_flags, "-l$lib")
    end

    for lib in compiler.config.libraries
        push!(link_flags, "-l$lib")
    end

    # Add rpath so executable can find .so at runtime
    push!(link_flags, "-Wl,-rpath,$(compiler.config.output_dir)")

    args = vcat(flags, link_flags, ["-o", output_exe, ir_file])

    output, exitcode = BuildBridge.execute(compiler.config.clang_path, args)

    if exitcode == 0
        println("  ✓ Created: $output_exe")
        # Make executable
        chmod(output_exe, 0o755)
        return output_exe
    else
        BuildBridge.ErrorLearning.record_error(
            db, "$(compiler.config.clang_path) $(join(args, " "))", output,
            project_path=compiler.config.build_dir)
        @error "Failed to create executable"
        println("Error output:\n$output")

        # Better error messages for common linking issues
        if contains(output, "undefined reference")
            println("\n💡 Linking Error - Undefined symbols:")
            println("  Check that all required libraries are specified in 'libraries = [...]'")
            println("  For multi-stage builds, use link_libs parameter to link .so from previous stage")
        elseif contains(output, "cannot find -l")
            missing_lib = match(r"cannot find -l(\w+)", output)
            if !isnothing(missing_lib)
                println("\n💡 Missing Library: lib$(missing_lib.captures[1])")
                println("  1. Install the library: sudo apt install lib$(missing_lib.captures[1])-dev")
                println("  2. Or add its path to lib_dirs in replibuild.toml")
            end
        end

        return nothing
    end
end

"""
Generate Julia bindings based on function information
"""
function generate_julia_bindings(compiler::LLVMJuliaCompiler, lib_name::String, functions::Vector)
    println("📝 Generating Julia bindings...")

    if compiler.config.binding_style == :simple
        return generate_simple_bindings(compiler, lib_name, functions)
    elseif compiler.config.binding_style == :advanced
        return generate_advanced_bindings(compiler, lib_name, functions)
    elseif compiler.config.binding_style == :cxxwrap
        return generate_cxxwrap_bindings(compiler, lib_name, functions)
    else
        error("Unknown binding style: $(compiler.config.binding_style)")
    end
end

"""
Generate simple Julia bindings
"""
function generate_simple_bindings(compiler::LLVMJuliaCompiler, lib_name::String, functions::Vector)
    module_name = titlecase(lib_name)

    content = """
    # Auto-generated Julia bindings for $lib_name
    # Generated on $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    # Compiler: LLVM/Clang Julia Compiler

    module $module_name

    using Libdl

    # Load the compiled shared library
    const _lib_path = joinpath(@__DIR__, "lib$lib_name.so")
    const _lib_handle = Libdl.dlopen(_lib_path)

    # Type mappings
    const CppTypeMap = Dict{String, DataType}(
    """

    # Add type mappings
    for (cpp_type, julia_type) in compiler.config.type_mappings
        content *= "    \"$cpp_type\" => $julia_type,\n"
    end

    # Add default mappings
    default_mappings = Dict(
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
        "void*" => "Ptr{Cvoid}",
        "const char*" => "Cstring"
    )

    for (cpp_type, julia_type) in default_mappings
        if !haskey(compiler.config.type_mappings, cpp_type)
            content *= "    \"$cpp_type\" => $julia_type,\n"
        end
    end

    content *= ")\n\n"

    # Generate function wrappers
    for func in functions
        func_name = func["name"]
        return_type = func["return_type"]
        params = func["params"]

        # Map return type
        julia_return_type = get(compiler.config.type_mappings, return_type,
            get(default_mappings, return_type, "Any"))

        # Generate parameter list
        param_names = String[]
        param_types = String[]
        ccall_types = String[]

        for (i, param) in enumerate(params)
            param_name = isempty(param["name"]) ? "arg$i" : param["name"]
            param_type = param["type"]

            # Clean parameter name
            param_name = replace(param_name, r"[^a-zA-Z0-9_]" => "_")
            push!(param_names, param_name)

            # Map parameter type
            julia_type = get(compiler.config.type_mappings, param_type,
                get(default_mappings, param_type, "Any"))
            push!(param_types, "$param_name::$julia_type")
            push!(ccall_types, julia_type)
        end

        # Generate function
        content *= """
        \"\"\"
            $func_name($(join(param_names, ", ")))

        Auto-generated wrapper for C++ function `$func_name`.
        \"\"\"
        function $func_name($(join(param_types, ", ")))
            ccall(
                (:$func_name, _lib_handle),
                $julia_return_type,
                ($(join(ccall_types, ", "))$(isempty(ccall_types) ? "" : ",")),
                $(join(param_names, ", "))
            )
        end

        """
    end

    # Export functions
    content *= "\n# Exports\nexport "
    content *= join([func["name"] for func in functions], ", ")
    content *= "\n\n"

    # Cleanup function
    content *= """
    # Cleanup
    function __cleanup__()
        Libdl.dlclose(_lib_handle)
    end

    end # module
    """

    # Write to file
    output_file = joinpath(compiler.config.output_dir, "$lib_name.jl")
    mkpath(dirname(output_file))

    open(output_file, "w") do f
        write(f, content)
    end

    println("  ✓ Generated: $output_file")
    return output_file
end

"""
Main compilation workflow
"""
function compile_project(compiler::LLVMJuliaCompiler;
    specific_files::Vector{String}=String[],
    components::Union{Vector{String},Nothing}=nothing)
    println("🚀 RepliBuild LLVMake - C++ to Julia Compiler")
    println("="^50)
    println("📁 Project: $(compiler.config.project_root)")
    println("📁 Source:  $(compiler.config.source_dir)")
    println("📁 Output:  $(compiler.config.output_dir)")
    println("🔧 Clang:   $(compiler.config.clang_path)")
    println("⚡ Target:  $(compiler.config.target.triple == "" ? "host" : compiler.config.target.triple)")
    println("="^50)

    # Create directories
    mkpath(compiler.config.output_dir)
    mkpath(compiler.config.build_dir)

    # Find C++ files
    cpp_files = if !isempty(specific_files)
        specific_files
    else
        find_cpp_files(compiler.config.source_dir)
    end

    println("\n📊 Found $(length(cpp_files)) C++ files")

    # Group files by component
    file_groups = group_files_by_component(cpp_files, components)

    # Process each component
    generated_modules = String[]

    for (component_name, files) in file_groups
        println("\n🔧 Processing component: $component_name")
        println("   Files: $(length(files))")

        # Parse all files to extract functions
        all_functions = []
        for file in files
            functions = parse_cpp_ast(compiler, file)
            append!(all_functions, functions)
        end

        # Remove duplicates
        unique_functions = unique(f -> f["name"], all_functions)
        println("   Functions: $(length(unique_functions))")

        if isempty(unique_functions)
            println("   ⚠️  No functions found, skipping...")
            continue
        end

        # Compile to IR
        ir_files = compile_to_ir(compiler, files)

        if isempty(ir_files)
            println("   ❌ Compilation failed")
            continue
        end

        # Link and optimize
        final_ir = optimize_and_link_ir(compiler, ir_files, component_name)

        if isnothing(final_ir)
            println("   ❌ Linking failed")
            continue
        end

        # Create shared library
        lib_path = compile_ir_to_shared_lib(compiler, final_ir, component_name)

        if isnothing(lib_path)
            println("   ❌ Library creation failed")
            continue
        end

        # Generate bindings
        binding_file = generate_julia_bindings(compiler, component_name, unique_functions)
        push!(generated_modules, component_name)

        println("   ✅ Component complete!")
    end

    # Generate main module
    if length(generated_modules) > 1
        generate_main_module(compiler, generated_modules)
    end

    # Save compilation metadata
    save_metadata(compiler, generated_modules)

    println("\n🎉 Compilation complete!")
    println("📁 Output: $(compiler.config.output_dir)")
    println("📦 Modules: $(join(generated_modules, ", "))")

    return generated_modules
end

"""
Find all C++ files in directory
"""
function find_cpp_files(dir::String)
    cpp_files = String[]

    for (root, dirs, files) in walkdir(dir)
        # Skip hidden directories
        filter!(d -> !startswith(d, "."), dirs)

        for file in files
            if endswith(file, ".cpp") || endswith(file, ".cc") ||
               endswith(file, ".cxx") || endswith(file, ".c++")
                push!(cpp_files, joinpath(root, file))
            end
        end
    end

    return cpp_files
end

"""
Group files by component
"""
function group_files_by_component(files::Vector{String},
    components::Union{Vector{String},Nothing}=nothing)
    if !isnothing(components)
        # User-specified components
        groups = Dict{String,Vector{String}}()

        for component in components
            groups[component] = filter(f -> contains(f, component), files)
        end

        # Add remaining files to "misc" component
        assigned_files = vcat(values(groups)...)
        remaining = setdiff(files, assigned_files)
        if !isempty(remaining)
            groups["misc"] = remaining
        end

        return groups
    else
        # Auto-detect components based on directory structure
        groups = Dict{String,Vector{String}}()

        for file in files
            dir_parts = splitpath(dirname(file))

            # Use immediate parent directory as component
            component = length(dir_parts) > 0 ? dir_parts[end] : "core"

            # Clean component name
            component = replace(component, r"[^a-zA-Z0-9_]" => "_")

            if !haskey(groups, component)
                groups[component] = String[]
            end

            push!(groups[component], file)
        end

        return groups
    end
end

"""
Generate main module that imports all components
"""
function generate_main_module(compiler::LLVMJuliaCompiler, components::Vector{String})
    content = """
    # Main module for compiled C++ components
    # Generated on $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))

    module CompiledCpp

    # Component modules
    """

    for component in components
        module_name = titlecase(component)
        content *= "include(\"$component.jl\")\n"
        content *= "using .$module_name\n"
        content *= "export $module_name\n"
    end

    content *= "\nend # module\n"

    output_file = joinpath(compiler.config.output_dir, "CompiledCpp.jl")
    open(output_file, "w") do f
        write(f, content)
    end

    println("\n✅ Generated main module: $output_file")
end

"""
Save compilation metadata
"""
function save_metadata(compiler::LLVMJuliaCompiler, components::Vector{String})
    metadata = Dict(
        "timestamp" => now(),
        "config" => compiler.config,
        "components" => components,
        "compiler_version" => read(`$(compiler.config.clang_path) --version`, String),
        "llvm_version" => read(`$(compiler.config.llvm_config_path) --version`, String)
    )

    metadata_file = joinpath(compiler.config.output_dir, "compilation_metadata.json")
    open(metadata_file, "w") do f
        JSON.print(f, metadata, 2)
    end
end

"""
Command-line interface
"""
function main()
    if length(ARGS) == 0
        println("""
        RepliBuild LLVMake - LLVM/Clang to Julia Compiler

        Usage:
            julia LLVMake.jl init [project_dir]
            julia LLVMake.jl compile [config_file]
            julia LLVMake.jl compile-file <file.cpp> [config_file]
            julia LLVMake.jl info [config_file]
            julia LLVMake.jl clean [config_file]

        Examples:
            julia LLVMake.jl init myproject
            julia LLVMake.jl compile
            julia LLVMake.jl compile-file src/math.cpp
            julia LLVMake.jl info
        """)
        return
    end

    command = ARGS[1]

    if command == "init"
        # Initialize new project
        project_dir = length(ARGS) >= 2 ? ARGS[2] : "."
        mkpath(project_dir)

        config_file = joinpath(project_dir, "replibuild.toml")
        create_default_config(config_file)

        # Create directory structure
        for dir in ["src", "include", "julia", "build", "test"]
            mkpath(joinpath(project_dir, dir))
        end

        println("✅ Initialized project in: $project_dir")
        println("📝 Edit $config_file to configure your project")

    elseif command == "compile"
        # Compile entire project
        config_file = length(ARGS) >= 2 ? ARGS[2] : "replibuild.toml"
        compiler = LLVMJuliaCompiler(config_file)
        compile_project(compiler)

    elseif command == "compile-file"
        # Compile specific file
        if length(ARGS) < 2
            println("Usage: julia LLVMake.jl compile-file <file.cpp> [config_file]")
            return
        end

        cpp_file = ARGS[2]
        config_file = length(ARGS) >= 3 ? ARGS[3] : "replibuild.toml"

        compiler = LLVMJuliaCompiler(config_file)
        compile_project(compiler, specific_files=[cpp_file])

    elseif command == "info"
        # Show project information
        config_file = length(ARGS) >= 2 ? ARGS[2] : "replibuild.toml"
        compiler = LLVMJuliaCompiler(config_file)

        println("RepliBuild Project Information:")
        println("="^50)
        println("Root:    $(compiler.config.project_root)")
        println("Source:  $(compiler.config.source_dir)")
        println("Output:  $(compiler.config.output_dir)")
        println("Target:  $(compiler.config.target.triple == "" ? "host" : compiler.config.target.triple)")
        println("CPU:     $(compiler.config.target.cpu)")
        println("Opt:     $(compiler.config.target.opt_level)")
        println("Debug:   $(compiler.config.target.debug)")
        println("LTO:     $(compiler.config.target.lto)")

        # Count files
        cpp_files = find_cpp_files(compiler.config.source_dir)
        println("\nFiles:   $(length(cpp_files)) C++ files")

        # Check for existing output
        if isdir(compiler.config.output_dir)
            jl_files = filter(f -> endswith(f, ".jl"), readdir(compiler.config.output_dir))
            println("Output:  $(length(jl_files)) Julia files")
        end

    elseif command == "clean"
        # Clean build artifacts
        config_file = length(ARGS) >= 2 ? ARGS[2] : "replibuild.toml"
        config = load_config(config_file)

        if isdir(config.build_dir)
            rm(config.build_dir, recursive=true)
            println("✅ Cleaned build directory")
        end

        if isdir(config.output_dir)
            print("Remove output directory? (y/N): ")
            response = readline()
            if lowercase(response) == "y"
                rm(config.output_dir, recursive=true)
                println("✅ Cleaned output directory")
            end
        end

    else
        println("Unknown command: $command")
    end
end

# Exports
export LLVMJuliaCompiler, CompilerConfig, TargetConfig
export compile_project, parse_cpp_ast, compile_to_ir
export optimize_and_link_ir, compile_ir_to_shared_lib, compile_ir_to_executable
export generate_julia_bindings, load_config, create_default_config
export find_cpp_files, group_files_by_component

end # module LLVMake

# Run if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    LLVMake.main()
end
