#!/usr/bin/env julia
# JuliaWrapItUp.jl
# Complements the RepliBuild LLVMake compiler by providing advanced binary introspection and wrapping

module JuliaWrapItUp

using Pkg
using JSON
using TOML
using Dates

"""
Configuration for binary wrapper generation
"""
struct WrapperConfig
    # Project paths
    project_root::String
    binary_dirs::Vector{String}
    output_dir::String
    header_dirs::Vector{String}

    # Wrapper settings
    wrapper_style::Symbol          # :basic, :advanced, :introspective
    symbol_detection::Symbol       # :nm, :objdump, :libclang, :all
    demangle_cpp::Bool
    generate_tests::Bool
    generate_docs::Bool
    safety_checks::Bool

    # Type inference
    use_headers::Bool
    header_parser::String         # Path to header parser tool
    type_hints::Dict{String,String}

    # Integration with stage 1
    stage1_metadata::Union{String,Nothing}  # Path to compilation_metadata.json
    inherit_mappings::Bool
end

"""
Binary information structure
"""
struct BinaryInfo
    path::String
    name::String
    type::Symbol  # :shared_lib, :static_lib, :executable, :object_file
    arch::String
    symbols::Vector{Dict{String,Any}}
    dependencies::Vector{String}
    metadata::Dict{String,Any}
end

"""
Universal Binary Wrapper Generator
"""
struct BinaryWrapper
    config::WrapperConfig
    binaries::Vector{BinaryInfo}
    type_registry::Dict{String,String}

    function BinaryWrapper(config_file::String="wrapper_config.toml")
        config = load_wrapper_config(config_file)

        # Load type mappings from stage 1 if available
        type_registry = load_type_registry(config)

        new(config, BinaryInfo[], type_registry)
    end
end

"""
Load wrapper configuration
"""
function load_wrapper_config(config_file::String)::WrapperConfig
    if !isfile(config_file)
        # Check if we're in a project with stage 1 config
        if isfile("replibuild.toml")
            config = create_wrapper_config_from_stage1()
            save_wrapper_config(config, config_file)
            println("📝 Created wrapper config from RepliBuild LLVMake: $config_file")
        else
            config = create_default_wrapper_config()
            save_wrapper_config(config, config_file)
            println("📝 Created default wrapper config: $config_file")
        end
        return config
    end

    data = TOML.parsefile(config_file)

    # Parse configuration
    project_root = get(data, "project_root", pwd())
    paths = get(data, "paths", Dict())
    binary_dirs = [joinpath(project_root, dir) for dir in get(paths, "binary_dirs", ["julia", "build", "lib", "bin"])]
    output_dir = joinpath(project_root, get(paths, "output", "julia_wrappers"))
    header_dirs = [joinpath(project_root, dir) for dir in get(paths, "header_dirs", ["include"])]

    wrapper = get(data, "wrapper", Dict())
    wrapper_style = Symbol(get(wrapper, "style", "advanced"))
    symbol_detection = Symbol(get(wrapper, "symbol_detection", "all"))
    demangle_cpp = get(wrapper, "demangle_cpp", true)
    generate_tests = get(wrapper, "generate_tests", true)
    generate_docs = get(wrapper, "generate_docs", true)
    safety_checks = get(wrapper, "safety_checks", true)

    type_inference = get(data, "type_inference", Dict())
    use_headers = get(type_inference, "use_headers", true)
    header_parser = get(type_inference, "header_parser", "")
    type_hints = Dict(String(k) => String(v) for (k, v) in get(type_inference, "hints", Dict()))

    integration = get(data, "integration", Dict())
    stage1_metadata = get(integration, "stage1_metadata", nothing)
    inherit_mappings = get(integration, "inherit_mappings", true)

    return WrapperConfig(
        project_root, binary_dirs, output_dir, header_dirs,
        wrapper_style, symbol_detection, demangle_cpp, generate_tests,
        generate_docs, safety_checks, use_headers, header_parser,
        type_hints, stage1_metadata, inherit_mappings
    )
end

"""
Create wrapper config from stage 1 LLVMake compiler config
"""
function create_wrapper_config_from_stage1()::WrapperConfig
    # Load stage 1 config
    stage1_data = TOML.parsefile("replibuild.toml")

    project_root = get(stage1_data, "project_root", pwd())
    paths = get(stage1_data, "paths", Dict())

    # Use stage 1 output as binary source
    binary_dirs = [
        joinpath(project_root, get(paths, "output", "julia")),
        joinpath(project_root, get(paths, "build", "build")),
        joinpath(project_root, "lib"),
        joinpath(project_root, "bin")
    ]

    output_dir = joinpath(project_root, "julia_wrappers")

    # Get include dirs from stage 1
    compile = get(stage1_data, "compile", Dict())
    header_dirs = [joinpath(project_root, dir) for dir in get(compile, "include_dirs", ["include"])]

    # Check for metadata
    metadata_path = joinpath(project_root, get(paths, "output", "julia"), "compilation_metadata.json")
    stage1_metadata = isfile(metadata_path) ? metadata_path : nothing

    return WrapperConfig(
        project_root, binary_dirs, output_dir, header_dirs,
        :advanced, :all, true, true, true, true,
        true, "", Dict{String,String}(),
        stage1_metadata, true
    )
end

"""
Create default wrapper configuration
"""
function create_default_wrapper_config()::WrapperConfig
    return WrapperConfig(
        pwd(),
        [".", "lib", "bin", "build"],
        "julia_wrappers",
        ["include"],
        :advanced, :all, true, true, true, true,
        false, "", Dict{String,String}(),
        nothing, false
    )
end

"""
Save wrapper configuration
"""
function save_wrapper_config(config::WrapperConfig, config_file::String)
    data = Dict(
        "project_root" => config.project_root,
        "paths" => Dict(
            "binary_dirs" => [relpath(dir, config.project_root) for dir in config.binary_dirs],
            "output" => relpath(config.output_dir, config.project_root),
            "header_dirs" => [relpath(dir, config.project_root) for dir in config.header_dirs]
        ),
        "wrapper" => Dict(
            "style" => string(config.wrapper_style),
            "symbol_detection" => string(config.symbol_detection),
            "demangle_cpp" => config.demangle_cpp,
            "generate_tests" => config.generate_tests,
            "generate_docs" => config.generate_docs,
            "safety_checks" => config.safety_checks
        ),
        "type_inference" => Dict(
            "use_headers" => config.use_headers,
            "header_parser" => config.header_parser,
            "hints" => config.type_hints
        ),
        "integration" => Dict(
            "stage1_metadata" => something(config.stage1_metadata, ""),
            "inherit_mappings" => config.inherit_mappings
        )
    )

    open(config_file, "w") do f
        TOML.print(f, data)
    end
end

"""
Load type registry from stage 1 or configuration
"""
function load_type_registry(config::WrapperConfig)::Dict{String,String}
    registry = Dict{String,String}()

    # Load from stage 1 metadata if available
    if config.inherit_mappings && !isnothing(config.stage1_metadata) && isfile(config.stage1_metadata)
        try
            metadata = JSON.parsefile(config.stage1_metadata)
            if haskey(metadata, "config") && haskey(metadata["config"], "type_mappings")
                merge!(registry, metadata["config"]["type_mappings"])
            end
        catch e
            @warn "Failed to load stage 1 metadata: $e"
        end
    end

    # Add configured type hints
    merge!(registry, config.type_hints)

    # Add default C/C++ to Julia mappings
    default_mappings = Dict(
        # Basic types
        "void" => "Cvoid",
        "bool" => "Bool",
        "_Bool" => "Bool",
        "char" => "Cchar",
        "signed char" => "Cchar",
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
        "long double" => "Cdouble",

        # Sized types
        "int8_t" => "Int8",
        "uint8_t" => "UInt8",
        "int16_t" => "Int16",
        "uint16_t" => "UInt16",
        "int32_t" => "Int32",
        "uint32_t" => "UInt32",
        "int64_t" => "Int64",
        "uint64_t" => "UInt64",

        # Pointers
        "void*" => "Ptr{Cvoid}",
        "const void*" => "Ptr{Cvoid}",
        "char*" => "Cstring",
        "const char*" => "Cstring",

        # Common types
        "size_t" => "Csize_t",
        "ssize_t" => "Cssize_t",
        "ptrdiff_t" => "Cptrdiff_t",
        "intptr_t" => "Cintptr_t",
        "uintptr_t" => "Cuintptr_t",

        # C++ STL types (basic)
        "std::string" => "String",
        "std::string&" => "String",
        "const std::string&" => "String"
    )

    for (k, v) in default_mappings
        if !haskey(registry, k)
            registry[k] = v
        end
    end

    return registry
end

"""
Scan directories for binaries
"""
function scan_binaries(wrapper::BinaryWrapper)
    println("🔍 Scanning for binaries...")
    binaries = BinaryInfo[]

    for dir in wrapper.config.binary_dirs
        if !isdir(dir)
            continue
        end

        println("  📁 Scanning: $dir")

        for (root, dirs, files) in walkdir(dir)
            # Skip hidden directories
            filter!(d -> !startswith(d, "."), dirs)

            for file in files
                file_path = joinpath(root, file)

                # Determine binary type
                binary_type = identify_binary_type(file_path)

                if binary_type != :unknown
                    info = analyze_binary(wrapper, file_path, binary_type)
                    if !isnothing(info)
                        push!(binaries, info)
                        println("    ✓ Found: $(info.name) ($(info.type))")
                    end
                end
            end
        end
    end

    println("  📊 Total binaries found: $(length(binaries))")
    return binaries
end

"""
Identify binary type from file
"""
function identify_binary_type(file_path::String)::Symbol
    if !isfile(file_path)
        return :unknown
    end

    ext = lowercase(splitext(file_path)[2])

    if ext in [".so", ".dylib"]
        return :shared_lib
    elseif ext == ".dll"
        return :shared_lib
    elseif ext == ".a"
        return :static_lib
    elseif ext == ".o"
        return :object_file
    elseif ext == ""
        # Check if executable
        try
            # Use file command to check
            result = read(`file $file_path`, String)
            if contains(result, "executable") || contains(result, "shared object")
                return isexecutable(file_path) ? :executable : :shared_lib
            end
        catch
            return :unknown
        end
    end

    return :unknown
end

"""
Check if file is executable
"""
function isexecutable(path::String)
    try
        if Sys.isunix()
            run(`test -x $path`)
            return true
        else
            # Windows: check for .exe extension
            return endswith(lowercase(path), ".exe")
        end
    catch
        return false
    end
end

"""
Analyze a binary file
"""
function analyze_binary(wrapper::BinaryWrapper, file_path::String, binary_type::Symbol)::Union{BinaryInfo,Nothing}
    try
        name = splitext(basename(file_path))[1]

        # Get architecture
        arch = get_binary_architecture(file_path)

        # Extract symbols
        symbols = extract_symbols(wrapper, file_path, binary_type)

        # Get dependencies
        dependencies = get_binary_dependencies(file_path, binary_type)

        # Additional metadata
        metadata = Dict{String,Any}(
            "size" => filesize(file_path),
            "mtime" => mtime(file_path),
            "real_path" => realpath(file_path)
        )

        return BinaryInfo(
            file_path, name, binary_type, arch,
            symbols, dependencies, metadata
        )
    catch e
        @warn "Failed to analyze $file_path: $e"
        return nothing
    end
end

"""
Get binary architecture
"""
function get_binary_architecture(file_path::String)::String
    try
        result = read(`file $file_path`, String)

        # Extract architecture info
        if contains(result, "x86-64")
            return "x86_64"
        elseif contains(result, "aarch64")
            return "aarch64"
        elseif contains(result, "arm64")
            return "arm64"
        elseif contains(result, "i386")
            return "i386"
        else
            return "unknown"
        end
    catch
        return "unknown"
    end
end

"""
Extract symbols from binary
"""
function extract_symbols(wrapper::BinaryWrapper, file_path::String, binary_type::Symbol)::Vector{Dict{String,Any}}
    symbols = Dict{String,Any}[]

    if binary_type == :executable
        # Executables might not export symbols
        return symbols
    end

    # Try multiple methods based on configuration
    methods = Symbol[]

    if wrapper.config.symbol_detection == :all
        methods = [:nm, :objdump, :libclang]
    else
        methods = [wrapper.config.symbol_detection]
    end

    for method in methods
        if method == :nm
            append!(symbols, extract_symbols_nm(wrapper, file_path))
        elseif method == :objdump
            append!(symbols, extract_symbols_objdump(wrapper, file_path))
        elseif method == :libclang && !isempty(wrapper.config.header_parser)
            append!(symbols, extract_symbols_libclang(wrapper, file_path))
        end
    end

    # Remove duplicates
    unique!(s -> s["name"], symbols)

    # Try to infer types from headers if configured
    if wrapper.config.use_headers
        enhance_symbols_with_headers(wrapper, symbols)
    end

    return symbols
end

"""
Extract symbols using nm
"""
function extract_symbols_nm(wrapper::BinaryWrapper, file_path::String)::Vector{Dict{String,Any}}
    symbols = Dict{String,Any}[]

    try
        # Use nm with demangling if requested
        cmd = wrapper.config.demangle_cpp ? `nm -DC $file_path` : `nm -D $file_path`
        result = read(cmd, String)

        for line in split(result, '\n')
            if isempty(strip(line))
                continue
            end

            parts = split(strip(line))
            if length(parts) >= 3
                # Symbol type is in the middle
                sym_type = parts[2]
                sym_name = join(parts[3:end], " ")  # Handle spaces in demangled names

                # Filter for exported functions (T, t) and data (D, d, B, b)
                if sym_type in ["T", "t", "D", "d", "B", "b"]
                    # Parse demangled name if available
                    parsed = parse_symbol_signature(sym_name)

                    symbol = Dict{String,Any}(
                        "name" => parsed["base_name"],
                        "mangled" => sym_name,
                        "type" => sym_type in ["T", "t"] ? "function" : "data",
                        "visibility" => sym_type in ["T", "D", "B"] ? "global" : "local",
                        "signature" => parsed["signature"],
                        "return_type" => parsed["return_type"],
                        "parameters" => parsed["parameters"]
                    )

                    push!(symbols, symbol)
                end
            end
        end
    catch e
        @debug "nm failed for $file_path: $e"
    end

    return symbols
end

"""
Extract symbols using objdump
"""
function extract_symbols_objdump(wrapper::BinaryWrapper, file_path::String)::Vector{Dict{String,Any}}
    symbols = Dict{String,Any}[]

    try
        # Use objdump with demangling
        cmd = wrapper.config.demangle_cpp ? `objdump -TC $file_path` : `objdump -T $file_path`
        result = read(cmd, String)

        for line in split(result, '\n')
            if contains(line, "*UND*") || !contains(line, ".text")
                continue
            end

            parts = split(strip(line))
            if length(parts) >= 6
                sym_name = join(parts[6:end], " ")

                parsed = parse_symbol_signature(sym_name)

                symbol = Dict{String,Any}(
                    "name" => parsed["base_name"],
                    "mangled" => sym_name,
                    "type" => "function",
                    "visibility" => "global",
                    "signature" => parsed["signature"],
                    "return_type" => parsed["return_type"],
                    "parameters" => parsed["parameters"]
                )

                push!(symbols, symbol)
            end
        end
    catch e
        @debug "objdump failed for $file_path: $e"
    end

    return symbols
end

"""
Parse C++ symbol signature
"""
function parse_symbol_signature(symbol::String)::Dict{String,Any}
    result = Dict{String,Any}(
        "base_name" => symbol,
        "signature" => "",
        "return_type" => nothing,
        "parameters" => []
    )

    # Check if it's a function signature
    paren_idx = findfirst('(', symbol)
    if !isnothing(paren_idx)
        # Extract base name
        result["base_name"] = strip(symbol[1:paren_idx-1])

        # Extract parameters
        close_idx = findlast(')', symbol)
        if !isnothing(close_idx) && close_idx > paren_idx
            param_str = symbol[paren_idx+1:close_idx-1]
            result["signature"] = symbol[paren_idx:close_idx]

            # Parse parameters
            if !isempty(strip(param_str)) && strip(param_str) != "void"
                # Simple parameter parsing (handles nested templates/parens)
                params = parse_parameter_list(param_str)
                result["parameters"] = params
            end
        end

        # Check for return type (C++ demangled format)
        space_idx = findfirst(' ', result["base_name"])
        if !isnothing(space_idx)
            potential_return = strip(result["base_name"][1:space_idx-1])
            if potential_return in ["void", "int", "float", "double", "bool"] ||
               contains(potential_return, "*") || contains(potential_return, "&")
                result["return_type"] = potential_return
                result["base_name"] = strip(result["base_name"][space_idx+1:end])
            end
        end
    end

    return result
end

"""
Parse parameter list handling nested templates and parentheses
"""
function parse_parameter_list(param_str::String)::Vector{Dict{String,String}}
    params = Dict{String,String}[]

    # Handle complex parameter lists with templates
    current_param = ""
    depth = 0

    for char in param_str
        if char == '<' || char == '('
            depth += 1
        elseif char == '>' || char == ')'
            depth -= 1
        elseif char == ',' && depth == 0
            # End of parameter
            push!(params, parse_single_parameter(strip(current_param)))
            current_param = ""
            continue
        end
        current_param *= char
    end

    # Don't forget the last parameter
    if !isempty(strip(current_param))
        push!(params, parse_single_parameter(strip(current_param)))
    end

    return params
end

"""
Parse a single parameter
"""
function parse_single_parameter(param::Union{String,SubString{String}})::Dict{String,String}
    # Remove const, volatile, etc.
    param = replace(param, r"^(const|volatile|mutable)\s+" => "")

    # Try to separate type and name
    # This is simplified - real C++ parsing is complex
    parts = split(param)

    if length(parts) >= 2 && !contains(parts[end], "*") && !contains(parts[end], "&")
        # Likely has a parameter name
        return Dict("type" => join(parts[1:end-1], " "), "name" => parts[end])
    else
        # Only type, no name
        return Dict("type" => param, "name" => "")
    end
end

"""
Get binary dependencies
"""
function get_binary_dependencies(file_path::String, binary_type::Symbol)::Vector{String}
    deps = String[]

    if binary_type in [:shared_lib, :executable]
        try
            if Sys.islinux()
                result = read(`ldd $file_path`, String)
                for line in split(result, '\n')
                    if contains(line, "=>")
                        parts = split(line, "=>")
                        if length(parts) >= 1
                            lib_name = strip(parts[1])
                            push!(deps, lib_name)
                        end
                    end
                end
            elseif Sys.isapple()
                result = read(`otool -L $file_path`, String)
                for line in split(result, '\n')[2:end]  # Skip first line
                    parts = split(strip(line))
                    if length(parts) >= 1
                        push!(deps, parts[1])
                    end
                end
            end
        catch e
            @debug "Failed to get dependencies for $file_path: $e"
        end
    end

    return deps
end

"""
Generate wrapper for a binary
"""
function generate_wrapper(wrapper::BinaryWrapper, binary::BinaryInfo)::String
    if wrapper.config.wrapper_style == :basic
        return generate_basic_wrapper(wrapper, binary)
    elseif wrapper.config.wrapper_style == :advanced
        return generate_advanced_wrapper(wrapper, binary)
    elseif wrapper.config.wrapper_style == :introspective
        return generate_introspective_wrapper(wrapper, binary)
    else
        error("Unknown wrapper style: $(wrapper.config.wrapper_style)")
    end
end

"""
Generate advanced wrapper with type inference
"""
function generate_advanced_wrapper(wrapper::BinaryWrapper, binary::BinaryInfo)::String
    module_name = generate_module_name(binary.name)

    content = """
    # Advanced Julia wrapper for $(binary.name)
    # Generated on $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    # Binary type: $(binary.type)
    # Architecture: $(binary.arch)

    module $module_name

    using Libdl

    # Library management
    const _lib_path = raw"$(binary.path)"
    const _lib_handle = Ref{Ptr{Nothing}}(C_NULL)
    const _load_errors = String[]

    function __init__()
        try
            _lib_handle[] = Libdl.dlopen(_lib_path, Libdl.RTLD_LAZY | Libdl.RTLD_GLOBAL)
            @debug "Loaded $(binary.name) from $_lib_path"
        catch e
            push!(_load_errors, string(e))
            @debug "Failed to load $(binary.name): \$e"
        end
    end

    \"\"\"
        is_loaded()

    Check if the library is successfully loaded.
    \"\"\"
    is_loaded() = _lib_handle[] != C_NULL

    \"\"\"
        get_load_errors()

    Get any errors that occurred during library loading.
    \"\"\"
    get_load_errors() = copy(_load_errors)

    \"\"\"
        get_lib_path()

    Get the path to the underlying binary.
    \"\"\"
    get_lib_path() = _lib_path

    """

    # Add safety check macro if enabled
    if wrapper.config.safety_checks
        content *= """
        # Safety check macro
        macro check_loaded()
            quote
                if !is_loaded()
                    error("Library $(binary.name) is not loaded. Check get_load_errors() for details.")
                end
            end
        end

        """
    end

    # Generate function wrappers
    functions_generated = 0
    for symbol in binary.symbols
        if symbol["type"] == "function"
            func_wrapper = generate_function_wrapper(wrapper, symbol, binary.name)
            if !isnothing(func_wrapper)
                content *= func_wrapper
                content *= "\n"
                functions_generated += 1
            end
        end
    end

    # Generate data accessors
    data_generated = 0
    for symbol in binary.symbols
        if symbol["type"] == "data"
            data_wrapper = generate_data_wrapper(wrapper, symbol, binary.name)
            if !isnothing(data_wrapper)
                content *= data_wrapper
                content *= "\n"
                data_generated += 1
            end
        end
    end

    # Add library info function
    content *= """
    \"\"\"
        library_info()

    Get information about the wrapped library.
    \"\"\"
    function library_info()
        return Dict(
            :name => "$(binary.name)",
            :path => _lib_path,
            :loaded => is_loaded(),
            :type => :$(binary.type),
            :arch => "$(binary.arch)",
            :functions => $functions_generated,
            :data => $data_generated,
            :dependencies => $(repr(binary.dependencies))
        )
    end

    """

    # Export statements
    if functions_generated > 0 || data_generated > 0
        exports = String[]

        # Add utility functions
        push!(exports, "is_loaded", "get_load_errors", "get_lib_path", "library_info")

        # Add wrapped functions
        for symbol in binary.symbols
            if symbol["type"] in ["function", "data"]
                push!(exports, make_julia_identifier(symbol["name"]))
            end
        end

        content *= "# Exports\n"
        content *= "export " * join(unique(exports), ", ") * "\n"
    end

    content *= "\nend # module\n"

    return content
end

"""
Generate function wrapper with type inference
"""
function generate_function_wrapper(wrapper::BinaryWrapper, symbol::Dict{String,Any}, lib_name::String)::Union{String,Nothing}
    func_name = symbol["name"]
    julia_name = make_julia_identifier(func_name)

    # Skip if name can't be made valid
    if isempty(julia_name)
        return nothing
    end

    # Infer types
    return_type = infer_julia_type(wrapper, symbol["return_type"])
    param_types = [infer_julia_type(wrapper, p["type"]) for p in symbol["parameters"]]

    # Generate parameter names
    param_names = String[]
    for (i, param) in enumerate(symbol["parameters"])
        if !isempty(param["name"])
            push!(param_names, make_julia_identifier(param["name"]))
        else
            push!(param_names, "arg$i")
        end
    end

    # Build function wrapper
    wrapper_content = """
    \"\"\"
        $julia_name($(join(param_names, ", ")))

    Wrapper for C/C++ function `$func_name`.

    Original signature: `$(symbol["signature"])`
    \"\"\"
    function $julia_name($(join(["$n::$t" for (n, t) in zip(param_names, param_types)], ", ")))
    """

    if wrapper.config.safety_checks
        wrapper_content *= "    @check_loaded()\n"
    end

    wrapper_content *= """
        return ccall(
            (:$func_name, _lib_handle[]),
            $return_type,
            ($(join(param_types, ", "))$(isempty(param_types) ? "" : ",")),
            $(join(param_names, ", "))
        )
    end
    """

    return wrapper_content
end

"""
Generate data wrapper
"""
function generate_data_wrapper(wrapper::BinaryWrapper, symbol::Dict{String,Any}, lib_name::String)::Union{String,Nothing}
    data_name = symbol["name"]
    julia_name = make_julia_identifier(data_name)

    if isempty(julia_name)
        return nothing
    end

    # For data symbols, we create accessor functions
    wrapper_content = """
    \"\"\"
        get_$julia_name()

    Get the value of global variable `$data_name`.
    \"\"\"
    function get_$julia_name()
    """

    if wrapper.config.safety_checks
        wrapper_content *= "    @check_loaded()\n"
    end

    wrapper_content *= """
        ptr = cglobal((:$data_name, _lib_handle[]), Ptr{Cvoid})
        # Note: You may need to adjust the type based on the actual data type
        return unsafe_load(ptr)
    end
    """

    return wrapper_content
end

"""
Infer Julia type from C/C++ type
"""
function infer_julia_type(wrapper::BinaryWrapper, cpp_type::Union{String,Nothing})::String
    if isnothing(cpp_type) || isempty(cpp_type)
        return "Any"  # Conservative default
    end

    # Check type registry
    if haskey(wrapper.type_registry, cpp_type)
        return wrapper.type_registry[cpp_type]
    end

    # Handle pointer types
    if endswith(cpp_type, "*")
        base_type = strip(cpp_type[1:end-1])
        if base_type == "void"
            return "Ptr{Cvoid}"
        elseif haskey(wrapper.type_registry, base_type)
            return "Ptr{$(wrapper.type_registry[base_type])}"
        else
            return "Ptr{Cvoid}"  # Conservative
        end
    end

    # Handle reference types
    if endswith(cpp_type, "&")
        base_type = strip(cpp_type[1:end-1])
        # References are treated as the base type in Julia
        return infer_julia_type(wrapper, base_type)
    end

    # Default to Any for unknown types
    return "Any"
end

"""
Make valid Julia identifier from C/C++ name
"""
function make_julia_identifier(name::String)::String
    # Remove common C++ namespace prefixes
    name = replace(name, r"^.*::" => "")

    # Replace invalid characters
    name = replace(name, r"[^a-zA-Z0-9_]" => "_")

    # Ensure it starts with a letter or underscore
    if !isempty(name) && isdigit(name[1])
        name = "_" * name
    end

    # Avoid Julia reserved words
    reserved = ["begin", "end", "if", "else", "elseif", "while", "for", "function",
        "return", "break", "continue", "macro", "module", "struct", "type",
        "abstract", "primitive", "mutable", "immutable", "using", "import"]

    if name in reserved
        name = name * "_"
    end

    return name
end

"""
Generate module name from binary name
"""
function generate_module_name(name::String)::String
    # Clean the name
    name = replace(name, r"^lib" => "")  # Remove lib prefix
    name = replace(name, r"[^a-zA-Z0-9]" => "_")  # Replace non-alphanumeric

    # Convert to CamelCase
    parts = split(name, "_")
    name = join([uppercasefirst(part) for part in parts if !isempty(part)], "")

    if isempty(name)
        name = "UnknownModule"
    end

    return name
end

"""
Generate test file for wrapper
"""
function generate_tests(wrapper::BinaryWrapper, binary::BinaryInfo, module_name::String)
    if !wrapper.config.generate_tests
        return
    end

    test_content = """
    # Tests for $module_name
    # Auto-generated test file

    using Test
    using $module_name

    @testset "$module_name Tests" begin
        @testset "Library Loading" begin
            @test $module_name.is_loaded() || !isempty($module_name.get_load_errors())
            @test isfile($module_name.get_lib_path())
        end

        @testset "Library Info" begin
            info = $module_name.library_info()
            @test info[:name] == "$(binary.name)"
            @test info[:type] == :$(binary.type)
            @test info[:arch] == "$(binary.arch)"
        end
    """

    # Add function tests
    function_count = 0
    for symbol in binary.symbols
        if symbol["type"] == "function" && function_count < 5  # Limit test generation
            julia_name = make_julia_identifier(symbol["name"])
            if !isempty(julia_name)
                test_content *= """

        @testset "$julia_name" begin
            if $module_name.is_loaded()
                # Test that the function exists
                @test isdefined($module_name, :$julia_name)
                @test isa($module_name.$julia_name, Function)
            end
        end
        """
                function_count += 1
            end
        end
    end

    test_content *= """
    end
    """

    # Write test file
    test_dir = joinpath(wrapper.config.output_dir, "test")
    mkpath(test_dir)
    test_file = joinpath(test_dir, "test_$module_name.jl")

    open(test_file, "w") do f
        write(f, test_content)
    end

    println("    📝 Generated tests: test_$module_name.jl")
end

"""
Generate documentation
"""
function generate_documentation(wrapper::BinaryWrapper, binaries::Vector{BinaryInfo})
    if !wrapper.config.generate_docs
        return
    end

    doc_content = """
    # Binary Wrappers Documentation

    Generated on: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))

    ## Overview

    This directory contains auto-generated Julia wrappers for the following binaries:

    | Module | Binary | Type | Functions | Data |
    |--------|--------|------|-----------|------|
    """

    for binary in binaries
        module_name = generate_module_name(binary.name)
        func_count = count(s -> s["type"] == "function", binary.symbols)
        data_count = count(s -> s["type"] == "data", binary.symbols)

        doc_content *= "| $module_name | $(binary.name) | $(binary.type) | $func_count | $data_count |\n"
    end

    doc_content *= """

    ## Usage

    To use a wrapped library:

    ```julia
    using .ModuleName

    # Check if library is loaded
    if ModuleName.is_loaded()
        # Use the wrapped functions
        result = ModuleName.some_function(args...)
    else
        # Check what went wrong
        errors = ModuleName.get_load_errors()
        println("Load errors: ", errors)
    end
    ```

    ## Binary Details

    """

    for binary in binaries
        module_name = generate_module_name(binary.name)
        doc_content *= """
        ### $module_name

        - **Path**: `$(binary.path)`
        - **Architecture**: $(binary.arch)
        - **Dependencies**: $(join(binary.dependencies, ", "))

        """

        # List some functions
        func_count = 0
        for symbol in binary.symbols
            if symbol["type"] == "function" && func_count < 10
                julia_name = make_julia_identifier(symbol["name"])
                if !isempty(julia_name)
                    doc_content *= "- `$julia_name$(symbol["signature"])`\n"
                    func_count += 1
                end
            end
        end

        if func_count == 10
            remaining = count(s -> s["type"] == "function", binary.symbols) - 10
            if remaining > 0
                doc_content *= "- ... and $remaining more functions\n"
            end
        end

        doc_content *= "\n"
    end

    # Write documentation
    doc_file = joinpath(wrapper.config.output_dir, "README.md")
    open(doc_file, "w") do f
        write(f, doc_content)
    end

    println("\n📚 Generated documentation: README.md")
end

"""
Generate main module that includes all wrappers
"""
function generate_main_module(wrapper::BinaryWrapper, module_names::Vector{String})
    content = """
    # Main module for binary wrappers
    # Generated on $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))

    module BinaryWrappers

    # Include all wrapper modules
    """

    for module_name in module_names
        content *= "include(\"$module_name.jl\")\n"
    end

    content *= "\n# Import modules\n"

    for module_name in module_names
        content *= "using .$module_name\n"
        content *= "export $module_name\n"
    end

    content *= """

    \"\"\"
        status()

    Check the status of all wrapped binaries.
    \"\"\"
    function status()
        println("Binary Wrappers Status")
        println("=" ^ 40)

    """

    for module_name in module_names
        content *= """
        try
            info = $module_name.library_info()
            status = info[:loaded] ? "✅ LOADED" : "❌ NOT LOADED"
            println("$(rpad(info[:name], 20)) \$status")
        catch e
            println("$(rpad("$module_name", 20)) ⚠️  ERROR: \$e")
        end

        """
    end

    content *= """
    end

    export status

    # Show status on load
    status()

    end # module
    """

    # Write main module
    main_file = joinpath(wrapper.config.output_dir, "BinaryWrappers.jl")
    open(main_file, "w") do f
        write(f, content)
    end

    println("✅ Generated main module: BinaryWrappers.jl")
end

"""
Main wrapper generation workflow
"""
function generate_wrappers(wrapper::BinaryWrapper; specific_binary::Union{String,Nothing}=nothing)
    println("🚀 Universal Binary Wrapper Generator")
    println("="^50)
    println("📁 Project: $(wrapper.config.project_root)")
    println("📁 Output:  $(wrapper.config.output_dir)")
    println("🔧 Style:   $(wrapper.config.wrapper_style)")
    println("🔍 Detection: $(wrapper.config.symbol_detection)")
    println("="^50)

    # Create output directory
    mkpath(wrapper.config.output_dir)

    # Scan for binaries
    binaries = if !isnothing(specific_binary)
        # Wrap specific binary
        binary_type = identify_binary_type(specific_binary)
        if binary_type != :unknown
            info = analyze_binary(wrapper, specific_binary, binary_type)
            isnothing(info) ? BinaryInfo[] : [info]
        else
            println("❌ Unknown binary type: $specific_binary")
            BinaryInfo[]
        end
    else
        scan_binaries(wrapper)
    end

    if isempty(binaries)
        println("❌ No binaries found to wrap")
        return
    end

    # Generate wrappers
    generated_modules = String[]

    for binary in binaries
        println("\n🔧 Processing: $(binary.name)")
        println("   Type: $(binary.type)")
        println("   Symbols: $(length(binary.symbols))")

        # Generate wrapper
        wrapper_content = generate_wrapper(wrapper, binary)
        module_name = generate_module_name(binary.name)

        # Write wrapper file
        wrapper_file = joinpath(wrapper.config.output_dir, "$module_name.jl")
        open(wrapper_file, "w") do f
            write(f, wrapper_content)
        end

        push!(generated_modules, module_name)
        println("   ✅ Generated: $module_name.jl")

        # Generate tests
        generate_tests(wrapper, binary, module_name)
    end

    # Generate main module
    if length(generated_modules) > 1
        generate_main_module(wrapper, generated_modules)
    end

    # Generate documentation
    generate_documentation(wrapper, binaries)

    # Save wrapper metadata
    save_wrapper_metadata(wrapper, binaries, generated_modules)

    println("\n🎉 Wrapper generation complete!")
    println("📁 Output: $(wrapper.config.output_dir)")
    println("📦 Modules: $(join(generated_modules, ", "))")

    return generated_modules
end

"""
Save wrapper generation metadata
"""
function save_wrapper_metadata(wrapper::BinaryWrapper, binaries::Vector{BinaryInfo}, modules::Vector{String})
    metadata = Dict(
        "timestamp" => now(),
        "config" => wrapper.config,
        "binaries" => [
            Dict(
                "name" => b.name,
                "path" => b.path,
                "type" => string(b.type),
                "arch" => b.arch,
                "symbols_count" => length(b.symbols),
                "dependencies" => b.dependencies
            ) for b in binaries
        ],
        "modules" => modules,
        "type_registry_size" => length(wrapper.type_registry)
    )

    metadata_file = joinpath(wrapper.config.output_dir, "wrapper_metadata.json")
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
        RepliBuild JuliaWrapItUp - Universal Binary Wrapper Generator

        Usage:
            julia JuliaWrapItUp.jl init [project_dir]
            julia JuliaWrapItUp.jl wrap [config_file]
            julia JuliaWrapItUp.jl wrap-binary <binary_path> [config_file]
            julia JuliaWrapItUp.jl integrate [llvmake_dir]
            julia JuliaWrapItUp.jl info [config_file]

        Examples:
            julia JuliaWrapItUp.jl init
            julia JuliaWrapItUp.jl wrap
            julia JuliaWrapItUp.jl wrap-binary ./lib/libmath.so
            julia JuliaWrapItUp.jl integrate ../llvmake_output
        """)
        return
    end

    command = ARGS[1]

    if command == "init"
        # Initialize wrapper configuration
        project_dir = length(ARGS) >= 2 ? ARGS[2] : "."
        cd(project_dir)

        config = create_default_wrapper_config()
        save_wrapper_config(config, "wrapper_config.toml")

        println("✅ Initialized wrapper configuration")
        println("📝 Edit wrapper_config.toml to customize settings")

    elseif command == "wrap"
        # Generate wrappers for all binaries
        config_file = length(ARGS) >= 2 ? ARGS[2] : "wrapper_config.toml"
        wrapper = BinaryWrapper(config_file)
        generate_wrappers(wrapper)

    elseif command == "wrap-binary"
        # Wrap specific binary
        if length(ARGS) < 2
            println("Usage: julia JuliaWrapItUp.jl wrap-binary <binary_path> [config_file]")
            return
        end

        binary_path = ARGS[2]
        config_file = length(ARGS) >= 3 ? ARGS[3] : "wrapper_config.toml"

        wrapper = BinaryWrapper(config_file)
        generate_wrappers(wrapper, specific_binary=binary_path)

    elseif command == "integrate"
        # Integrate with LLVMake output
        if length(ARGS) < 2
            println("Usage: julia JuliaWrapItUp.jl integrate [llvmake_dir]")
            return
        end

        llvmake_dir = ARGS[2]

        # Look for LLVMake metadata
        metadata_path = joinpath(llvmake_dir, "compilation_metadata.json")
        if !isfile(metadata_path)
            println("❌ LLVMake metadata not found: $metadata_path")
            return
        end

        # Create configuration that inherits from LLVMake
        config = create_wrapper_config_from_stage1()
        config = WrapperConfig(
            config.project_root,
            [llvmake_dir],  # Use LLVMake output as binary source
            joinpath(dirname(llvmake_dir), "julia_wrappers_stage2"),
            config.header_dirs,
            config.wrapper_style,
            config.symbol_detection,
            config.demangle_cpp,
            config.generate_tests,
            config.generate_docs,
            config.safety_checks,
            config.use_headers,
            config.header_parser,
            config.type_hints,
            metadata_path,
            true
        )

        save_wrapper_config(config, "wrapper_config_stage2.toml")

        # Generate wrappers
        wrapper = BinaryWrapper("wrapper_config_stage2.toml")
        generate_wrappers(wrapper)

    elseif command == "info"
        # Show wrapper information
        config_file = length(ARGS) >= 2 ? ARGS[2] : "wrapper_config.toml"
        wrapper = BinaryWrapper(config_file)

        println("Wrapper Configuration:")
        println("="^50)
        println("Project: $(wrapper.config.project_root)")
        println("Binary dirs: $(join(wrapper.config.binary_dirs, ", "))")
        println("Output: $(wrapper.config.output_dir)")
        println("Style: $(wrapper.config.wrapper_style)")
        println("Symbol detection: $(wrapper.config.symbol_detection)")
        println("Type registry: $(length(wrapper.type_registry)) mappings")

        if !isnothing(wrapper.config.stage1_metadata)
            println("\nStage 1 Integration:")
            println("Metadata: $(wrapper.config.stage1_metadata)")
            println("Inherit mappings: $(wrapper.config.inherit_mappings)")
        end

    else
        println("Unknown command: $command")
    end
end

# Exports
export BinaryWrapper, WrapperConfig, BinaryInfo
export generate_wrappers, scan_binaries, analyze_binary
export extract_symbols, generate_wrapper, generate_documentation
export load_wrapper_config, save_wrapper_config
export create_wrapper_config_from_stage1, create_default_wrapper_config

end # module JuliaWrapItUp

# Run if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    JuliaWrapItUp.main()
end
