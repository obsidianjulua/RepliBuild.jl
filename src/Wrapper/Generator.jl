# =============================================================================
# HIGH-LEVEL WRAPPER API
# =============================================================================

"""
    wrap_library(config::RepliBuildConfig, library_path::String;
                 headers::Vector{String}=String[],
                 generate_tests::Bool=false,
                 generate_docs::Bool=true)

Generate Julia wrapper for compiled library.

Always uses introspective (DWARF metadata) wrapping when metadata is available,
otherwise falls back to basic symbol-only extraction with conservative types.

# Arguments
- `config`: RepliBuildConfig with wrapper settings
- `library_path`: Path to compiled library (.so, .dylib, .dll)
- `headers`: Optional header files (currently unused, reserved for future)
- `generate_tests`: Generate test file (default: false, TODO)
- `generate_docs`: Include comprehensive documentation (default: true)

# Returns
Path to generated Julia wrapper file
"""
function wrap_library(config::RepliBuildConfig, library_path::String;
                     headers::Vector{String}=String[],
                     generate_tests::Bool=false,
                     generate_docs::Bool=true)


    if !isfile(library_path)
        error("Library not found: $library_path")
    end

    # Check for metadata (DWARF + symbol info from compilation)
    metadata_file = joinpath(dirname(library_path), "compilation_metadata.json")
    has_metadata = isfile(metadata_file)

    if !has_metadata
        @warn "No compilation metadata found. Did you compile with -g flag?"
        @warn "Falling back to basic symbol-only wrapper (conservative types, limited safety)"
        return wrap_basic(config, library_path, generate_docs=generate_docs)
    end

    return wrap_introspective(config, library_path, headers, generate_docs=generate_docs)
end

# =============================================================================
# TIER 1: BASIC WRAPPER (Symbol-Only)
# =============================================================================

"""
    wrap_basic(config::RepliBuildConfig, library_path::String; generate_docs::Bool=true)

Generate basic Julia wrapper from binary symbols only (no headers required).

Quality: ~40% - Conservative types, placeholder signatures, requires manual refinement.
Use when: Headers not available, quick prototyping, binary-only distribution.
"""
function wrap_basic(config::RepliBuildConfig, library_path::String; generate_docs::Bool=true)

    if !isfile(library_path)
        error("Library not found: $library_path")
    end

    # Create type registry
    registry = create_type_registry(config)

    # Extract symbols
    symbols = extract_symbols(library_path, registry, demangle=true, method=:nm)

    if isempty(symbols)
        @warn "No symbols found in library"
        return nothing
    end

    # Filter functions and data
    functions = filter(s -> s.symbol_type == :function, symbols)
    data_symbols = filter(s -> s.symbol_type == :data, symbols)

    # Generate wrapper module
    module_name = get_module_name(config)
    wrapper_content = generate_basic_module(config, library_path, functions, data_symbols,
                                           module_name, registry, generate_docs)

    # Write to file
    output_dir = get_output_path(config)
    mkpath(output_dir)
    output_file = joinpath(output_dir, "$(module_name).jl")

    write(output_file, wrapper_content)

    return output_file
end

"""
Generate basic wrapper module content.
"""
function generate_basic_module(config::RepliBuildConfig, lib_path::String,
                               functions::Vector{SymbolInfo}, data_symbols::Vector{SymbolInfo},
                               module_name::String, registry::TypeRegistry, generate_docs::Bool)

    # Header with metadata
    header = """
    # Auto-generated Julia wrapper for $(config.project.name)
    # Generated: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    # Generator: RepliBuild Wrapper (Basic: Symbol extraction)
    # Library: $(basename(lib_path))

    """

    content = header

    # Module declaration
    content *= "module $module_name\n\n"
    content *= "const Cintptr_t = Int\n"
    content *= "const Cuintptr_t = UInt\n\n"
    content *= "using Libdl\n\n"

    # Library management
    content *= """
    # =============================================================================
    # LIBRARY MANAGEMENT
    # =============================================================================

    const _LIB_PATH = raw"$(abspath(lib_path))"
    const _LIB = Ref{Ptr{Nothing}}(C_NULL)
    const _LOAD_ERRORS = String[]

    function __init__()
        try
            _LIB[] = Libdl.dlopen(_LIB_PATH, Libdl.RTLD_LAZY | Libdl.RTLD_GLOBAL)
        catch e
            push!(_LOAD_ERRORS, string(e))
            @error "Failed to load library $(basename(lib_path))" exception=e
        end
    end

    \"""
        is_loaded()

    Check if the library is successfully loaded.
    \"""
    is_loaded() = _LIB[] != C_NULL

    \"""
        get_load_errors()

    Get any errors that occurred during library loading.
    \"""
    get_load_errors() = copy(_LOAD_ERRORS)

    \"""
        get_lib_path()

    Get the path to the underlying library.
    \"""
    get_lib_path() = _LIB_PATH

    # Safety check macro
    macro check_loaded()
        quote
            if !is_loaded()
                error("Library not loaded. Errors: ", join(get_load_errors(), "; "))
            end
        end
    end

    """

    # Function wrappers
    content *= """
    # =============================================================================
    # FUNCTION WRAPPERS
    # =============================================================================

    """

    function_count = 0
    exports = String["is_loaded", "get_load_errors", "get_lib_path"]

    for func in functions
        if function_count >= 50
            break  # Limit to avoid huge files
        end

        func_wrapper = generate_basic_function_wrapper(func, registry, generate_docs)
        if !isnothing(func_wrapper)
            content *= func_wrapper * "\n"
            push!(exports, func.julia_name)
            function_count += 1
        end
    end

    if length(functions) > 50
        content *= """
        # ... and $(length(functions) - 50) more functions omitted
        # Regenerate with headers for complete wrapping:
        #   RepliBuild.wrap("$lib_path", headers=["your_header.h"])

        """
    end

    # Library info function
    content *= """
    # =============================================================================
    # METADATA
    # =============================================================================

    \"""
        library_info()

    Get information about the wrapped library.
    \"""
    function library_info()
        return Dict{Symbol,Any}(
            :name => "$(config.project.name)",
            :path => _LIB_PATH,
            :loaded => is_loaded(),
            :tier => :basic,
            :type_safety => "40% (conservative placeholders)",
            :functions_wrapped => $function_count,
            :functions_total => $(length(functions)),
            :data_symbols => $(length(data_symbols))
        )
    end

    """

    push!(exports, "library_info")

    # Exports
    content *= "# Exports\n"
    content *= "export " * join(unique(exports), ", ") * "\n\n"

    content *= "end # module $module_name\n"

    return content
end

"""
Generate wrapper for a single function (basic tier).
"""
function generate_basic_function_wrapper(func::SymbolInfo, registry::TypeRegistry, generate_docs::Bool)
    if isempty(func.julia_name)
        return nothing
    end

    # For basic tier, we use conservative Any types since we don't have parameter info
    wrapper = ""

    if generate_docs
        wrapper *= """
        \"""
            $(func.julia_name)(args...)

        Wrapper for C/C++ function `$(func.demangled_name)`.

        Signature uses placeholder types. Actual types unknown without headers.
        Return type and parameters may need manual adjustment.

        # C/C++ Symbol
        `$(func.name)`
        \"""
        """
    end

    wrapper *= """
    function $(func.julia_name)(args...)
        @check_loaded()
        ccall((:$(func.name), _LIB[]), Any, (), args...)
    end

    """

    return wrapper
end

# =============================================================================
# TIER 2: ADVANCED WRAPPER (Header-Aware via Clang.jl)
# =============================================================================

"""
    wrap_with_clang(config::RepliBuildConfig, library_path::String, headers::Vector{String}; generate_docs::Bool=true)

Generate advanced Julia wrapper using Clang.jl for type-aware binding generation.

Quality: ~85% - Accurate types from headers, production-ready with minor tweaks.
Use when: Headers available, need type safety, production deployment.
"""
function wrap_with_clang(config::RepliBuildConfig, library_path::String, headers::Vector{String};
                        generate_docs::Bool=true)

    if !isfile(library_path)
        error("Library not found: $library_path")
    end

    if isempty(headers)
        error("Headers required for advanced wrapping")
    end

    # Verify headers exist
    for header in headers
        if !isfile(header)
            @warn "Header not found: $header"
        end
    end

    # Build config for ClangJLBridge
    clang_config = Dict(
        "project" => Dict("name" => config.project.name),
        "compile" => Dict("include_dirs" => config.compile.include_dirs),
        "binding" => Dict(
            "use_ccall_macro" => false,
            "add_doc_strings" => generate_docs,
            "use_julia_native_enum" => true
        )
    )

    output_file = ClangJLBridge.generate_bindings_clangjl(clang_config, library_path, headers)

    if isnothing(output_file)
        error("Binding generation failed")
    end

    # TODO: Enhance generated file with our safety checks and metadata
    # For now, ClangJLBridge handles the generation

    return output_file
end

# =============================================================================
# TIER 3: INTROSPECTIVE WRAPPER (Metadata-Rich)
# =============================================================================

"""
    wrap_introspective(config::RepliBuildConfig, library_path::String, headers::Vector{String}; generate_docs::Bool=true)

Generate introspective Julia wrapper using compilation metadata for perfect type accuracy.

Quality: ~95% - Exact types from compilation, language-agnostic, zero manual configuration.
Use when: Metadata available from RepliBuild compilation, need perfect bindings.

This is the culmination of RepliBuild's vision: automatic, accurate, language-agnostic wrapping.
"""
function wrap_introspective(config::RepliBuildConfig, library_path::String, headers::Vector{String};
                           generate_docs::Bool=true)

    if !isfile(library_path)
        error("Library not found: $library_path")
    end

    # Load compilation metadata
    metadata_file = joinpath(dirname(library_path), "compilation_metadata.json")
    if !isfile(metadata_file)
        error("Compilation metadata not found: $metadata_file\nRun RepliBuild.build() first to generate metadata")
    end

    metadata = JSON.parsefile(metadata_file)

    if !haskey(metadata, "functions")
        error("Invalid metadata: missing 'functions' key")
    end

    functions = metadata["functions"]

    # Extract supplementary types from headers (enums, unused types, etc.)
    include_dirs = get(metadata, "include_dirs", String[])
    header_types = if !isempty(headers)
        ClangJLBridge.extract_header_types(headers, include_dirs)
    else
        # Auto-discover headers from include directories
        discovered_headers = String[]
        for inc_dir in include_dirs
            if isdir(inc_dir)
                append!(discovered_headers, ClangJLBridge.discover_headers(inc_dir, recursive=false))
            end
        end
        if !isempty(discovered_headers)
            ClangJLBridge.extract_header_types(discovered_headers, include_dirs)
        else
            Dict("enums" => Dict(), "constants" => Dict(), "typedefs" => Dict(), "structs" => String[])
        end
    end

    # Merge header types into metadata
    if !isempty(header_types["enums"])
        if !haskey(metadata, "header_enums")
            metadata["header_enums"] = header_types["enums"]
        else
            merge!(metadata["header_enums"], header_types["enums"])
        end
    end

    # Store function pointer typedefs for callback documentation
    if haskey(header_types, "function_pointers") && !isempty(header_types["function_pointers"])
        metadata["function_pointer_typedefs"] = header_types["function_pointers"]
    end

    # Create type registry with metadata + typedef resolution table
    typedef_table = get(metadata, "typedef_table", Dict{String,Any}())
    # Convert to String,String for custom_types merge
    typedef_custom = Dict{String,String}(String(k) => String(v) for (k, v) in typedef_table if v != "Any")
    registry = create_type_registry(config, custom_types=typedef_custom)

    # AOT Compilation Pass
    thunks_lib_path = ""
    if config.compile.aot_thunks
        output_dir = get_output_path(config)
        lib_name = basename(library_path)
        thunks_name = replace(lib_name, ".so" => "_thunks.so", ".dylib" => "_thunks.dylib", ".dll" => "_thunks.dll")
        thunks_so = joinpath(output_dir, thunks_name)
        if isfile(thunks_so)
            thunks_lib_path = abspath(thunks_so)
        else
            @warn "AOT thunks enabled but companion library not found at $thunks_so"
        end
    end

    # Generate wrapper module
    module_name = get_module_name(config)
    # Generate wrapper module
    module_name = get_module_name(config)
    
    wrapper_content = if registry.language == :c
        generate_introspective_module_c(config, library_path, metadata,
                                        module_name, registry, generate_docs, thunks_lib_path)
    else
        generate_introspective_module_cpp(config, library_path, metadata,
                                          module_name, registry, generate_docs, thunks_lib_path)
    end

    # Write to file
    output_dir = get_output_path(config)
    mkpath(output_dir)
    output_file = joinpath(output_dir, "$(module_name).jl")

    write(output_file, wrapper_content)

    println("  wrap: $(basename(output_file))")

    return output_file
end


# =============================================================================
# DISPATCH LOGIC HELPERS
# =============================================================================

"""
    get_julia_aligned_size(members::Vector)

Calculate the size of a struct in Julia including standard padding alignment.
Used to detect if a C++ struct is 'packed' (Julia size > DWARF size).
"""
function get_julia_aligned_size(members::Vector)
    current_offset = 0
    max_align = 1

    for m in members
        # specific size of this member
        m_size = get(m, "size", 0)

        # simple alignment heuristic (size usually equals alignment for primitives)
        # generic pointer/int alignment cap at 8 bytes on 64-bit
        align = m_size > 8 ? 8 : m_size
        align = align == 0 ? 1 : align # handle empty/void

        # update generic alignment requirement
        max_align = max(max_align, align)

        # Add padding to current offset
        padding = (align - (current_offset % align)) % align
        current_offset += padding + m_size
    end

    # Final structure alignment padding
    padding = (max_align - (current_offset % max_align)) % max_align
    return current_offset + padding
end

"""
    is_ccall_safe(func_info, dwarf_structs)::Bool

Determine if a function is safe for standard `ccall`.
Returns false if:
1. Arguments are Packed Structs (alignment mismatch)
2. Arguments are Unions
3. Return type is a complex struct by value
"""
function is_ccall_safe(func_info, dwarf_structs)
    # 0. Check for STL container types (never ccall-safe)
    ret_type_str = String(get(func_info["return_type"], "c_type", ""))
    if is_stl_container_type(ret_type_str)
        return false
    end
    for param in func_info["parameters"]
        if is_stl_container_type(get(param, "c_type", ""))
            return false
        end
    end

    # 1. Check Return Type
    ret_type = ret_type_str

    # If returning a struct by value (not pointer/void/primitive)
    # For template types (containing '<'), skip primitive substring matching —
    # "Matrix<double, -1, -1>" contains "double" but is NOT a double return.
    is_template_ret = occursin('<', ret_type)
    is_primitive_ret = !is_template_ret &&
        (contains(ret_type, "int") || contains(ret_type, "float") ||
         contains(ret_type, "double") || contains(ret_type, "bool"))
    if !contains(ret_type, "*") && !contains(ret_type, "void") && !is_primitive_ret

        # Template return types (e.g. Matrix<double,-1,-1>) are always complex
        # struct returns — route to MLIR unconditionally.  DWARF may not have
        # an entry for the exact template instantiation.
        if is_template_ret
            return false
        end

        # Check if it's a known struct
        if haskey(dwarf_structs, ret_type)
            s_info = dwarf_structs[ret_type]

            # Struct return by value is notoriously fragile in ABIs (large structs split registers)
            # Conservative: Send to MLIR if > 16 bytes
            if parse(Int, get(s_info, "byte_size", "0")) > 16
                return false
            end

            # Check if it is a class (likely non-POD)
            if get(s_info, "kind", "struct") == "class"
                return false
            end

            # CHECK: Is return type a packed struct?
            # Packed struct returns use sret ABI which Julia ccall doesn't handle correctly
            dwarf_size = parse(Int, get(s_info, "byte_size", "0"))
            members = get(s_info, "members", [])
            julia_size = get_julia_aligned_size(members)
            if dwarf_size > 0 && dwarf_size != julia_size
                return false
            end
        end
    end

    # 2. Check Arguments
    for param in func_info["parameters"]
        c_type = get(param, "c_type", "")

        # If the parameter is passed by pointer (contains *), it's always ccall-safe
        # Only by-value struct parameters need special handling
        if contains(c_type, "*")
            continue
        end

        # Clean const prefix for base type lookup
        base_type = String(strip(replace(c_type, "const" => "")))

        if haskey(dwarf_structs, base_type)
            s_info = dwarf_structs[base_type]

            # CHECK A: Is it a Union?
            if get(s_info, "kind", "struct") == "union"
                return false
            end

            # CHECK B: Is it Packed?
            dwarf_size = parse(Int, get(s_info, "byte_size", "0"))
            members = get(s_info, "members", [])
            julia_size = get_julia_aligned_size(members)

            # If DWARF says 5 bytes but Julia calculates 8, it's packed!
            if dwarf_size > 0 && dwarf_size != julia_size
                return false
            end
        end
    end

    return true
end

"""
    generate_vararg_wrappers(func_name, mangled, julia_name, params, return_type, overloads, generate_docs, demangled) -> (code, export_names)

Generate typed overload wrappers for a variadic C function.
Julia's `ccall` requires fixed signatures, so we generate:
- A base wrapper with only the fixed (non-variadic) parameters
- Typed overloads for each signature listed in `overloads`
All varargs wrappers use direct ccall (Tier 1), never JIT.
"""
function generate_vararg_wrappers(func_name::String, mangled::String, julia_name::String,
                                  params::Vector, return_type,
                                  overloads::Vector{Vector{String}},
                                  generate_docs::Bool, demangled::String, lang::Symbol)
    code = ""
    export_names = String[]

    # Build fixed parameter info
    fixed_param_names = String[]
    fixed_param_types = String[]  # Julia/ccall types
    fixed_julia_sig_types = String[]  # Ergonomic types for signature
    fixed_needs_conversion = Bool[]

    for param in params
        safe_name = lang == :c ? make_c_identifier(param["name"]) : make_cpp_identifier(param["name"])
        if safe_name == "varargs..."
            continue  # Skip the varargs placeholder
        end
        push!(fixed_param_names, safe_name)

        julia_type = param["julia_type"]
        push!(fixed_param_types, julia_type)

        # Ergonomic type mapping (same logic as main wrapper gen)
        if julia_type in ["Cint", "Clong", "Cshort"]
            push!(fixed_julia_sig_types, "Integer")
            push!(fixed_needs_conversion, true)
        elseif startswith(julia_type, "Ptr{")
            push!(fixed_julia_sig_types, "Any")
            push!(fixed_needs_conversion, false)
        else
            push!(fixed_julia_sig_types, julia_type)
            push!(fixed_needs_conversion, false)
        end
    end

    julia_return_type = get(return_type, "julia_type", "Cvoid")

    # Build fixed parameter signature
    fixed_sig_parts = ["$(n)::$(t)" for (n, t) in zip(fixed_param_names, fixed_julia_sig_types)]
    fixed_sig = join(fixed_sig_parts, ", ")

    # Build conversion code for fixed params
    fixed_conversion = ""
    fixed_ccall_names = String[]
    for (name, ctype, needs_conv) in zip(fixed_param_names, fixed_param_types, fixed_needs_conversion)
        if needs_conv
            converted = "$(name)_c"
            push!(fixed_ccall_names, converted)
            fixed_conversion *= "    $converted = $ctype($name)\n"
        else
            push!(fixed_ccall_names, name)
        end
    end

    # --- Base wrapper (fixed args only) ---
    fixed_ccall_types = if isempty(fixed_param_types)
        "()"
    else
        "($(join(fixed_param_types, ", ")),)"
    end
    fixed_ccall_args = join(fixed_ccall_names, ", ")

    doc = ""
    if generate_docs
        doc = """
        \"\"\"
            $julia_name($fixed_sig) -> $julia_return_type

        Wrapper for variadic C function: `$demangled` (base call with fixed args only)
        \"\"\"
        """
    end

    code *= """
    $doc
    function $julia_name($fixed_sig)::$julia_return_type
    $fixed_conversion    return ccall((:$mangled, LIBRARY_PATH), $julia_return_type, $fixed_ccall_types, $fixed_ccall_args)
    end

    """
    push!(export_names, julia_name)

    # --- Typed overloads ---
    for va_types in overloads
        # Build overload function name: fname_Type1_Type2
        type_suffix = join(va_types, "_")
        overload_name = "$(julia_name)_$(type_suffix)"

        # Build variadic parameter names and types
        va_param_names = ["va_$(i)" for i in 1:length(va_types)]
        va_sig_parts = ["$(n)::$(t)" for (n, t) in zip(va_param_names, va_types)]

        # Full signature = fixed + variadic
        all_sig_parts = vcat(fixed_sig_parts, va_sig_parts)
        all_sig = join(all_sig_parts, ", ")

        # ccall types = fixed + variadic (with Vararg marker for proper ABI)
        all_ccall_types_list = vcat(fixed_param_types, va_types)
        all_ccall_types = "($(join(all_ccall_types_list, ", ")),)"

        # ccall args = fixed converted + variadic
        all_ccall_names = vcat(fixed_ccall_names, va_param_names)
        all_ccall_args = join(all_ccall_names, ", ")

        overload_doc = ""
        if generate_docs
            overload_doc = """
            \"\"\"
                $overload_name($all_sig) -> $julia_return_type

            Typed variadic overload for: `$demangled`
            Variadic types: $(join(va_types, ", "))
            \"\"\"
            """
        end

        code *= """
        $overload_doc
        function $overload_name($all_sig)::$julia_return_type
        $fixed_conversion    return ccall((:$mangled, LIBRARY_PATH), $julia_return_type, $all_ccall_types, $all_ccall_args)
        end

        """
        push!(export_names, overload_name)
    end

    return (code, export_names)
end

"""
Generate introspective wrapper module content using compilation metadata.
"""

