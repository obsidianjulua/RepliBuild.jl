# =============================================================================
# RUST WRAPPER GENERATOR
# =============================================================================

import Dates
using JSON

"""
    generate_introspective_module_rust(config::RepliBuildConfig, lib_path::String, metadata, module_name::String, registry::TypeRegistry, generate_docs::Bool, thunks_lib_path::String="")

Generate an introspective Julia wrapper module for a Rust library.
Rust uses a cleaner DWARF output compared to C++, avoiding STL leakages and mangling.
"""
function generate_introspective_module_rust(config::RepliBuildConfig, lib_path::String,
                                      metadata, module_name::String,
                                      registry::TypeRegistry, generate_docs::Bool,
                                      thunks_lib_path::String="")

    exports = String[]

    header = """
    # Auto-generated Julia wrapper for $(config.project.name)
    # Generated: $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))
    # Generator: RepliBuild Wrapper (Rust Introspective)
    # Library: $(basename(lib_path))
    # Metadata: compilation_metadata.json

    module $module_name

    const Cintptr_t = Int
    const Cuintptr_t = UInt
    const Cssize_t = Int
    const Csize_t = UInt

    using Libdl
    import RepliBuild
    import Base: unsafe_convert

    const LIBRARY_PATH = "$(abspath(lib_path))"
    const THUNKS_LIBRARY_PATH = "$(thunks_lib_path)"

    # Verify library exists
    if !isfile(LIBRARY_PATH)
        error("Library not found: \$LIBRARY_PATH")
    end

    """

    content = header

    # Extract structs and enums
    all_structs = get(metadata, "struct_definitions", Dict{String,Any}())
    dwarf_structs = Dict{String,Any}()
    dwarf_enums = Dict{String,Any}()
    for (k, v) in all_structs
        if startswith(k, "__enum__")
            dwarf_enums[replace(k, "__enum__" => "")] = v
        else
            dwarf_structs[k] = v
        end
    end
    functions = get(metadata, "functions", [])

    # DWARF Structs
    if !isempty(dwarf_structs)
        content *= "# =============================================================================\n"
        content *= "# STRUCTS & OPAQUE TYPES\n"
        content *= "# =============================================================================\n\n"

        for (name, info) in dwarf_structs
            # Filter out Rust internal stdlib/core types that leak into DWARF
            if occursin("::", name) || occursin("<", name) || name in ["TryReserveErrorKind", "NulError", "Formatter", "Count", "CStr", "Placeholder", "Arguments", "Error", "Result", "Option"]
                continue
            end

            julia_name = make_rust_identifier(name)
            
            # Check if it's an opaque type (no members or marked as such)
            members = get(info, "members", [])
            if isempty(members)
                # Opaque type forward declaration
                content *= "struct $julia_name end\n"
                push!(exports, julia_name)
                continue
            end

            # It's a regular struct
            content *= "struct $julia_name\n"
            for member in members
                m_name = make_rust_identifier(get(member, "name", "unknown"))
                m_type = get(member, "c_type", "void")
                julia_type = infer_rust_type(registry, m_type, context="Struct $name member $m_name")
                content *= "    $m_name::$julia_type\n"
            end
            content *= "end\n\n"
            push!(exports, julia_name)
        end
    end

    # DWARF Enums
    if !isempty(dwarf_enums)
        content *= "# =============================================================================\n"
        content *= "# ENUMS\n"
        content *= "# =============================================================================\n\n"

        for (name, info) in dwarf_enums
            members = get(info, "members", [])
            if isempty(members)
                continue
            end
            
            julia_name = make_rust_identifier(name)
            
            # In Rust, enums in DWARF are usually standard C-compatible enums 
            # if they have DW_TAG_enumerator. Let's assume Cint for the enum base type.
            content *= "@enum $julia_name::Cint begin\n"
            
            for member in members
                m_name = make_rust_identifier(get(member, "name", "UNKNOWN"))
                m_val = get(member, "value", "0")
                content *= "    $(m_name) = $m_val\n"
            end
            content *= "end\n\n"
            push!(exports, julia_name)
        end
    end

    # Functions
    if !isempty(functions)
        content *= "# =============================================================================\n"
        content *= "# FUNCTIONS\n"
        content *= "# =============================================================================\n\n"

        for func in functions
            # Rust extern "C" functions are clean, we use them directly
            raw_name = get(func, "name", "")
            julia_name = make_rust_identifier(raw_name)
            
            if isempty(raw_name) || startswith(raw_name, "_")
                continue # Skip internal rust symbols
            end

            # Return type
            ret_info = get(func, "return_type", Dict())
            c_ret_type = get(ret_info, "c_type", "void")
            julia_ret_type = infer_rust_type(registry, c_ret_type, context="Return type of $raw_name")

            # Parameters
            params = get(func, "parameters", [])
            param_names = String[]
            param_types = String[]
            
            for (i, p) in enumerate(params)
                p_name = get(p, "name", "arg$i")
                p_c_type = get(p, "c_type", "void")
                p_julia_name = make_rust_identifier(p_name)
                p_julia_type = infer_rust_type(registry, p_c_type, context="Parameter $p_name of $raw_name")
                
                push!(param_names, p_julia_name)
                push!(param_types, p_julia_type)
            end

            # Docs
            if generate_docs
                content *= "\"\"\"\n"
                content *= "    $julia_name($(join(param_names, ", ")))\n\n"
                content *= "Wrapper for Rust function `$raw_name`.\n"
                content *= "\"\"\"\n"
            end

            # Function body
            sig = join(["$(n)::$(t)" for (n, t) in zip(param_names, param_types)], ", ")
            ccall_types = isempty(param_types) ? "()" : "($(join(param_types, ", ")),)"
            ccall_args = isempty(param_names) ? "" : ", $(join(param_names, ", "))"

            content *= "function $julia_name($sig)::$julia_ret_type\n"
            content *= "    return ccall((:$raw_name, LIBRARY_PATH), $julia_ret_type, $ccall_types$ccall_args)\n"
            content *= "end\n\n"
            
            push!(exports, julia_name)
        end
    end

    # Exports
    content *= "# =============================================================================\n"
    content *= "# EXPORTS\n"
    content *= "# =============================================================================\n\n"
    content *= "export " * join(unique(exports), ", ") * "\n\n"

    content *= "end # module $module_name\n"

    return content
end
