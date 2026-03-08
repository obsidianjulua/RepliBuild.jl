# =============================================================================
# FUNCTION POINTER SIGNATURE PARSING
# =============================================================================

"""
    parse_function_pointer_signature(fp_sig::String)::Union{String, Nothing}

Parse DWARF function pointer signature into Julia cfunction format.

# Format
DWARF format: `"function_ptr(return_type; param1, param2, ...)"`
Julia format: `"my_callback, return_type, (param1, param2, ...)"`

# Example
```julia
parse_function_pointer_signature("function_ptr(void; Ray*, AABB*, RayHit*)")
# => "my_callback, Cvoid, (Ptr{Ray}, Ptr{AABB}, Ptr{RayHit})"
```
"""
function parse_function_pointer_signature(fp_sig::String, registry::TypeRegistry)::Union{String, Nothing}
    # Match: function_ptr(return_type) or function_ptr(return_type; params...)
    m = match(r"^function_ptr\(([^;)]+)(?:;\s*(.+))?\)$", fp_sig)

    if isnothing(m)
        return nothing
    end

    return_type = String(strip(m.captures[1]))
    params_str = isnothing(m.captures[2]) ? nothing : String(strip(m.captures[2]))

    # Map return type
    julia_return = infer_julia_type(registry, return_type, context="callback return type")

    # Map parameter types
    if isnothing(params_str) || isempty(strip(params_str))
        # No parameters - () in Julia
        return "my_callback, $julia_return, ()"
    else
        # Parse and map each parameter
        param_types = split(params_str, ",")
        julia_params = String[]

        for p in param_types
            p_stripped = String(strip(p))
            if !isempty(p_stripped)
                # Extract just the type by removing the parameter name
                # For "const double* x" or "size_t n", we need just the type part
                # Split by whitespace and take all but the last token (which is the param name)
                tokens = split(p_stripped)
                if length(tokens) > 1
                    # Check if last token looks like a parameter name (alphanumeric, not a type modifier)
                    last_token = tokens[end]
                    # If it doesn't end with *, &, or [], it's likely a parameter name
                    if !endswith(last_token, '*') && !endswith(last_token, '&') && !occursin('[', last_token)
                        # Remove the parameter name, keep the type
                        p_type = join(tokens[1:end-1], " ")
                    else
                        p_type = p_stripped
                    end
                else
                    p_type = p_stripped
                end

                julia_p = infer_julia_type(registry, p_type, context="callback parameter")
                push!(julia_params, julia_p)
            end
        end

        if isempty(julia_params)
            return "my_callback, $julia_return, ()"
        else
            param_tuple = "(" * join(julia_params, ", ") * ",)"  # Trailing comma for proper tuple
            return "my_callback, $julia_return, $param_tuple"
        end
    end
end

