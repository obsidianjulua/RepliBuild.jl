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

import ..Compiler: cpp_to_julia_type

"""
    _callback_typedef_signature(param, metadata) -> Union{String,Nothing}

The signature a callback parameter's `@cfunction` example is built from, in
`parse_function_pointer_signature`'s input form: the header signature
(`function_pointer_typedefs`) of the typedef the parameter is declared with
(`function_pointer_typedef`, recorded from DWARF). `nothing` when the parameter
has no typedef, or the headers do not spell that typedef, and then the example
comes from DWARF's signature (`_dwarf_callback_signature`). Both generators ask
this.

The example used to come from whichever header typedef's name best matched the
parameter's or the function's name. The score tied often and fell back to the
first typedef in the table, both in `Dict` order, so the example changed with
the Julia release, and it was often another callback's: llama.cpp's
`ggml_backend_compare_graph_backend` documented a signature unrelated to its
`ggml_backend_eval_callback`. A wrong example is worse than none, since a
callback built from it is called with the wrong arguments.
"""
function _callback_typedef_signature(param, metadata)::Union{String,Nothing}
    td = get(param, "function_pointer_typedef", nothing)
    td isa AbstractString || return nothing
    table = get(metadata, "function_pointer_typedefs", nothing)
    table isa AbstractDict || return nothing
    info = get(table, td, nothing)
    info isa AbstractDict || return nothing
    ret = info["return_type"]
    params = info["parameters"]
    return isempty(params) ? "function_ptr($ret)" : "function_ptr($ret; $(join(params, ", ")))"
end

"""
    _split_c_type_list(s, sep) -> Vector{String}

Split a list of C type spellings on `sep` at depth 0 of `()`, `<>` and `[]`. A
type holds separators of its own: `std::map<int, int>*` has a comma, and a
function pointer has a `;` and commas inside its parentheses.
"""
function _split_c_type_list(s::AbstractString, sep::Char)::Vector{String}
    parts = String[]
    depth = 0
    start = firstindex(s)
    for i in eachindex(s)
        c = s[i]
        if c in ('(', '<', '[')
            depth += 1
        elseif c in (')', '>', ']')
            depth = max(0, depth - 1)
        elseif c == sep && depth == 0
            push!(parts, String(strip(s[start:prevind(s, i)])))
            start = nextind(s, i)
        end
    end
    push!(parts, String(strip(s[start:end])))
    return parts
end

"""
    _dwarf_callback_signature(sig, param_type) -> Union{String,Nothing}

The `@cfunction` arguments for a callback whose headers spell no typedef, built
from DWARF's signature `sig` (`function_ptr(R; P1, P2)*`), in
`parse_function_pointer_signature`'s output form: `"my_callback, R, (P1, P2,)"`.
`param_type(c_type)` spells a type the way the wrapper spells a parameter of that
type in its own `ccall` (`_callback_param_type`), or answers `nothing`, and then
there is no example: none is better than a wrong one.

Exactly one pointer level is stripped, so a pointer to a callback
(`function_ptr(…)**`, an out-parameter) and a reference to a function have no
example, and nor has a variadic callback (`…, ...`), which `@cfunction` cannot
make. The list is split at depth 0, since a parameter can be a template with
commas or another function pointer, and so can the return type
(`function_ptr(function_ptr(double; float)*; int)*`). Nothing is removed from a
DWARF type: it has no parameter name, which `parse_function_pointer_signature`
drops from header text as a last word, so `unsigned int` would be `unsigned`.
"""
function _dwarf_callback_signature(sig::AbstractString, param_type)::Union{String,Nothing}
    m = match(r"^function_ptr\((.*)\)\*$", strip(sig))
    m === nothing && return nothing
    parts = _split_c_type_list(m.captures[1], ';')
    length(parts) <= 2 || return nothing
    params = length(parts) == 2 ? _split_c_type_list(parts[2], ',') : String[]
    "..." in params && return nothing
    ret = parts[1] == "void" ? "Cvoid" : param_type(parts[1])
    ret === nothing && return nothing
    jps = String[]
    for p in params
        jp = param_type(p)
        jp === nothing && return nothing
        push!(jps, jp)
    end
    return isempty(jps) ? "my_callback, $ret, ()" : "my_callback, $ret, ($(join(jps, ", ")),)"
end

"""
    _callback_param_type(ccall_param_type, dwarf_structs) -> Function

`c_type -> Union{String,Nothing}` for a type in a callback's DWARF signature,
spelled as the wrapper spells a parameter's `ccall` type. `cpp_to_julia_type`
names it, against the records and enums in `dwarf_structs`, the way it named
every parameter's metadata `julia_type` at build time. Then
`ccall_param_type(julia_type, c_type)` applies the generator's own parameter
steps (sanitizing, `Ptr{Cvoid}` for an undeclared pointee). A type that comes
out `Any` gets `nothing`.
"""
function _callback_param_type(ccall_param_type, dwarf_structs)
    structs = Set{String}()
    enums = Set{String}()
    for k in keys(dwarf_structs)
        s = String(k)
        startswith(s, "__enum__") ? push!(enums, s[length("__enum__")+1:end]) : push!(structs, s)
    end
    return function (c_type::AbstractString)
        t = ccall_param_type(cpp_to_julia_type(c_type, structs, enums), String(c_type))
        return t in ("Any", "unknown", "") ? nothing : t
    end
end

"""
    _param_docs!(arg_docs, callback_docs, param, metadata, registry, callback_param_type)

A parameter's line in a wrapper docstring's `# Arguments`, and for a callback
its `@cfunction` example. Both generators write their docstrings through this.

A callback's example comes from the header signature of the typedef it is
declared with (`_callback_typedef_signature`), and without one from DWARF's
signature (`_dwarf_callback_signature`, with `callback_param_type` from
`_callback_param_type`). That needs metadata whose function-pointer signatures
list their parameters (`function_pointer_parameters`): in older metadata every
one is `function_ptr(R)*`, and a four-argument callback would be documented as
taking none. Without an example the line shows the signature as text, escaped
for the docstring it is written into, where `\\` escapes and `\$` interpolates.
"""
function _param_docs!(arg_docs::Vector{String}, callback_docs::Vector{String}, param, metadata,
                      registry::TypeRegistry, callback_param_type)
    name = param["name"]
    c_type_name = get(param, "c_type", "")
    julia_type = _pointee_safe_type(param["julia_type"])     # what the signature takes, not the metadata's `Ref{Any}`

    # Check if this is a callback parameter (Ptr{Cvoid} from function pointer)
    if julia_type == "Ptr{Cvoid}" && (startswith(c_type_name, "function_ptr") || contains(c_type_name, "(*"))
        dwarf_sig = get(param, "function_pointer_signature", nothing)
        td_sig = _callback_typedef_signature(param, metadata)
        julia_sig = if td_sig !== nothing
            parse_function_pointer_signature(td_sig, registry)
        elseif dwarf_sig isa AbstractString && get(metadata, "function_pointer_parameters", false) === true
            _dwarf_callback_signature(dwarf_sig, callback_param_type)
        else
            nothing
        end
        fp_sig = something(td_sig, dwarf_sig, Some(nothing))

        if !isnothing(julia_sig)
            push!(arg_docs, "- `$(name)::Ptr{Cvoid}` - Callback function")
            push!(callback_docs, """
                **Callback `$name`**: Create using `@cfunction`
                ```julia
                callback = @cfunction($julia_sig) Ptr{Cvoid}
                ```""")
        elseif !isnothing(fp_sig)
            push!(arg_docs, "- `$(name)::$(julia_type)` - Callback function (signature: $(escape_string(fp_sig, '$')))")
        else
            # No signature info available
            push!(arg_docs, "- `$(name)::$(julia_type)` - Callback function (signature unknown)")
        end
    else
        # Regular parameter
        push!(arg_docs, "- `$(name)::$(julia_type)`")
    end
    return nothing
end
