module FunctionGen

using ..TypeUtils
using ..StructGen # for get_struct_type_string

export generate_function_thunks

"""
    _fuzzy_struct_lookup(c_type, structs) -> Union{String, Nothing}

Fuzzy-match a C type name against struct definition keys.
DWARF keys often have trailing " >" nesting artifacts from template types.
"""
function _fuzzy_struct_lookup(c_type::String, structs::Any)
    haskey(structs, c_type) && return c_type
    key = c_type * " >"
    haskey(structs, key) && return key
    c_norm = rstrip(c_type, [' ', '>'])
    for k in keys(structs)
        rstrip(String(k), [' ', '>']) == c_norm && return String(k)
    end
    return nothing
end

"""
    _byte_blob_type(byte_size) -> String

Generate a packed MLIR struct type of exactly `byte_size` bytes.
Uses i64 chunks with i8 remainder for compactness.
"""
function _byte_blob_type(byte_size::Int)
    byte_size <= 0 && return "!llvm.ptr"
    n_i64 = div(byte_size, 8)
    n_rem = byte_size % 8
    parts = fill("i64", n_i64)
    for _ in 1:n_rem
        push!(parts, "i8")
    end
    return "!llvm.struct<packed ($(join(parts, ", ")))>"
end

function _parse_byte_size(s::AbstractString)
    startswith(s, "0x") ? parse(Int, s[3:end], base=16) : parse(Int, s)
end

function generate_function_thunks(functions::Vector, structs::Any=Dict())
    io = IOBuffer()
    println(io, "// Function Thunks")

    generated = Set{String}()

    for func in functions
        name = get(func, "name", "")
        mangled = get(func, "mangled", name)

        if mangled in generated
            continue
        end

        # Skip varargs functions — they use va_start/va_end which MLIR cannot handle.
        # These are already routed through direct ccall via varargs interception.
        if get(func, "is_vararg", false)
            continue
        end

        push!(generated, mangled)

        params = copy(get(func, "parameters", []))
        ret_info = get(func, "return_type", Dict())
        is_method = get(func, "is_method", false)

        if is_method
            if isempty(params) || get(params[1], "name", "") != "this"
                pushfirst!(params, Dict("c_type" => "void*", "name" => "this"))
            end
        end

        arg_types = String[]
        for p in params
            t = get(p, "c_type", "void*")
            mlir_t = map_cpp_type(t)

            # Resolve full struct type if available
            if startswith(mlir_t, "!llvm.struct<\"") && endswith(mlir_t, "\">")
                s_name = mlir_t[15:end-2]
                lookup_key = haskey(structs, s_name) ? s_name : (haskey(structs, "__enum__$(s_name)") ? "__enum__$(s_name)" : nothing)
                if lookup_key !== nothing
                    # For packed structs, use LLVM packed type to avoid type mismatch with thunk marshalling
                    if StructGen.is_struct_packed(structs[lookup_key])
                        mlir_t = StructGen.get_llvm_equivalent_type_string(s_name, structs[lookup_key])
                    else
                        # Use FULL definition string to avoid alias parser issues in function signatures
                        mlir_t = StructGen.get_struct_definition_string(s_name, structs[lookup_key])
                    end
                end
            end

            push!(arg_types, mlir_t)
        end

        ret_c_type = get(ret_info, "c_type", "void")
        ret_type = map_cpp_type(ret_c_type)

        # Track packed struct return for thunk conversion
        is_packed_ret = false
        ret_packed_type = ""
        ret_aligned_type = ""
        ret_struct_info = nothing
        ret_num_members = 0

        # Resolve full struct type for return
        if startswith(ret_type, "!llvm.struct<\"") && endswith(ret_type, "\">")
            s_name = ret_type[15:end-2]
            lookup_key = haskey(structs, s_name) ? s_name : (haskey(structs, "__enum__$(s_name)") ? "__enum__$(s_name)" : nothing)
            if lookup_key !== nothing
                # For packed structs, use LLVM packed type for external decl/ffe_call
                if StructGen.is_struct_packed(structs[lookup_key])
                    ret_type = StructGen.get_llvm_equivalent_type_string(s_name, structs[lookup_key])
                    is_packed_ret = true
                    ret_packed_type = ret_type
                    ret_aligned_type = StructGen.get_llvm_aligned_type_string(s_name, structs[lookup_key])
                    ret_struct_info = structs[lookup_key]
                    ret_num_members = length(get(ret_struct_info, "members", []))
                else
                    # Check if the struct has members that can't be sized correctly in MLIR.
                    # When members are template types with size=0, get_struct_definition_string
                    # would produce a struct with wrong total size. Use a byte blob instead.
                    info = structs[lookup_key]
                    byte_size = try _parse_byte_size(get(info, "byte_size", "0")) catch; 0 end
                    members = get(info, "members", [])
                    has_unsizable = any(m -> begin
                        ms = try; s = get(m, "size", 0); s isa String ? parse(Int, s) : s; catch; 0; end
                        ct = strip(replace(get(m, "c_type", ""), r"\bconst\b" => ""))
                        ms == 0 && !endswith(ct, "*") && !endswith(ct, "&") && !occursin(r"^[A-Za-z0-9_]+$", ct)
                    end, members)
                    if has_unsizable && byte_size > 0
                        ret_type = _byte_blob_type(byte_size)
                    else
                        ret_type = StructGen.get_struct_definition_string(s_name, structs[lookup_key])
                    end
                end
            end
        elseif ret_type == "!llvm.ptr" && !contains(ret_c_type, "*") && ret_c_type != "void" && ret_c_type != "unknown"
            # Template types (e.g. "Matrix<double, -1, -1>") fall through map_cpp_type
            # to !llvm.ptr because they contain non-alphanumeric chars.
            # Fuzzy-match against DWARF struct definitions to recover the correct size.
            matched_key = _fuzzy_struct_lookup(ret_c_type, structs)
            if matched_key !== nothing
                info = structs[matched_key]
                byte_size = try _parse_byte_size(get(info, "byte_size", "0")) catch; 0 end
                if byte_size > 0
                    ret_type = _byte_blob_type(byte_size)
                end
            else
                # Fallback for known STL containers that might be missing from DWARF struct_defs
                # We dynamically invoke the function from the Wrapper module
                # Note: FunctionGen is in the ir_gen submodule, so we must qualify
                stl_size = try 
                    Main.RepliBuild.Wrapper.get_stl_container_size(ret_c_type)
                catch
                    # If called from a context where Wrapper isn't loaded, fallback
                    0
                end
                if stl_size > 0
                    ret_type = _byte_blob_type(stl_size)
                end
            end
        end

        # External declaration uses the actual C type (packed for packed structs)
        ext_ret_type = ret_type
        ext_mlir_ret = ext_ret_type == "" || ext_ret_type == "!llvm.void" ? "" : "-> $ext_ret_type"

        # Thunk returns aligned type so Julia can read it correctly
        thunk_ret_type = is_packed_ret ? ret_aligned_type : ret_type
        thunk_mlir_ret = thunk_ret_type == "" || thunk_ret_type == "!llvm.void" ? "" : "-> $thunk_ret_type"
        func_ret = ret_type == "" || ret_type == "!llvm.void" ? "" : ret_type

        # 1. External Declaration (Real C++ Symbol)
        # Use real mangled name so JIT can link to it
        println(io, "func.func private @$(mangled)($(join(arg_types, ", "))) $ext_mlir_ret")

        # 2. Thunk (Exposed to JIT)
        # Append _thunk suffix to avoid collision
        # Add llvm.emit_c_interface to generate _mlir_ciface_ wrapper for invokePacked
        println(io, "func.func @$(mangled)_thunk(%args_ptr: !llvm.ptr) $(thunk_mlir_ret) attributes { llvm.emit_c_interface } {")

        call_args = String[]

        for (i, p) in enumerate(params)
            t = get(p, "c_type", "void*")
            mlir_t = map_cpp_type(t)

            # Resolve full struct type if available (must match arg_types)
            is_packed_struct = false
            struct_info = nothing
            s_name = ""

            if startswith(mlir_t, "!llvm.struct<\"") && endswith(mlir_t, "\">")
                s_name = mlir_t[15:end-2]
                lookup_key = haskey(structs, s_name) ? s_name : (haskey(structs, "__enum__$(s_name)") ? "__enum__$(s_name)" : nothing)
                if lookup_key !== nothing
                    struct_info = structs[lookup_key]
                    is_packed_struct = StructGen.is_struct_packed(struct_info)
                    # Use the same type as arg_types for consistency
                    if is_packed_struct
                        mlir_t = StructGen.get_llvm_equivalent_type_string(s_name, struct_info)
                    else
                        mlir_t = StructGen.get_struct_definition_string(s_name, struct_info)
                    end
                end
            end

            idx = i - 1
            println(io, "  %idx_$(i) = arith.constant $(idx) : i64")
            println(io, "  %arg_ptr_$(i) = llvm.getelementptr %args_ptr[%idx_$(i)] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr")
            println(io, "  %val_ptr_$(i) = llvm.load %arg_ptr_$(i) : !llvm.ptr -> !llvm.ptr")

            if is_packed_struct
                # Layout mismatch handling: Julia passes aligned struct pointer, C++ expects packed struct by value.
                # We must load fields using Julia alignment offsets and reconstruct the packed struct.
                println(io, "  // Marshalling packed struct from aligned Julia pointer")

                llvm_t = mlir_t  # Already the LLVM packed type

                # 1. Create undefined struct (LLVM type)
                println(io, "  %s_undef_$(i) = llvm.mlir.undef : $(llvm_t)")

                # 2. Get Julia offsets
                offsets = StructGen.get_julia_offsets(struct_info)
                members = get(struct_info, "members", [])

                prev_val = "%s_undef_$(i)"

                for (m_idx, (member, offset)) in enumerate(zip(members, offsets))
                    m_idx_zero = m_idx - 1
                    m_type_c = get(member, "c_type", "void*")
                    m_type_mlir = map_cpp_type(m_type_c)

                    # 3. GEP into Julia pointer (byte offset)
                    println(io, "  %off_$(i)_$(m_idx) = arith.constant $(offset) : i64")
                    println(io, "  %field_ptr_raw_$(i)_$(m_idx) = llvm.getelementptr %val_ptr_$(i)[%off_$(i)_$(m_idx)] : (!llvm.ptr, i64) -> !llvm.ptr, i8")

                    # 4. Load field (explicitly unaligned load since we are reading packed bytes or unaligned offsets)
                    println(io, "  %field_val_$(i)_$(m_idx) = llvm.load %field_ptr_raw_$(i)_$(m_idx) {alignment = 1 : i64} : !llvm.ptr -> $(m_type_mlir)")

                    # 5. Insert into packed struct (LLVM type)
                    curr_val = "%s_packed_$(i)_$(m_idx)"
                    println(io, "  $(curr_val) = llvm.insertvalue %field_val_$(i)_$(m_idx), $(prev_val)[$(m_idx_zero)] : $(llvm_t)")
                    prev_val = curr_val
                end

                # Use the fully reconstructed packed struct directly
                arg_value_name = prev_val
            else
                println(io, "  %val_$(i) = llvm.load %val_ptr_$(i) : !llvm.ptr -> $(mlir_t)")
                arg_value_name = "%val_$(i)"
            end

            push!(call_args, arg_value_name)
        end

        # Call using jlcs.ffe_call (via Dialect)
        if func_ret == ""
             # Void return
             println(io, "  jlcs.ffe_call $(join(call_args, ", ")) { callee = @$(mangled) } : ($(join(arg_types, ", "))) -> ()")
             println(io, "  return")
        elseif is_packed_ret
             # Packed struct return: call returns packed type, convert to aligned for Julia
             println(io, "  %ret_packed = jlcs.ffe_call $(join(call_args, ", ")) { callee = @$(mangled) } : ($(join(arg_types, ", "))) -> $(ret_packed_type)")
             # Extract fields from packed struct, insert into aligned struct
             println(io, "  %ret_aligned_undef = llvm.mlir.undef : $(ret_aligned_type)")
             prev_ret = "%ret_aligned_undef"
             for m_idx in 1:ret_num_members
                 m_idx_zero = m_idx - 1
                 m = get(ret_struct_info, "members", [])[m_idx]
                 m_type_mlir = map_cpp_type(get(m, "c_type", "void*"))
                 println(io, "  %ret_field_$(m_idx) = llvm.extractvalue %ret_packed[$(m_idx_zero)] : $(ret_packed_type)")
                 curr_ret = "%ret_aligned_$(m_idx)"
                 println(io, "  $(curr_ret) = llvm.insertvalue %ret_field_$(m_idx), $(prev_ret)[$(m_idx_zero)] : $(ret_aligned_type)")
                 prev_ret = curr_ret
             end
             println(io, "  return $(prev_ret) : $(ret_aligned_type)")
        else
             # Value return
             println(io, "  %ret_val = jlcs.ffe_call $(join(call_args, ", ")) { callee = @$(mangled) } : ($(join(arg_types, ", "))) -> $(func_ret)")
             println(io, "  return %ret_val : $(func_ret)")
        end

        println(io, "}")
        println(io, "")
    end

    return String(take!(io))
end

end
