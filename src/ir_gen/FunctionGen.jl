module FunctionGen

using ..TypeUtils
using ..StructGen # for get_struct_type_string

export generate_function_thunks

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
                if haskey(structs, s_name)
                    # Use FULL definition string to avoid alias parser issues in function signatures
                    mlir_t = StructGen.get_struct_definition_string(s_name, structs[s_name])
                end
            end
            
            push!(arg_types, mlir_t)
        end
        
        ret_c_type = get(ret_info, "c_type", "void")
        ret_type = map_cpp_type(ret_c_type)
        
        # Resolve full struct type for return
        if startswith(ret_type, "!llvm.struct<\"") && endswith(ret_type, "\">")
            s_name = ret_type[15:end-2]
            if haskey(structs, s_name)
                # Use FULL definition string
                ret_type = StructGen.get_struct_definition_string(s_name, structs[s_name])
            end
        end

        # func.func uses () for void, not !llvm.void
        mlir_ret = ret_type == "" || ret_type == "!llvm.void" ? "" : "-> $ret_type"
        func_ret = ret_type == "" || ret_type == "!llvm.void" ? "" : ret_type
        
        # 1. External Declaration (Real C++ Symbol)
        # Use real mangled name so JIT can link to it
        println(io, "func.func private @$(mangled)($(join(arg_types, ", "))) $mlir_ret")
        
        # Pre-scan: Check if any argument is a packed struct
        # If so, skip thunk generation to avoid parser issues with manual marshalling
        skip_thunk = false
        for p in params
            t = get(p, "c_type", "void*")
            check_t = map_cpp_type(t)
            if startswith(check_t, "!llvm.struct<\"") && endswith(check_t, "\">")
                s_name = check_t[15:end-2]
                if haskey(structs, s_name)
                    if StructGen.is_struct_packed(structs[s_name])
                        skip_thunk = true
                        break
                    end
                end
            end
        end

        if skip_thunk
            println(io, "// Thunk for @$(mangled) skipped due to packed struct argument")
            println(io, "")
            continue
        end
        
        # 2. Thunk (Exposed to JIT)
        # Append _thunk suffix to avoid collision
        # Add llvm.emit_c_interface to generate _mlir_ciface_ wrapper for invokePacked
        println(io, "func.func @$(mangled)_thunk(%args_ptr: !llvm.ptr) $(mlir_ret) attributes { llvm.emit_c_interface } {")
        
        call_args = String[]
        
        for (i, p) in enumerate(params)
            t = get(p, "c_type", "void*")
            mlir_t = map_cpp_type(t)
            
            # Resolve full struct type if available (must match arg_types)
            is_packed_struct = false
            struct_info = nothing
            
            if startswith(mlir_t, "!llvm.struct<\"") && endswith(mlir_t, "\">")
                s_name = mlir_t[15:end-2]
                if haskey(structs, s_name)
                    # Use definition string
                    mlir_t = StructGen.get_struct_definition_string(s_name, structs[s_name])
                    struct_info = structs[s_name]
                    is_packed_struct = StructGen.is_struct_packed(struct_info)
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
                
                # We cannot build !jlcs.c_struct directly with llvm.insertvalue.
                # Build an equivalent LLVM packed struct and cast it.
                s_name = mlir_t[15:end-2] # Extract name from !jlcs.c_struct<"Name"...> if needed, but we have struct_info
                # Wait, mlir_t is the FULL string !jlcs.c_struct<...>. Extracting name is hard.
                # But we have s_name from the check above! 
                
                llvm_t = StructGen.get_llvm_equivalent_type_string(s_name, struct_info)

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
                
                # Cast to JLCS type using generic syntax
                # mlir_t is now an alias (!Struct_Name), so this should parse cleanly.
                println(io, "  %s_jlcs_$(i) = llvm.mlir.undef : $(mlir_t)")
                
                # Use the reconstructed struct
                arg_value_name = "%s_jlcs_$(i)"
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
