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
                    mlir_t = StructGen.get_struct_type_string(s_name, structs[s_name])
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
                ret_type = StructGen.get_struct_type_string(s_name, structs[s_name])
            end
        end

        # func.func uses () for void, not !llvm.void
        mlir_ret = ret_type == "" || ret_type == "!llvm.void" ? "" : "-> $ret_type"
        func_ret = ret_type == "" || ret_type == "!llvm.void" ? "" : ret_type
        
        # 1. External Declaration (Real C++ Symbol)
        # Use real mangled name so JIT can link to it
        println(io, "func.func private @$(mangled)($(join(arg_types, ", "))) $mlir_ret")
        
        # 2. Thunk (Exposed to JIT)
        # Append _thunk suffix to avoid collision
        # Add llvm.emit_c_interface to generate _mlir_ciface_ wrapper for invokePacked
        println(io, "func.func @$(mangled)_thunk(%args_ptr: !llvm.ptr) attributes { llvm.emit_c_interface } {")
        
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
                    mlir_t = StructGen.get_struct_type_string(s_name, structs[s_name])
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
                
                # 1. Create undefined struct
                println(io, "  %s_undef_$(i) = llvm.mlir.undef : $(mlir_t)")
                
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
                    
                    # 4. Load field
                    println(io, "  %field_val_$(i)_$(m_idx) = llvm.load %field_ptr_raw_$(i)_$(m_idx) : !llvm.ptr -> $(m_type_mlir)")
                    
                    # 5. Insert into packed struct
                    curr_val = "%s_packed_$(i)_$(m_idx)"
                    println(io, "  $(curr_val) = llvm.insertvalue %field_val_$(i)_$(m_idx), $(prev_val)[$(m_idx_zero)] : $(mlir_t)")
                    prev_val = curr_val
                end
                
                # Use the reconstructed struct
                println(io, "  %val_$(i) = llvm.mlir.constant $(prev_val) : $(mlir_t)") # No, insertvalue returns SSA value
                # Wait, prev_val IS the value. We can just alias it or use it directly.
                # Let's just define %val_$(i) as the last result? No, we can't redefine.
                # We need to map %val_$(i) to the last result.
                # Let's output an alias/bitcast (identity) just to keep naming consistent or update push! logic.
                # Simplest: Just reuse variable naming convention
                # We need `call_args` to have the value.
                # Let's adjust the loop to output the final value as %val_$(i)
                # Hack: Just do a bitcast or move? MLIR has no move.
                # But we can just use the SSA name.
                
                # REVISION:
                # Use SSA name directly in call_args. But call_args is built at the end of loop?
                # No, call_args is built in the loop.
                # So we just need to ensure the final value is assigned to a known name or we capture it.
                
                # Let's redefine the loop variable `push!(call_args, ...)`
                
                # Actually, I can just not print `%val_$(i) = ...` line for packed case
                # and instead use the last `curr_val`
                
                # But wait, `prev_val` is the name of the SSA value.
                # I can't assign to `%val_$(i)` because SSA.
                # I will store the final SSA name in a variable `arg_value_name`
                arg_value_name = prev_val
            else
                println(io, "  %val_$(i) = llvm.load %val_ptr_$(i) : !llvm.ptr -> $(mlir_t)")
                arg_value_name = "%val_$(i)"
            end
            
            push!(call_args, arg_value_name)
        end
        
        # Call using standard 'call' op (since target is func.func)
        if func_ret == ""
             println(io, "  call @$(mangled)($(join(call_args, ", "))) : ($(join(arg_types, ", "))) -> ()")
             println(io, "  return")
        else
             println(io, "  %ret_val = call @$(mangled)($(join(call_args, ", "))) : ($(join(arg_types, ", "))) -> $(func_ret)")
             println(io, "  return")
        end
        
        println(io, "}")
        println(io, "")
    end
    
    return String(take!(io))
end

end
