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
            push!(arg_types, map_cpp_type(t))
        end
        
        ret_type = map_cpp_type(get(ret_info, "c_type", "void"))
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
            idx = i - 1
            
            println(io, "  %idx_$(i) = arith.constant $(idx) : i64")
            println(io, "  %arg_ptr_$(i) = llvm.getelementptr %args_ptr[%idx_$(i)] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr") 
            println(io, "  %val_ptr_$(i) = llvm.load %arg_ptr_$(i) : !llvm.ptr -> !llvm.ptr")
            println(io, "  %val_$(i) = llvm.load %val_ptr_$(i) : !llvm.ptr -> $(mlir_t)")
            
            push!(call_args, "%val_$(i)")
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
