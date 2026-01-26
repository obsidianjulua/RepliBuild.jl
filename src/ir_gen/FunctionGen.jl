module FunctionGen

using ..TypeUtils
using ..StructGen # for get_struct_type_string

export generate_function_thunks

function generate_function_thunks(functions::Vector, structs::Any=Dict())
    io = IOBuffer()
    println(io, "// Function Thunks")
    
    for func in functions
        # Skip methods (handled by VtableGen)
        if get(func, "is_method", false)
            continue
        end
        
        name = get(func, "name", "")
        # Use mangled name for external call
        mangled = get(func, "mangled", name)
        
        # 1. External Declaration
        params = get(func, "parameters", [])
        ret_info = get(func, "return_type", Dict())
        
        arg_types = String[]
        for p in params
            t = get(p, "c_type", "void*")
            # For external decl, we can use opaque pointers or strict types?
            # LLVM allows mismatch if opaque pointers, but better safe.
            push!(arg_types, map_cpp_type(t))
        end
        
        ret_type = map_cpp_type(get(ret_info, "c_type", "void"))
        mlir_ret = ret_type == "" ? "!llvm.void" : ret_type
        
        println(io, "llvm.func @$(mangled)($(join(arg_types, ", "))) -> $(mlir_ret) attributes { sym_visibility = \"private\" }")
        
        # 2. Thunk
        println(io, "func.func @$(name)(%args_ptr: !llvm.ptr) {")
        
        # Unpack arguments
        call_args = String[]
        
        for (i, p) in enumerate(params)
            t = get(p, "c_type", "void*")
            mlir_t = map_cpp_type(t)
            
            # GEP to get pointer to argument i in the void** array
            println(io, "  %idx_$(i) = arith.constant $(i-1) : i64")
            println(io, "  %arg_ptr_$(i) = llvm.getelementptr %args_ptr[%idx_$(i)] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr") 
            
            # Load the pointer TO the value (args[i] is void*, which is a pointer to the value)
            println(io, "  %val_ptr_$(i) = llvm.load %arg_ptr_$(i) : !llvm.ptr -> !llvm.ptr")
            
            # Now %val_ptr_$(i) points to the actual data (e.g. PackedStruct)
            # We need to load the data to pass by value
            println(io, "  %val_$(i) = llvm.load %val_ptr_$(i) : !llvm.ptr -> $(mlir_t)")
            
            push!(call_args, "%val_$(i)")
        end
        
        # Call
        call_prefix = ret_type == "" ? "llvm.call" : "%ret_val ="
        call_sig = "($(join(arg_types, ", "))) -> $(mlir_ret)"
        
        println(io, "  $(call_prefix) llvm.call @$(mangled)($(join(call_args, ", "))) : $(call_sig)")
        
        println(io, "  return")
        println(io, "}")
        println(io, "")
    end
    
    return String(take!(io))
end

end
