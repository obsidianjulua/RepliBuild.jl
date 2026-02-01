module FunctionGen

using ..TypeUtils
using ..StructGen # for get_struct_type_string

export generate_function_thunks

function generate_function_thunks(functions::Vector, structs::Any=Dict())
    io = IOBuffer()
    println(io, "// Function Thunks")

    # Track generated thunks to avoid duplicates (e.g. constructors)
    generated = Set{String}()

    for func in functions
        name = get(func, "name", "")
        mangled = get(func, "mangled", name)
        
        if mangled in generated
            # println("Skipping duplicate function: $mangled") # Debug only
            continue
        end
        push!(generated, mangled)
        
        params = copy(get(func, "parameters", []))
        ret_info = get(func, "return_type", Dict())
        is_method = get(func, "is_method", false)
        
        # Inject 'this' pointer for methods if not present
        if is_method
            # Check if first param is already 'this' (some DWARF info includes it)
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
        # Use !llvm.void instead of () for consistency with VtableGen, 
        # but ensure parser accepts it. Actually () is standard for void return in MLIR func.
        # But for llvm.func, !llvm.void is typical.
        mlir_ret = ret_type == "" ? "!llvm.void" : ret_type
        
        # 1. External Declaration (Real C++ Symbol)
        real_symbol = "__real_$(mangled)"
        # Note: Removing attributes to avoid "expected attribute value" parsing error
        println(io, "llvm.func @$(real_symbol)($(join(arg_types, ", "))) -> $(mlir_ret)")
        
        # 2. Thunk (Exposed to JIT)
        println(io, "func.func @$(mangled)(%args_ptr: !llvm.ptr) {")
        
        # Unpack arguments
        call_args = String[]
        
        for (i, p) in enumerate(params)
            t = get(p, "c_type", "void*")
            mlir_t = map_cpp_type(t)
            
            # i is 1-based index in params list
            # We map this to 0-based index in args array
            idx = i - 1
            
            println(io, "  %idx_$(i) = arith.constant $(idx) : i64")
            println(io, "  %arg_ptr_$(i) = llvm.getelementptr %args_ptr[%idx_$(i)] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr") 
            println(io, "  %val_ptr_$(i) = llvm.load %arg_ptr_$(i) : !llvm.ptr -> !llvm.ptr")
            println(io, "  %val_$(i) = llvm.load %val_ptr_$(i) : !llvm.ptr -> $(mlir_t)")
            
            push!(call_args, "%val_$(i)")
        end
        
        # Call
        call_prefix = ret_type == "" ? "llvm.call" : "%ret_val ="
        call_sig = "($(join(arg_types, ", "))) -> $(mlir_ret)"
        
        println(io, "  $(call_prefix) llvm.call @$(real_symbol)($(join(call_args, ", "))) : $(call_sig)")
        
        println(io, "  return")
        println(io, "}")
        println(io, "")
    end
    
    return String(take!(io))
end

end