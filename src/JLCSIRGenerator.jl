#!/usr/bin/env julia
# JLCSIRGenerator.jl - Generate MLIR JLCS dialect IR from DWARF vtable data
# The bridge between binary analysis and executable IR

module JLCSIRGenerator

using ..DWARFParser

export generate_jlcs_ir, generate_mlir_module, save_mlir_module, generate_vcall_example

"""
    generate_type_info_ir(class_name::String, info::ClassInfo, vtable_addr::UInt64) -> String

Generate JLCS type_info operation for a class.

# Example Output
```mlir
jlcs.type_info @Base {
    size = 8 : i64,
    vtable_offset = 0 : i64,
    vtable_addr = 0x3d20 : i64
}
```
"""
function generate_type_info_ir(class_name::String, info::DWARFParser.ClassInfo, vtable_addr::UInt64)
    # Sanitize class name for MLIR (no ::, <, >, space, comma allowed in identifiers)
    # Convert "std::vector<int>" -> "std_vector_int_"
    mlir_name = replace(class_name, 
        "::" => "_", 
        "<" => "_", 
        ">" => "_",
        "(" => "_",
        ")" => "_",
        " " => "",
        "," => "_",
        "*" => "Ptr",
        "&" => "Ref"
    )

    # Ensure starts with letter or underscore
    if !isempty(mlir_name) && !isletter(mlir_name[1]) && mlir_name[1] != '_'
        mlir_name = "_" * mlir_name
    end
    
    # Empty name fallback
    if isempty(mlir_name)
        mlir_name = "anonymous_type_$(hash(class_name))"
    end

    ir = """
  jlcs.type_info @"$(mlir_name)" {
    size = $(info.size) : i64,
    vtable_offset = $(info.vtable_ptr_offset) : i64,
    vtable_addr = $(vtable_addr) : i64
  }"""

    return ir
end

"""
    map_cpp_type(type_str::String) -> String

Map C++ type string to MLIR type.
"""
function map_cpp_type(type_str::String)
    # Basic types
    if type_str == "void"
        return "" # Void return usually means no value
    elseif type_str == "int" || type_str == "int32_t"
        return "i32"
    elseif type_str == "long" || type_str == "long long" || type_str == "int64_t" || type_str == "size_t"
        return "i64"
    elseif type_str == "float"
        return "f32"
    elseif type_str == "double"
        return "f64"
    elseif type_str == "bool"
        return "i1"
    elseif type_str == "char" || type_str == "int8_t"
        return "i8"
    elseif endswith(type_str, "*") || contains(type_str, "*") # simplified pointer check
        return "!llvm.ptr"
    end

    # Fallback for unknown types (assume pointer or i64 for safety/placeholder)
    # Ideally we should log a warning
    return "!llvm.ptr" 
end

"""
    generate_virtual_method_ir(method::VirtualMethod, addr::UInt64) -> String

Generate IR for a virtual method declaration.

# Example Output
```mlir
func.func @Base_foo(%arg0: !llvm.ptr) -> i32 {
    %addr = arith.constant 0x12a0 : i64
    %fptr = llvm.inttoptr %addr : i64 to !llvm.ptr
    %result = llvm.call %fptr(%arg0) : !llvm.ptr, (i32) -> i32
    return %result : i32
}
```
"""
function generate_virtual_method_ir(method::DWARFParser.VirtualMethod, addr::UInt64)
    # Sanitize names
    mlir_name = replace(method.mangled_name, "::" => "_", "(" => "_", ")" => "_")
    
    # Map types
    ret_type = map_cpp_type(method.return_type)
    
    # Build argument list
    # Implicit 'this' pointer is always first
    arg_types = ["!llvm.ptr"]
    arg_names = ["%arg0"]
    arg_sig_parts = ["%arg0: !llvm.ptr"]
    
    for (i, param) in enumerate(method.parameters)
        type_str = map_cpp_type(param)
        push!(arg_types, type_str)
        push!(arg_names, "%arg$i")
        push!(arg_sig_parts, "%arg$i: $type_str")
    end
    
    args_sig = "(" * join(arg_sig_parts, ", ") * ")"
    args_vals = "(" * join(arg_names, ", ") * ")"
    call_sig = "(" * join(arg_types, ", ") * ")" # Signature for call instruction (types only)
    
    # Function body
    # We create a wrapper that calls the absolute address
    # This is a static dispatch wrapper for testing/direct access
    
    call_stmt = ret_type == "" ? 
        "llvm.call %fptr$(args_vals) : !llvm.ptr, $(call_sig) -> ()" : 
        "%result = llvm.call %fptr$(args_vals) : !llvm.ptr, $(call_sig) -> $(ret_type)"
    
    return_stmt = ret_type == "" ? "return" : "return %result : $(ret_type)"

    body = """
  func.func @$(mlir_name)$(args_sig) -> $(ret_type == "" ? "()" : ret_type) {
    %addr = arith.constant $(addr) : i64
    %fptr = llvm.inttoptr %addr : i64 to !llvm.ptr
    $(call_stmt)
    $(return_stmt)
  }"""

    return body
end

"""
    generate_vcall_example(class_name::String, method_name::String, slot::Int,
                          vtable_offset::Int, return_type::String) -> String

Generate example MLIR IR showing how to call a virtual method using jlcs.vcall.

# Example Output
```mlir
func.func @call_Base_foo(%obj: !llvm.ptr) -> i32 {
  %result = jlcs.vcall @Base::foo(%obj)
    { vtable_offset = 0 : i64, slot = 0 : i64 }
    : (!llvm.ptr) -> i32
  return %result : i32
}
```
"""
function generate_vcall_example(class_name::String, method_name::String, slot::Int,
                                vtable_offset::Int, return_type::String)
    safe_class = replace(class_name, "::" => "_")
    safe_method = replace(method_name, "::" => "_", "(" => "_", ")" => "_")
    func_name = "call_$(safe_class)_$(safe_method)"

    # Map C++ types to MLIR types
    mlir_return = map_cpp_type(return_type)
    ret_sig = mlir_return == "" ? "()" : mlir_return

    ir = """
  func.func @$(func_name)(%obj: !llvm.ptr) -> $(ret_sig) {
    %result = jlcs.vcall @$(class_name)::$(method_name)(%obj)
      { vtable_offset = $(vtable_offset) : i64, slot = $(slot) : i64 }
      : (!llvm.ptr) -> $(ret_sig)
    return $(mlir_return == "" ? "" : "%result : " * mlir_return)
  }"""

    return ir
end

"""
    generate_jlcs_ir(vtinfo::VtableInfo) -> String

Generate complete JLCS MLIR module from vtable information.

# Arguments
- `vtinfo`: VtableInfo from DWARFParser

# Returns
MLIR module text in JLCS dialect
"""
function generate_jlcs_ir(vtinfo::DWARFParser.VtableInfo)
    io = IOBuffer()

    # Module header
    println(io, "// JLCS IR - Generated from DWARF debug info")
    println(io, "// Universal FFI via MLIR Dialects")
    println(io, "")
    println(io, "module {")

    # Generate type info for each class
    for (class_name, class_info) in vtinfo.classes
        # Skip invalid/incomplete classes
        if class_info.size == 0
            continue
        end

        # Get vtable address if available
        vtable_addr = get(vtinfo.vtable_addresses, class_name, UInt64(0))

        # Generate type info (even for non-polymorphic types)
        println(io, generate_type_info_ir(class_name, class_info, vtable_addr))
        println(io, "")

        # Generate method IR for virtual methods
        for method in class_info.virtual_methods
            method_addr = get(vtinfo.method_addresses, method.mangled_name, UInt64(0))
            if method_addr != 0
                println(io, generate_virtual_method_ir(method, method_addr))
                println(io, "")
            end
        end
    end

    # Add exported method addresses as constants
    println(io, "  // Method Address Table")
    for (mangled_name, addr) in vtinfo.method_addresses
        # Only include actual methods (skip _init, _fini, etc)
        if contains(mangled_name, "::") || contains(mangled_name, "main")
            safe_name = replace(replace(mangled_name, "::" => "_"),
                              "(" => "_", ")" => "_", "~" => "dtor_")
            println(io, "  // $safe_name = 0x$(string(addr, base=16))")
        end
    end

    println(io, "}")

    return String(take!(io))
end

"""
    generate_mlir_module(binary_path::String) -> String

Complete pipeline: Binary → DWARF → JLCS IR

# Arguments
- `binary_path`: Path to C++ binary with debug info

# Returns
JLCS MLIR IR module text
"""
function generate_mlir_module(binary_path::String)
    # Parse vtables from binary
    vtinfo = DWARFParser.parse_vtables(binary_path)

    # Generate JLCS IR
    ir = generate_jlcs_ir(vtinfo)

    return ir
end

"""
    save_mlir_module(binary_path::String, output_path::String)

Generate JLCS IR from binary and save to file.
"""
function save_mlir_module(binary_path::String, output_path::String)
    ir = generate_mlir_module(binary_path)

    write(output_path, ir)

    println("Generated JLCS IR: $output_path")
    println("$(count(c -> c == '\n', ir)) lines")
end

end # module JLCSIRGenerator
