#!/usr/bin/env julia
# JLCSIRGenerator.jl - Generate MLIR JLCS dialect IR from DWARF vtable data
# The bridge between binary analysis and executable IR

module JLCSIRGenerator

using ..DWARFParser
import JSON

# Modular includes
include("ir_gen/TypeUtils.jl")
include("ir_gen/StructGen.jl")
include("ir_gen/FunctionGen.jl")

using .TypeUtils
using .StructGen
using .FunctionGen

export generate_jlcs_ir, generate_mlir_module

"""
    map_cpp_type_to_mlir(cpp_type::String) -> String

Map C++ DWARF type names to MLIR types.
"""
function map_cpp_type_to_mlir(cpp_type::String)
    t = strip(cpp_type)
    
    if endswith(t, "*") || endswith(t, "&") || contains(t, "(*)")
        return "!llvm.ptr"
    end
    
    if t == "double"
        return "f64"
    elseif t == "float"
        return "f32"
    elseif t == "int" || t == "unsigned int" || t == "int32_t" || t == "uint32_t"
        return "i32"
    elseif t == "short" || t == "unsigned short" || t == "int16_t" || t == "uint16_t"
        return "i16"
    elseif t == "char" || t == "unsigned char" || t == "int8_t" || t == "uint8_t" || t == "bool"
        return "i8"
    elseif t == "long" || t == "unsigned long" || t == "long long" || t == "unsigned long long" || t == "int64_t" || t == "uint64_t" || t == "size_t"
        return "i64"
    elseif t == "void"
        return "none" # Should handle carefully
    end
    
    # Fallback for unknown structs/classes
    # Ideally we would map them to their own named type, but for now pointer or opaque.
    # If it's a value type member, we might want to assume it's a struct and handle it.
    # But DWARF parser might not give us full nested info easily yet.
    # Let's fallback to i8 array of correct size? No, that's hard.
    # Let's fallback to !llvm.ptr if we can't determine.
    # Actually, for members, if it is a struct by value, we need to know.
    # For now, let's assume primitives or pointers.
    return "!llvm.ptr" 
end

"""
    generate_type_info_ir(class_name::String, info::ClassInfo, vtable_addr::UInt64) -> String

Generate JLCS type_info operation for a class.
"""
function generate_type_info_ir(class_name::String, info::DWARFParser.ClassInfo, vtable_addr::UInt64)
    mlir_name = replace(class_name, 
        "::" => "_", "<" => "_", ">" => "_", "(" => "_", ")" => "_",
        " " => "", "," => "_", "*" => "Ptr", "&" => "Ref"
    )

    if !isempty(mlir_name) && !isletter(mlir_name[1]) && mlir_name[1] != "_"
        mlir_name = "_" * mlir_name
    end
    
    if isempty(mlir_name)
        mlir_name = "anonymous_type_$(hash(class_name))"
    end

    # Build field types and offsets lists
    field_types = String[]
    field_offsets = Int[]
    
    # 1. Add vtable pointer if present (usually at offset 0)
    # We check if info.vtable_ptr_offset is valid (e.g. 0).
    # Some classes might not have vtable.
    has_vptr = !isempty(info.virtual_methods)
    
    # We need to sort members by offset to ensure correct order?
    # !jlcs.c_struct takes lists, so order matters for the list, but offsets are explicit.
    # However, for readability, sorting is good.
    sorted_members = sort(info.members, by = m -> m.offset)
    
    # Explicitly add members
    for m in sorted_members
        push!(field_types, map_cpp_type_to_mlir(m.type_name))
        push!(field_offsets, m.offset)
    end
    
    # If we have a vptr but no member covering it (usually implicit), we should probably add it?
    # DWARF usually exposes vptr as a member like "_vptr$Shape" or similar.
    # Our DWARFParser handles it.
    # If not found in members, and we expect one...
    # Let's rely on what DWARFParser gives us.

    field_types_str = join(field_types, ", ")
    field_offsets_str = join(field_offsets, ", ")
    
    # Supertype
    super_type = isempty(info.base_classes) ? "" : info.base_classes[1]
    
    # Construct the CStruct type string
    # !jlcs.c_struct<"Name", [T1, T2], [O1, O2], packed=false>
    struct_type_str = "!jlcs.c_struct<\"$(class_name)\", [$(field_types_str)], [$(field_offsets_str)], packed=false>"

    ir = """
  jlcs.type_info "$(mlir_name)", $(struct_type_str), "$(super_type)" """

    return ir
end

"""
    generate_virtual_method_ir(method::VirtualMethod, addr::UInt64) -> String

Generate IR for a virtual method declaration.
"""
function generate_virtual_method_ir(method::DWARFParser.VirtualMethod, addr::UInt64)
    mlir_name = replace(method.mangled_name, "::" => "_", "(" => "_", ")" => "_")
    dispatch_name = "dispatch_$(replace(method.mangled_name, "::" => "_", "(" => "_", ")" => "_"))"
    
    (ret_type, arg_types_str) = get_llvm_signature(method)
    
    arg_names = ["%arg$i" for i in 0:length(method.parameters)]
    arg_sig_parts = ["$(arg_names[i]): $(t)" for (i, t) in enumerate(split(arg_types_str, ", "))]
    
    args_sig = "(" * join(arg_sig_parts, ", ") * ")"
    args_vals = "(" * join(arg_names, ", ") * ")"
    call_sig = "(" * arg_types_str * ")"
    
    call_stmt = ret_type == "" ? 
        "llvm.call @$(dispatch_name)$(args_vals) : $(call_sig) -> ()" : 
        "%result = llvm.call @$(dispatch_name)$(args_vals) : $(call_sig) -> $(ret_type)"
    
    return_stmt = ret_type == "" ? "return" : "return %result : $(ret_type)"

    body = """
  func.func @$(mlir_name)$(args_sig) -> $(ret_type == "" ? "()" : ret_type) {
    $(call_stmt)
    $(return_stmt)
  }"""

    return body
end

"""
    generate_jlcs_ir(vtinfo::VtableInfo, metadata::Any=Dict()) -> String

Generate complete JLCS MLIR module from vtable information and metadata.
"""
function generate_jlcs_ir(vtinfo::DWARFParser.VtableInfo, metadata::Any=Dict())
    io = IOBuffer()

    println(io, "// JLCS IR - Generated from DWARF debug info")
    println(io, "// Universal FFI via MLIR Dialects")
    println(io, "")

    # 1. Generate Struct Definitions (Aliases & Registration)
    if haskey(metadata, "struct_definitions")
        (aliases_ir, regs_ir) = generate_struct_definitions(metadata["struct_definitions"])
        println(io, aliases_ir)
        println(io, "")
    else
        regs_ir = ""
    end

    println(io, "module {")
    
    if !isempty(regs_ir)
        println(io, regs_ir)
    end

    # 2. External Dispatch Declarations (Virtual Methods)
    println(io, "  // External Dispatch Declarations (Virtual Methods)")
    for (class_name, class_info) in vtinfo.classes
        for method in class_info.virtual_methods
             method_addr = get(vtinfo.method_addresses, method.mangled_name, UInt64(0))
             if method_addr != 0
                 dispatch_name = "dispatch_$(replace(method.mangled_name, "::" => "_", "(" => "_", ")" => "_"))"
                 (ret_type, arg_types) = get_llvm_signature(method)
                 
                 decl_ret = ret_type == "" ? "!llvm.void" : ret_type
                 println(io, "  llvm.func @$(dispatch_name)($(arg_types)) -> $(decl_ret)")
             end
        end
    end
    println(io, "")

    # 3. Generate Type Info & VMethods
    generated_symbols = Set{String}()
    
    for (class_name, class_info) in vtinfo.classes
        if class_info.size == 0; continue; end

        # vtable_addr might be useful metadata, but TypeInfoOp doesn't store it in the new format.
        # We could add it as an attribute if needed.
        vtable_addr = get(vtinfo.vtable_addresses, class_name, UInt64(0))
        
        println(io, generate_type_info_ir(class_name, class_info, vtable_addr))
        println(io, "")

        for method in class_info.virtual_methods
            method_addr = get(vtinfo.method_addresses, method.mangled_name, UInt64(0))
            if method_addr != 0
                println(io, generate_virtual_method_ir(method, method_addr))
                println(io, "")
                push!(generated_symbols, method.mangled_name)
            end
        end
    end

    # 4. Generate Function Thunks (for regular functions)
    if haskey(metadata, "functions")
        structs_meta = get(metadata, "struct_definitions", Dict())
        # Filter out functions already generated by VtableGen
        filtered_functions = filter(f -> !(get(f, "mangled", "") in generated_symbols), metadata["functions"])
        println(io, generate_function_thunks(filtered_functions, structs_meta))
    end

    println(io, "}")

    ir = String(take!(io))
    println("DEBUG: Generated IR:\n$ir")
    return ir
end

end # module JLCSIRGenerator
