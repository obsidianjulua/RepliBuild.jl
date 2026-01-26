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

    ir = """
  jlcs.type_info @"$(mlir_name)" {
    size = $(info.size) : i64,
    vtable_offset = $(info.vtable_ptr_offset) : i64,
    vtable_addr = $(vtable_addr) : i64
  }"""

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

    println(io, "module {")

    # 1. Generate Struct Definitions (Registration)
    if haskey(metadata, "struct_definitions")
        println(io, generate_struct_definitions(metadata["struct_definitions"]))
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
                 println(io, "  llvm.func @$(dispatch_name)($(arg_types)) -> $(decl_ret) attributes { sym_visibility = \"private\" }")
             end
        end
    end
    println(io, "")

    # 3. Generate Type Info & VMethods
    for (class_name, class_info) in vtinfo.classes
        if class_info.size == 0; continue; end

        vtable_addr = get(vtinfo.vtable_addresses, class_name, UInt64(0))
        println(io, generate_type_info_ir(class_name, class_info, vtable_addr))
        println(io, "")

        for method in class_info.virtual_methods
            method_addr = get(vtinfo.method_addresses, method.mangled_name, UInt64(0))
            if method_addr != 0
                println(io, generate_virtual_method_ir(method, method_addr))
                println(io, "")
            end
        end
    end

    # 4. Generate Function Thunks (for regular functions)
    if haskey(metadata, "functions")
        structs_meta = get(metadata, "struct_definitions", Dict())
        println(io, generate_function_thunks(metadata["functions"], structs_meta))
    end

    println(io, "}")

    ir = String(take!(io))
    println("DEBUG: Generated IR:\n$ir")
    return ir
end

end # module JLCSIRGenerator