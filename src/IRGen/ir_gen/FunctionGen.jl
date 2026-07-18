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

function generate_function_thunks(functions::Vector, structs::Any=Dict(); may_throw::Bool=false,
                                  class_raii::Dict{String,Dict{Symbol,String}}=Dict{String,Dict{Symbol,String}}(),
                                  vcall_info::AbstractDict=Dict{String,Any}())
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

        # Per-param scope-RAII decision: a by-value class param whose class has
        # an emitted destructor is non-trivial for the purposes of calls under
        # the Itanium ABI — the callee expects a POINTER to a caller-owned
        # temporary, not raw bits in registers. Passing the raw struct (the old
        # behavior) miscompiled those calls. The producer builds the temporary
        # (copy-ctor when resolvable, bit-copy when the copy is trivial), passes
        # its address, and destructs it after the call via jlcs.scope.
        # nothing = not an RAII param; NamedTuple = transform it.
        raii_specs = Vector{Any}(nothing, length(params))

        arg_types = String[]
        for (pi, p) in enumerate(params)
            t = get(p, "c_type", "void*")
            mlir_t = map_cpp_type(t)

            # Resolve full struct type if available
            if startswith(mlir_t, "!llvm.struct<\"") && endswith(mlir_t, "\">")
                s_name = mlir_t[15:end-2]
                lookup_key = haskey(structs, s_name) ? s_name : (haskey(structs, "__enum__$(s_name)") ? "__enum__$(s_name)" : nothing)
                if lookup_key !== nothing
                    # Scope-RAII takes precedence over both by-value marshalling
                    # paths: a class with an emitted destructor is non-trivial
                    # for the purposes of calls (Itanium), so the callee expects
                    # a POINTER to a caller-owned temporary regardless of byte
                    # layout. (is_struct_packed classifies any padding-free
                    # struct as "packed", so it must not gate this decision.)
                    cls_key = strip(replace(t, r"\bconst\b" => ""))
                    bsz = try _parse_byte_size(get(structs[lookup_key], "byte_size", "0")) catch; 0 end
                    if haskey(class_raii, cls_key) && bsz > 0
                        raii_specs[pi] = (cls = cls_key, size = bsz,
                                          dtor = class_raii[cls_key][:dtor],
                                          copy_ctor = get(class_raii[cls_key], :copy_ctor, ""))
                        mlir_t = "!llvm.ptr"   # Itanium: pass address of the temporary
                    elseif StructGen.is_struct_packed(structs[lookup_key])
                        # For packed structs, use LLVM packed type to avoid type mismatch with thunk marshalling
                        mlir_t = StructGen.get_llvm_equivalent_type_string(s_name, structs[lookup_key], structs)
                    else
                        # Use FULL definition string to avoid alias parser issues in function signatures
                        mlir_t = StructGen.get_struct_definition_string(s_name, structs[lookup_key], structs)
                    end
                end
            end

            push!(arg_types, mlir_t)
        end
        has_raii = any(!isnothing, raii_specs)

        ret_c_type = get(ret_info, "c_type", "void")
        ret_type = map_cpp_type(ret_c_type)

        # Track packed struct return for thunk conversion
        is_packed_ret = false
        ret_packed_type = ""
        ret_aligned_type = ""
        ret_struct_info = nothing
        ret_num_members = 0

        # Enum return → bare underlying integer, NOT a single-member struct.
        # A struct result (even `struct<(i32)>`) makes MLIR's emit_c_interface
        # use the sret convention (void ciface(T* sret, void** args)), but the
        # Julia side calls the @enum back as a scalar (T ciface(void** args)) —
        # the args pointer lands in the sret slot and the call derefs garbage.
        # Found live: tinyxml2 XMLDocument::Parse → XMLError. Bare int returns
        # by value, matching the scalar ccall; @enum is ABI-identical to its base.
        if startswith(ret_type, "!llvm.struct<\"") && endswith(ret_type, "\">")
            _ename = ret_type[15:end-2]
            if haskey(structs, "__enum__$(_ename)")
                ju = String(get(structs["__enum__$(_ename)"], "julia_type", "Cint"))
                ret_type = map_cpp_type(ju == "Any" ? "int" : ju)
                ret_type == "" && (ret_type = "i32")
            end
        end

        # Resolve full struct type for return
        if startswith(ret_type, "!llvm.struct<\"") && endswith(ret_type, "\">")
            s_name = ret_type[15:end-2]
            lookup_key = haskey(structs, s_name) ? s_name : (haskey(structs, "__enum__$(s_name)") ? "__enum__$(s_name)" : nothing)
            if lookup_key !== nothing
                # For packed structs, use LLVM packed type for external decl/ffe_call
                if StructGen.is_struct_packed(structs[lookup_key])
                    ret_type = StructGen.get_llvm_equivalent_type_string(s_name, structs[lookup_key], structs)
                    is_packed_ret = true
                    ret_packed_type = ret_type
                    ret_aligned_type = StructGen.get_llvm_aligned_type_string(s_name, structs[lookup_key], structs)
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
                        ret_type = StructGen.get_struct_definition_string(s_name, structs[lookup_key], structs)
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
                stl_size = get_stl_container_size(ret_c_type)
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

        # Scope-RAII entry allocas: one caller-owned temporary per non-trivial
        # by-value param, plus (for non-void calls) a slot the call result
        # escapes through — jlcs.scope has no results, so values leave via memory.
        raii_ret_slot_type = ""
        if has_raii
            println(io, "  %raii_one = llvm.mlir.constant(1 : i64) : i64")
            for (pi, spec) in enumerate(raii_specs)
                isnothing(spec) && continue
                println(io, "  %raii_tmp_$(pi) = llvm.alloca %raii_one x !llvm.array<$(spec.size) x i8> : (i64) -> !llvm.ptr")
            end
            raii_ret_slot_type = is_packed_ret ? ret_packed_type : func_ret
            if raii_ret_slot_type != ""
                println(io, "  %raii_retslot = llvm.alloca %raii_one x $(raii_ret_slot_type) : (i64) -> !llvm.ptr")
            end
        end

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
                        mlir_t = StructGen.get_llvm_equivalent_type_string(s_name, struct_info, structs)
                    else
                        mlir_t = StructGen.get_struct_definition_string(s_name, struct_info, structs)
                    end
                end
            end

            idx = i - 1
            println(io, "  %idx_$(i) = arith.constant $(idx) : i64")
            println(io, "  %arg_ptr_$(i) = llvm.getelementptr %args_ptr[%idx_$(i)] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr")
            println(io, "  %val_ptr_$(i) = llvm.load %arg_ptr_$(i) : !llvm.ptr -> !llvm.ptr")

            if !isnothing(raii_specs[i])
                # Non-trivial by-value param: the temporary is copy-constructed
                # inside the jlcs.scope (emitted at the call site below); here we
                # only record the source pointer. The callee receives %raii_tmp_i.
                push!(call_args, "%raii_tmp_$(i)")
                continue
            end

            if is_packed_struct
                # Layout mismatch: Julia passes aligned struct pointer, C++ expects packed struct by value.
                # Emit jlcs.marshal_arg op — the MLIR lowering pass handles the field-by-field reconstruction.
                llvm_t = mlir_t  # Already the LLVM packed type

                offsets = StructGen.get_julia_offsets(struct_info)
                members = get(struct_info, "members", [])
                member_types_strs = [map_cpp_type(get(m, "c_type", "void*")) for m in members]

                member_types_attr = "[" * join(member_types_strs, ", ") * "]"
                offsets_attr = "[" * join(["$(o) : i64" for o in offsets], ", ") * "]"

                println(io, "  %packed_$(i) = jlcs.marshal_arg %val_ptr_$(i) { memberTypes = $(member_types_attr), juliaOffsets = $(offsets_attr) } : (!llvm.ptr) -> $(llvm_t)")
                arg_value_name = "%packed_$(i)"
            else
                println(io, "  %val_$(i) = llvm.load %val_ptr_$(i) : !llvm.ptr -> $(mlir_t)")
                arg_value_name = "%val_$(i)"
            end

            push!(call_args, arg_value_name)
        end

        # Determine whether to use jlcs.try_call (exception-safe) or jlcs.ffe_call
        # Per-function noexcept: if the function is marked noexcept, use ffe_call even when may_throw is set
        func_is_noexcept = get(func, "is_noexcept", false)
        use_try_call = may_throw && !func_is_noexcept
        call_op = use_try_call ? "jlcs.try_call" : "jlcs.ffe_call"

        # Virtual dispatch: methods with vtable coordinates get a jlcs.vcall
        # (read vptr → index slot → indirect call) so OVERRIDES are honored —
        # a direct symbol call is `p->Class::method()` static semantics. The
        # vcall lowering does no sret/packed ABI coercion, so only
        # scalar/pointer signatures are eligible; struct-shaped ones keep the
        # direct-call path (rare for virtual methods). may_throw on the op
        # picks the invoke+landing-pad lowering (same sentinel-continue EH
        # model as try_call).
        vc = get(vcall_info, mangled, nothing)
        use_vcall = vc !== nothing && is_method && !has_raii && !is_packed_ret &&
                    !isempty(call_args) &&
                    !occursin("struct", func_ret) &&
                    !any(t -> occursin("struct", t) || occursin("array", t), arg_types)
        if use_vcall
            vcall_attrs = "class_name = @$(vc.class), vtable_offset = $(vc.vtable_offset) : i64, " *
                          "slot = $(vc.slot) : i64, this_offset = 0 : i64"
            use_try_call && (vcall_attrs *= ", may_throw")
        end

        # Call using jlcs.ffe_call or jlcs.try_call (via Dialect)
        if has_raii
            # Scope-RAII: copy-construct temporaries, call inside the scope,
            # destructors fire in reverse order at scope exit. try_call converts
            # C++ exceptions to sentinel-return-and-continue, so the normal path
            # (where the scope emits dtors) is the only path — coverage is total.
            raii_idx = [i for i in 1:length(params) if !isnothing(raii_specs[i])]
            tmp_list = join(["%raii_tmp_$(i)" for i in raii_idx], ", ")
            tmp_types = join(fill("!llvm.ptr", length(raii_idx)), ", ")
            # managed_ptrs and dtors are co-generated from the same index list —
            # the arity invariant the (still unverified) op relies on
            dtor_list = join(["@$(raii_specs[i].dtor)" for i in raii_idx], ", ")

            println(io, "  jlcs.scope($(tmp_list) : $(tmp_types)) dtors([$(dtor_list)]) {")
            for i in raii_idx
                spec = raii_specs[i]
                if !isempty(spec.copy_ctor)
                    println(io, "    jlcs.ctor_call @$(spec.copy_ctor)(%raii_tmp_$(i), %val_ptr_$(i)) : (!llvm.ptr, !llvm.ptr) -> ()")
                else
                    # No copy-ctor symbol → copy is trivial; bit-copy the bytes
                    println(io, "    %raii_blob_$(i) = llvm.load %val_ptr_$(i) : !llvm.ptr -> !llvm.array<$(spec.size) x i8>")
                    println(io, "    llvm.store %raii_blob_$(i), %raii_tmp_$(i) : !llvm.array<$(spec.size) x i8>, !llvm.ptr")
                end
            end
            if func_ret == ""
                println(io, "    $(call_op) $(join(call_args, ", ")) { callee = @$(mangled) } : ($(join(arg_types, ", "))) -> ()")
            else
                inner_ret = is_packed_ret ? ret_packed_type : func_ret
                println(io, "    %raii_ret = $(call_op) $(join(call_args, ", ")) { callee = @$(mangled) } : ($(join(arg_types, ", "))) -> $(inner_ret)")
                println(io, "    llvm.store %raii_ret, %raii_retslot : $(inner_ret), !llvm.ptr")
            end
            println(io, "    jlcs.yield")
            println(io, "  }")
            if func_ret == ""
                println(io, "  return")
            elseif is_packed_ret
                println(io, "  %ret_packed = llvm.load %raii_retslot : !llvm.ptr -> $(ret_packed_type)")
                println(io, "  %ret_aligned = jlcs.marshal_ret %ret_packed { numMembers = $(ret_num_members) : i64 } : ($(ret_packed_type)) -> $(ret_aligned_type)")
                println(io, "  return %ret_aligned : $(ret_aligned_type)")
            else
                println(io, "  %ret_val = llvm.load %raii_retslot : !llvm.ptr -> $(func_ret)")
                println(io, "  return %ret_val : $(func_ret)")
            end
        elseif func_ret == ""
             # Void return
             if use_vcall
                 println(io, "  \"jlcs.vcall\"($(join(call_args, ", "))) { $(vcall_attrs) } : ($(join(arg_types, ", "))) -> ()")
             else
                 println(io, "  $(call_op) $(join(call_args, ", ")) { callee = @$(mangled) } : ($(join(arg_types, ", "))) -> ()")
             end
             println(io, "  return")
        elseif is_packed_ret
             # Packed struct return: call returns packed type, marshal to aligned for Julia
             println(io, "  %ret_packed = $(call_op) $(join(call_args, ", ")) { callee = @$(mangled) } : ($(join(arg_types, ", "))) -> $(ret_packed_type)")
             println(io, "  %ret_aligned = jlcs.marshal_ret %ret_packed { numMembers = $(ret_num_members) : i64 } : ($(ret_packed_type)) -> $(ret_aligned_type)")
             println(io, "  return %ret_aligned : $(ret_aligned_type)")
        else
             # Value return
             if use_vcall
                 println(io, "  %ret_val = \"jlcs.vcall\"($(join(call_args, ", "))) { $(vcall_attrs) } : ($(join(arg_types, ", "))) -> $(func_ret)")
             else
                 println(io, "  %ret_val = $(call_op) $(join(call_args, ", ")) { callee = @$(mangled) } : ($(join(arg_types, ", "))) -> $(func_ret)")
             end
             println(io, "  return %ret_val : $(func_ret)")
        end

        println(io, "}")
        println(io, "")
    end

    return String(take!(io))
end

end
