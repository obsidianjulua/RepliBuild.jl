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
include("ir_gen/STLContainerGen.jl")
include("ir_gen/ArrayViewGen.jl")

using .TypeUtils
using .StructGen
using .FunctionGen
using .STLContainerGen
using .ArrayViewGen

export generate_jlcs_ir, generate_mlir_module

"""
    _collect_class_raii(metadata) -> Dict{String,Dict{Symbol,String}}

Resolve per-class RAII symbols from DWARF metadata: the complete-object
destructor (`D1` preferred over `D2`/`D0`) and, when present, the copy
constructor (`C1` preferred, signature `(const T&)`).

Symbol presence is the gate (the repo's symbol-presence law): a truly trivial
destructor is never emitted as a symbol, so a class appearing here is
non-trivial for the purposes of calls under the Itanium ABI — by-value
crossings must go through a caller-owned temporary that is destructed after
the call, which is exactly what the scope-RAII producer emits.
"""
function _collect_class_raii(metadata)::Dict{String,Dict{Symbol,String}}
    raii = Dict{String,Dict{Symbol,String}}()
    for f in get(metadata, "functions", [])
        get(f, "is_method", false) || continue
        cls = String(get(f, "class", ""))
        isempty(cls) && continue
        mangled = String(get(f, "mangled", ""))
        isempty(mangled) && continue
        demangled = String(get(f, "demangled", ""))

        entry = get!(raii, cls, Dict{Symbol,String}())

        if occursin("~", demangled)
            # Destructor: prefer the complete-object destructor (D1)
            if !haskey(entry, :dtor) || (occursin("D1E", mangled) && !occursin("D1E", entry[:dtor]))
                entry[:dtor] = mangled
            end
        elseif occursin("$(cls)::$(cls)(", demangled)
            # Constructor: a copy ctor takes exactly one param of `const cls&`
            # (metadata params exclude the implicit `this`)
            params = get(f, "parameters", [])
            explicit = [p for p in params if get(p, "name", "") != "this"]
            if length(explicit) == 1
                pct = strip(String(get(explicit[1], "c_type", "")))
                pct_clean = strip(replace(pct, r"\bconst\b" => ""))
                if pct_clean == "$(cls)&" || pct_clean == "$(cls) &"
                    if !haskey(entry, :copy_ctor) || (occursin("C1E", mangled) && !occursin("C1E", entry[:copy_ctor]))
                        entry[:copy_ctor] = mangled
                    end
                end
            end
        end
    end
    # Only classes with a resolvable destructor participate
    filter!(p -> haskey(p.second, :dtor), raii)
    return raii
end

"""
    _sanitize_mlir_symbol(name) -> String

C++ type name → valid MLIR symbol identifier (shared by type_info emission
and the vcall producer's class_name attr).
"""
function _sanitize_mlir_symbol(name::String)
    mlir_name = replace(name,
        "::" => "_", "<" => "_", ">" => "_", "(" => "_", ")" => "_",
        " " => "", "," => "_", "*" => "Ptr", "&" => "Ref"
    )
    if !isempty(mlir_name) && !isletter(mlir_name[1]) && mlir_name[1] != '_'
        mlir_name = "_" * mlir_name
    end
    if isempty(mlir_name)
        mlir_name = "anonymous_type_$(hash(name))"
    end
    return mlir_name
end

"""
    generate_type_info_ir(class_name::String, info::ClassInfo, vtable_addr::UInt64;
                          destructor::String="") -> String

Generate JLCS type_info operation for a class. `destructor` is the
DWARF-resolved complete-object destructor symbol (empty when none exists).
"""
function generate_type_info_ir(class_name::String, info::DWARFParser.ClassInfo, vtable_addr::UInt64;
                               destructor::String="")
    mlir_name = _sanitize_mlir_symbol(class_name)

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
        mlir_t = map_cpp_type(m.type_name)
        # Convert bare named struct references (!llvm.struct<"Name">) to aliases (!Struct_Name)
        # because bare named struct refs are only valid in recursive struct contexts in MLIR
        if startswith(mlir_t, "!llvm.struct<\"") && endswith(mlir_t, "\">")
            s_name = mlir_t[15:end-2]
            safe_name = replace(s_name, r"[^a-zA-Z0-9_]" => "_")
            mlir_t = "!Struct_$(safe_name)"
        end
        push!(field_types, mlir_t)
        push!(field_offsets, m.offset)
    end
    
    # If we have a vptr but no member covering it (usually implicit), we should probably add it?
    # DWARF usually exposes vptr as a member like "_vptr$Shape" or similar.
    # Our DWARFParser handles it.
    # If not found in members, and we expect one...
    # Let's rely on what DWARFParser gives us.

    field_types_str = join(field_types, ", ")
    # fieldOffsets is an ArrayAttr — parser calls FieldParser<ArrayAttr>::parse after consuming
    # the outer '[', so it needs a full MLIR attribute: [N : i64, ...] inside the outer brackets.
    field_offsets_attr = "[" * join(["$(o) : i64" for o in field_offsets], ", ") * "]"

    # Supertype (primary base) — kept for single-inheritance consumers
    super_type = isempty(info.base_classes) ? "" : info.base_classes[1]

    # Construct the CStruct type string
    # !jlcs.c_struct<"Name", [T1, T2], [[O1 : i64, O2 : i64]], packed = false>
    struct_type_str = "!jlcs.c_struct<\"$(class_name)\", [$(field_types_str)], [$(field_offsets_attr)], packed = false>"

    # Base tables: non-virtual bases carry static subobject offsets in
    # baseNames/baseOffsets; virtual bases have NO static offset (it is
    # vtable-resident) and go in vbaseNames/vbaseVtableOffsets, recording the
    # negative byte position of the vbase-offset entry relative to the vtable
    # address point — the coordinate a dynamic upcast reads through the vptr.
    base_table = ""
    if !isempty(info.base_classes)
        nv_names = String[]; nv_offs = Int[]
        vb_names = String[]; vb_offs = Int[]
        for (i, b) in enumerate(info.base_classes)
            if info.virtual_bases[i]
                # Only representable with a parsed vtable coordinate
                if info.vbase_vtable_offsets[i] != 0
                    push!(vb_names, b); push!(vb_offs, info.vbase_vtable_offsets[i])
                else
                    @warn "Class $(class_name): virtual base $(b) has no resolvable vbase vtable offset — omitted from type_info"
                end
            else
                push!(nv_names, b); push!(nv_offs, info.base_offsets[i])
            end
        end
        parts = String[]
        if !isempty(nv_names)
            push!(parts, "baseNames = [$(join(["\"$(n)\"" for n in nv_names], ", "))]")
            push!(parts, "baseOffsets = [$(join(["$(o) : i64" for o in nv_offs], ", "))]")
        end
        if !isempty(vb_names)
            push!(parts, "vbaseNames = [$(join(["\"$(n)\"" for n in vb_names], ", "))]")
            push!(parts, "vbaseVtableOffsets = [$(join(["$(o) : i64" for o in vb_offs], ", "))]")
        end
        isempty(parts) || (base_table = " {$(join(parts, ", "))}")
    end

    ir = """
  jlcs.type_info "$(mlir_name)", $(struct_type_str), "$(super_type)", "$(destructor)"$(base_table) """

    return ir
end

"""
    generate_virtual_method_ir(method::VirtualMethod, addr::UInt64) -> String

Generate IR for a virtual method declaration.
"""
function generate_virtual_method_ir(method::DWARFParser.VirtualMethod, addr::UInt64)
    # The actual C++ function we want to call
    call_target = method.mangled_name
    
    # The MLIR wrapper function name (must not conflict with the call target)
    mlir_name = "thunk_$(call_target)"
    
    (ret_type, arg_types_str) = get_llvm_signature(method)
    
    arg_names = ["%arg$i" for i in 0:length(method.parameters)]
    arg_sig_parts = ["$(arg_names[i]): $(t)" for (i, t) in enumerate(split(arg_types_str, ", "))]
    
    args_sig = "(" * join(arg_sig_parts, ", ") * ")"
    args_vals = "(" * join(arg_names, ", ") * ")"
    call_sig = "(" * arg_types_str * ")"
    
    call_stmt = ret_type == "" ? 
        "llvm.call @$(call_target)$(args_vals) : $(call_sig) -> ()" : 
        "%result = llvm.call @$(call_target)$(args_vals) : $(call_sig) -> $(ret_type)"
    
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
function generate_jlcs_ir(vtinfo::DWARFParser.VtableInfo, metadata::Any=Dict();
                          needed_symbols::Union{Set{String}, Nothing}=nothing)
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

    # 1b. Exception handling helper declarations (for C++ projects)
    lang = get(metadata, "language", "")
    is_cpp = lang in ("c++", "cpp", "cxx") ||
             (haskey(metadata, "functions") && any(f -> get(f, "is_method", false) ||
              startswith(get(f, "mangled", ""), "_Z"), get(metadata, "functions", [])))
    if is_cpp
        println(io, "  // Exception handling helper declarations")
        println(io, "  llvm.func @__gxx_personality_v0(...) -> i32")
        println(io, "  llvm.func @__cxa_begin_catch(!llvm.ptr) -> !llvm.ptr")
        println(io, "  llvm.func @__cxa_end_catch()")
        println(io, "  llvm.func @jlcs_set_pending_exception(!llvm.ptr)")
        println(io, "  llvm.func @jlcs_catch_current_exception() -> !llvm.ptr")
        println(io, "")
    end

    # 2. External Dispatch Declarations (Virtual Methods)
    # Each callee must be declared exactly once, with the op type its caller
    # needs. Two later passes declare callees:
    #   • the vtable vmethod-IR pass emits `llvm.call @X`, needing `llvm.func @X`
    #     (these symbols — `gen_pre` below — are the ones THIS loop must declare);
    #   • the function-thunk pass emits `func.func private @X` + `jlcs.try_call`
    #     (these — `fthunk_decls` — own their own declaration).
    # The two sets are mutually exclusive. Declaring an `fthunk_decls` symbol here
    # too produces a redefinition with a conflicting arity; so skip those. (The
    # vmethod-IR pass and function-thunk filter below replicate this same gating.)
    # Virtual methods the WRAPPER needs must go through the function-thunk
    # pass: the generated wrapper dispatches Tier 2 via
    # JITManager.invoke("_mlir_ciface_<mangled>_thunk", ...), and only
    # FunctionGen emits that ciface convention. The legacy vmethod-IR pass
    # emits `thunk_<mangled>` direct-call wrappers no wrapper ever looks up —
    # every virtual instance method was silently uncallable through invoke()
    # until the MI fixture drove one (2026-07-17). Note the resulting call is
    # the statically-named class's implementation (`p->Base2::get_b()`
    # semantics); override-honoring dispatch is the future vcall producer.
    gen_pre = Set{String}()
    for (class_name, class_info) in vtinfo.classes
        (class_info.size == 0 || isempty(class_info.members)) && continue
        for method in class_info.virtual_methods
            get(vtinfo.method_addresses, method.mangled_name, UInt64(0)) == 0 && continue
            needed_symbols !== nothing && method.mangled_name in needed_symbols && continue
            push!(gen_pre, method.mangled_name)
        end
    end
    fthunk_decls = Set{String}()
    for f in get(metadata, "functions", [])
        m = get(f, "mangled", "")
        isempty(m) && continue
        m in gen_pre && continue
        needed_symbols !== nothing && !(m in needed_symbols) && continue
        get(f, "is_vararg", false) && continue
        push!(fthunk_decls, m)
    end
    println(io, "  // External Dispatch Declarations (Virtual Methods)")
    for (class_name, class_info) in vtinfo.classes
        for method in class_info.virtual_methods
             method_addr = get(vtinfo.method_addresses, method.mangled_name, UInt64(0))
             if method_addr != 0
                 dispatch_name = "$(method.mangled_name)"
                 dispatch_name in fthunk_decls && continue
                 (ret_type, arg_types) = get_llvm_signature(method)

                 decl_ret = ret_type == "" ? "!llvm.void" : ret_type
                 println(io, "  llvm.func @$(dispatch_name)($(arg_types)) -> $(decl_ret)")
             end
        end
    end
    println(io, "")

    # Per-class RAII symbols (destructor + copy ctor) resolved once from DWARF.
    # Drives type_info's destructorName and the scope-RAII producer in FunctionGen.
    class_raii = _collect_class_raii(metadata)

    # 3. Generate Type Info & VMethods
    generated_symbols = Set{String}()

    for (class_name, class_info) in vtinfo.classes
        if class_info.size == 0; continue; end
        # Skip classes with no members (empty tag types, stdlib internals, etc.)
        if isempty(class_info.members); continue; end

        # vtable_addr might be useful metadata, but TypeInfoOp doesn't store it in the new format.
        # We could add it as an attribute if needed.
        vtable_addr = get(vtinfo.vtable_addresses, class_name, UInt64(0))

        class_dtor = get(get(class_raii, class_name, Dict{Symbol,String}()), :dtor, "")
        println(io, generate_type_info_ir(class_name, class_info, vtable_addr; destructor=class_dtor))
        println(io, "")

        for method in class_info.virtual_methods
            method_addr = get(vtinfo.method_addresses, method.mangled_name, UInt64(0))
            if method_addr != 0
                # Wrapper-needed methods are FunctionGen's (ciface thunks) —
                # emitting the legacy thunk here would llvm.call a symbol whose
                # declaration now belongs to the function-thunk pass.
                method.mangled_name in fthunk_decls && continue
                println(io, generate_virtual_method_ir(method, method_addr))
                println(io, "")
                push!(generated_symbols, method.mangled_name)
            end
        end
    end

    # 4. Generate Function Thunks (for regular functions)
    if haskey(metadata, "functions")
        structs_meta = get(metadata, "struct_definitions", Dict())
        # Filter out functions already generated by VtableGen, and (when a
        # thunk manifest is present) functions the wrapper doesn't need thunks for.
        filtered_functions = filter(metadata["functions"]) do f
            m = get(f, "mangled", "")
            m in generated_symbols && return false
            needed_symbols !== nothing && !(m in needed_symbols) && return false
            return true
        end

        # Detect if project is C++ (may throw exceptions unless noexcept)
        lang = get(metadata, "language", "")
        is_cpp = lang in ("c++", "cpp", "cxx") ||
                 any(f -> get(f, "is_method", false) || startswith(get(f, "mangled", ""), "_Z"), filtered_functions)

        # Virtual-dispatch table for the vcall producer: mangled symbol →
        # class-LOCAL dispatch coordinates. DWARF's DW_AT_vtable_elem_location
        # is the slot in the DECLARING class's own primary vtable (Itanium
        # even re-homes overrides of non-primary-base methods into the
        # derived class's primary vtable — verified against dwarfdump
        # 2026-07-17), and every ClassName_method wrapper takes a
        # ClassName-relative `this` — so vtable_offset = the class's vptr
        # offset (0 under Itanium) and this_offset = 0, always. MI callers
        # upcast first (as_Base2), and the vtable entries' own thunks handle
        # any further adjustment. Destructors are EXCLUDED: Managed
        # finalizers and the RAII producer require exact-class direct calls,
        # not dynamic dispatch.
        vcall_info = Dict{String, NamedTuple{(:class, :slot, :vtable_offset), Tuple{String,Int,Int}}}()
        for (class_name, class_info) in vtinfo.classes
            (class_info.size == 0 || isempty(class_info.members)) && continue
            cls_sym = _sanitize_mlir_symbol(class_name)
            for method in class_info.virtual_methods
                method.slot < 0 && continue
                startswith(method.name, "~") && continue
                isempty(method.mangled_name) && continue
                vcall_info[method.mangled_name] =
                    (class = cls_sym, slot = method.slot,
                     vtable_offset = class_info.vtable_ptr_offset)
            end
        end

        println(io, generate_function_thunks(filtered_functions, structs_meta;
                                             may_throw=is_cpp, class_raii=class_raii,
                                             vcall_info=vcall_info))
    end

    # 5. Generate STL Container Accessor Thunks
    if haskey(metadata, "stl_methods") && !isempty(metadata["stl_methods"])
        println(io, generate_stl_thunks(metadata["stl_methods"], metadata))
    end

    # 6. Generate strided array-view accessor thunks for fixed-size array members
    if haskey(metadata, "struct_definitions")
        av_ir = generate_array_view_thunks(metadata["struct_definitions"])
        isempty(av_ir) || println(io, av_ir)
    end

    println(io, "}")

    ir = String(take!(io))

    # Bodyless named-struct refs (`!llvm.struct<"X">`) leak in from the
    # get_llvm_signature path (dispatch decls / vmethod bodies), which — unlike
    # FunctionGen — doesn't resolve struct types to a bodied form. These appear in
    # LLVM-dialect ops (`llvm.func`/`llvm.call`), so they must become a literal
    # LLVM struct (`!llvm.struct<(…)>`) — NOT the `!Struct_X` jlcs alias, which is
    # a `!jlcs.c_struct` and is rejected as an `llvm.func` result. Resolve known
    # by-value structs to their LLVM-equivalent; degrade unknown/opaque to
    # !llvm.ptr. Skip alias-definition lines (`!Struct_X = …`), whose recursive
    # forward-refs are legitimately bodyless.
    struct_defs = get(metadata, "struct_definitions", Dict())
    llvm_struct_of(name::AbstractString) = haskey(struct_defs, String(name)) ?
        StructGen.get_llvm_equivalent_type_string(String(name), struct_defs[String(name)], struct_defs) : "!llvm.ptr"
    ir = join(map(split(ir, '\n')) do ln
        (startswith(lstrip(ln), "!Struct_") && occursin(" = ", ln)) && return ln
        replace(ln, r"!llvm\.struct<\"([A-Za-z0-9_]+)\">" =>
                    m -> llvm_struct_of(match(r"\"([A-Za-z0-9_]+)\"", m).captures[1]))
    end, '\n')

    # Sanitize dangling struct-alias references. Member rewriting in
    # generate_type_info_ir emits `!Struct_<name>` for any named-struct field,
    # but opaque/incomplete structs (e.g. glibc's `__off_t` pulled in via FILE*)
    # never get an alias definition (StructGen emits them as `opaque>` and skips
    # the alias). A reference with no definition makes MLIR reject the whole
    # module ("undefined symbol alias id"). Mirror StructGen's fallback: any
    # `!Struct_*` that isn't actually defined here degrades to `!llvm.ptr`.
    defined_aliases = Set{String}()
    for m in eachmatch(r"(!Struct_[A-Za-z0-9_]+)\s*=", ir)
        push!(defined_aliases, m.captures[1])
    end
    ir = replace(ir, r"!Struct_[A-Za-z0-9_]+" =>
                     s -> s in defined_aliases ? s : "!llvm.ptr")

    # println("DEBUG: Generated IR:\n$ir")
    return ir
end

end # module JLCSIRGenerator
