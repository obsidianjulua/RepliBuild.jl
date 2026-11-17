# LibDW.jl — DWARF read in-process through elfutils' libdw.
#
# `Compiler.parse_dwarf_dump` reads the text of `readelf --debug-dump=info`.
# `info_dump` walks the same DIE tree through libdw and writes the same text:
# one header line per DIE, one line per attribute, a terminator where a
# sibling list ends, in GNU readelf's spelling for everything the parser
# reads. So the parser, its invariants and its synthetic-dump tests are
# untouched, and the two front ends can be held to one answer: the tables
# parsed from either must be equal (test_libdw_dwarf_live.jl).
#
# What is NOT reproduced is what the parser never reads: abbreviation numbers,
# attribute offsets, the annotation after an enumerated constant
# (`8\t(unsigned char)`), register names.
#
# `file_tables` is `_dwarf_file_tables`' answer (one `index => path` map per
# compile unit) from libdw's own line-table reader.

module LibDW

import Elfutils_jll, Libdl

struct Die
    addr::Ptr{Cvoid}
    cu::Ptr{Cvoid}
    abbrev::Ptr{Cvoid}
    padding::Clong
end
Die() = Die(C_NULL, C_NULL, C_NULL, 0)

struct Attr
    code::Cuint
    form::Cuint
    valp::Ptr{UInt8}
    cu::Ptr{Cvoid}
end

struct Block
    length::UInt64
    data::Ptr{UInt8}
end

const libdw = Elfutils_jll.libdw

"Whether libdw can be used on this host."
available()::Bool = try
    Elfutils_jll.is_available() && Libdl.dlopen(libdw; throw_error = false) !== nothing
catch
    false
end

_errmsg() = (p = ccall((:dwarf_errmsg, libdw), Cstring, (Cint,), -1); p == C_NULL ? "unknown error" : unsafe_string(p))

# ── names, in binutils' spelling ────────────────────────────────────────────

const TAGS = Dict{Int, String}(
    0x01 => "array_type", 0x02 => "class_type", 0x03 => "entry_point", 0x04 => "enumeration_type",
    0x05 => "formal_parameter", 0x08 => "imported_declaration", 0x0a => "label", 0x0b => "lexical_block",
    0x0d => "member", 0x0f => "pointer_type", 0x10 => "reference_type", 0x11 => "compile_unit",
    0x12 => "string_type", 0x13 => "structure_type", 0x15 => "subroutine_type", 0x16 => "typedef",
    0x17 => "union_type", 0x18 => "unspecified_parameters", 0x19 => "variant", 0x1a => "common_block",
    0x1b => "common_inclusion", 0x1c => "inheritance", 0x1d => "inlined_subroutine", 0x1e => "module",
    0x1f => "ptr_to_member_type", 0x20 => "set_type", 0x21 => "subrange_type", 0x22 => "with_stmt",
    0x23 => "access_declaration", 0x24 => "base_type", 0x25 => "catch_block", 0x26 => "const_type",
    0x27 => "constant", 0x28 => "enumerator", 0x29 => "file_type", 0x2a => "friend", 0x2b => "namelist",
    0x2c => "namelist_item", 0x2d => "packed_type", 0x2e => "subprogram", 0x2f => "template_type_param",
    0x30 => "template_value_param", 0x31 => "thrown_type", 0x32 => "try_block", 0x33 => "variant_part",
    0x34 => "variable", 0x35 => "volatile_type", 0x36 => "dwarf_procedure", 0x37 => "restrict_type",
    0x38 => "interface_type", 0x39 => "namespace", 0x3a => "imported_module", 0x3b => "unspecified_type",
    0x3c => "partial_unit", 0x3d => "imported_unit", 0x3f => "condition", 0x40 => "shared_type",
    0x41 => "type_unit", 0x42 => "rvalue_reference_type", 0x43 => "template_alias", 0x44 => "coarray_type",
    0x45 => "generic_subrange", 0x46 => "dynamic_type", 0x47 => "atomic_type", 0x48 => "call_site",
    0x49 => "call_site_parameter", 0x4a => "skeleton_unit", 0x4b => "immutable_type",
    0x4106 => "GNU_template_template_param", 0x4107 => "GNU_template_parameter_pack",
    0x4108 => "GNU_formal_parameter_pack", 0x4109 => "GNU_call_site", 0x410a => "GNU_call_site_parameter",
)

const ATTRS = Dict{Int, String}(
    0x01 => "sibling", 0x02 => "location", 0x03 => "name", 0x09 => "ordering", 0x0b => "byte_size",
    0x0c => "bit_offset", 0x0d => "bit_size", 0x10 => "stmt_list", 0x11 => "low_pc", 0x12 => "high_pc",
    0x13 => "language", 0x15 => "discr", 0x16 => "discr_value", 0x17 => "visibility", 0x18 => "import",
    0x19 => "string_length", 0x1a => "common_reference", 0x1b => "comp_dir", 0x1c => "const_value",
    0x1d => "containing_type", 0x1e => "default_value", 0x20 => "inline", 0x21 => "is_optional",
    0x22 => "lower_bound", 0x25 => "producer", 0x27 => "prototyped", 0x2a => "return_addr",
    0x2c => "start_scope", 0x2e => "bit_stride", 0x2f => "upper_bound", 0x31 => "abstract_origin",
    0x32 => "accessibility", 0x33 => "address_class", 0x34 => "artificial", 0x35 => "base_types",
    0x36 => "calling_convention", 0x37 => "count", 0x38 => "data_member_location", 0x39 => "decl_column",
    0x3a => "decl_file", 0x3b => "decl_line", 0x3c => "declaration", 0x3d => "discr_list",
    0x3e => "encoding", 0x3f => "external", 0x40 => "frame_base", 0x41 => "friend",
    0x42 => "identifier_case", 0x43 => "macro_info", 0x44 => "namelist_item", 0x45 => "priority",
    0x46 => "segment", 0x47 => "specification", 0x48 => "static_link", 0x49 => "type",
    0x4a => "use_location", 0x4b => "variable_parameter", 0x4c => "virtuality",
    0x4d => "vtable_elem_location", 0x4e => "allocated", 0x4f => "associated", 0x50 => "data_location",
    0x51 => "byte_stride", 0x52 => "entry_pc", 0x53 => "use_UTF8", 0x54 => "extension", 0x55 => "ranges",
    0x56 => "trampoline", 0x57 => "call_column", 0x58 => "call_file", 0x59 => "call_line",
    0x5a => "description", 0x5b => "binary_scale", 0x5c => "decimal_scale", 0x5d => "small",
    0x5e => "decimal_sign", 0x5f => "digit_count", 0x60 => "picture_string", 0x61 => "mutable",
    0x62 => "threads_scaled", 0x63 => "explicit", 0x64 => "object_pointer", 0x65 => "endianity",
    0x66 => "elemental", 0x67 => "pure", 0x68 => "recursive", 0x69 => "signature",
    0x6a => "main_subprogram", 0x6b => "data_bit_offset", 0x6c => "const_expr", 0x6d => "enum_class",
    0x6e => "linkage_name", 0x6f => "string_length_bit_size", 0x70 => "string_length_byte_size",
    0x71 => "rank", 0x72 => "str_offsets_base", 0x73 => "addr_base", 0x74 => "rnglists_base",
    0x76 => "dwo_name", 0x77 => "reference", 0x78 => "rvalue_reference", 0x79 => "macros",
    0x7a => "call_all_calls", 0x7b => "call_all_source_calls", 0x7c => "call_all_tail_calls",
    0x7d => "call_return_pc", 0x7e => "call_value", 0x7f => "call_origin", 0x80 => "call_parameter",
    0x81 => "call_pc", 0x82 => "call_tail_call", 0x83 => "call_target", 0x84 => "call_target_clobbered",
    0x85 => "call_data_location", 0x86 => "call_data_value", 0x87 => "noreturn", 0x88 => "alignment",
    0x89 => "export_symbols", 0x8a => "deleted", 0x8b => "defaulted", 0x8c => "loclists_base",
    0x2007 => "MIPS_linkage_name", 0x2107 => "GNU_vector", 0x2111 => "GNU_call_site_value",
    0x2115 => "GNU_tail_call", 0x2116 => "GNU_all_tail_call_sites", 0x2117 => "GNU_all_call_sites",
    0x2119 => "GNU_macros", 0x211a => "GNU_deleted", 0x2133 => "GNU_addr_base",
    0x2134 => "GNU_pubnames", 0x2135 => "GNU_pubtypes", 0x2137 => "GNU_locviews", 0x2138 => "GNU_entry_view",
)

# Form classes.
const _STRING_FORMS = (0x08,)
const _INDIRECT_STRING_FORMS = (0x0e, 0x1a, 0x1d, 0x1f, 0x25, 0x26, 0x27, 0x28, 0x1f02, 0x1f21)
const _REF_FORMS = (0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x1c, 0x20, 0x24, 0x1f20)
const _FLAG_FORMS = (0x0c, 0x19)
const _UDATA_FORMS = (0x05, 0x06, 0x07, 0x0b, 0x0f, 0x21)
const _SDATA_FORMS = (0x0d,)
const _BLOCK_FORMS = (0x03, 0x04, 0x09, 0x0a, 0x18)
const _ADDR_FORMS = (0x01, 0x1b, 0x29, 0x2a, 0x2b, 0x2c, 0x1f01)
const _OFFSET_FORMS = (0x17, 0x22, 0x23)

# ── location expressions ────────────────────────────────────────────────────

function _uleb(b::Vector{UInt8}, i::Int)
    v = UInt64(0); shift = 0
    while i <= length(b)
        x = b[i]; i += 1
        v |= UInt64(x & 0x7f) << shift
        shift += 7
        x & 0x80 == 0 && break
    end
    return v, i
end

function _sleb(b::Vector{UInt8}, i::Int)
    v = Int64(0); shift = 0; x = 0x00
    while i <= length(b)
        x = b[i]; i += 1
        v |= Int64(x & 0x7f) << shift
        shift += 7
        x & 0x80 == 0 && break
    end
    (shift < 64 && x & 0x40 != 0) && (v |= -(Int64(1) << shift))
    return v, i
end

_le(b::Vector{UInt8}, i::Int, n::Int) = (v = UInt64(0); for k in 0:n-1; v |= UInt64(b[i+k]) << (8k); end; v)

"`b` as readelf prints an expression: `DW_OP_dup; DW_OP_deref; DW_OP_constu: 24`."
function expression(b::Vector{UInt8})::String
    parts = String[]
    i = 1
    n = length(b)
    while i <= n
        op = b[i]; i += 1
        if op == 0x03 && i + 7 <= n
            push!(parts, "DW_OP_addr: " * string(_le(b, i, 8); base = 16)); i += 8
        elseif op == 0x06; push!(parts, "DW_OP_deref")
        elseif op in (0x08, 0x0a, 0x0c, 0x0e)
            w = (1, 2, 4, 8)[(Int(op) - 0x08) ÷ 2 + 1]
            i + w - 1 <= n || break
            push!(parts, "DW_OP_const$(w)u: " * string(_le(b, i, w))); i += w
        elseif op in (0x09, 0x0b, 0x0d, 0x0f)
            w = (1, 2, 4, 8)[(Int(op) - 0x09) ÷ 2 + 1]
            i + w - 1 <= n || break
            u = _le(b, i, w); i += w
            s = w == 8 ? reinterpret(Int64, u) : Int64(u) - (u >> (8w - 1) == 1 ? Int64(1) << (8w) : 0)
            push!(parts, "DW_OP_const$(w)s: " * string(s))
        elseif op == 0x10; (v, i) = _uleb(b, i); push!(parts, "DW_OP_constu: $v")
        elseif op == 0x11; (v, i) = _sleb(b, i); push!(parts, "DW_OP_consts: $v")
        elseif op == 0x12; push!(parts, "DW_OP_dup")
        elseif op == 0x13; push!(parts, "DW_OP_drop")
        elseif op == 0x1a; push!(parts, "DW_OP_and")
        elseif op == 0x1c; push!(parts, "DW_OP_minus")
        elseif op == 0x1e; push!(parts, "DW_OP_mul")
        elseif op == 0x22; push!(parts, "DW_OP_plus")
        elseif op == 0x23; (v, i) = _uleb(b, i); push!(parts, "DW_OP_plus_uconst: $v")
        elseif 0x30 <= op <= 0x4f; push!(parts, "DW_OP_lit$(Int(op) - 0x30)")
        elseif 0x50 <= op <= 0x6f; r = Int(op) - 0x50; push!(parts, "DW_OP_reg$r (r$r)")
        elseif 0x70 <= op <= 0x8f; r = Int(op) - 0x70; (v, i) = _sleb(b, i); push!(parts, "DW_OP_breg$r (r$r): $v")
        elseif op == 0x91; (v, i) = _sleb(b, i); push!(parts, "DW_OP_fbreg: $v")
        elseif op == 0x93; (v, i) = _uleb(b, i); push!(parts, "DW_OP_piece: $v")
        elseif op == 0x94 && i <= n; push!(parts, "DW_OP_deref_size: $(b[i])"); i += 1
        elseif op == 0x96; push!(parts, "DW_OP_nop")
        elseif op == 0x9c; push!(parts, "DW_OP_call_frame_cfa")
        elseif op == 0xa1; (v, i) = _uleb(b, i); push!(parts, "DW_OP_addrx <0x" * string(v; base = 16) * ">")
        elseif op == 0xa2; (v, i) = _uleb(b, i); push!(parts, "DW_OP_constx <0x" * string(v; base = 16) * ">")
        elseif op == 0x9f; push!(parts, "DW_OP_stack_value")
        elseif op == 0xa3
            (len, i) = _uleb(b, i)
            j = min(n, i + Int(len) - 1)
            push!(parts, "DW_OP_entry_value: (" * expression(b[i:j]) * ")"); i = j + 1
        else
            # An operation this printer does not decode: its operands are
            # unknown, so nothing after it can be read.
            push!(parts, "DW_OP_unknown: 0x" * string(op; base = 16)); break
        end
    end
    return join(parts, "; ")
end

# ── attribute values ────────────────────────────────────────────────────────

function _attr_callback(attr::Ptr{Attr}, arg::Ptr{Cvoid})::Cint
    push!(unsafe_pointer_to_objref(arg)::Vector{Attr}, unsafe_load(attr))
    return Cint(0)
end

function _value(io::IO, a::Attr)
    form = Int(a.form)
    ra = Ref(a)
    if form in _STRING_FORMS || form in _INDIRECT_STRING_FORMS
        p = ccall((:dwarf_formstring, libdw), Cstring, (Ref{Attr},), ra)
        form in _INDIRECT_STRING_FORMS && print(io, "(indirect string, offset: 0): ")
        p == C_NULL || print(io, unsafe_string(p))
    elseif form in _REF_FORMS
        target = Ref(Die())
        ok = ccall((:dwarf_formref_die, libdw), Ptr{Die}, (Ref{Attr}, Ref{Die}), ra, target)
        if ok == C_NULL
            print(io, "<unresolved>")
        else
            off = ccall((:dwarf_dieoffset, libdw), UInt64, (Ref{Die},), target)
            print(io, "<0x", string(off; base = 16), ">")
        end
    elseif form in _FLAG_FORMS
        f = Ref{Bool}(false)
        ccall((:dwarf_formflag, libdw), Cint, (Ref{Attr}, Ref{Bool}), ra, f)
        print(io, f[] ? "1" : "0")
    elseif form in _SDATA_FORMS
        v = Ref{Int64}(0)
        ccall((:dwarf_formsdata, libdw), Cint, (Ref{Attr}, Ref{Int64}), ra, v)
        print(io, v[])
    elseif form in _UDATA_FORMS
        v = Ref{UInt64}(0)
        ccall((:dwarf_formudata, libdw), Cint, (Ref{Attr}, Ref{UInt64}), ra, v)
        print(io, v[])
    elseif form in _BLOCK_FORMS
        blk = Ref(Block(0, C_NULL))
        if ccall((:dwarf_formblock, libdw), Cint, (Ref{Attr}, Ref{Block}), ra, blk) == 0
            bytes = unsafe_wrap(Array, blk[].data, Int(blk[].length)) |> copy
            print(io, length(bytes), " byte block: ")
            for x in bytes
                print(io, string(x; base = 16), " ")
            end
            print(io, "\t(", expression(bytes), ")")
        end
    elseif form in _ADDR_FORMS
        v = Ref{UInt64}(0)
        ccall((:dwarf_formaddr, libdw), Cint, (Ref{Attr}, Ref{UInt64}), ra, v)
        print(io, "0x", string(v[]; base = 16))
    elseif form in _OFFSET_FORMS
        v = Ref{UInt64}(0)
        ccall((:dwarf_formudata, libdw), Cint, (Ref{Attr}, Ref{UInt64}), ra, v)
        print(io, "0x", string(v[]; base = 16))
    elseif form == 0x1e
        print(io, "<data16>")
    else
        print(io, "<form 0x", string(form; base = 16), ">")
    end
end

function _die(io::IO, die::Base.RefValue{Die}, depth::Int, attrs::Vector{Attr}, cb::Ptr{Cvoid})
    off = ccall((:dwarf_dieoffset, libdw), UInt64, (Ref{Die},), die)
    tag = Int(ccall((:dwarf_tag, libdw), Cint, (Ref{Die},), die))
    hex = string(off; base = 16)
    name = get(TAGS, tag, nothing)
    print(io, " <", depth, "><", hex, ">: Abbrev Number: 1 (")
    name === nothing ? print(io, "Unknown TAG value: ", string(tag; base = 16)) : print(io, "DW_TAG_", name)
    print(io, ")\n")
    empty!(attrs)
    GC.@preserve attrs ccall((:dwarf_getattrs, libdw), Cptrdiff_t,
                             (Ref{Die}, Ptr{Cvoid}, Ptr{Cvoid}, Cptrdiff_t),
                             die, cb, pointer_from_objref(attrs), 0)
    for a in attrs
        aname = get(ATTRS, Int(a.code), nothing)
        label = aname === nothing ? "Unknown AT value: " * string(a.code; base = 16) : "DW_AT_" * aname
        # readelf pads the name to 18 columns and writes ": " after it.
        print(io, "    <", hex, ">   ", rpad(label, 18), ": ")
        _value(io, a)
        print(io, "\n")
    end
    return nothing
end

"""
    info_dump(io, path)

Write the `.debug_info` of the ELF file at `path` to `io`, as
`readelf --debug-dump=info` does for everything `parse_dwarf_dump` reads.
"""
function info_dump(io::IO, path::AbstractString)
    with_dwarf(path) do dbg
        cb = @cfunction(_attr_callback, Cint, (Ptr{Attr}, Ptr{Cvoid}))
        attrs = Attr[]
        print(io, "Contents of the .debug_info section:\n\n")
        off = UInt64(0)
        next = Ref{UInt64}(0); hsize = Ref{Csize_t}(0); version = Ref{UInt16}(0)
        abbrev = Ref{UInt64}(0); asize = Ref{UInt8}(0); osize = Ref{UInt8}(0)
        while true
            rc = ccall((:dwarf_next_unit, libdw), Cint,
                       (Ptr{Cvoid}, UInt64, Ref{UInt64}, Ref{Csize_t}, Ref{UInt16}, Ref{UInt64},
                        Ref{UInt8}, Ref{UInt8}, Ptr{UInt64}, Ptr{UInt64}),
                       dbg, off, next, hsize, version, abbrev, asize, osize, C_NULL, C_NULL)
            rc == 0 || (rc < 0 && error("libdw: reading a unit of $path: $(_errmsg())"); break)
            print(io, "  Compilation Unit @ offset 0x", string(off; base = 16), ":\n",
                  "   Version:       ", version[], "\n",
                  "   Pointer Size:  ", asize[], "\n")
            root = Ref(Die())
            ccall((:dwarf_offdie, libdw), Ptr{Die}, (Ptr{Cvoid}, UInt64, Ref{Die}), dbg, off + hsize[], root) == C_NULL &&
                error("libdw: no unit DIE at 0x$(string(off + hsize[]; base = 16)) of $path: $(_errmsg())")
            _walk(io, root, attrs, cb)
            off = next[]
        end
    end
    return nothing
end

info_dump(path::AbstractString)::String = (io = IOBuffer(); info_dump(io, path); String(take!(io)))

# Depth first, without recursion: a DIE tree is as deep as the source nests.
function _walk(io::IO, root::Base.RefValue{Die}, attrs::Vector{Attr}, cb::Ptr{Cvoid})
    stack = Base.RefValue{Die}[root]       # the open DIE at each depth
    _die(io, root, 0, attrs, cb)
    descend = true
    while !isempty(stack)
        depth = length(stack) - 1
        cur = stack[end]
        nxt = Ref(Die())
        if descend && ccall((:dwarf_child, libdw), Cint, (Ref{Die}, Ref{Die}), cur, nxt) == 0
            push!(stack, nxt)
            _die(io, nxt, depth + 1, attrs, cb)
            continue
        end
        # No (more) children here: its sibling, or close the list it is in.
        if depth > 0 && ccall((:dwarf_siblingof, libdw), Cint, (Ref{Die}, Ref{Die}), cur, nxt) == 0
            stack[end] = nxt
            _die(io, nxt, depth, attrs, cb)
            descend = true
        else
            pop!(stack)
            depth > 0 && print(io, " <", depth, "><0>: Abbrev Number: 0\n")
            descend = false
        end
    end
    return nothing
end

"""
    file_tables(path) -> Vector{Dict{Int,String}}

One `file index => path` map per compile unit, in unit order: what
`DW_AT_decl_file` indexes.
"""
function file_tables(path::AbstractString)::Vector{Dict{Int, String}}
    tables = Dict{Int, String}[]
    with_dwarf(path) do dbg
        off = UInt64(0)
        next = Ref{UInt64}(0); hsize = Ref{Csize_t}(0)
        while ccall((:dwarf_next_unit, libdw), Cint,
                    (Ptr{Cvoid}, UInt64, Ref{UInt64}, Ref{Csize_t}, Ptr{UInt16}, Ptr{UInt64},
                     Ptr{UInt8}, Ptr{UInt8}, Ptr{UInt64}, Ptr{UInt64}),
                    dbg, off, next, hsize, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL) == 0
            table = Dict{Int, String}()
            root = Ref(Die())
            if ccall((:dwarf_offdie, libdw), Ptr{Die}, (Ptr{Cvoid}, UInt64, Ref{Die}), dbg, off + hsize[], root) != C_NULL
                files = Ref{Ptr{Cvoid}}(C_NULL); n = Ref{Csize_t}(0)
                if ccall((:dwarf_getsrcfiles, libdw), Cint, (Ref{Die}, Ref{Ptr{Cvoid}}, Ref{Csize_t}), root, files, n) == 0
                    # libdw leaves a name relative when its directory entry is
                    # the unit's own; the unit's DW_AT_comp_dir is what it is
                    # relative to.
                    comp_dir = _comp_dir(root)
                    for i in 0:Int(n[])-1
                        p = ccall((:dwarf_filesrc, libdw), Cstring, (Ptr{Cvoid}, Csize_t, Ptr{Cvoid}, Ptr{Cvoid}),
                                  files[], i, C_NULL, C_NULL)
                        p == C_NULL && continue
                        name = unsafe_string(p)
                        isempty(name) && continue
                        table[i] = (startswith(name, "/") || isempty(comp_dir)) ? name : joinpath(comp_dir, name)
                    end
                end
            end
            push!(tables, table)
            off = next[]
        end
    end
    return tables
end

# DW_AT_comp_dir of a unit DIE, or "".
function _comp_dir(root::Base.RefValue{Die})::String
    attr = Ref(Attr(0, 0, C_NULL, C_NULL))
    ccall((:dwarf_attr, libdw), Ptr{Attr}, (Ref{Die}, Cuint, Ref{Attr}), root, 0x1b, attr) == C_NULL && return ""
    p = ccall((:dwarf_formstring, libdw), Cstring, (Ref{Attr},), attr)
    return p == C_NULL ? "" : unsafe_string(p)
end

function with_dwarf(f::Function, path::AbstractString)
    isfile(path) || error("libdw: no such file: $path")
    open(path, "r") do file
        dbg = ccall((:dwarf_begin, libdw), Ptr{Cvoid}, (Cint, Cint), fd(file), 0)
        dbg == C_NULL && error("libdw: cannot read DWARF from $path: $(_errmsg())")
        try
            return f(dbg)
        finally
            ccall((:dwarf_end, libdw), Cint, (Ptr{Cvoid},), dbg)
        end
    end
end

end # module LibDW
