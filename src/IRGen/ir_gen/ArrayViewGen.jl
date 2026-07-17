"""
Strided array-view accessor thunk generation — the DWARF-driven producer for
`!jlcs.array_view` and the `jlcs.load_array_element`/`store_array_element` ops.

For every struct member that is a fixed-size array of a primitive element type
(`float vals[4]`, `int32_t ids[16]`, …), emits a get/set thunk pair that
materializes an ArrayView descriptor over the member in place (zero-copy: the
data pointer aims into the caller's struct) and accesses elements through the
strided ops. Rank 1 today; the descriptor layout already carries dims/strides/
rank for the multi-dimensional generalization.

ArrayView runtime layout (matches the op lowering, which reads data at +0 and
strides at +16; strides are in ELEMENTS):

    struct ArrayView { T* data; int64_t* dims; int64_t* strides; int64_t rank; }

Thunk calling convention matches FunctionGen (`%args_ptr` array of pointers,
`llvm.emit_c_interface`):
    <base>_get_thunk: args = [obj_ptr, index_i64]            -> element
    <base>_set_thunk: args = [obj_ptr, index_i64, value_ptr] -> ()

The array ops have no custom assembly, so they are emitted in generic form —
the same form `test_jlcs_invariants.jl` proves parses and lowers.
"""
module ArrayViewGen

export generate_array_view_thunks

# Primitive C element types the producer handles, mapped to MLIR types
const _AV_ELEM_MLIR = Dict{String,String}(
    "float" => "f32", "double" => "f64",
    "int" => "i32", "unsigned int" => "i32", "int32_t" => "i32", "uint32_t" => "i32",
    "long" => "i64", "unsigned long" => "i64", "int64_t" => "i64", "uint64_t" => "i64",
    "long long" => "i64", "unsigned long long" => "i64", "size_t" => "i64",
    "short" => "i16", "unsigned short" => "i16", "int16_t" => "i16", "uint16_t" => "i16",
    "char" => "i8", "signed char" => "i8", "unsigned char" => "i8",
    "int8_t" => "i8", "uint8_t" => "i8",
)

function _parse_int_maybe_hex(raw)::Int
    s = raw isa String ? raw : string(raw)
    try
        (startswith(s, "0x") || startswith(s, "0X")) ? parse(Int, s[3:end], base=16) : parse(Int, s)
    catch
        -1
    end
end

"""
Shared thunk prologue: load `obj` (args[0]) and `index` (args[1], i64→index),
then build the ArrayView descriptor on the stack with the data pointer aimed
at the array member (byte offset `member_off`), dims=[n], strides=[1], rank=1.
"""
function _emit_view_prologue(member_off::Int, n::Int)::String
    io = IOBuffer()
    println(io, "  %one = llvm.mlir.constant(1 : i64) : i64")
    println(io, "  %i0 = arith.constant 0 : i64")
    println(io, "  %oslot = llvm.getelementptr %args_ptr[%i0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr")
    println(io, "  %optr = llvm.load %oslot : !llvm.ptr -> !llvm.ptr")
    println(io, "  %obj = llvm.load %optr : !llvm.ptr -> !llvm.ptr")
    println(io, "  %i1 = arith.constant 1 : i64")
    println(io, "  %islot = llvm.getelementptr %args_ptr[%i1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr")
    println(io, "  %iptr = llvm.load %islot : !llvm.ptr -> !llvm.ptr")
    println(io, "  %idx64 = llvm.load %iptr : !llvm.ptr -> i64")
    println(io, "  %index = arith.index_cast %idx64 : i64 to index")
    # Data pointer: the array member inside the caller's struct (zero-copy)
    println(io, "  %data = llvm.getelementptr %obj[$(member_off)] : (!llvm.ptr) -> !llvm.ptr, i8")
    # dims = [n], strides = [1] (in elements)
    println(io, "  %dims = llvm.alloca %one x i64 : (i64) -> !llvm.ptr")
    println(io, "  %dimval = llvm.mlir.constant($(n) : i64) : i64")
    println(io, "  llvm.store %dimval, %dims : i64, !llvm.ptr")
    println(io, "  %strides = llvm.alloca %one x i64 : (i64) -> !llvm.ptr")
    println(io, "  llvm.store %one, %strides : i64, !llvm.ptr")
    # ArrayView descriptor: data@0, dims@8, strides@16, rank@24
    println(io, "  %view = llvm.alloca %one x !llvm.struct<(ptr, ptr, ptr, i64)> : (i64) -> !llvm.ptr")
    println(io, "  llvm.store %data, %view : !llvm.ptr, !llvm.ptr")
    println(io, "  %dimsfield = llvm.getelementptr %view[8] : (!llvm.ptr) -> !llvm.ptr, i8")
    println(io, "  llvm.store %dims, %dimsfield : !llvm.ptr, !llvm.ptr")
    println(io, "  %stridesfield = llvm.getelementptr %view[16] : (!llvm.ptr) -> !llvm.ptr, i8")
    println(io, "  llvm.store %strides, %stridesfield : !llvm.ptr, !llvm.ptr")
    println(io, "  %rankfield = llvm.getelementptr %view[24] : (!llvm.ptr) -> !llvm.ptr, i8")
    println(io, "  llvm.store %one, %rankfield : i64, !llvm.ptr")
    return String(take!(io))
end

"""
    generate_array_view_thunks(structs) -> String

Emit get/set accessor thunk pairs for every rank-1 fixed-size primitive array
member found in the DWARF struct definitions. Returns "" when nothing applies.
"""
function generate_array_view_thunks(structs)::String
    io = IOBuffer()
    emitted = Set{String}()

    for (sname_raw, info) in structs
        sname = String(sname_raw)
        startswith(sname, "__enum__") && continue
        info isa AbstractDict || continue
        for m in get(info, "members", [])
            ct = strip(String(get(m, "c_type", "")))
            am = match(r"^([A-Za-z_][A-Za-z0-9_ ]*?)\s*\[(\d+)\]$", ct)
            am === nothing && continue
            elem_c = String(strip(am.captures[1]))
            haskey(_AV_ELEM_MLIR, elem_c) || continue
            n = parse(Int, am.captures[2])
            n > 0 || continue
            off = _parse_int_maybe_hex(get(m, "offset", "-1"))
            off >= 0 || continue
            mname = String(get(m, "name", ""))
            isempty(mname) && continue

            safe_s = replace(sname, r"[^A-Za-z0-9_]" => "_")
            safe_m = replace(mname, r"[^A-Za-z0-9_]" => "_")
            base = "jlcs_av_$(safe_s)_$(safe_m)"
            # Distinct DWARF keys can sanitize identically — one pair per name
            base in emitted && continue
            push!(emitted, base)

            elt = _AV_ELEM_MLIR[elem_c]
            if length(emitted) == 1
                println(io, "// Strided array-view accessor thunks (fixed-size array members)")
                println(io, "")
            end

            println(io, "func.func @$(base)_get_thunk(%args_ptr: !llvm.ptr) -> $(elt) attributes { llvm.emit_c_interface } {")
            print(io, _emit_view_prologue(off, n))
            println(io, "  %elem = \"jlcs.load_array_element\"(%view, %index) : (!llvm.ptr, index) -> $(elt)")
            println(io, "  return %elem : $(elt)")
            println(io, "}")
            println(io, "")

            println(io, "func.func @$(base)_set_thunk(%args_ptr: !llvm.ptr) attributes { llvm.emit_c_interface } {")
            print(io, _emit_view_prologue(off, n))
            println(io, "  %vidx = arith.constant 2 : i64")
            println(io, "  %vslot = llvm.getelementptr %args_ptr[%vidx] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr")
            println(io, "  %vptr = llvm.load %vslot : !llvm.ptr -> !llvm.ptr")
            println(io, "  %value = llvm.load %vptr : !llvm.ptr -> $(elt)")
            println(io, "  \"jlcs.store_array_element\"(%value, %view, %index) : ($(elt), !llvm.ptr, index) -> ()")
            println(io, "  return")
            println(io, "}")
            println(io, "")
        end
    end

    return isempty(emitted) ? "" : String(take!(io))
end

end # module ArrayViewGen
