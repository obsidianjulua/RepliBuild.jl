# The thunk a Rust function gets (`RustThunks`), as text, without a toolchain:
# generated from the vendored builds of test/fixtures/rust/lean.rs that
# test_rust_abi.jl reads. What a thunk states is the `define` line and nothing
# else, so each check here pins the one line that is the call, or the one
# store that is the result, not a whole thunk.

using Test
using RepliBuild

isdefined(@__MODULE__, :rust_extract) || include(joinpath(@__DIR__, "support", "rust_fixture.jl"))

const RT = RepliBuild.RustThunks

# The lines of one function's thunk, and how many slots it reads.
function thunk_lines(x, tail)
    row = rust_row(x, tail)
    t = RT.thunk(row, x.types)
    t isa String && error("no thunk for $tail: $t")
    return (lines = String[String(strip(l)) for l in split(t[1], '\n') if !isempty(strip(l))], slots = t[2], row = row)
end
call_line(t) = only(l for l in t.lines if occursin("jlcs.native_call", l))

@testset "Rust thunks: LLVM types and attributes as MLIR" begin
    @test RT.mlir_type("i32") == "i32" && RT.mlir_type("i1") == "i1" && RT.mlir_type("i24") == "i24"
    @test RT.mlir_type("i128") == "i128" && RT.mlir_type("double") == "f64" && RT.mlir_type("float") == "f32"
    @test RT.mlir_type("ptr") == "!llvm.ptr"
    @test RT.mlir_type("{ i32, double }") == "!llvm.struct<(i32, f64)>"
    @test RT.mlir_type("{ ptr, i64 }") == "!llvm.struct<(!llvm.ptr, i64)>"
    # What a thunk does not carry is said to be so, not approximated.
    @test RT.mlir_type("<4 x float>") === nothing
    @test RT.mlir_type("[2 x i64]") === nothing
    @test RT.mlir_type("%\"core::fmt::Formatter\"") === nothing
    @test RT.mlir_type("{ i32, <2 x double> }") === nothing

    # Only what changes how an argument is passed.
    @test RT._arg_attrs(String[]) == "{}"
    @test RT._arg_attrs(["noundef", "nonnull", "readonly", "captures(none)", "dereferenceable(16)", "align 8"]) == "{}"
    @test RT._arg_attrs(["noundef", "zeroext"]) == "{llvm.zeroext}"
    @test RT._arg_attrs(["signext"]) == "{llvm.signext}"
    @test RT._arg_attrs(["dead_on_unwind", "noalias", "sret([24 x i8])", "align 8", "dereferenceable(24)"]) ==
          "{llvm.sret = !llvm.array<24 x i8>, llvm.align = 8 : i64}"
    @test RT._arg_attrs(["byval([32 x i8])", "align 16"]) == "{llvm.byval = !llvm.array<32 x i8>, llvm.align = 16 : i64}"
    # A hidden pointer described by a type this does not size, or a convention it does not carry.
    @test RT._arg_attrs(["sret(%\"Thing\")", "align 8"]) === nothing
    @test RT._arg_attrs(["inalloca(i32)"]) === nothing

    @test RT.thunk_name("_RNvCsX_1m3add") == "_RNvCsX_1m3add_rbthunk"
    @test RT._sym("c_add") == "@c_add" && RT._sym("odd name") == "@\"odd name\""
end

@testset "Rust thunks: the call is the define line's" begin
    for opt in (0, 3)
        x = rust_extract(opt)
        # Scalars: one slot each, loaded and passed as they are. The function
        # itself is the slot after them: its address, called through. The
        # thunk names no function.
        t = thunk_lines(x, "4lean3add")
        @test replace(call_line(t), "%r = " => "") == "jlcs.native_call %fn, %v1, %v2 : (!llvm.ptr, i32, i32) -> i32"
        @test t.slots == 3
        @test "%s1 = llvm.getelementptr %args[0] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr" in t.lines
        @test "%s2 = llvm.getelementptr %args[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr" in t.lines
        @test "%sf = llvm.getelementptr %args[2] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr" in t.lines
        @test "%fn = llvm.load %sf : !llvm.ptr -> !llvm.ptr" in t.lines
        @test !any(l -> occursin("callee", l) || occursin(t.row["symbol"] * " ", l), t.lines)
        @test "llvm.store %r, %ret {alignment = 1 : i64} : i32, !llvm.ptr" in t.lines
        @test first(t.lines) == "func.func @$(t.row["symbol"])_rbthunk(%args: !llvm.ptr, %ret: !llvm.ptr) {"

        # Two scalars come back as a first-class pair: that type, and no C rule on it.
        t = thunk_lines(x, "4lean5maybe")
        @test endswith(call_line(t), ": (!llvm.ptr, i32) -> !llvm.struct<(i32, i32)>")
        @test !occursin("arg_attrs", call_line(t)) && !occursin("sysv", call_line(t))
        # ...and each half is stored where DWARF puts it in the type (tag at 0, payload at 4).
        @test "llvm.store %r1, %ret {alignment = 1 : i64} : i32, !llvm.ptr" in t.lines
        @test "%r2q = llvm.getelementptr %ret[4] : (!llvm.ptr) -> !llvm.ptr, i8" in t.lines
        @test "llvm.store %r2, %r2q {alignment = 1 : i64} : i32, !llvm.ptr" in t.lines
        t = thunk_lines(x, "4lean4pair")
        @test endswith(call_line(t), ": (!llvm.ptr, i32, f64) -> !llvm.struct<(i32, f64)>")
        @test "%r2q = llvm.getelementptr %ret[8] : (!llvm.ptr) -> !llvm.ptr, i8" in t.lines
        t = thunk_lines(x, "4lean5hello")
        @test endswith(call_line(t), ": (!llvm.ptr) -> !llvm.struct<(!llvm.ptr, i64)>") && t.slots == 1

        # A hidden return pointer IS `%ret`, with its attribute, and takes no slot.
        # One dictionary of attributes per argument of the call; the function's address is not one.
        t = thunk_lines(x, "4lean8rec_make")
        @test occursin("jlcs.native_call %fn, %ret, %v2, %v3, %v4, %v5 { arg_attrs = ", call_line(t))
        @test occursin("arg_attrs = [{llvm.sret = !llvm.array<16 x i8>, llvm.align = 8 : i64}, {}, {}, {}, {}]", call_line(t))
        @test endswith(call_line(t), ": (!llvm.ptr, !llvm.ptr, i8, f64, i16, i32) -> ()")
        @test "%s2 = llvm.getelementptr %args[0] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr" in t.lines   # the first slot is the second IR parameter
        @test "%sf = llvm.getelementptr %args[4] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr" in t.lines && t.slots == 5
        @test !any(l -> occursin("llvm.store", l), t.lines)           # Rust writes the result; the thunk stores nothing

        # A record by value is a pointer to the caller's copy: a pointer in a slot, like any pointer.
        t = thunk_lines(x, "4lean7rec_sum")
        @test "%v1 = llvm.load %p1 {alignment = 1 : i64} : !llvm.ptr -> !llvm.ptr" in t.lines
        @test occursin(": (!llvm.ptr, !llvm.ptr) -> f64", call_line(t))

        # A bool is a byte in storage and one bit, zero-extended, in the call.
        t = thunk_lines(x, "4lean4flag")
        @test "%b1 = llvm.load %p1 {alignment = 1 : i64} : !llvm.ptr -> i8" in t.lines
        @test "%v1 = llvm.trunc %b1 : i8 to i1" in t.lines
        @test occursin("arg_attrs = [{llvm.zeroext}]", call_line(t)) && endswith(call_line(t), ": (!llvm.ptr, i1) -> i1")
        @test "%rz = llvm.zext %r : i1 to i8" in t.lines
        @test "llvm.store %rz, %ret {alignment = 1 : i64} : i8, !llvm.ptr" in t.lines

        # Three bytes are an `i24`, and sixteen an `i128`: no Julia call type says either.
        @test endswith(call_line(thunk_lines(x, "4lean6b3_rot")), ": (!llvm.ptr, i24) -> i24")
        @test endswith(call_line(thunk_lines(x, "4lean4wide")), ": (!llvm.ptr, i128) -> i128")
        # A zero-sized parameter has no IR parameter, so no slot; nothing comes back.
        t = thunk_lines(x, "4lean1z")
        @test endswith(call_line(t), ": (!llvm.ptr, i32) -> ()") && startswith(call_line(t), "jlcs.native_call %fn, %v1 ") && t.slots == 2
        # A fat pointer is two IR parameters, so two slots.
        t = thunk_lines(x, "4lean9slice_sum")
        @test occursin("jlcs.native_call %fn, %v1, %v2 ", call_line(t)) && endswith(call_line(t), ": (!llvm.ptr, !llvm.ptr, i64) -> f64")
    end
end

@testset "Rust thunks: the module" begin
    for opt in (0, 3)
        x = rust_extract(opt)
        rows = [x.rows[s] for s in x.symbols]
        (ir, made, skipped) = RT.thunk_module(rows, x.types)
        @test length(made) == 29 && length(skipped) == 2
        @test made == [s for s in x.symbols if x.rows[s]["refusal"] === nothing]      # the rows' order
        # A refused function has no thunk, and its reason is the refusal's.
        located = only(s for s in x.symbols if endswith(s, "4lean7located"))
        @test occursin("track_caller", skipped[located]) && occursin("refused", skipped["cpoint_norm"])
        @test !occursin(located * "_rbthunk", ir)
        # One native_call per thunk, and none of the calls that apply a C rule.
        @test count("jlcs.native_call", ir) == 29
        @test count(r"func\.func @\S+_rbthunk\(%args: !llvm\.ptr, %ret: !llvm\.ptr\) \{", ir) == 29
        @test !occursin("jlcs.ffe_call", ir) && !occursin("jlcs.try_call", ir)
        # The module names no function of the library: nothing is declared,
        # no call has a `callee`, and no symbol of the crate is in the text
        # except inside a thunk's own name. So a loader or a JIT has nothing
        # to resolve, and the same text serves a function the library exports
        # and one it has no symbol for.
        @test !occursin("func.func private", ir) && !occursin("callee", ir)
        @test all(s -> count(s, ir) == 1, made)
        @test isempty(RepliBuild.JLCSIRGenerator.undeclared_callees(ir))
        @test startswith(ir, "module {\n") && endswith(ir, "}\n")
        # `only` narrows it: what the library cannot bind gets no thunk.
        (_, some, _) = RT.thunk_module(rows, x.types; only = Set(x.symbols[1:3]))
        @test some == [s for s in x.symbols[1:3] if x.rows[s]["refusal"] === nothing]
        # The same rows give the same text.
        @test RT.thunk_module(rows, x.types)[1] == ir
    end
    # Optimized, `c_add` is an alias of `add`: it has a thunk of its own name
    # and of the same call, and the caller hands it `c_add`'s own address.
    x = rust_extract(3)
    t = RT.thunk(x.rows["c_add"], x.types)
    @test occursin("func.func @c_add_rbthunk(", t[1]) && t[2] == 3
    @test occursin("jlcs.native_call %fn, %v1, %v2 : (!llvm.ptr, i32, i32) -> i32", t[1])
    # An ahead-of-time thunks library is named after the library it is for.
    @test RT.aot_library_name("/a/b/librb_fix.so") == "librb_fix_thunks.so"
    @test RT.aot_library_name("rb_fix.dll") == "rb_fix_thunks.dll"
end

@testset "Rust thunks: what a thunk does not carry is a reason" begin
    x = rust_extract(0)
    row = deepcopy(rust_row(x, "4lean3add"))
    row["ir"]["parameters"][1]["type"] = "<4 x float>"
    @test occursin("<4 x float>", RT.thunk(row, x.types))
    row = deepcopy(rust_row(x, "4lean3add"))
    row["ir"]["return"]["type"] = "[2 x i64]"
    @test occursin("[2 x i64]", RT.thunk(row, x.types))
    row = deepcopy(rust_row(x, "4lean8rec_make"))
    row["ir"]["parameters"][1]["attrs"] = ["sret(%\"lean::Rec\")", "align 8"]
    @test occursin("attribute a thunk does not", RT.thunk(row, x.types))
    row = deepcopy(rust_row(x, "4lean5maybe"))
    row["return"]["type"] = "no::such::Type"
    @test occursin("two places", RT.thunk(row, x.types))
    row = deepcopy(rust_row(x, "4lean3add"))
    row["refusal"] = "for the test"
    @test RT.thunk(row, x.types) == "the function is refused: for the test"
end

# The thunks library is linked with no C runtime, so nothing defines the entry
# point a DLL must name (MinGW's lld asks for `DllMainCRTStartup` and has no
# switch for a library without one): on Windows the link failed, and every
# wrapper fell back to the JIT (2026-11-28, the first run on a Windows box).
# The link there also compiles an entry that does nothing.
@testset "the thunks library's link: no C runtime, and a DLL's own entry point" begin
    elf = RT._aot_link_args("out/libx_thunks.so", "out/rust_thunks.o")
    @test elf[1:6] == ["-shared", "-fPIC", "-nostdlib", "-o", "out/libx_thunks.so", "out/rust_thunks.o"]
    @test !any(a -> endswith(a, ".c"), elf)
    pe = RT._aot_link_args("out/libx_thunks.dll", "out/rust_thunks.o"; entry = "out/rust_thunks_entry.c")
    @test pe[1:7] == ["-shared", "-fPIC", "-nostdlib", "-o", "out/libx_thunks.dll", "out/rust_thunks.o",
                      "out/rust_thunks_entry.c"]
    # The same reproducible-link flags after either.
    @test elf[7:end] == pe[8:end] == RepliBuild.Compiler._reproducible_link_flags()
    # The entry: the name the linker asks for, and a load that goes on.
    @test occursin(r"\bint DllMainCRTStartup\(void \*\w+, unsigned long \w+, void \*\w+\)", RT.AOT_DLL_ENTRY)
    @test occursin("return 1;", RT.AOT_DLL_ENTRY)
    @test !occursin("#include", RT.AOT_DLL_ENTRY)        # -nostdlib: it asks for no header's help
end
