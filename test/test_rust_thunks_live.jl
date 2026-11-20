# test_rust_thunks_live.jl — `jlcs.native_call`, and a Rust crate called through it.
#
# The dialect's other call ops apply the host's C record rule to every struct.
# `native_call` applies none: its lowering must emit exactly the call it was
# given, with the parameter attributes it was given. The first half holds the
# op to that on the LLVM IR it lowers to, which needs libJLCS and no Rust.
#
# The second half is the reason the op exists. test/rust_test/crate is built
# with cargo, thunks are generated from its metadata (`RustThunks`), JIT'd,
# and test/rust_test/verify_thunks.jl calls through them in a child process:
# `Option<i32>` back in two registers, a `&str`, a trait object, a three-byte
# integer, a `u128`, a record through its hidden pointer. Unoptimized and
# optimized, since the two builds state the same calls on different lines.

using Test
using RepliBuild

isdefined(@__MODULE__, :TestSupport) ||
    include(joinpath(@__DIR__, "support", "TestSupport.jl"))
using .TestSupport

haskey(ENV, "REPLIBUILD_HOME") || (ENV["REPLIBUILD_HOME"] = mktempdir())

const _RTL_DIR = joinpath(@__DIR__, "rust_test")
const _RTL_TOML = joinpath(_RTL_DIR, "replibuild.toml")
const _RTL_N = RepliBuild.MLIRNative

# The LLVM IR a module lowers to: its `declare` and `call` lines.
function _rtl_lowered(ir::AbstractString)
    ctx = _RTL_N.create_context()
    mod = _RTL_N.parse_module(ctx, String(ir); source_name = "native_call_test")
    _RTL_N.lower_to_llvm(mod) || error(_RTL_N._with_diagnostics("the module did not lower"))
    ll = tempname() * ".ll"
    _RTL_N.emit_llvmir(mod, ll)
    lines = [String(strip(replace(l, r",? ?!dbg !\d+" => ""))) for l in eachline(ll)]
    rm(ll; force = true)
    return (declares = filter(l -> startswith(l, "declare"), lines), calls = filter(l -> occursin(r"\bcall ", l), lines))
end
_rtl_error(f) = try
    f()
    ""
catch e
    sprint(showerror, e)
end

if requires("jlcs.native_call lowering", :libJLCS)

@testset "jlcs.native_call lowers to the call it states" begin
    # No library is loaded: these are the lines of the lowered module.
    ir = """
    module {
      func.func private @pair_ret(i32) -> !llvm.struct<(i32, i32)>
      func.func private @hidden(!llvm.ptr, i8, f64)
      func.func private @narrow(i1, i8, i24, i128) -> i1
      func.func private @copy(!llvm.ptr) -> f64
      func.func @t(%x: i32, %p: !llvm.ptr, %a: i8, %d: f64, %b: i1, %w3: i24, %w16: i128) {
        %r = jlcs.native_call %x { callee = @pair_ret } : (i32) -> !llvm.struct<(i32, i32)>
        jlcs.native_call %p, %a, %d { callee = @hidden, arg_attrs = [{llvm.sret = !llvm.array<16 x i8>, llvm.align = 8 : i64}, {}, {}] } : (!llvm.ptr, i8, f64) -> ()
        %n = jlcs.native_call %b, %a, %w3, %w16 { callee = @narrow, arg_attrs = [{llvm.zeroext}, {llvm.signext}, {}, {}] } : (i1, i8, i24, i128) -> i1
        %c = jlcs.native_call %p { callee = @copy, arg_attrs = [{llvm.byval = !llvm.array<32 x i8>, llvm.align = 16 : i64}] } : (!llvm.ptr) -> f64
        return
      }
    }
    """
    low = _rtl_lowered(ir)
    @test length(low.calls) == 4 && length(low.declares) == 4
    # A first-class pair stays one: no hidden pointer, no packing into an integer.
    @test any(l -> occursin(r"= call \{ i32, i32 \} @pair_ret\(i32 %\d+\)$", l), low.calls)
    @test "declare { i32, i32 } @pair_ret(i32)" in low.declares
    # The attributes are on the call site AND on the declaration, as written.
    @test any(l -> occursin(r"^call void @hidden\(ptr sret\(\[16 x i8\]\) align 8 %\d+, i8 %\d+, double %\d+\)$", l), low.calls)
    @test "declare void @hidden(ptr sret([16 x i8]) align 8, i8, double)" in low.declares
    @test any(l -> occursin(r"= call i1 @narrow\(i1 zeroext %\d+, i8 signext %\d+, i24 %\d+, i128 %\d+\)$", l), low.calls)
    @test "declare i1 @narrow(i1 zeroext, i8 signext, i24, i128)" in low.declares
    @test any(l -> occursin(r"= call double @copy\(ptr byval\(\[32 x i8\]\) align 16 %\d+\)$", l), low.calls)

    # The same function through `ffe_call` is the C rule's call, not the stated
    # one, on every host: that difference is why this op exists.
    ffe = _rtl_lowered("""
    module {
      func.func private @pair_ret(i32) -> !llvm.struct<(i32, i32)>
      func.func @t(%x: i32) -> !llvm.struct<(i32, i32)> {
        %r = jlcs.ffe_call %x { callee = @pair_ret } : (i32) -> !llvm.struct<(i32, i32)>
        return %r : !llvm.struct<(i32, i32)>
      }
    }
    """)
    @test length(ffe.calls) == 1 && !occursin("call { i32, i32 }", ffe.calls[1])

    # An `llvm.func` declaration is retyped and stamped the same way.
    low = _rtl_lowered("""
    module {
      llvm.func @hidden(!llvm.ptr, i32)
      func.func @t(%p: !llvm.ptr, %x: i32) {
        jlcs.native_call %p, %x { callee = @hidden, arg_attrs = [{llvm.sret = !llvm.array<8 x i8>, llvm.align = 4 : i64}, {}] } : (!llvm.ptr, i32) -> ()
        return
      }
    }
    """)
    @test "declare void @hidden(ptr sret([8 x i8]) align 4, i32)" in low.declares
end

@testset "jlcs.native_call refuses what it cannot state" begin
    head = "module {\n  func.func private @f(i32, i32) -> i32\n  func.func @t(%x: i32) {\n"
    tail = "\n    return\n  }\n}\n"
    parse(body) = _rtl_error(() -> _RTL_N.parse_module(_RTL_N.create_context(), head * body * tail;
                                                       source_name = "native_call_bad"))
    # No callee, attributes for another number of operands, a string where a dictionary goes.
    @test occursin("callee", parse("    %r = jlcs.native_call %x, %x : (i32, i32) -> i32"))
    @test occursin("arg_attrs", parse("    %r = jlcs.native_call %x, %x { callee = @f, arg_attrs = [{}] } : (i32, i32) -> i32"))
    @test occursin("arg_attrs", parse("    %r = jlcs.native_call %x, %x { callee = @f, arg_attrs = [\"zeroext\", {}] } : (i32, i32) -> i32"))
    @test occursin("arg_attrs", parse("    %r = jlcs.native_call %x, %x { callee = @f, arg_attrs = {} } : (i32, i32) -> i32"))
    @test parse("    %r = jlcs.native_call %x, %x { callee = @f, arg_attrs = [{}, {}] } : (i32, i32) -> i32") == ""

    # A callee the module does not declare is an error in the lowering. MLIR's
    # own translation would dereference null for it, which is a dead process.
    ctx = _RTL_N.create_context()
    mod = _RTL_N.parse_module(ctx, """
    module {
      func.func @t(%x: i32) {
        %r = jlcs.native_call %x { callee = @nowhere } : (i32) -> i32
        return
      }
    }
    """; source_name = "native_call_undeclared")
    @test _RTL_N.lower_to_llvm(mod) == false
    @test occursin("callee @nowhere is not declared", _RTL_N.take_diagnostics())
    # The Julia side refuses such a module before it gets that far.
    @test RepliBuild.JLCSIRGenerator.undeclared_callees(
        "%r = jlcs.native_call %x { callee = @nowhere } : (i32) -> i32") == ["nowhere"]
end

end # requires

if requires("Rust thunks (live)", :libJLCS, :cargo, :rustc, :nm)

@testset "Rust thunks (live)" begin
    for opt in (0, 3)
        @testset "opt-level $opt" begin
            # test/*/*.toml is untracked, so the project file is written here.
            write(_RTL_TOML, "[project]\nname = \"rb_fix\"\n\n[rust]\nmanifest = \"crate/Cargo.toml\"\nopt_level = $opt\ninstantiate = [\"rb_fix::midpoint::<f64>\", \"rb_fix::Pair::<i32>::new\", \"rb_fix::Pair::<i32>::sum\", \"(rb_fix::into_f64 as fn(f32) -> f64)\"]\n")
            lib = withenv("REPLIBUILD_JULIA_ENV" => "none") do
                redirect_stdout(() -> RepliBuild.build(_RTL_TOML), devnull)
            end
            meta = joinpath(dirname(lib), "compilation_metadata.json")
            md = RepliBuild._read_json(meta)
            @test md["build"]["opt_level"] == opt

            # The module the thunks are JIT'd from: one native_call per callable function.
            callable = [f for f in md["functions"] if f["refusal"] === nothing]
            (ir, made, skipped) = RepliBuild.RustThunks.thunk_module(md["functions"], md["types"])
            @test length(made) == length(callable) >= 37
            @test count("jlcs.native_call", ir) == length(made)
            @test sort!(collect(keys(skipped))) == sort!([f["symbol"] for f in md["functions"] if f["refusal"] !== nothing])
            @test isempty(RepliBuild.JLCSIRGenerator.undeclared_callees(ir))
            # The fixture has every way a result comes back, so the probes below mean something.
            @test Set(f["return"]["pass"] for f in callable) == Set(["ignore", "direct", "pair", "cast", "indirect"])

            r = run_isolated(`$(joinpath(_RTL_DIR, "verify_thunks.jl")) $lib $meta`; timeout = 600)
            v = probe_verdicts(r.output)
            @test r.ok
            @test haskey(v, "PROBE_DONE")
            expected = ["thunks_for_every_callable", "functions_with_no_symbol", "option_i32", "two_field_record", "record_of_two_doubles", "tuple",
                        "str_returned", "method_returning_a_record", "trait_object", "three_byte_integer", "u128",
                        "bool", "fieldless_enum", "char_and_plain_scalars", "one_bit_values", "zero_sized",
                        "record_through_hidden_pointer", "enum_with_data", "string_out_and_in", "slices",
                        "box_and_niche", "wrong_calls_are_errors"]
            for name in expected
                @test (name, startswith(get(v, name, "no PROBE line"), "PASS") ? "PASS" : get(v, name, "no PROBE line")) ==
                      (name, "PASS")
            end
            @test length(v) == length(expected) + 1
        end
    end
end

end # requires
