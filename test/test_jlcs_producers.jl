# test/test_jlcs_producers.jl — DWARF-driven producers for the JLCS RAII and
# array-view ops.
#
# Until now jlcs.scope/ctor_call/dtor_call and load/store_array_element +
# !jlcs.array_view lowered correctly but had NO producer — nothing emitted
# them from DWARF ("Not Yet Built" ledger). This suite pins the producers:
#
#  A. type_info carries the DWARF-resolved destructor (was always "")
#  B. Scope-RAII producer: a by-value param of a class with an emitted dtor is
#     non-trivial for calls (Itanium) — the thunk copy-constructs a
#     caller-owned temporary, passes its ADDRESS, and destructs it after the
#     call via jlcs.scope. Verified end-to-end: copy-ctor + dtor tallies fire,
#     the callee's mutation stays in the temporary, the result round-trips.
#     (The old raw-bits by-value pass was a miscompile for these classes.)
#  C. Array-view producer: fixed-size primitive array members get zero-copy
#     get/set thunks through the strided ops. Verified end-to-end against a
#     raw Julia buffer.
#  E. dtor_call producer (2026-08-13): a destructor thunk is exactly the
#     single-pointer-arg void-return shape jlcs.dtor_call encodes, so it stops
#     going out as a generic ffe_call/try_call. Includes the Itanium VTT gate.
#  F. get_field / set_field producers (2026-08-13): thunk argument slots are
#     fields of the ciface `void**` at byte offset 8*slot, and the ArrayView
#     descriptor is four fields at 0/8/16/24. Both were previously raw GEP
#     chains. Pinned by MACHINE-CODE equality against the old spelling — the
#     LLVM IR differs cosmetically (GEP element type, !dbg numbering), so a
#     text diff would prove nothing.
#
# Requires libJLCS.so + clang++ (devtests tier).

using Test
using Libdl

const MLIR_AVAILABLE = try
    using RepliBuild
    isfile(RepliBuild.MLIRNative.libJLCS)
catch
    false
end

if !MLIR_AVAILABLE
    @info "libJLCS not found — skipping JLCS producer tests"
    exit(0)
end

using RepliBuild.MLIRNative
using RepliBuild.JLCSIRGenerator
using RepliBuild.DWARFParser

# ── Fixture build ─────────────────────────────────────────────────────────────

const PROD_DIR = joinpath(@__DIR__, "jlcs_producers")
const PROD_SRC = joinpath(PROD_DIR, "src", "raii_fixture.cpp")
const PROD_LIB = joinpath(PROD_DIR, "libraii_fixture.so")

if !isfile(PROD_LIB) || mtime(PROD_SRC) > mtime(PROD_LIB)
    @info "Building JLCS producer fixture..."
    run(`clang++ -shared -fPIC -O0 -std=c++17 -o $PROD_LIB $PROD_SRC`)
end
const FIXTURE = abspath(PROD_LIB)

# Resolve the fixture's mangled symbols from nm (no hardcoded mangling)
_nm = read(`nm -g --defined-only $FIXTURE`, String)
_sym(pat) = begin
    m = match(Regex("\\b(\\S*$(pat)\\S*)\\b"), _nm)
    m === nothing ? "" : String(m.captures[1])
end
const SYM_COPYCTOR = let m = match(r"(_ZN4GripC1ERKS_|_ZN4GripC2ERKS_)", _nm)
    m === nothing ? "" : String(m.captures[1])
end
const SYM_DTOR = let m = match(r"(_ZN4GripD1Ev|_ZN4GripD2Ev)", _nm)
    m === nothing ? "" : String(m.captures[1])
end
const SYM_CONSUME = let m = match(r"(_Z\S*consume_grip\S*)", _nm)
    m === nothing ? "" : String(m.captures[1])
end
@assert !isempty(SYM_COPYCTOR) && !isempty(SYM_DTOR) && !isempty(SYM_CONSUME) "fixture symbols not found:\n$_nm"

const FIXTURE_HANDLE = Libdl.dlopen(FIXTURE)
_tally() = ccall(Libdl.dlsym(FIXTURE_HANDLE, :jlcs_get_tally), Int64, ())
_reset_tally() = ccall(Libdl.dlsym(FIXTURE_HANDLE, :jlcs_reset_tally), Cvoid, ())

# ── Synthetic DWARF metadata (what Compiler.jl would extract) ─────────────────

const GRIP_STRUCTS = Dict{String,Any}(
    "Grip" => Dict{String,Any}(
        "kind" => "struct", "byte_size" => "0x8",
        "members" => Any[Dict{String,Any}("name" => "v", "c_type" => "int64_t",
                                          "offset" => "0x0", "size" => 8)]),
    # Array-view target: int64 tag at 0, double vals[4] at 8
    "Quad" => Dict{String,Any}(
        "kind" => "struct", "byte_size" => "0x28",
        "members" => Any[
            Dict{String,Any}("name" => "tag", "c_type" => "int64_t", "offset" => "0x0", "size" => 8),
            Dict{String,Any}("name" => "vals", "c_type" => "double [4]", "offset" => "0x8", "size" => 32),
        ]),
)

const GRIP_FUNCTIONS = Any[
    Dict{String,Any}("name" => "consume_grip", "mangled" => SYM_CONSUME,
        "demangled" => "consume_grip(Grip)", "is_method" => false,
        "parameters" => Any[Dict{String,Any}("name" => "g", "c_type" => "Grip")],
        "return_type" => Dict{String,Any}("c_type" => "int64_t")),
    Dict{String,Any}("name" => "Grip", "mangled" => SYM_COPYCTOR,
        "demangled" => "Grip::Grip(const Grip&)", "is_method" => true, "class" => "Grip",
        "parameters" => Any[Dict{String,Any}("name" => "o", "c_type" => "const Grip&")],
        "return_type" => Dict{String,Any}("c_type" => "void")),
    Dict{String,Any}("name" => "~Grip", "mangled" => SYM_DTOR,
        "demangled" => "Grip::~Grip()", "is_method" => true, "class" => "Grip",
        "parameters" => Any[],
        "return_type" => Dict{String,Any}("c_type" => "void")),
]

const METADATA = Dict{String,Any}(
    "language" => "cpp",
    "functions" => GRIP_FUNCTIONS,
    "struct_definitions" => GRIP_STRUCTS,
)

const EMPTY_VT = DWARFParser.VtableInfo(Dict{String,DWARFParser.ClassInfo}(),
                                        Dict{String,UInt64}(), Dict{String,UInt64}())

const IR = JLCSIRGenerator.generate_jlcs_ir(EMPTY_VT, METADATA)

@testset "JLCS producers" begin

    @testset "A. type_info destructorName wiring" begin
        vt = DWARFParser.VtableInfo(
            Dict("Grip" => DWARFParser.ClassInfo("Grip", 0, String[],
                                                 DWARFParser.VirtualMethod[],
                                                 [DWARFParser.MemberInfo("v", "int64_t", 0)], 8)),
            Dict{String,UInt64}(), Dict{String,UInt64}())
        ir_ti = JLCSIRGenerator.generate_jlcs_ir(vt, METADATA)
        @test occursin("jlcs.type_info \"Grip\"", ir_ti)
        @test occursin("\"$(SYM_DTOR)\"", ir_ti)          # dtor resolved, not ""
    end

    @testset "B. scope-RAII producer — emission" begin
        # The by-value Grip param becomes a caller-owned temporary
        @test occursin("%raii_tmp_1 = llvm.alloca", IR)
        @test occursin("jlcs.scope(%raii_tmp_1 : !llvm.ptr) dtors([@$(SYM_DTOR)])", IR)
        @test occursin("jlcs.ctor_call @$(SYM_COPYCTOR)(%raii_tmp_1, %val_ptr_1)", IR)
        # Callee now takes a pointer (Itanium indirect pass), not raw struct bits
        @test occursin("func.func private @$(SYM_CONSUME)(!llvm.ptr) -> i64", IR)
        # Result escapes the (result-less) scope through the slot
        @test occursin("llvm.store %raii_ret, %raii_retslot", IR)
    end

    @testset "C. array-view producer — emission" begin
        @test occursin("@jlcs_av_Quad_vals_get_thunk", IR)
        @test occursin("@jlcs_av_Quad_vals_set_thunk", IR)
        @test occursin("\"jlcs.load_array_element\"(%view, %index) : (!llvm.ptr, index) -> f64", IR)
        @test occursin("\"jlcs.store_array_element\"(%value, %view, %index) : (f64, !llvm.ptr, index) -> ()", IR)
        # No thunks for the scalar member — only fixed-size array members qualify
        @test !occursin("jlcs_av_Quad_tag", IR)
        # Grip's scalar member produces nothing either
        @test !occursin("jlcs_av_Grip", IR)
    end

    @testset "C2. array-view producer — reachability and type screens" begin
        AVG = RepliBuild.JLCSIRGenerator.ArrayViewGen
        structs = Dict(
            "Good"     => Dict("members" => [Dict("name"=>"vals",   "c_type"=>"float [4]", "offset"=>"0")]),
            "_IO_FILE" => Dict("members" => [Dict("name"=>"backup", "c_type"=>"char [8]",  "offset"=>"0")]),
        )

        # No manifest keeps the documented fallback: emit everything. This is
        # the path `IR` above takes, so it must not change.
        all_ir = AVG.generate_array_view_thunks(structs)
        @test occursin("jlcs_av_Good_vals_get_thunk", all_ir)

        # ...but the blocklist applies even then. A wrapper never declares a
        # type for _IO_FILE, so an accessor over its members could not have been
        # called even in principle.
        @test !occursin("_IO_FILE", all_ir)

        # Reachability. This producer never received `needed_symbols`, so it
        # emitted a pair per qualifying member whether or not anything could
        # call it: on llamacpp that was 424 thunks — 212 pairs, 208 owning
        # types, zero references from the generated wrapper — lowered and
        # JIT-compiled on every load. `generate_function_thunks` has honoured
        # the manifest since it existed; this is the second producer catching up.
        @test isempty(strip(AVG.generate_array_view_thunks(structs; needed_symbols=Set{String}())))

        # Asking for either half emits the pair — they share a prologue, and a
        # consumer that reads a member almost always writes it too.
        for half in ("get", "set")
            ir = AVG.generate_array_view_thunks(structs;
                     needed_symbols=Set(["jlcs_av_Good_vals_$(half)_thunk"]))
            @test occursin("jlcs_av_Good_vals_get_thunk", ir)
            @test occursin("jlcs_av_Good_vals_set_thunk", ir)
        end

        # An unrelated request must not drag anything in.
        @test isempty(strip(AVG.generate_array_view_thunks(structs;
                            needed_symbols=Set(["some_other_thunk"]))))

        # One derivation: the screen IRGen applies is the object Wrapper uses,
        # not a copy that can drift.
        @test RepliBuild.Wrapper._INTERNAL_TYPE_BLOCKLIST === RepliBuild.INTERNAL_TYPE_BLOCKLIST
        @test "_IO_FILE" in RepliBuild.INTERNAL_TYPE_BLOCKLIST
    end

    @testset "D. producers lower and execute" begin
        ctx = create_context()
        try
            mod = parse_module(ctx, IR)
            @test mod != C_NULL
            mod_jit = clone_module(mod)
            @test lower_to_llvm(mod_jit)

            # libJLCS must be resolvable: try_call lowering references
            # jlcs_set_pending_exception, which lives there (same pairing the
            # production JITManager uses: [binary, libJLCS])
            jit = create_jit(mod_jit, shared_libs=[FIXTURE, RepliBuild.MLIRNative.libJLCS])
            @test jit != C_NULL

            # ── scope-RAII end-to-end ────────────────────────────────────────
            _reset_tally()
            grip_buf = Int64[21]                       # a live Grip{v=21}
            result = Ref(Int64(0))
            GC.@preserve grip_buf result begin
                # FunctionGen slot convention for a BY-VALUE struct arg:
                # the slot holds a pointer to the struct storage directly
                # (one load yields the struct's address / its value)
                inner = Ptr{Cvoid}[Ptr{Cvoid}(pointer(grip_buf))]
                GC.@preserve inner begin
                    args_ref = Ref(Ptr{Cvoid}(pointer(inner)))
                    ptrs = Ptr{Cvoid}[Base.unsafe_convert(Ptr{Cvoid}, args_ref),
                                      Base.unsafe_convert(Ptr{Cvoid}, result)]
                    GC.@preserve args_ref begin
                        @test jit_invoke(jit, "$(SYM_CONSUME)_thunk", ptrs)
                    end
                end
            end
            # consume_grip mutates its copy (+1000) then returns v*2
            @test result[] == (21 + 1000) * 2
            # Copy-ctor (+100) and dtor (+1) both fired on the temporary
            @test _tally() == 101
            # The caller's object was isolated from the callee's mutation
            @test grip_buf[1] == 21

            # ── array-view end-to-end ────────────────────────────────────────
            # Quad memory: [i64 tag][4 x f64 vals]
            quad = zeros(UInt8, 40)
            GC.@preserve quad begin
                qp = pointer(quad)
                # set vals[2] = 7.5 (0-based index 2)
                val = Ref(Float64(7.5))
                obj_ptr = Ref(Ptr{Cvoid}(qp))
                idx = Ref(Int64(2))
                inner = Ptr{Cvoid}[Base.unsafe_convert(Ptr{Cvoid}, obj_ptr),
                                   Base.unsafe_convert(Ptr{Cvoid}, idx),
                                   Base.unsafe_convert(Ptr{Cvoid}, val)]
                GC.@preserve obj_ptr idx val inner begin
                    args_ref = Ref(Ptr{Cvoid}(pointer(inner)))
                    ptrs = Ptr{Cvoid}[Base.unsafe_convert(Ptr{Cvoid}, args_ref)]
                    GC.@preserve args_ref begin
                        @test jit_invoke(jit, "jlcs_av_Quad_vals_set_thunk", ptrs)
                    end
                end
                # Raw memory agrees: element 2 of the member at byte offset 8
                @test unsafe_load(Ptr{Float64}(qp + 8), 3) == 7.5

                # get vals[2] back through the view
                got = Ref(Float64(0.0))
                inner2 = Ptr{Cvoid}[Base.unsafe_convert(Ptr{Cvoid}, obj_ptr),
                                    Base.unsafe_convert(Ptr{Cvoid}, idx)]
                GC.@preserve obj_ptr idx inner2 got begin
                    args_ref2 = Ref(Ptr{Cvoid}(pointer(inner2)))
                    ptrs2 = Ptr{Cvoid}[Base.unsafe_convert(Ptr{Cvoid}, args_ref2),
                                       Base.unsafe_convert(Ptr{Cvoid}, got)]
                    GC.@preserve args_ref2 begin
                        @test jit_invoke(jit, "jlcs_av_Quad_vals_get_thunk", ptrs2)
                    end
                end
                @test got[] == 7.5
                # Neighbors untouched
                @test unsafe_load(Ptr{Float64}(qp + 8), 1) == 0.0
                @test unsafe_load(Ptr{Int64}(qp), 1) == 0
            end

            # ── dtor_call end-to-end ─────────────────────────────────────────
            # The destructor thunk now goes out as jlcs.dtor_call rather than
            # try_call. It must still destruct: Grip::~Grip() bumps the tally
            # by 1, and the may_throw lowering (invoke + landing pad) is the
            # path actually taken here, since DWARF marks no dtor noexcept.
            _reset_tally()
            dgrip = Int64[5]
            GC.@preserve dgrip begin
                obj = Ref(Ptr{Cvoid}(pointer(dgrip)))
                inner = Ptr{Cvoid}[Base.unsafe_convert(Ptr{Cvoid}, obj)]
                GC.@preserve obj inner begin
                    args_ref = Ref(Ptr{Cvoid}(pointer(inner)))
                    ptrs = Ptr{Cvoid}[Base.unsafe_convert(Ptr{Cvoid}, args_ref)]
                    GC.@preserve args_ref begin
                        @test jit_invoke(jit, "$(SYM_DTOR)_thunk", ptrs)
                    end
                end
            end
            @test _tally() == 1

            destroy_jit(jit)
            println("  ✓ scope-RAII + array-view + dtor_call producers execute end-to-end")
        finally
            destroy_context(ctx)
        end
    end

    @testset "E. dtor_call producer — emission and the VTT arity gate" begin
        # The destructor thunk carries the op, with may_throw (the module is
        # C++ and DWARF marks no destructor noexcept, which is the same rule
        # that used to pick try_call over ffe_call).
        @test occursin("jlcs.dtor_call @$(SYM_DTOR)(%val_1) { may_throw } : (!llvm.ptr) -> ()", IR)
        # ...and it is no longer routed through the generic call ops.
        @test !occursin("try_call %val_1 { callee = @$(SYM_DTOR) }", IR)
        # Non-destructors are untouched.
        @test occursin("callee = @$(SYM_CONSUME)", IR)

        FG = RepliBuild.JLCSIRGenerator.FunctionGen

        # THE ARITY GATE. Under Itanium a base-object destructor (D2) of a
        # class with virtual bases takes a second VTT argument — vi_test has
        # three of these — and jlcs.dtor_call has no operand to carry it.
        # Such a destructor must keep the ffe_call/try_call path; converting
        # it would drop the VTT silently, which is a miscompile, not a
        # cosmetic difference. Negative-checked below by the paired case.
        vtt_dtor = Any[Dict{String,Any}(
            "name" => "~Vb", "mangled" => "_ZN2VbD2Ev",
            "demangled" => "Vb::~Vb()", "is_method" => true, "class" => "Vb",
            "parameters" => Any[
                Dict{String,Any}("name" => "this", "c_type" => "Vb*"),
                Dict{String,Any}("name" => "vtt",  "c_type" => "void**")],
            "return_type" => Dict{String,Any}("c_type" => "void"))]
        vtt_ir = FG.generate_function_thunks(vtt_dtor, GRIP_STRUCTS; may_throw=true)
        @test !occursin("jlcs.dtor_call", vtt_ir)
        @test occursin("callee = @_ZN2VbD2Ev", vtt_ir)

        # The same destructor WITHOUT the VTT parameter does convert — so the
        # exclusion above is the arity, not the name or the class.
        plain_dtor = Any[Dict{String,Any}(
            "name" => "~Vb", "mangled" => "_ZN2VbD1Ev",
            "demangled" => "Vb::~Vb()", "is_method" => true, "class" => "Vb",
            "parameters" => Any[Dict{String,Any}("name" => "this", "c_type" => "Vb*")],
            "return_type" => Dict{String,Any}("c_type" => "void"))]
        plain_ir = FG.generate_function_thunks(plain_dtor, GRIP_STRUCTS; may_throw=true)
        @test occursin("jlcs.dtor_call @_ZN2VbD1Ev(%val_1) { may_throw }", plain_ir)

        # may_throw tracks the module setting, so a noexcept-clean module gets
        # the plain-call lowering rather than an invoke it does not need.
        noeh_ir = FG.generate_function_thunks(plain_dtor, GRIP_STRUCTS; may_throw=false)
        @test occursin("jlcs.dtor_call @_ZN2VbD1Ev(%val_1) : (!llvm.ptr) -> ()", noeh_ir)
        @test !occursin("may_throw", noeh_ir)
    end

    @testset "F. get_field / set_field producers" begin
        AVG = RepliBuild.JLCSIRGenerator.ArrayViewGen

        # Argument slots are fields of the ciface `void**`: slot i at byte 8i.
        @test occursin("\"jlcs.get_field\"(%args_ptr) {fieldOffset = 0 : i64} : (!llvm.ptr) -> !llvm.ptr", IR)
        # The raw spelling this replaced is gone from both producers.
        @test !occursin("llvm.getelementptr %args_ptr", IR)

        av = AVG.generate_array_view_thunks(GRIP_STRUCTS)
        @test occursin("\"jlcs.get_field\"(%args_ptr) {fieldOffset = 8 : i64}", av)   # index slot
        @test occursin("\"jlcs.get_field\"(%args_ptr) {fieldOffset = 16 : i64}", av)  # value slot (set thunk)
        # ArrayView descriptor built field-by-field, in the same terms the
        # load/store_array_element lowering reads it back with (data@0,
        # strides@16 via getStructField in JLCSPasses.cpp).
        for (off, val) in ((0, "%data"), (8, "%dims"), (16, "%strides"), (24, "%one"))
            @test occursin("\"jlcs.set_field\"(%view, $(val)) {fieldOffset = $(off) : i64}", av)
        end
        @test !occursin("llvm.getelementptr %view", av)
    end

    @testset "G. get_field / set_field are machine-code identical to raw GEP" begin
        # The whole justification for moving these producers onto the dialect
        # is that it changes what the IR SAYS, not what it DOES. Text and even
        # LLVM IR differ (GEP element type, !dbg numbering); the instructions
        # must not. Compare the exact shapes the producers emit.
        function insns(src, tag)
            ctx = create_context()
            obj = tempname() * ".o"
            try
                mod = parse_module(ctx, src; source_name="equiv_$tag.mlir")
                @assert lower_to_llvm(mod)
                RepliBuild.MLIRNative.emit_object(mod, obj)
                out = String[]
                for l in split(read(`objdump -d --no-show-raw-insn $obj`, String), '\n')
                    m = match(r"^\s+[0-9a-f]+:\s+(.*)$", l)
                    m === nothing && continue
                    push!(out, replace(strip(m.captures[1]), r"\s+#.*$" => ""))
                end
                return out
            finally
                rm(obj, force=true); destroy_context(ctx)
            end
        end
        wrap(body) = """
        module {
          func.func private @sink(!llvm.ptr)
          func.func @t(%args_ptr: !llvm.ptr) attributes { llvm.emit_c_interface } {
        $body
            return
          }
        }
        """
        raw_slot = wrap("""
            %idx_2 = arith.constant 1 : i64
            %arg_ptr_2 = llvm.getelementptr %args_ptr[%idx_2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
            %val_ptr_2 = llvm.load %arg_ptr_2 : !llvm.ptr -> !llvm.ptr
            jlcs.ffe_call %val_ptr_2 { callee = @sink } : (!llvm.ptr) -> ()""")
        new_slot = wrap("""
            %val_ptr_2 = "jlcs.get_field"(%args_ptr) {fieldOffset = 8 : i64} : (!llvm.ptr) -> !llvm.ptr
            jlcs.ffe_call %val_ptr_2 { callee = @sink } : (!llvm.ptr) -> ()""")
        @test insns(raw_slot, "raw_slot") == insns(new_slot, "new_slot")

        raw_store = wrap("""
            %f = llvm.getelementptr %args_ptr[16] : (!llvm.ptr) -> !llvm.ptr, i8
            llvm.store %args_ptr, %f : !llvm.ptr, !llvm.ptr""")
        new_store = wrap("""
            "jlcs.set_field"(%args_ptr, %args_ptr) {fieldOffset = 16 : i64} : (!llvm.ptr, !llvm.ptr) -> ()""")
        @test insns(raw_store, "raw_store") == insns(new_store, "new_store")
    end
end
