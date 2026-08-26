#!/usr/bin/env julia
# test/test_introspection.jl — the two surfaces Hub consumers hand-rolled
# before the generator emitted them (2026-08-12).
#
# `kernel_emits_llvmcall` — reach into the private `_TIER1_*` kernel, call
# `code_typed`, string-match "llvmcall" — appeared in 12 consumer files, 11 of
# them byte-identical, because `TIER1_FUNCTIONS` records what the generator
# INTENDED and a `@generated` kernel can demote at generation time. And
# `struct_size`/`meta_offset` — re-parsing `compilation_metadata.json` for facts
# the generator held while emitting — appeared in four more.
#
# So the wrapper emits `DISPATCH_TIER` (intent) + `dispatch_tier` (reality), and
# `STRUCT_SIZES`/`STRUCT_OFFSETS` + `struct_size`/`member_offset`.
#
# Library-free: drives the emitters and EXECUTES what they produce.

using Test
using RepliBuild

const W = RepliBuild.Wrapper

@testset "Dispatch + layout introspection" begin

    @testset "Tiers are classified from the emitted chunks" begin
        chunks = [
            # Tier 1: the public wrapper delegates to a @generated kernel
            """
            @generated function _TIER1_fast(x::Cint)
                return :(Base.llvmcall((IR, "fast"), Cint, Tuple{Cint}, x))
            end
            function fast(x::Cint)::Cint
                return _TIER1_fast(x)
            end
            """,
            # Tier 2, JIT
            """
            function thunked(a::Any)
                return RepliBuild.JITManager.invoke("_mlir_ciface__Z7thunkedv_thunk", Cint, a)
            end
            """,
            # Tier 2, AOT companion library — VIRTUAL dispatch shape.
            # Note this body contains `ccall((:`, so it is only classified
            # correctly while the Tier-2 test precedes the Tier-3 one.
            """
            function aot(a::Any)
                ret = ccall((:_mlir_ciface__Z3aotv_thunk, THUNKS_LIBRARY_PATH), Cint, (Ptr{Ptr{Cvoid}},), inner_ptrs)
                return ret
            end
            """,
            # Tier 2, AOT companion library — ORDINARY function shape. A
            # SECOND, unrelated spelling: `invoke_aot` through the dlopened
            # handle, which shares no substring with the virtual shape above.
            # Untested until 2026-08-23, and untested is how it broke: the
            # classifier knew only the virtual spelling, so an AOT wrapper
            # with no virtual methods lost EVERY Tier-2 row and then read as a
            # uniform Tier-3 module.
            """
            function aot_plain(a::Any)
                return RepliBuild.JITManager.invoke_aot(THUNKS_HANDLE[], "_mlir_ciface__Z9aot_plainv_thunk", Cint, a)
            end
            """,
            # Tier 2, AOT companion library — ORDINARY function, EAGER shape.
            # A THIRD spelling: since thunk addresses are resolved in `__init__`
            # the call site names neither the handle nor the symbol, so it shares
            # no substring with either shape above — `invoke_aot_ptr(` does not
            # contain `invoke_aot(`. The shape above is kept alongside it rather
            # than replaced: `invoke_aot` is still a supported entry point, and a
            # wrapper generated before the change is a file on someone's disk.
            """
            function aot_eager(a::Any)
                return RepliBuild.JITManager.invoke_aot_ptr(_FPTR__Z9aot_eagerv[], Cint, a)
            end
            """,
            # Tier 3, plain and variadic
            """
            function plain(x::Cint)::Cint
                return ccall((:plain, LIBRARY_PATH), Cint, (Cint,), x)
            end
            """,
            """
            function va(x::Cint)::Cint
                return @ccall LIBRARY_PATH.var"va"(x::Cint;)::Cint
            end
            """,
        ]
        (tier, kernels) = W._dispatch_facts(chunks)

        @test tier["fast"]      === :tier1
        @test tier["thunked"]   === :tier2
        @test tier["aot"]       === :tier2
        @test tier["aot_plain"] === :tier2
        @test tier["aot_eager"] === :tier2
        @test tier["plain"]     === :tier3
        @test tier["va"]        === :tier3

        # A DISPATCHING FUNCTION MUST NEVER BE MERELY ABSENT. Omission is worse
        # than misfiling: a short table can read as UNIFORM, and a uniform
        # table is emitted as a sentence asserting the tier, so one unclassified
        # shape makes the wrapper describe itself falsely. Keyed on naming the
        # dispatch machinery, so an ordinary helper stays unclassified in peace.
        @test_throws ErrorException W._dispatch_facts([
            """
            function future_shape(a::Any)
                return RepliBuild.JITManager.invoke_by_some_new_means(H[], "sym", Cint, a)
            end
            """,
        ])

        helper_only = W._dispatch_facts([
            """
            function helper(p::Ptr{Cvoid})
                return unsafe_load(Base.unsafe_convert(Ptr{Cint}, p))
            end
            """,
        ])
        @test isempty(helper_only[1])

        # The kernel is recorded so `dispatch_tier` can check it, and is itself
        # NOT classified — reaching for it is exactly what this replaces.
        @test kernels["fast"] == "_TIER1_fast"
        @test !haskey(tier, "_TIER1_fast")

        # One Julia name, two tiers: a last-write-wins table would answer
        # confidently and wrongly. 48 names on the Hub have this shape.
        (t3, k3) = W._dispatch_facts([
            "function overloaded(x::Cint)::Cint\n    return _TIER1_overloaded(x)\nend\n",
            "function overloaded(x::Any)::Cint\n    return ccall((:overloaded, LIBRARY_PATH), Cint, (Cint,), x)\nend\n",
        ])
        @test t3["overloaded"] === :mixed
        @test !haskey(k3, "overloaded")      # no single kernel can answer for it

        # Agreeing definitions are NOT mixed — the pair emitted for a Cstring
        # return is two functions on one tier and must stay precise.
        (t4, _) = W._dispatch_facts([
            "function s()::Union{String,Nothing}\n    ptr = ccall((:s, LIBRARY_PATH), Cstring, (), )\nend\n",
            "function s_ptr()::Cstring\n    return ccall((:s, LIBRARY_PATH), Cstring, (), )\nend\n",
        ])
        @test t4["s"] === :tier3 && t4["s_ptr"] === :tier3

        # A function that calls nothing foreign is not a dispatch site.
        (t2, _) = W._dispatch_facts(["function helper(x)\n    return x + 1\nend\n"])
        @test !haskey(t2, "helper")
    end

    @testset "dispatch_tier answers from emitted code, not from the table" begin
        chunk = W._dispatch_tier_chunk(
            Dict("real" => :tier1, "demoted" => :tier1, "thunk" => :tier2, "direct" => :tier3),
            Dict("real" => "_TIER1_real", "demoted" => "_TIER1_demoted"))
        @test occursin("const DISPATCH_TIER", chunk)
        @test occursin("function dispatch_tier(f)", chunk)

        m = Module(:LayoutProbe)
        Base.include_string(m, """
        # A kernel that really does emit an llvmcall, and one that does not —
        # standing in for a slice that shipped and a slice that went missing.
        _TIER1_real() = Base.llvmcall("ret i32 42", Int32, Tuple{})
        _TIER1_demoted() = Int32(42)
        real() = _TIER1_real()
        demoted() = _TIER1_demoted()
        thunk() = nothing
        direct() = nothing
        """ * chunk)

        @test m.real() == Int32(42)          # the llvmcall kernel actually runs
        @test m.dispatch_tier(:real)    === :tier1
        @test m.dispatch_tier(m.real)   === :tier1   # by function, not just Symbol
        @test m.dispatch_tier(:thunk)   === :tier2
        @test m.dispatch_tier(:direct)  === :tier3
        @test m.dispatch_tier(:nope)    === :unknown

        # THE POINT: emitted intent says tier1, the emitted code says otherwise,
        # and the honest answer is what will actually run.
        @test m.DISPATCH_TIER[:demoted] === :tier1
        @test m.dispatch_tier(:demoted) === :tier3
    end

    @testset "Observation is refused in output mode, not frozen" begin
        # dispatch_tier is NOT read-only: probing forces the @generated kernel to
        # generate. Inside a precompile worker that kernel deliberately splices
        # its ccall body, so an answer taken there describes the WORKER, not the
        # session that runs — `const T = dispatch_tier(:f)` at module scope froze
        # :tier3 into the pkgimage for a function that re-generates to llvmcall on
        # load. Measured with an Observer/Control package pair before the guard:
        # Observer baked :tier3, runtime kernel was llvmcall. After: :deferred.
        chunk = W._dispatch_tier_chunk(Dict("f" => :tier1), Dict("f" => "_TIER1_f"))
        @test occursin("jl_generating_output", chunk)
        @test occursin(":deferred", chunk)
        # `ccall` is syntax, not a binding: qualifying it as `Base.ccall` is an
        # UndefVarError at load. Every other Base call in the emitted helper IS
        # qualified, so this one must be checked explicitly.
        @test !occursin("Base.ccall(", chunk)
        # The guard has to precede the probe, or it does not prevent generation.
        @test findfirst("jl_generating_output", chunk).start <
              findfirst("code_typed", chunk).start
    end

    @testset "Layout facts are emitted for the types the module declares" begin
        structs = Dict(
            "Vec3"     => Dict("byte_size" => 12,
                               "members" => [Dict("name" => "x", "offset" => 0),
                                             Dict("name" => "y", "offset" => 4),
                                             Dict("name" => "z", "offset" => 8)]),
            "Big"      => Dict("byte_size" => "0x100", "members" => [Dict("name" => "f", "offset" => 16)]),
            "NotMine"  => Dict("byte_size" => 8, "members" => [Dict("name" => "q", "offset" => 0)]),
            "Sizeless" => Dict("byte_size" => 0, "members" => []),
        )
        chunk = W._layout_chunk(structs, Set(["Vec3", "Big", "Sizeless"]), identity)

        m = Module(:LayoutOnly)
        Base.include_string(m, chunk)

        @test m.struct_size(:Vec3) == 12
        @test m.struct_size("Vec3") == 12          # accepts a string too
        @test m.struct_size(:Big) == 256           # hex byte_size parsed
        @test m.member_offset(:Vec3, :z) == 8
        @test m.member_offset(:Big, :f) == 16

        # Scoped to the emitted set — a metadata struct the module never
        # declared is not something a caller can use (llama.cpp: 434 of 2864).
        @test !haskey(m.STRUCT_SIZES, :NotMine)
        # A zero/absent size would silently under-allocate, so it is omitted.
        @test !haskey(m.STRUCT_SIZES, :Sizeless)

        @test_throws ErrorException m.struct_size(:NotMine)
        @test_throws ErrorException m.member_offset(:Vec3, :nope)
        # The error names the alternatives rather than just failing
        err = try m.member_offset(:Vec3, :nope) catch e; sprint(showerror, e) end
        @test occursin("x, y, z", err)
    end

    @testset "A DWARF member name is not always a Julia identifier" begin
        # A polymorphic class carries a synthesized `_vptr$Class`. Emitted raw
        # into a `const` Dict literal, `$` is interpolation — `UndefVarError: $`
        # at module load, taking the entire wrapper with it. Caught live on
        # llama.cpp, whose metadata has 110 of them.
        structs = Dict("Poly" => Dict("byte_size" => 16,
                        "members" => [Dict("name" => "_vptr\$Poly", "offset" => 0),
                                      Dict("name" => "value", "offset" => 8)]))
        chunk = W._layout_chunk(structs, Set(["Poly"]), identity)

        @test !occursin("\$", chunk)               # nothing interpolatable survives
        @test occursin(":_vptr_Poly", chunk)       # sanitized, not dropped

        m = Module(:VptrProbe)
        Base.include_string(m, chunk)              # would throw pre-fix
        @test m.struct_size(:Poly) == 16
        @test m.member_offset(:Poly, :value) == 8
        @test m.member_offset(:Poly, :_vptr_Poly) == 0
    end

    @testset "Keys come from the caller's own sanitizer, not a re-derivation" begin
        # The generators collapse `_+` and trim; a generic [^A-Za-z0-9_] => "_"
        # leaves a trailing underscore on any templated name, misses the
        # emitted-name gate, and drops the type SILENTLY. Measured on the Hub
        # before this was threaded through: 100 of imgui's 262 declared structs,
        # 29 of tinyxml2's 42 — exactly the templated types whose size a caller
        # cannot compute any other way.
        structs = Dict("ImChunkStream<ImGuiTableSettings>" =>
                       Dict("byte_size" => 24, "members" => [Dict("name" => "Buf", "offset" => 0)]))
        emitted = Set(["ImChunkStream_ImGuiTableSettings"])          # what the generator emitted
        generic(s) = replace(String(s), r"[^A-Za-z0-9_]" => "_")     # the wrong re-derivation
        real(s)    = W._sanitize_cpp_type_name(String(s))            # what the generator used

        @test generic("ImChunkStream<ImGuiTableSettings>") != real("ImChunkStream<ImGuiTableSettings>")
        @test W._layout_chunk(structs, emitted, generic) == ""       # silently dropped
        chunk = W._layout_chunk(structs, emitted, real)
        @test occursin(":ImChunkStream_ImGuiTableSettings =>", chunk)

        m = Module(:SanitizerProbe)
        Base.include_string(m, chunk)
        @test m.struct_size(:ImChunkStream_ImGuiTableSettings) == 24
    end

    @testset "Empty inputs emit nothing rather than empty scaffolding" begin
        @test W._dispatch_tier_chunk(Dict{String,Symbol}(), Dict{String,String}()) == ""
        @test W._layout_chunk(Dict(), Set{String}(), identity) == ""
    end

    @testset "AOT thunk slots are derived from the final text" begin
        # The slot namer is keyed on the MANGLED symbol so two functions cannot
        # share an address. `_ZN1A1fEv` and `_ZN1A1f.Ev` sanitize to the same
        # identifier; without the `taken` registry the second would silently
        # rebind the first's slot and call the wrong thunk.
        taken = Dict{String,String}()
        a = W._aot_fptr_const_name!(taken, "_ZN1A1fEv")
        b = W._aot_fptr_const_name!(taken, "_ZN1A1f.Ev")
        @test a == "_FPTR__ZN1A1fEv"
        @test a != b
        # Idempotent: asking again for the same symbol returns the same slot.
        @test W._aot_fptr_const_name!(taken, "_ZN1A1fEv") == a

        # Derivation is from the text that ships, not from `taken` — a symbol
        # the generator registered but whose chunk lost the dedup must not
        # leave a slot behind.
        W._aot_fptr_const_name!(taken, "_Z7droppedv")
        text = """
        function f(a::Any)
            return RepliBuild.JITManager.invoke_aot_ptr($a[], Cint, a)
        end
        function g(a::Any)
            return RepliBuild.JITManager.invoke_aot_ptr($b[], Cint, a)
        end
        """
        chunk = W._aot_thunk_slot_chunk(text, taken)
        @test occursin("const $a = Ref{Ptr{Cvoid}}(C_NULL)", chunk)
        @test occursin("const $b = Ref{Ptr{Cvoid}}(C_NULL)", chunk)
        @test !occursin("_FPTR__Z7droppedv", chunk)
        @test occursin("\"_mlir_ciface__ZN1A1fEv_thunk\"", chunk)

        # A cstring pair emits TWO functions reading ONE thunk; that must be one
        # slot, not a duplicate definition (which is a hard error at load).
        pair = """
        function h(); ptr = RepliBuild.JITManager.invoke_aot_ptr($a[], Cstring); end
        function h_ptr(); return RepliBuild.JITManager.invoke_aot_ptr($a[], Cstring); end
        """
        pair_chunk = W._aot_thunk_slot_chunk(pair, taken)
        @test count(_ -> true, eachmatch(Regex("const \\Q$a\\E = Ref"), pair_chunk)) == 1

        # The table exists even with nothing in it — `__init__` iterates it
        # unconditionally, so a sometimes-absent table is a second thing to
        # keep in sync.
        @test occursin("const _THUNK_SLOTS", W._aot_thunk_slot_chunk("", taken))

        # A slot in the text that no emission site registered is generator
        # drift, and must raise at wrap time rather than become an
        # UndefVarError on whichever call the user happens to make first.
        @test_throws ErrorException W._aot_thunk_slot_chunk(
            "RepliBuild.JITManager.invoke_aot_ptr(_FPTR__Z7unknownv[], Cint)",
            Dict{String,String}())
    end
end
