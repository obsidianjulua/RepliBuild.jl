# =============================================================================
# The temporary of a by-value class parameter (2026-11-17)
#
# C++ passes a class with a destructor as a pointer to a temporary the caller
# makes and ends, so a Tier 2 thunk makes one (`jlcs.scope`, `jlcs.ctor_call`).
# Two bugs lived there:
#
#   * The thunk module called the complete constructor and destructor (`C1`,
#     `D1`) and relied on their own thunks to declare them. A thunk the wrapper
#     does not bind is not emitted, and the wrapper binds `C2`/`D2`, so every
#     real module called a symbol it never declared; translating it to LLVM IR
#     dereferenced null and `using` the wrapper killed the process. The
#     producer test generated with no manifest, which declares everything.
#   * "No copy-constructor symbol" was read as "the copy is trivial". A deleted
#     copy constructor has no symbol either, so a move-only class was
#     duplicated bytewise and one of the two destroyed.
#
# Toolchain-free: the table (`_collect_class_raii`), the verdict
# (`value_temporary`, `by_value_crossing`), the module text and its
# declarations, the pre-flight set and the undeclared-callee check, over
# hand-made metadata that mirrors test/byvalue_class_test.
# test_byvalue_class_live.jl builds that library and calls through it.
# =============================================================================

using Test
using RepliBuild

const _BVT_J = RepliBuild.JLCSIRGenerator
const _BVT_FG = _BVT_J.FunctionGen
const _BVT_JM = RepliBuild.JITManager

_bvt_fn(mangled, demangled; class = "", name = "", params = (), ret = "void") = Dict{String,Any}(
    "mangled" => mangled, "demangled" => demangled, "class" => class, "name" => name,
    "is_method" => !isempty(class),
    "parameters" => Any[Dict{String,Any}("name" => n, "c_type" => t) for (n, t) in params],
    "return_type" => Dict{String,Any}("c_type" => ret))
_bvt_row(sym; acc = "public", deleted = false, artificial = false) =
    Dict{String,Any}("symbol" => sym, "accessibility" => acc, "deleted" => deleted, "artificial" => artificial)
_bvt_method(sym; virtual = false) = merge(_bvt_row(sym), Dict{String,Any}("virtual" => virtual))
_bvt_owner(scope = "") = Dict{String,Any}("kind" => "struct", "byte_size" => "0x8", "scope" => scope, "scopes" => [scope],
    "members" => Any[Dict{String,Any}("name" => "p", "c_type" => "int*", "offset" => "0x0", "size" => 8)])
# The constructor and destructor rows of class `cls` (as the demangler spells
# it; `bare` is how DWARF spells a parameter of it), symbols `pre…`.
function _bvt_lifetime(cls, bare, pre, self; copy = false, move = false, d0_only = false)
    short = last(split(cls, "::"))
    short = first(split(short, "<"))
    out = Any[]
    for v in ("C1", "C2")
        copy && push!(out, _bvt_fn("$(pre)$(v)ERK$(self)", "$cls::$short($cls const&)"; class = cls, name = short,
                                   params = (("this", "$bare*"), ("o", "const $bare&"))))
        move && push!(out, _bvt_fn("$(pre)$(v)EO$(self)", "$cls::$short($cls&&)"; class = cls, name = short,
                                   params = (("this", "$bare*"), ("o", "$bare&&"))))
        push!(out, _bvt_fn("$(pre)$(v)Ei", "$cls::$short(int)"; class = cls, name = short,
                           params = (("this", "$bare*"), ("v", "int"))))
    end
    for v in (d0_only ? ("D0",) : ("D0", "D1", "D2"))
        push!(out, _bvt_fn("$(pre)$(v)Ev", "$cls::~$short()"; class = cls, name = "~$short", params = (("this", "$bare*"),)))
    end
    return out
end
_bvt_take(sym, name, bare) = _bvt_fn(sym, "$name($bare)"; name = name, params = (("x", bare),), ret = "int")

const _BVT_STRUCTS = Dict{String,Any}(
    "Res" => _bvt_owner(), "Uniq" => _bvt_owner(), "Pinned" => _bvt_owner(), "Guarded" => _bvt_owner(),
    "Dflt" => _bvt_owner(), "Undefined" => _bvt_owner(), "Legacy" => _bvt_owner(), "OldCopy" => _bvt_owner(),
    "Virt" => _bvt_owner(), "Assigned" => _bvt_owner(),
    "Plain" => Dict{String,Any}("kind" => "struct", "byte_size" => "0x4", "scope" => "", "scopes" => [""],
        "members" => Any[Dict{String,Any}("name" => "v", "c_type" => "int", "offset" => "0x0", "size" => 4)]),
    "Holder" => Dict{String,Any}("kind" => "struct", "byte_size" => "0x10", "scope" => "", "scopes" => [""],
        "members" => Any[Dict{String,Any}("name" => "r", "c_type" => "Res", "offset" => "0x0", "size" => 8),
                         Dict{String,Any}("name" => "id", "c_type" => "int", "offset" => "0x8", "size" => 4)]),
    "Poly" => Dict{String,Any}("kind" => "struct", "byte_size" => "0x10", "scope" => "", "scopes" => [""],
        "members" => Any[Dict{String,Any}("name" => "_vptr\$Poly", "c_type" => "function_ptr(int)**", "offset" => "0x0", "size" => 8),
                         Dict{String,Any}("name" => "v", "c_type" => "int", "offset" => "0x8", "size" => 4)]),
    "NRes" => _bvt_owner("ns"), "NUniq" => _bvt_owner("ns"), "Box<int>" => _bvt_owner("ns"),
    "Inner" => _bvt_owner("Outer"),
    "Twin" => merge(_bvt_owner("one"), Dict{String,Any}("scopes" => ["one", "two"])))

_bvt_facts(; ctors = nothing, methods = Any[]) = (d = Dict{String,Any}("byte_size" => "0x8", "pass" => "reference", "empty" => false,
                                                                         "methods" => methods);
                                                  ctors === nothing || (d["constructors"] = ctors); d)
const _BVT_ABI = Dict{String,Any}(
    "Res" => _bvt_facts(ctors = [_bvt_row("_ZN3ResC4ERKS_"), _bvt_row("_ZN3ResC4Ei")]),
    "Uniq" => _bvt_facts(ctors = [_bvt_row("_ZN4UniqC4EOS_"), _bvt_row("_ZN4UniqC4ERKS_"; deleted = true), _bvt_row("_ZN4UniqC4Ei")]),
    "Pinned" => _bvt_facts(ctors = [_bvt_row("_ZN6PinnedC4ERKS_"; deleted = true), _bvt_row("_ZN6PinnedC4Ei")]),
    "Guarded" => _bvt_facts(ctors = [_bvt_row("_ZN7GuardedC4ERKS_"; acc = "private"), _bvt_row("_ZN7GuardedC4Ei")]),
    # No constructor declared at all, and a destructor: the copy is implicit.
    "Plain" => _bvt_facts(methods = [_bvt_method("_ZN5PlainD4Ev")]),
    "Holder" => _bvt_facts(ctors = [_bvt_row("_ZN6HolderC4Ei")]),
    # A copy constructor declared, public, not deleted, and in no symbol table:
    # `= default` over a pointer (`Dflt`, which the forced unit was asked for),
    # or an inline one nothing emitted (`Undefined`, which it was not).
    "Dflt" => _bvt_facts(ctors = [_bvt_row("_ZN4DfltC4ERKS_"), _bvt_row("_ZN4DfltC4Ei")]),
    "Undefined" => _bvt_facts(ctors = [_bvt_row("_ZN9UndefinedC4ERKS_"), _bvt_row("_ZN9UndefinedC4Ei")]),
    "Poly" => _bvt_facts(ctors = [_bvt_row("_ZN4PolyC4ERKS_"), _bvt_row("_ZN4PolyC4Ei")],
                         methods = [_bvt_method("_ZN4PolyD4Ev"; virtual = true)]),
    "Virt" => _bvt_facts(methods = [_bvt_method("_ZN4VirtD4Ev"; virtual = true)]),
    # `Assigned& operator=(Assigned&&)` and no constructor of either kind.
    "Assigned" => _bvt_facts(methods = [_bvt_method("_ZN8AssignedD4Ev"), _bvt_method("_ZN8AssignedaSEOS_")]),
    "NRes" => _bvt_facts(ctors = [_bvt_row("_ZN2ns4NResC4ERKS0_"), _bvt_row("_ZN2ns4NResC4Ei")]),
    "NUniq" => _bvt_facts(ctors = [_bvt_row("_ZN2ns5NUniqC4EOS0_"), _bvt_row("_ZN2ns5NUniqC4ERKS0_"; deleted = true)]),
    "Box<int>" => _bvt_facts(ctors = [_bvt_row("_ZN2ns3BoxIiEC4ERKS1_"), _bvt_row("_ZN2ns3BoxIiEC4Ei")]),
    "Inner" => _bvt_facts(ctors = [_bvt_row("_ZN5Outer5InnerC4ERKS0_"), _bvt_row("_ZN5Outer5InnerC4Ei")]),
    "Twin" => _bvt_facts(ctors = [_bvt_row("_ZN3one4TwinC4ERKS0_")]),
    # Metadata from before declarations were recorded: only the convention.
    "Legacy" => Dict{String,Any}("byte_size" => "0x8", "pass" => "reference", "empty" => false),
    "OldCopy" => Dict{String,Any}("byte_size" => "0x8", "pass" => "reference", "empty" => false))

const _BVT_FUNCTIONS = vcat(
    _bvt_lifetime("Res", "Res", "_ZN3Res", "S_"; copy = true),
    _bvt_lifetime("Uniq", "Uniq", "_ZN4Uniq", "S_"; move = true),
    _bvt_lifetime("Pinned", "Pinned", "_ZN6Pinned", "S_"),
    _bvt_lifetime("Guarded", "Guarded", "_ZN7Guarded", "S_"; copy = true),
    _bvt_lifetime("Plain", "Plain", "_ZN5Plain", "S_"),
    _bvt_lifetime("Holder", "Holder", "_ZN6Holder", "S_"),
    _bvt_lifetime("Dflt", "Dflt", "_ZN4Dflt", "S_"),
    _bvt_lifetime("Undefined", "Undefined", "_ZN9Undefined", "S_"),
    _bvt_lifetime("Poly", "Poly", "_ZN4Poly", "S_"; copy = true),
    _bvt_lifetime("Virt", "Virt", "_ZN4Virt", "S_"; d0_only = true),
    _bvt_lifetime("Assigned", "Assigned", "_ZN8Assigned", "S_"),
    _bvt_lifetime("Legacy", "Legacy", "_ZN6Legacy", "S_"; move = true),
    _bvt_lifetime("OldCopy", "OldCopy", "_ZN7OldCopy", "S_"; copy = true),
    _bvt_lifetime("ns::NRes", "NRes", "_ZN2ns4NRes", "S0_"; copy = true),
    _bvt_lifetime("ns::NUniq", "NUniq", "_ZN2ns5NUniq", "S0_"; move = true),
    _bvt_lifetime("ns::Box<int>", "Box<int>", "_ZN2ns3BoxIiE", "S1_"; copy = true),
    _bvt_lifetime("Outer::Inner", "Inner", "_ZN5Outer5Inner", "S0_"; copy = true),
    _bvt_lifetime("one::Twin", "Twin", "_ZN3one4Twin", "S0_"; copy = true),
    _bvt_lifetime("two::Twin", "Twin", "_ZN3two4Twin", "S0_"; copy = true),
    Any[_bvt_take("_Z8take_res3Res", "take_res", "Res"),
        _bvt_take("_Z12consume_uniq4Uniq", "consume_uniq", "Uniq"),
        _bvt_take("_Z11take_pinned6Pinned", "take_pinned", "Pinned"),
        _bvt_take("_Z12take_guarded7Guarded", "take_guarded", "Guarded"),
        _bvt_take("_Z10take_plain5Plain", "take_plain", "Plain"),
        _bvt_take("_Z11take_holder6Holder", "take_holder", "Holder"),
        _bvt_take("_Z9take_dflt4Dflt", "take_dflt", "Dflt"),
        _bvt_take("_Z9take_poly4Poly", "take_poly", "Poly"),
        _bvt_take("_ZN2ns9take_nresENS_4NResE", "ns::take_nres", "NRes"),
        _bvt_take("_ZN2ns13consume_nuniqENS_5NUniqE", "ns::consume_nuniq", "NUniq"),
        _bvt_take("_ZN2ns8take_boxENS_3BoxIiEE", "ns::take_box", "Box<int>"),
        _bvt_take("_Z10take_innerN5Outer5InnerE", "take_inner", "Inner"),
        _bvt_take("_Z9take_twinN3one4TwinE", "take_twin", "Twin"),
        _bvt_fn("_Z9take_both3Resi4Uniq", "take_both(Res, int, Uniq)"; name = "take_both",
                params = (("a", "Res"), ("k", "int"), ("b", "Uniq")), ret = "int"),
        # A reference, not a value: no temporary.
        _bvt_fn("_Z8peek_resRK3Res", "peek_res(Res const&)"; name = "peek_res", params = (("r", "const Res&"),), ret = "int")])

const _BVT_METADATA = Dict{String,Any}(
    "language" => "cpp", "functions" => _BVT_FUNCTIONS, "struct_definitions" => _BVT_STRUCTS,
    "record_abi" => _BVT_ABI,
    "inline_constructors" => Dict{String,Any}(
        "forced" => Any[Dict{String,Any}("class" => "Dflt", "symbol" => "_ZN4DfltC4ERKS_", "emitted" => false)],
        "skipped" => Any[]))

const _BVT_EMPTY_VT = RepliBuild.DWARFParser.VtableInfo(Dict{String,RepliBuild.DWARFParser.ClassInfo}(),
                                                        Dict{String,UInt64}(), Dict{String,UInt64}())

# The wrapper binds one constructor and one destructor per Julia method, and
# its dedup has kept the base-object variants: the manifest lists `C2` and
# `D2`, never the `C1` and `D1` a temporary needs.
const _BVT_MANIFEST = Set{String}(m for m in (String(f["mangled"]) for f in _BVT_FUNCTIONS)
                                  if !occursin(r"C1E|D1E|D0E", m))

# The body of `sym`'s thunk in a module's text.
function _bvt_thunk(ir, sym)
    i = findfirst("func.func @$(sym)_thunk(", ir)
    i === nothing && return ""
    j = findnext("\n}\n", ir, i.start)
    return ir[i.start:(j === nothing ? lastindex(ir) : j.stop)]
end

@testset "The temporary of a by-value class parameter" begin
    raii = _BVT_J._collect_class_raii(_BVT_METADATA)

    @testset "a class is keyed as a parameter names it, namespace or not" begin
        @test Set(keys(raii)) == Set(["Res", "Uniq", "Pinned", "Guarded", "Plain", "Holder", "Dflt", "Undefined",
                                      "Poly", "Virt", "Assigned", "Legacy", "OldCopy", "NRes", "NUniq",
                                      "Box<int>", "Inner", "Twin"])
        @test raii["NRes"][:class] == "ns::NRes"
        @test raii["Inner"][:class] == "Outer::Inner"
        @test raii["Box<int>"][:class] == "ns::Box<int>"
        # The complete-object variants, for a class in a namespace too.
        @test raii["Res"][:dtor] == "_ZN3ResD1Ev" && raii["Res"][:copy_ctor] == "_ZN3ResC1ERKS_"
        @test raii["NRes"][:dtor] == "_ZN2ns4NResD1Ev" && raii["NRes"][:copy_ctor] == "_ZN2ns4NResC1ERKS0_"
        @test raii["Box<int>"][:copy_ctor] == "_ZN2ns3BoxIiEC1ERKS1_"
        @test raii["Uniq"][:move_ctor] == "_ZN4UniqC1EOS_" && !haskey(raii["Uniq"], :copy_ctor)
        @test raii["NUniq"][:move_ctor] == "_ZN2ns5NUniqC1EOS0_"
        # A constructor from an `int` is neither.
        @test !haskey(raii["Pinned"], :copy_ctor) && !haskey(raii["Pinned"], :move_ctor)
        # Every symbol an entry was chosen from, for the pre-flight.
        @test Set(split(raii["Res"][:symbols])) ==
              Set(["_ZN3ResC1ERKS_", "_ZN3ResC2ERKS_", "_ZN3ResD0Ev", "_ZN3ResD1Ev", "_ZN3ResD2Ev"])
        # Two classes on one key: the first keeps the entry and both are named.
        @test raii["Twin"][:class] == "one::Twin" && raii["Twin"][:ambiguous] == "one::Twin; two::Twin"
        @test raii["Twin"][:dtor] == "_ZN3one4TwinD1Ev"
        # A class with no entry in the table keeps the demangler's spelling.
        bare = _BVT_J._collect_class_raii(Dict{String,Any}("functions" => _bvt_lifetime("ns::NRes", "NRes", "_ZN2ns4NRes", "S0_"; copy = true)))
        @test collect(keys(bare)) == ["ns::NRes"] && bare["ns::NRes"][:copy_ctor] == "_ZN2ns4NResC1ERKS0_"
    end

    @testset "how the temporary is made is read from the declarations" begin
        how(k) = (raii[k][:temp], get(raii[k], :ctor, ""))
        @test how("Res") == ("copy", "_ZN3ResC1ERKS_")
        @test how("NRes") == ("copy", "_ZN2ns4NResC1ERKS0_")
        @test how("Box<int>") == ("copy", "_ZN2ns3BoxIiEC1ERKS1_")
        @test how("Inner") == ("copy", "_ZN5Outer5InnerC1ERKS0_")
        @test how("Poly") == ("copy", "_ZN4PolyC1ERKS_")
        # The copy constructor is deleted and a move constructor is defined.
        @test how("Uniq") == ("move", "_ZN4UniqC1EOS_")
        @test how("NUniq") == ("move", "_ZN2ns5NUniqC1EOS0_")
        # The bytes only where the copy is trivial: implicit over a scalar, or
        # declared and the forced unit had nothing to emit for it.
        @test how("Plain") == ("bits", "")
        @test how("Dflt") == ("bits", "")
        # Everything else is no temporary, and says why.
        why(k) = (raii[k][:temp] == "none" ? raii[k][:why] : "not refused: $(raii[k][:temp])")
        @test why("Pinned") == "its copy constructor is deleted and it has no move constructor"
        @test why("Guarded") == "its copy constructor is private and it has no move constructor"
        @test occursin("implicit and not trivial", why("Holder"))
        @test why("Undefined") == "the library does not define its copy constructor"
        @test occursin("move assignment operator", why("Assigned")) && occursin("no move constructor", why("Assigned"))
        @test occursin("deleting destructor", why("Virt"))
        @test occursin("named `Twin`", why("Twin")) && occursin("one::Twin; two::Twin", why("Twin"))
        # Metadata that records no declarations: a copy constructor the library
        # defines is used as before, and without one nothing says the copy is
        # not deleted, although a move constructor is there.
        @test how("OldCopy") == ("copy", "_ZN7OldCopyC1ERKS_")
        @test haskey(raii["Legacy"], :move_ctor) && occursin("rebuild", why("Legacy"))
        # A symbol's absence decides nothing: every class refused or moved
        # here has no copy-constructor symbol, which used to mean "its bytes".
        @test sort!([k for k in keys(raii) if !haskey(raii[k], :copy_ctor) && raii[k][:temp] != "bits"]) ==
              ["Assigned", "Holder", "Legacy", "NUniq", "Pinned", "Undefined", "Uniq", "Virt"]
    end

    @testset "the copy is its bytes only over members that are" begin
        structs = Dict{String,Any}(
            "__enum__Color" => Dict{String,Any}("kind" => "enum", "byte_size" => "0x4", "julia_type" => "Cint"),
            "Pod" => Dict{String,Any}("kind" => "struct", "byte_size" => "0x8", "members" => Any[]),
            "Res" => _bvt_owner())
        abi = Dict{String,Any}("Pod" => Dict{String,Any}("pass" => "value"), "Res" => Dict{String,Any}("pass" => "reference"),
                               "basic_string<char, std::char_traits<char>, std::allocator<char> >" => Dict{String,Any}("pass" => "reference"))
        rec(members...; bases = Any[]) = Dict{String,Any}("kind" => "struct", "byte_size" => "0x20", "base_classes" => bases,
            "members" => Any[Dict{String,Any}("name" => "m$i", "c_type" => t, "offset" => "0x$(string(8 * (i - 1); base = 16))", "size" => 8)
                             for (i, t) in enumerate(members)])
        yes(r) = (structs["T"] = r; _BVT_FG._bitwise_copyable("T", structs, abi))
        @test yes(rec("int", "double*", "const char*", "Color", "Pod", "int[4]", "Pod[2]", "function_ptr(int)*"))
        @test yes(rec("int&"))                                        # a reference member copies as its bytes
        @test !yes(rec("int", "Res"))                                 # a member with a copy of its own
        @test !yes(rec("Res[2]"))
        @test !yes(rec("std::string"))                                # a container is never its bytes
        @test !yes(rec("std::vector<int>"))
        @test !yes(rec("Unknown"))                                    # no facts: not known to be
        @test !yes(rec("int"; bases = Any[Dict{String,Any}("type" => "Res", "virtual" => false)]))
        @test yes(rec("int"; bases = Any[Dict{String,Any}("type" => "struct Pod", "virtual" => false)]))
        @test !yes(rec("int"; bases = Any[Dict{String,Any}("type" => "Pod", "virtual" => true)]))
        @test !_BVT_FG._bitwise_copyable("Poly", _BVT_STRUCTS, _BVT_ABI)   # a vptr
        @test !_BVT_FG._bitwise_copyable("Missing", structs, abi)
        # A static member is not part of the object.
        s = rec("int"); push!(s["members"], Dict{String,Any}("name" => "count", "c_type" => "Res"))
        @test yes(s)
    end

    @testset "one verdict for both generators: a class no temporary can be made for is a trap" begin
        func(sym) = _BVT_FUNCTIONS[findfirst(f -> f["mangled"] == sym, _BVT_FUNCTIONS)]
        cross(sym) = _BVT_FG.thunk_crossings(func(sym), _BVT_STRUCTS, _BVT_ABI; abi = :sysv, class_raii = raii)
        checked = 0
        for (sym, part) in (("_Z11take_pinned6Pinned", "its copy constructor is deleted and it has no move constructor"),
                            ("_Z12take_guarded7Guarded", "its copy constructor is private"),
                            ("_Z11take_holder6Holder", "implicit and not trivial"),
                            ("_Z9take_twinN3one4TwinE", "named `Twin`"))
            c = cross(sym)
            @test (sym, c.params[1][1]) == (sym, :uncopyable)
            @test (sym, c.trap !== nothing && c.trap == c.abi_trap) == (sym, true)
            @test (sym, c.layout_trap, c.conflict_trap) == (sym, nothing, nothing)
            c.trap === nothing && continue
            @test occursin("cannot be passed by value", c.trap) && occursin(part, c.trap)
            checked += 1
        end
        @test checked == 4
        @test cross("_Z11take_pinned6Pinned").abi_trap ==
              "takes `Pinned` by value, and `Pinned` cannot be passed by value: " *
              "its copy constructor is deleted and it has no move constructor"
        for sym in ("_Z8take_res3Res", "_Z12consume_uniq4Uniq", "_Z10take_plain5Plain", "_Z9take_dflt4Dflt",
                    "_ZN2ns9take_nresENS_4NResE", "_ZN2ns13consume_nuniqENS_5NUniqE", "_Z9take_both3Resi4Uniq")
            @test (sym, cross(sym).trap) == (sym, nothing)
        end
        # A reference to the class, and a return of it, need no temporary.
        @test _BVT_FG.by_value_crossing("const Pinned&", _BVT_STRUCTS, _BVT_ABI; class_raii = raii)[1] === :direct
        @test _BVT_FG.by_value_crossing("Pinned", _BVT_STRUCTS, _BVT_ABI; ret = true, class_raii = raii)[1] === :layout
        # Without the table the crossing is what it was: only a caller that
        # passes it can be told, and both generators do.
        @test _BVT_FG.by_value_crossing("Pinned", _BVT_STRUCTS, _BVT_ABI)[1] === :layout
        src = read(joinpath(@__DIR__, "..", "src", "Wrapper", "Cpp", "GeneratorCpp.jl"), String)
        @test occursin(r"thunk_crossings\([^\n]*;[^\n]*class_raii", src)
        @test occursin("JLCSIRGenerator._collect_class_raii(metadata)", src)
    end

    # Every symbol the module's scopes and constructor calls name.
    lifetime = ["_ZN3ResC1ERKS_", "_ZN3ResD1Ev", "_ZN4UniqC1EOS_", "_ZN4UniqD1Ev", "_ZN5PlainD1Ev", "_ZN4DfltD1Ev",
                "_ZN4PolyC1ERKS_", "_ZN4PolyD1Ev", "_ZN2ns4NResC1ERKS0_", "_ZN2ns4NResD1Ev", "_ZN2ns5NUniqC1EOS0_",
                "_ZN2ns5NUniqD1Ev", "_ZN2ns3BoxIiEC1ERKS1_", "_ZN2ns3BoxIiED1Ev", "_ZN5Outer5InnerC1ERKS0_",
                "_ZN5Outer5InnerD1Ev"]
    declarations(ir, sym) = count("func.func private @$(sym)(", ir) + count("llvm.func @$(sym)(", ir)

    @testset "the module declares every constructor and destructor it calls, once, whatever the manifest" begin
        ir = _BVT_J.generate_jlcs_ir(_BVT_EMPTY_VT, _BVT_METADATA; needed_symbols = copy(_BVT_MANIFEST), abi = :sysv)
        # The manifest has none of them, so none has a thunk to declare it.
        @test isempty(intersect(lifetime, _BVT_MANIFEST))
        @test !any(s -> occursin("func.func @$(s)_thunk(", ir), lifetime)
        called = Set{String}()
        for m in eachmatch(r"dtors\(\[([^\]]*)\]\)", ir), s in eachmatch(r"@([^\s,\]]+)", m.captures[1])
            push!(called, String(s.captures[1]))
        end
        for m in eachmatch(r"jlcs\.ctor_call @([^\s(]+)\(", ir)
            push!(called, String(m.captures[1]))
        end
        @test called == Set(lifetime)
        @test [(s, declarations(ir, s)) for s in lifetime] == [(s, 1) for s in lifetime]
        @test _BVT_J.undeclared_callees(ir) == String[]

        # With no manifest each has a thunk of its own, which declares it: still once.
        all_ir = _BVT_J.generate_jlcs_ir(_BVT_EMPTY_VT, _BVT_METADATA; abi = :sysv)
        @test all(s -> occursin("func.func @$(s)_thunk(", all_ir), lifetime)
        @test [(s, declarations(all_ir, s)) for s in lifetime] == [(s, 1) for s in lifetime]
        @test _BVT_J.undeclared_callees(all_ir) == String[]
        @test !occursin("Constructors and destructors of by-value temporaries", all_ir)

        # A callee the module declared before the thunks (a virtual destructor
        # on the legacy path) is not declared again.
        take = Any[f for f in _BVT_FUNCTIONS if f["mangled"] == "_Z8take_res3Res"]
        pre = _BVT_FG.generate_function_thunks(take, _BVT_STRUCTS; may_throw = true, class_raii = raii,
                                               record_abi = _BVT_ABI, abi = :sysv, predeclared = ["_ZN3ResD1Ev"])
        @test declarations(pre, "_ZN3ResD1Ev") == 0 && declarations(pre, "_ZN3ResC1ERKS_") == 1
        alone = _BVT_FG.generate_function_thunks(take, _BVT_STRUCTS; may_throw = true, class_raii = raii,
                                                 record_abi = _BVT_ABI, abi = :sysv)
        @test declarations(alone, "_ZN3ResD1Ev") == 1 && declarations(alone, "_ZN3ResC1ERKS_") == 1
        @test occursin("func.func private @_ZN3ResD1Ev(!llvm.ptr) \n", alone)
        @test occursin("func.func private @_ZN3ResC1ERKS_(!llvm.ptr, !llvm.ptr) \n", alone)
    end

    @testset "each thunk makes its temporary the way its class needs" begin
        ir = _BVT_J.generate_jlcs_ir(_BVT_EMPTY_VT, _BVT_METADATA; needed_symbols = copy(_BVT_MANIFEST), abi = :sysv)
        t = _bvt_thunk(ir, "_Z8take_res3Res")
        @test occursin("jlcs.scope(%raii_tmp_1 : !llvm.ptr) dtors([@_ZN3ResD1Ev]) {", t)
        @test occursin("jlcs.ctor_call @_ZN3ResC1ERKS_(%raii_tmp_1, %val_ptr_1) : (!llvm.ptr, !llvm.ptr) -> ()", t)
        # Move-only: the move constructor, from the argument's storage, and
        # never the bytes.
        t = _bvt_thunk(ir, "_Z12consume_uniq4Uniq")
        @test occursin("jlcs.scope(%raii_tmp_1 : !llvm.ptr) dtors([@_ZN4UniqD1Ev]) {", t)
        @test occursin("jlcs.ctor_call @_ZN4UniqC1EOS_(%raii_tmp_1, %val_ptr_1) : (!llvm.ptr, !llvm.ptr) -> ()", t)
        @test !occursin("raii_blob", t)
        # A trivial copy: the bytes, and the destructor.
        for (sym, dtor, n) in (("_Z10take_plain5Plain", "_ZN5PlainD1Ev", 4), ("_Z9take_dflt4Dflt", "_ZN4DfltD1Ev", 8))
            t = _bvt_thunk(ir, sym)
            @test occursin("dtors([@$(dtor)])", t) && !occursin("jlcs.ctor_call", t)
            @test occursin("%raii_blob_1 = llvm.load %val_ptr_1 : !llvm.ptr -> !llvm.array<$(n) x i8>", t)
        end
        # A namespace, a nested class and a template's specialization take the
        # same path: they used to be passed as the bytes behind the slot, with
        # no constructor and no destructor.
        for (sym, ctor, dtor) in (("_ZN2ns9take_nresENS_4NResE", "_ZN2ns4NResC1ERKS0_", "_ZN2ns4NResD1Ev"),
                                  ("_ZN2ns13consume_nuniqENS_5NUniqE", "_ZN2ns5NUniqC1EOS0_", "_ZN2ns5NUniqD1Ev"),
                                  ("_ZN2ns8take_boxENS_3BoxIiEE", "_ZN2ns3BoxIiEC1ERKS1_", "_ZN2ns3BoxIiED1Ev"),
                                  ("_Z10take_innerN5Outer5InnerE", "_ZN5Outer5InnerC1ERKS0_", "_ZN5Outer5InnerD1Ev"))
            t = _bvt_thunk(ir, sym)
            @test (sym, occursin("dtors([@$(dtor)])", t), occursin("jlcs.ctor_call @$(ctor)(%raii_tmp_1, %val_ptr_1)", t)) ==
                  (sym, true, true)
        end
        # Two temporaries: a copy and a move, one destructor each.
        t = _bvt_thunk(ir, "_Z9take_both3Resi4Uniq")
        @test occursin("jlcs.scope(%raii_tmp_1, %raii_tmp_3 : !llvm.ptr, !llvm.ptr) dtors([@_ZN3ResD1Ev, @_ZN4UniqD1Ev]) {", t)
        @test occursin("jlcs.ctor_call @_ZN3ResC1ERKS_(%raii_tmp_1, %val_ptr_1)", t)
        @test occursin("jlcs.ctor_call @_ZN4UniqC1EOS_(%raii_tmp_3, %val_ptr_3)", t)
        # No thunk where no temporary can be made, and the module says why.
        for sym in ("_Z11take_pinned6Pinned", "_Z12take_guarded7Guarded", "_Z11take_holder6Holder", "_Z9take_twinN3one4TwinE")
            @test (sym, _bvt_thunk(ir, sym)) == (sym, "")
            @test occursin(Regex("// no thunk for " * sym * ": .*cannot be passed by value"), ir)
        end
        # A reference is a pointer, as it was.
        @test !occursin("raii", _bvt_thunk(ir, "_Z8peek_resRK3Res"))
    end

    @testset "a module that calls what it never declares is refused by name, before the JIT sees it" begin
        # The module as it was generated before: the scope and the constructor
        # call name C1 and D1, and only C2 and D2 are declared.
        broken = """
        module {
          llvm.func @__gxx_personality_v0(...) -> i32
        func.func private @_Z4take3Res(!llvm.ptr) -> i32
        func.func @_Z4take3Res_thunk(%args_ptr: !llvm.ptr) -> i32 attributes { llvm.emit_c_interface } {
          %raii_one = llvm.mlir.constant(1 : i64) : i64
          %raii_tmp_1 = llvm.alloca %raii_one x !llvm.array<8 x i8> : (i64) -> !llvm.ptr
          %raii_retslot = llvm.alloca %raii_one x i32 : (i64) -> !llvm.ptr
          %val_ptr_1 = "jlcs.get_field"(%args_ptr) {fieldOffset = 0 : i64} : (!llvm.ptr) -> !llvm.ptr
          jlcs.scope(%raii_tmp_1 : !llvm.ptr) dtors([@_ZN3ResD1Ev]) {
            jlcs.ctor_call @_ZN3ResC1ERKS_(%raii_tmp_1, %val_ptr_1) : (!llvm.ptr, !llvm.ptr) -> ()
            %raii_ret = jlcs.try_call %raii_tmp_1 { callee = @_Z4take3Res } : (!llvm.ptr) -> i32
            llvm.store %raii_ret, %raii_retslot : i32, !llvm.ptr
            jlcs.yield
          }
          %ret_val = llvm.load %raii_retslot : !llvm.ptr -> i32
          return %ret_val : i32
        }
        func.func private @_ZN3ResD2Ev(!llvm.ptr)
        func.func @_ZN3ResD2Ev_thunk(%args_ptr: !llvm.ptr)  attributes { llvm.emit_c_interface } {
          %val_ptr_1 = "jlcs.get_field"(%args_ptr) {fieldOffset = 0 : i64} : (!llvm.ptr) -> !llvm.ptr
          %val_1 = llvm.load %val_ptr_1 : !llvm.ptr -> !llvm.ptr
          jlcs.dtor_call @_ZN3ResD2Ev(%val_1) { may_throw } : (!llvm.ptr) -> ()
          return
        }
        }
        """
        @test _BVT_J.undeclared_callees(broken) == ["_ZN3ResC1ERKS_", "_ZN3ResD1Ev"]
        err = try
            _BVT_JM._assert_callees_declared(broken, "libbv.so")
            ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("libbv.so", err) && occursin("_ZN3ResC1ERKS_, _ZN3ResD1Ev", err) && occursin("never declares", err)
        fixed = replace(broken, "func.func private @_ZN3ResD2Ev(!llvm.ptr)\n" =>
                        "func.func private @_ZN3ResD2Ev(!llvm.ptr)\nfunc.func private @_ZN3ResD1Ev(!llvm.ptr)\n" *
                        "  llvm.func @_ZN3ResC1ERKS_(!llvm.ptr, !llvm.ptr)\n")
        @test _BVT_J.undeclared_callees(fixed) == String[]
        @test _BVT_JM._assert_callees_declared(fixed, "libbv.so") === nothing
        # Every form of callee is read: a plain call, a destructor call, a
        # legacy virtual thunk's `llvm.call`, and each member of a scope's list.
        forms = """
          %r = jlcs.ffe_call %a { callee = @plain, sysv_args = ["mem:8"] } : (!llvm.ptr) -> i32
          jlcs.dtor_call @dtor(%a) : (!llvm.ptr) -> ()
          %x = llvm.call @legacy(%arg0) : (!llvm.ptr) -> i32
          jlcs.scope(%a, %b : !llvm.ptr, !llvm.ptr) dtors([@first, @second]) {
          "jlcs.vcall"(%a) { class_name = @SomeClass, vtable_offset = 0 : i64, slot = 1 : i64, this_offset = 0 : i64 } : (!llvm.ptr) -> ()
        """
        @test _BVT_J.undeclared_callees(forms) == ["dtor", "first", "legacy", "plain", "second"]
    end

    @testset "the JIT pre-flights the constructors and destructors a thunk will call" begin
        needed = Set(["_Z8take_res3Res", "_ZN2ns13consume_nuniqENS_5NUniqE", "_Z8peek_resRK3Res"])
        life = _BVT_J.lifetime_callees(_BVT_METADATA, needed)
        # Every variant the entry was chosen from, so that one that does not
        # resolve cannot hand the call to an unchecked sibling.
        @test life == Set(["_ZN3ResC1ERKS_", "_ZN3ResC2ERKS_", "_ZN3ResD0Ev", "_ZN3ResD1Ev", "_ZN3ResD2Ev",
                           "_ZN2ns5NUniqC1EOS0_", "_ZN2ns5NUniqC2EOS0_", "_ZN2ns5NUniqD0Ev", "_ZN2ns5NUniqD1Ev",
                           "_ZN2ns5NUniqD2Ev"])
        @test _BVT_JM._module_callees(_BVT_METADATA, needed) == union(needed, life)
        # A reference alone asks for nothing.
        @test isempty(_BVT_J.lifetime_callees(_BVT_METADATA, Set(["_Z8peek_resRK3Res"])))
        # With no manifest, every function's.
        every = _BVT_J.lifetime_callees(_BVT_METADATA, nothing)
        @test issubset(life, every) && "_ZN5Outer5InnerD1Ev" in every && "_ZN2ns3BoxIiEC1ERKS1_" in every

        # One the library's handle does not resolve is dropped with its row,
        # so the next table no longer offers it.
        md = deepcopy(_BVT_METADATA)
        own = Set{String}(String(f["mangled"]) for f in md["functions"])
        gone = ["_ZN3ResC1ERKS_", "_ZN3ResC2ERKS_"]
        (bound, unresolved) = _BVT_JM._preflight_callees(_ -> nothing, _BVT_JM._module_callees(md, needed), setdiff(own, gone))
        @test isempty(bound) && unresolved == gone
        _BVT_JM._drop_functions!(md, unresolved)
        @test !any(f -> f["mangled"] in gone, md["functions"]) && length(md["functions"]) == length(_BVT_FUNCTIONS) - 2
        after = _BVT_J._collect_class_raii(md)
        @test !haskey(after["Res"], :copy_ctor) && after["Res"][:temp] == "none"
        ir = _BVT_J.generate_jlcs_ir(_BVT_EMPTY_VT, md; needed_symbols = needed, abi = :sysv)
        @test !any(s -> occursin(s, replace(ir, r"//[^\n]*" => "")), gone)
        @test _BVT_J.undeclared_callees(ir) == String[]
    end
end
