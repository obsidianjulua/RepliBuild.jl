# The Julia module a Rust library gets (`GeneratorRust`), as text, without a
# toolchain: generated from the vendored builds of test/fixtures/rust/lean.rs.
# Each check pins the line that is the claim (the call, the type's bytes, the
# reason a function is not wrapped), not a whole function.
#
# The module itself is loaded and called in test_rust_wrap_live.jl: a wrapper
# needs its library beside it.

using Test
using RepliBuild

isdefined(@__MODULE__, :rust_extract) || include(joinpath(@__DIR__, "support", "rust_fixture.jl"))

const GR = RepliBuild.GeneratorRust

wrapper(opt) = GR.generate(rust_metadata(opt); module_name = "Lean", library_path = "/somewhere/liblean.so", generator = "test")
tier(report, name) = only(r for r in report if r["name"] == name)["tier"]
# The text of one generated function: from its `function` line to its `end`.
function body(text, name)
    m = match(Regex("\\nfunction \\Q$name\\E\\(.*?\\n(?:.*\\n)*?end\\n"), text)
    m === nothing && error("no function $name in the wrapper")
    return String(m.match)
end

@testset "Rust wrapper: names" begin
    name(path) = GR.function_name(Dict("path" => path))
    @test name("lean::add") == "add"
    @test name("libm::math::sin::sin") == "sin"                       # where it is defined does not name it
    @test name("<lean::Acc>::push") == "Acc_push"
    @test name("<lean::Shape as lean::Area>::area") == "Shape_Area_area"
    @test name("<kurbo::point::Point as core::fmt::Debug>::fmt") == "Point_Debug_fmt"
    @test name("lean::largest::<f64>") == "largest_f64"
    @test name("<lean::Pair<i32>>::sum") == "Pair_i32_sum"
    @test name("c_add") == "c_add"
    @test GR.sanitize("core::option::Option<&lean::Rec>") == "Option_Rec"
    @test GR.sanitize("(i32, f64)") == "i32_f64" && GR.sanitize("3d") == "_3d" && GR.sanitize("<>") == "unnamed"
    # Two functions with one name are told apart by where they are, and a
    # function never takes a type's name or a word Julia keeps.
    md = rust_metadata(0)
    twin = deepcopy(only(r for r in md["functions"] if r["path"] == "lean::add"))
    twin["path"] = "lean::deep::add"; twin["scope"] = ["lean", "deep"]; twin["symbol"] = "_RNvNtCsX_4lean4deep3add"
    kw = deepcopy(twin); kw["path"] = "lean::end"; kw["scope"] = ["lean"]; kw["symbol"] = "_RNvCsX_4lean3end"
    ty = deepcopy(twin); ty["path"] = "lean::Rec"; ty["scope"] = ["lean"]; ty["symbol"] = "_RNvCsX_4lean3Rec"
    append!(md["functions"], [twin, kw, ty])
    sort!(md["functions"]; by = r -> (r["path"], r["symbol"]))
    (_, report) = GR.generate(md; module_name = "Lean", library_path = "/x/liblean.so")
    names = [r["name"] for r in report]
    @test allunique(names)
    @test only(r for r in report if r["path"] == "lean::deep::add")["name"] == "deep_add"
    @test only(r for r in report if r["path"] == "lean::end")["name"] == "end_"
    @test only(r for r in report if r["path"] == "lean::Rec")["name"] == "Rec_"
end

@testset "Rust wrapper: the module" begin
    for opt in (0, 3)
        (text, report) = wrapper(opt)
        # It parses, as one module.
        ex = Meta.parseall(text)
        @test !any(a -> a isa Expr && a.head in (:incomplete, :error), ex.args)
        mods = [a for a in ex.args if a isa Expr && a.head === :module]
        @test length(mods) == 1 && mods[1].args[2] === :Lean
        @test occursin("const LIBRARY_SHA = \"$("0"^64)\"", text)
        @test occursin("const LIBRARY_PATH = let baked = \"/somewhere/liblean.so\"", text)
        # Every function of the library is in the module, wrapped or as a stub that says why.
        @test length(report) == 31 && allunique(r["name"] for r in report)
        @test count(r -> r["tier"] == "unsupported", report) == 3
        @test sort!([r["name"] for r in report if r["tier"] == "unsupported"]) == ["cpoint_norm", "dyn_area", "located"]
        @test occursin("track_caller", only(r for r in report if r["name"] == "located")["reason"])
        @test occursin("trait object", only(r for r in report if r["name"] == "dyn_area")["reason"])
        @test occursin("located(args...) = Base.error(", text)
        @test occursin(":located => ", text) && occursin(":add => ", text)
        # The library's names fill the module, so the generator's own calls into
        # Base are qualified: a crate with a function `convert` or `length`
        # must not change what the wrapper's glue calls.
        code = join((l for l in split(text, '\n') if !startswith(l, "#")), "\n")
        for f in ("error", "convert", "unsafe_load", "unsafe_store!", "pointer", "length", "ncodeunits", "ntuple",
                  "reinterpret", "isfile", "joinpath", "basename", "open", "bytes2hex", "unsafe_convert", "getfield")
            @test (f, occursin(Regex("(?<![\\w.!])\\Q$f\\E\\("), code)) == (f, false)
        end
        # The same metadata gives the same text, apart from the line that says when.
        again = wrapper(opt)[1]
        strip_time(t) = replace(t, r"^# Generated: .*$"m => "")
        @test strip_time(again) == strip_time(text)
    end
end

@testset "Rust wrapper: how each function calls" begin
    (text0, report0) = wrapper(0)
    (text3, report3) = wrapper(3)
    # Unoptimized, no Rust function is proved not to panic: every call is inside
    # the catching frame. (`c_add` is `extern "C"`, which cannot unwind by definition.)
    @test all(r -> r["tier"] in ("guarded", "unsupported") || r["name"] == "c_add", report0)
    @test tier(report0, "c_add") == "ccall"
    # Optimized: a plain `ccall` where the signature is scalars and pointers and
    # the function cannot panic; a thunk where it is not; the frame where it can.
    @test tier(report3, "add") == "ccall" && tier(report3, "slice_sum") == "ccall" && tier(report3, "rec_sum_ref") == "ccall"
    @test tier(report3, "maybe") == "thunk" && tier(report3, "pair") == "thunk" && tier(report3, "hello") == "thunk"
    @test tier(report3, "rec_make") == "thunk"            # a hidden return pointer is the thunk's
    @test tier(report3, "flag") == "thunk"                # one bit is not a `ccall` type
    @test tier(report3, "wide") == "thunk" && tier(report3, "b3_rot") == "thunk"
    @test tier(report3, "checked_div") == "guarded"
    @test tier(report0, "add") == "guarded"

    add = body(text3, "add")
    @test occursin(r"ccall\(\(:_R\w+4lean3add, LIBRARY_PATH\), Base\.UInt32, \(Base\.UInt32, Base\.UInt32\), ", add)
    @test occursin("Base.reinterpret(Base.UInt32, Base.convert(Base.Int32, a1))", add)
    @test startswith(add, "\nfunction add(a1::Base.Integer, a2::Base.Integer)")
    @test !occursin("RustRuntime.invoke", add)
    # The same function unoptimized goes through its thunk, guarded.
    add0 = body(text0, "add")
    @test occursin("RepliBuild.RustRuntime.invoke(_rb_library(), RepliBuild.RustRuntime.thunk(_THUNKS[", add0)
    @test occursin(", true, _slots, _ret_p)", add0) && !occursin("ccall(", add0)
    @test occursin(", false, _slots, _ret_p)", body(text3, "maybe"))
    @test occursin(", true, _slots, _ret_p)", body(text3, "checked_div"))
end

@testset "Rust wrapper: values in and out" begin
    (text, _) = wrapper(3)
    # A string is a pointer and a length, kept alive across the call.
    s = body(text, "str_len")
    @test occursin("function str_len(a1::Base.AbstractString)", s)
    @test occursin("a1_s = Base.String(a1)", s) && occursin("Base.ncodeunits(a1_s)", s) && occursin("GC.@preserve a1_s", s)
    @test occursin("function slice_sum(a1::Base.DenseVector{Base.Float64})", body(text, "slice_sum"))
    # ...and comes back as a Julia string.
    @test occursin("return Base.unsafe_string(Base.Ptr{Base.UInt8}(", body(text, "hello"))
    # `Option<i32>` is the value or nothing: the tag at 0, the value at 4.
    @test occursin("return (Base.unsafe_load(Base.Ptr{Base.UInt32}(_ret_p)) == 1 ? Base.unsafe_load(Base.Ptr{Base.Int32}(_ret_p + 4)) : nothing)",
                   body(text, "maybe"))
    # A tuple is a Julia tuple, each part read where the compiler put it.
    @test occursin("return (Base.unsafe_load(Base.Ptr{Base.Int32}(_ret_p)), Base.unsafe_load(Base.Ptr{Base.Float64}(_ret_p + 8)))",
                   body(text, "pair"))
    # A bool is a Bool on both sides and a byte in between.
    @test occursin("function flag(a1::Base.Bool)", body(text, "flag"))
    @test occursin("return (Base.unsafe_load(Base.Ptr{Base.UInt8}(_ret_p)) != 0x00)", body(text, "flag"))
    @test occursin("Base.UInt32(Base.Char(a1))", body(text, "char_code"))
    # A record by value is its bytes; the two halves of a pair are read from them at their offsets.
    dot = body(text, "f2_dot")
    @test occursin("function f2_dot(a1::F2, a2::F2)", dot)
    @test occursin("Base.unsafe_store!(Base.Ptr{F2}(a1_p), a1::F2)", dot)
    @test occursin("Base.unsafe_load(Base.Ptr{Base.Float64}(a1_p + 8))", dot)
    # A record over a word goes as a pointer to those bytes.
    @test occursin("Base.Float64, (Base.Ptr{Base.Cvoid},), a1_p)", body(text, "rec_sum"))
    # A reference takes the value, a Ref of it, or a pointer; a mutable one, not the value.
    @test occursin("function rec_sum_ref(a1::Base.Union{Rec,Base.RefValue{Rec},Base.Ptr})", body(text, "rec_sum_ref"))
    @test occursin("function Acc_push(a1::Base.Union{Base.RefValue,Base.Ptr}, a2::Base.Real)", body(text, "Acc_push"))
    @test occursin("function opt_ref_d(a1::Base.Union{Base.Nothing,Base.RefValue,Base.Ptr})", body(text, "opt_ref_d"))
    @test occursin("(a1_r === nothing ? Base.C_NULL : _rb_address(a1_r))", body(text, "opt_ref_d"))
    # A zero-sized argument is not an argument.
    @test occursin("function z(a2::Base.Integer)", body(text, "z"))
end

@testset "Rust wrapper: types" begin
    (text, _) = wrapper(0)
    # A record is its bytes, at the size and offsets the compiler chose.
    @test occursin("struct Rec\n    _rb_bytes::Base.NTuple{16,Base.UInt8}\n    Rec(::_RbRaw, bytes::Base.NTuple{16,Base.UInt8}) = new(bytes)\nend", text)
    @test occursin("f === :b && return Base.unsafe_load(Base.Ptr{Base.Float64}(_p))", text)
    @test occursin("f === :a && return Base.unsafe_load(Base.Ptr{Base.UInt8}(_p + 14))", text)
    @test occursin("Base.propertynames(::Rec) = (:a, :b, :c, :d,)", text)
    @test occursin("function Rec(f1, f2, f3, f4)", text)
    # A fieldless enum is a Julia enum, its members under its name.
    @test occursin("@enum Level::Base.UInt8 begin\n    Level_Low = 0\n    Level_Mid = 1\n    Level_High = 2\nend", text)
    # An enum with data says which variant it is, and gives that variant's record.
    @test occursin("function RepliBuild.RustRuntime.variant(x::Shape)", text)
    @test occursin("_d == Base.UInt64(0x1) && return :Rect", text)
    @test occursin("struct Shape_Rect\n", text) && occursin("f === :w && return Base.unsafe_load(Base.Ptr{Base.Float64}(_p + 8))", text)
    @test occursin("f === :Rect && return (RepliBuild.RustRuntime.variant(x) === :Rect ? Base.unsafe_load(Base.Ptr{Shape_Rect}(_p))", text)
    # A fat pointer and a tuple get no type of their own.
    @test !occursin(r"struct \w*str\w*\n", text) && !occursin("struct i32_f64", text)
end

@testset "Rust wrapper: what may own memory is not moved" begin
    ptr(kind, target) = Dict{String,Any}("kind" => "pointer", "pointer" => kind, "target" => target, "size" => 8)
    rec(fields...) = Dict{String,Any}("kind" => "struct", "size" => 8 * length(fields), "name" => "S", "scope" => Any[],
                                      "fields" => Any[Dict{String,Any}("name" => "f$i", "offset" => 8 * (i - 1), "type" => t)
                                                      for (i, t) in enumerate(fields)])
    types = Dict{String,Any}(
        "u8" => Dict{String,Any}("kind" => "base", "size" => 1, "encoding" => "unsigned"),
        "u64" => Dict{String,Any}("kind" => "base", "size" => 8, "encoding" => "unsigned"),
        "&u8" => ptr("ref", "u8"), "*const u8" => ptr("const_ptr", "u8"), "Box<u8>" => ptr("box", "u8"),
        "Plain" => rec("u64", "u64"), "Borrows" => rec("&u8", "u64"), "Raw" => rec("*const u8", "u64"),
        "Boxed" => rec("Box<u8>"), "Nested" => rec("Boxed", "u64"),
        "alloc::string::String" => rec("u64", "u64", "u64"),
        "&[u8]" => merge(rec("*const u8", "u64"), Dict{String,Any}("fat" => "slice")))
    ctx = GR.Ctx(Dict{String,Any}("types" => types))
    @test !GR.owns(ctx, "u64") && !GR.owns(ctx, "Plain") && !GR.owns(ctx, "Borrows") && !GR.owns(ctx, nothing)
    @test !GR.owns(ctx, "&u8") && !GR.owns(ctx, "*const u8")          # a pointer handed over as itself is the caller's
    @test !GR.owns(ctx, "&[u8]")                                      # a borrowed slice
    @test GR.owns(ctx, "Box<u8>") && GR.owns(ctx, "Boxed") && GR.owns(ctx, "Nested")
    @test GR.owns(ctx, "Raw")                                         # a raw pointer inside a record may be owned
    @test GR.owns(ctx, "alloc::string::String")
    @test GR.owns(ctx, "no::such::Type")                              # what is not described is not assumed plain

    # That is the fallback. The compiler's answer, where the build has it, is the one used:
    # a record with a raw pointer is plain when the compiler says nothing drops it, and a
    # type is movable when the build has its drop function.
    @test GR.unmovable(ctx, "Plain") === nothing && GR.unmovable(ctx, "&u8") === nothing && GR.unmovable(ctx, "u64") === nothing
    @test occursin("may own memory", GR.unmovable(ctx, "Raw")) && occursin("built without a drop function", GR.unmovable(ctx, "Raw"))
    @test occursin("owns its value", GR.unmovable(ctx, "Box<u8>"))
    answered = GR.Ctx(Dict{String,Any}("types" => types,
        "glue" => Dict{String,Any}("Raw" => Dict{String,Any}("drop" => "__rb_drop_1", "owns" => false, "spelling" => "Raw"),
                                   "Boxed" => Dict{String,Any}("drop" => "__rb_drop_2", "owns" => true, "spelling" => "Boxed")),
        "glue_refused" => Dict{String,Any}("Nested" => "no path names it")))
    @test GR.unmovable(answered, "Raw") === nothing && GR.unmovable(answered, "Boxed") === nothing
    @test GR.classify(answered, "Raw").kind === :blob && GR.classify(answered, "Boxed").kind === :owned
    @test occursin("the build has no drop function for it: no path names it", GR.unmovable(answered, "Nested"))
end

@testset "Rust wrapper: what Rust must drop" begin
    md = RepliBuild._read_json(rust_fixture("owned_o3.metadata.json"))
    (text, report) = GR.generate(md; module_name = "Owned", library_path = "/somewhere/librb_fix.so", generator = "test")
    ex = Meta.parseall(text)
    @test !any(a -> a isa Expr && a.head in (:incomplete, :error), ex.args)
    reasons = Dict(r["name"] => r["reason"] for r in report if r["tier"] == "unsupported")
    reasons_all = copy(reasons)
    delete!(reasons, "Guard_Drop_drop")
    delete!(reasons, "text_callback")
    # Beside a destructor, the one function left out takes the one type nothing can name or get.
    @test collect(keys(reasons)) == ["sealed_len"]
    @test occursin("moves a `rb_fix::hidden::Sealed`, which may own memory", reasons["sealed_len"]) &&
          occursin("no path names `rb_fix::hidden::Sealed` from outside the crate", reasons["sealed_len"]) &&
          occursin("no function with a public path returns one", reasons["sealed_len"])
    # A type no path spells is an owning object like any other; its drop function is the one the
    # compiler picked, called where the library's static says it is.
    @test occursin("return Secret(_RbRaw(), ", body(text, "secret")) && occursin("_rb_take!(a1::Secret)", body(text, "secret_take"))
    at = md["glue"]["rb_fix::hidden::Secret"]["drop"]
    @test occursin("$(at[2:end])(p::Base.Ptr{Base.Cvoid}) = ccall(Base.unsafe_load(Base.cglobal((:$at, LIBRARY_PATH), Base.Ptr{Base.Cvoid})), Base.UInt, ", text)
    @test occursin("Base.throw(RepliBuild.RustRuntime.RustError(Fault(_RbRaw(), ", body(text, "parse_byte"))
    @test occursin("function fault_text(a1::Base.Union{Fault,Base.Ptr})", body(text, "fault_text"))

    # A `String` out is a Julia String: copied, and Rust's dropped there.
    @test occursin("return _rb_string_take!(_ret_p)", body(text, "make_greeting"))
    # By value in, it is built by Rust from the Julia string, for the callee to drop.
    owned = body(text, "string_len_owned")
    @test occursin("function string_len_owned(a1::Base.Union{Base.AbstractString,RustString})", owned)
    @test occursin("_rb_string_new!(a1_p, a1)", owned)
    # A shared reference takes the object that holds one, or a Julia string as one made for the call.
    ref = body(text, "string_ref_len")
    @test occursin("function string_ref_len(a1::Base.Union{RustString,Base.AbstractString,Base.Ptr})", ref)
    @test occursin("a1_r = a1 isa Base.AbstractString ? RustString(a1) : a1", ref)
    @test occursin("a1 isa Base.AbstractString && RepliBuild.RustRuntime.drop!(a1_r)", ref)
    # A mutable reference must be to a String that stays.
    @test occursin("function shout(a1::Base.Union{RustString,Base.Ptr})", body(text, "shout"))
    # A string is checked before it is Rust's.
    @test occursin("Base.isvalid(a1_s) || Base.throw(Base.ArgumentError(", body(text, "parse_int"))

    # A `Vec` of plain data is copied whole and dropped; of strings, read element by element.
    squares = body(text, "squares")
    @test occursin("RepliBuild.RustRuntime.vec_copy(Base.Float64, _rb_vec_view_", squares)
    @test occursin(r"_rb_drop_\d+\(_q\); _v end\)", squares)
    @test occursin("RepliBuild.RustRuntime.vec_copy(Rec, _rb_vec_view_", body(text, "recs"))
    @test occursin("RepliBuild.RustRuntime.vec_read(Base.String, _e -> _rb_string_at(_e), _rb_vec_view_", body(text, "words"))
    @test occursin("function vec_sum(a1::Base.Union{Base.AbstractVector,Vec_f64})", body(text, "vec_sum"))
    @test occursin("function vec_push(a1::Base.Union{Vec_f64,Base.Ptr}, a2::Base.Real)", body(text, "vec_push"))

    # `Result`: the value, or the error thrown. `Option` and a tuple: taken apart.
    parse_int = body(text, "parse_int")
    @test occursin(r"\? Base\.unsafe_load\(Base\.Ptr\{Base\.Int32\}\(_ret_p( \+ \d+)?\)\) : Base\.throw\(RepliBuild\.RustRuntime\.RustError\(_rb_string_take!\(_ret_p\)\)\)\)", parse_int)
    @test occursin("? nothing : Base.throw(RepliBuild.RustRuntime.RustError(nothing)))", body(text, "check"))
    @test occursin(r"== Base\.UInt64\(0x[0-9a-f]+\) \? nothing : _rb_string_take!\(_ret_p\)\)", body(text, "find_word"))
    @test occursin(r"return \(_rb_string_take!\(_ret_p( \+ \d+)?\), Base\.unsafe_load\(Base\.Ptr\{Base\.UInt64\}", body(text, "name_and_len"))

    # A `Box` is an object holding the pointer: returned, moved in, and passed where `&Rec` is taken.
    @test occursin("return Box_Rec(_RbRaw(), ", body(text, "boxed"))
    @test occursin("_rb_take!(a1::Box_Rec)", body(text, "unbox_d"))
    box_d = body(text, "box_d")
    @test occursin("function box_d(a1::Base.Union{Rec,Base.RefValue{Rec},Base.Ptr,Box_Rec})", box_d)
    @test occursin("(a1_r isa Box_Rec ? _rb_target(a1_r) : _rb_address(a1_r))", box_d)
    @test occursin(" == 0 ? nothing : Box_Rec(_RbRaw(), ", body(text, "opt_box"))
    @test occursin("return Slot(_RbRaw(), ", body(text, "slot_make")) && occursin("_rb_take!(a1::Slot)", body(text, "slot_take"))
    @test occursin("function slot_d(a1::Base.Union{Slot,Base.Ptr})", body(text, "slot_d"))

    # The types. What is dropped is the compiler's answer, not the shape's: `Guard` is one
    # integer with drop code, `View` a raw pointer without.
    @test occursin("mutable struct Named\n    _rb_bytes::Base.NTuple{32,Base.UInt8}\n    _rb_owned::Base.Bool\n", text)
    @test occursin("mutable struct Guard\n", text) && occursin("\nstruct View\n", text) && occursin("\nstruct Rec\n", text)
    @test count("Base.finalizer(RepliBuild.RustRuntime.drop!, x)", text) == count("\nmutable struct ", text) > 5
    @test occursin("f === :name && return _rb_string_at(_p)", text) && occursin("Base.propertynames(::Named) = (:name, :id,)", text)
    @test occursin("function RustString(s::Base.AbstractString)", text) && occursin("Base.String(x::RustString) = ", text)
    @test occursin("function Vec_f64(v::Base.AbstractVector)", text) && occursin("Base.collect(x::Vec_f64) = ", text)
    @test occursin("Base.getindex(x::Box_Rec) = ", text) && occursin("_rb_target(x::Box_Rec) = ", text)
    @test occursin("function RepliBuild.RustRuntime.variant(x::Slot)", text)
    # Each drop is the build's function for that type, called as the C function it is.
    for key in ("rb_fix::Named", "alloc::string::String", "alloc::boxed::Box<rb_fix::Rec, alloc::alloc::Global>")
        sym = md["glue"][key]["drop"]
        @test occursin("$(sym[2:end])(p::Base.Ptr{Base.Cvoid}) = ccall((:$sym, LIBRARY_PATH), Base.UInt, ", text)
    end
    @test !occursin("$(md["glue"]["rb_fix::Rec"]["drop"][2:end])(p::", text)          # plain data has nothing to call
    @test occursin(":Named => \"rb_fix::Named\"", text) && occursin(":Token => \"rb_fix::inner::Token\"", text)
    @test !occursin(":View =>", text) && !occursin(":Rec =>", text)
    # A type's name does not carry the default allocator, and Rust's String is not named `String`.
    @test occursin("mutable struct Vec_f64\n", text) && occursin("mutable struct Box_Rec\n", text)
    @test !occursin("Vec_f64_Global", text) && !occursin(r"struct String\b", text)

    # A fieldless enum is stored as the integer the compiler says: the standard library's
    # `Ordering` is a signed byte, and an unsigned Julia enum of it would not load.
    @test occursin("@enum Ordering::Base.Int8 begin\n    Ordering_Less = -1\n    Ordering_Equal = 0\n    Ordering_Greater = 1\nend", text)
    @test occursin("@enum Level::Base.UInt8 begin", text)
    @test occursin("Base.reinterpret(Base.UInt8, Base.Int8(Base.Integer(a1)))", body(text, "ordering_sign"))
    # Every type and enum of the module evaluates: `@enum` checks its values when it is expanded.
    types_text = text[findfirst("# ── Types", text).start:findfirst("# ── What Rust must drop", text).start-1]
    sandbox = Module(:OwnedTypes)
    Core.eval(sandbox, :(import RepliBuild; struct _RbRaw end; _rb_address(x) = x))
    evaluated = 0
    for ex in Meta.parseall(types_text).args
        ex isa LineNumberNode && continue
        Core.eval(sandbox, ex)
        evaluated += 1
    end
    @test evaluated > 50 && Int(getfield(sandbox, :Ordering_Less)) == -1 && isdefined(sandbox, :Named)
    @test GR._enum_value(255, "Base.Int8", 1) == -1 && GR._enum_value(-1, "Base.UInt8", 1) == 255 && GR._enum_value(3, "Base.Int8", 1) == 3

    # `[T; N]` is read and written element by element, through a place bound before the
    # elements are reached, so an array of arrays reads through its own.
    @test occursin("(let _a = _ret_p; Base.ntuple(_i -> Base.unsafe_load(Base.Ptr{Base.UInt8}((_a + (_i - 1) * 1))), Base.Val(16)) end)",
                   body(text, "bytes16"))
    @test occursin("(let _a = _ret_p; Base.ntuple(_i -> (let _a = (_a + (_i - 1) * 8); Base.ntuple(_i -> ", body(text, "grid"))
    sum16 = body(text, "bytes16_sum")
    @test occursin("function bytes16_sum(a1::Base.Union{Base.Tuple,Base.AbstractVector})", sum16)
    @test occursin("Base.length(_t) == 16 || Base.throw(Base.ArgumentError(", sum16)
    # A slice of plain records is a vector of them.
    @test occursin("function recs_total(a1::Base.DenseVector{Rec})", body(text, "recs_total"))

    # A function the library has no symbol for is called where the library's static says it is:
    # a plain call through that address, or its thunk, which is given the address under the function's name.
    small = body(text, "small")
    address = only(f for f in md["functions"] if f["path"] == "rb_fix::small")["address"]
    @test occursin("ccall(Base.unsafe_load(Base.cglobal((:$address, LIBRARY_PATH), Base.Ptr{Base.Cvoid})), Base.UInt32, (Base.UInt32,), ", small)
    @test !occursin("4small, LIBRARY_PATH", small)
    @test occursin("RepliBuild.RustRuntime.thunk(_THUNKS[", body(text, "sum_diff")) && !occursin("cglobal", body(text, "sum_diff"))
    @test occursin(", true, _slots, _ret_p)", body(text, "divide"))                    # it may panic: the catching frame
    @test occursin("function Named_rename(a1::Base.Union{Named,Base.Ptr}, a2::Base.AbstractString)", body(text, "Named_rename"))
    @test occursin("return Tag(_RbRaw(), ", body(text, "Tag_Clone_clone"))             # a clone is a value of its own
    @test occursin("function midpoint_f64(a1::Base.Real, a2::Base.Real)", body(text, "midpoint_f64"))
    @test occursin("(an inline function: the library has it because the build named it)", text)
    @test occursin("(compiled for these types because `[rust] instantiate` asks for it)", text)
    # A function the library exports is still called by its name.
    @test occursin(r"ccall\(\(:_R\w+3add, LIBRARY_PATH\), ", body(text, "add"))
    # A destructor is in the library, and nothing may call it by name.
    @test occursin("Rust does not let a destructor be called directly", reasons_all["Guard_Drop_drop"])

    # A `static` is read where it is, each time it is asked for; one that may change says so.
    @test occursin("LIMIT_ptr() = Base.cglobal((:", text) && occursin(", LIBRARY_PATH), Base.UInt32)", text)
    @test occursin("LIMIT() = (let _p = Base.Ptr{Base.Cvoid}(LIMIT_ptr()); Base.unsafe_load(Base.Ptr{Base.UInt32}(_p)) end)", text)
    @test occursin("MOTTO() = (let _p = Base.Ptr{Base.Cvoid}(MOTTO_ptr()); Base.unsafe_string(", text)
    @test occursin("Where the Rust static `rb_fix::TICKS` (`u64`) is in the library. It may change", text)
    @test occursin(":LIMIT => \"rb_fix::LIMIT\"", text) && occursin("const STATICS = ", text)

    # A reference to a trait object is two pointers, written by the helper for its trait from what it
    # is given: a value of a type that implements the trait (the library writes it), or a `Box` of one.
    size = body(text, "dyn_size")
    @test occursin("function dyn_size(a1)", size) && occursin("a1_k = _rb_dyn_Scale(a1)", size)
    @test occursin("_rb_dyn_Scale!(a1_p, a1_k)", size)
    @test occursin("a1_k = _rb_dyn_mut_Scale(a1)", body(text, "dyn_grow"))
    for (key, jl) in (("rb_fix::F2", "Base.RefValue{F2}"), ("rb_fix::Named", "Named"), ("f64", "Base.RefValue{Base.Float64}"))
        sym = md["dyn"]["dyn rb_fix::Scale"]["implementors"][key]
        @test occursin("_rb_dyn_Scale!(p::Base.Ptr{Base.Cvoid}, k::$jl) = (ccall((:$sym, LIBRARY_PATH), ", text)
    end
    @test occursin("_rb_dyn_Scale(x::F2) = Base.RefValue{F2}(x)", text) && occursin("_rb_dyn_Scale(x::Box_dyn_Scale) = x", text)
    # A mutable reference changes what it is given: a plain value is refused, a `Ref` or an owner is not.
    @test occursin("_rb_dyn_mut_Scale(x::F2) = Base.error(", text) && !occursin("_rb_dyn_mut_Scale(x::Named) = Base.error(", text)
    # A trait's method called through a trait object is the trait's, and takes what a trait object is made from.
    @test occursin("function Scale_label(a1)", body(text, "Scale_label")) && occursin("function Area_area(a1)", body(text, "Area_area"))
    @test occursin("(the trait's method, called on a trait object)", text)

    # A callback. Julia's function is made a C function; an error in it is kept and thrown after
    # the call. A Rust function pointer is the library's own function of that signature, which
    # calls on to it: what that calls is set around the call, and put back whatever happens.
    apply = body(text, "apply")
    cb = md["callbacks"]["fn(i32) -> i32"]
    @test occursin("function apply(a1::Base.Union{Base.Function,Base.Ptr}, a2::Base.Integer)", apply)
    @test occursin("a1_c = a1 isa Base.Ptr ? a1 : Base.@cfunction(\$(a1_w), Base.Int32, (Base.Int32,))", apply)
    @test occursin("a1_o = ccall((:$(cb["set"]), LIBRARY_PATH), Base.Ptr{Base.Cvoid}, (Base.Ptr{Base.Cvoid},), a1_f)", apply)
    @test occursin("Base.unsafe_load(Base.cglobal((:$(cb["at"]), LIBRARY_PATH), Base.Ptr{Base.Cvoid}))", apply)
    @test occursin("    try\n", apply) && occursin("    finally\n        ccall((:$(cb["set"]), LIBRARY_PATH), Base.Ptr{Base.Cvoid}, (Base.Ptr{Base.Cvoid},), a1_o)\n        Base.unlock(_RB_CALLBACKS)\n    end\n", apply)
    @test occursin("a1_e[] === nothing || Base.throw(a1_e[])", apply) && occursin("Base.lock(_RB_CALLBACKS)", apply)
    # A C function pointer is called as C calls: Julia's is passed as it is, with nothing set.
    c = body(text, "apply_c")
    @test occursin("Base.@cfunction(\$(a1_w), Base.Float64, (Base.Float64, Base.Float64))", c) && !occursin("_RB_CALLBACKS", c) && !occursin("finally", c)
    # What the callback is given is what the library passes: a record by reference is a pointer to one.
    @test occursin("a2(b1, Base.Ptr{Rec}(b2))", body(text, "count_if")) && occursin("Base.Bool, (Base.UInt32, Base.Ptr{Base.Cvoid})", body(text, "count_if"))
    @test occursin("Base.Cvoid, (Base.UInt32,))", body(text, "each"))
    @test occursin("is a callback that takes a `&str`", reasons_all["text_callback"])

    # Two functions with the same code keep their own types in this build.
    @test occursin("function flip(a1::Side)", body(text, "flip")) && occursin("function flag(a1::Base.Bool)", body(text, "flag"))

    # A library built before there were drop functions: nothing that may own is moved, and each says so.
    old = deepcopy(md)
    foreach(k -> delete!(old, k), ("glue", "glue_refused", "glue_helpers"))
    (old_text, old_report) = GR.generate(old; module_name = "Owned", library_path = "/somewhere/librb_fix.so")
    old_reasons = Dict(r["name"] => r["reason"] for r in old_report if r["tier"] == "unsupported")
    for name in ("make_greeting", "string_len_owned", "squares", "vec_sum", "boxed", "unbox_d", "named", "slot_make", "view", "parse_int")
        @test occursin("built without a drop function", get(old_reasons, name, "wrapped"))
    end
    @test !haskey(old_reasons, "rec_sum_ref") && !haskey(old_reasons, "flag") && !haskey(old_reasons, "check")
    @test !occursin("mutable struct", old_text) && !occursin("_rb_take!", old_text)
    @test !any(a -> a isa Expr && a.head in (:incomplete, :error), Meta.parseall(old_text).args)
end
