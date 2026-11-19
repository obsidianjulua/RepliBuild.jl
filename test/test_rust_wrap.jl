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
    @test occursin("@enum Level::UInt8 begin\n    Level_Low = 0\n    Level_Mid = 1\n    Level_High = 2\nend", text)
    # An enum with data says which variant it is, and gives that variant's record.
    @test occursin("function RepliBuild.RustRuntime.variant(x::Shape)", text)
    @test occursin("_d == 1 && return :Rect", text)
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
    ctx = GR.Ctx(types, Dict{String,String}(), Dict{String,GR.T}())
    @test !GR.owns(ctx, "u64") && !GR.owns(ctx, "Plain") && !GR.owns(ctx, "Borrows") && !GR.owns(ctx, nothing)
    @test !GR.owns(ctx, "&u8") && !GR.owns(ctx, "*const u8")          # a pointer handed over as itself is the caller's
    @test !GR.owns(ctx, "&[u8]")                                      # a borrowed slice
    @test GR.owns(ctx, "Box<u8>") && GR.owns(ctx, "Boxed") && GR.owns(ctx, "Nested")
    @test GR.owns(ctx, "Raw")                                         # a raw pointer inside a record may be owned
    @test GR.owns(ctx, "alloc::string::String")
    @test GR.owns(ctx, "no::such::Type")                              # what is not described is not assumed plain
end
