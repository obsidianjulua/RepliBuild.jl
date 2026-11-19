# The Rust front end, without a toolchain: what a crate's functions are and how
# each one is called, read off vendored text.
#
#   test/fixtures/rust/lean.rs           the source
#   test/fixtures/rust/lean_o<N>.dump    GNU readelf's dump of the crate's compile
#                                        unit, and of one unit of the standard library
#   test/fixtures/rust/lean_o<N>.ir.txt  the `define`, `attributes` and alias lines
#   test/fixtures/rust/lean_o<N>.exports the dynamic symbol table
#
# for an unoptimized build (N = 0) and an optimized one (N = 3), which describe
# the same functions differently (test/tools/gen_rust_fixtures.jl writes them).
# The claim under test is the front end's whole premise: the call is read from
# the compiler's own `define` line and matched to DWARF's parameters, and a
# function that does not match in exactly one way is refused.

using Test
using RepliBuild

isdefined(@__MODULE__, :rust_extract) || include(joinpath(@__DIR__, "support", "rust_fixture.jl"))

@testset "Rust ABI: v0 symbol roots" begin
    r = RA.v0_root("_RNvCs9Nq2ISRkAlj_5probe11checked_div")
    @test r !== nothing && r.crate == "probe" && r.disambiguator == "9Nq2ISRkAlj" && !r.generic && r.namespace == 'v'
    # A method: the path goes through the impl block, whose own path comes first.
    r = RA.v0_root("_RNvMCs9Nq2ISRkAlj_5probeNtB2_3Acc3new")
    @test r !== nothing && r.crate == "probe" && r.namespace == 'v'
    # A trait impl in the crate.
    r = RA.v0_root("_RNvXs_Cs1hrDFCe5rCe_4leanNtB4_5ShapeNtB4_4Area4area")
    @test r !== nothing && r.crate == "lean"
    # An instantiation is rooted where the generic is DEFINED, not where it was
    # instantiated: the crate's name is in this symbol, at the end.
    r = RA.v0_root("_RINvNtCs4NRVxsYgnAr_4core3ptr13drop_in_placeNtNtCscdodAO9FK5_5alloc6string6StringECsadWKTVoPK0X_7rbprobe")
    @test r !== nothing && r.crate == "core" && r.generic
    # A trait's provided method on some type (`Y`): the trait's crate, past the type.
    r = RA.v0_root("_RNvYNtNtCscdodAO9FK5_5alloc6string6StringNtNtCs4NRVxsYgnAr_4core3fmt5Write9write_fmtCsadWKTVoPK0X_7rbprobe")
    @test r !== nothing && r.crate == "core"
    r = RA.v0_root("_RINvYINtNtNtCs4NRVxsYgnAr_4core5slice4iter4IterdENtNtNtNtBa_4iter6traits8iterator8Iterator3sumdECsadWKTVoPK0X_7rbprobe")
    @test r !== nothing && r.crate == "core" && r.generic
    # A closure is not a function path.
    r = RA.v0_root("_RNCNvCs9Nq2ISRkAlj_5probe7rec_sum0B3_")
    @test r !== nothing && r.crate == "probe" && r.namespace == 'C'
    # Not v0, and never guessed at.
    @test RA.v0_root("_ZN5probe3add17h0123456789abcdefE") === nothing
    @test RA.v0_root("c_add") === nothing
    @test RA.v0_root("_R") === nothing
    @test RA.v0_root("_RNvCs9Nq2ISRkAlj_99probe") === nothing        # a length past the end
end

@testset "Rust ABI: define lines" begin
    mktempdir() do dir
        ll = joinpath(dir, "m.ll")
        write(ll, """
            define void @_RNvCsX_1m8rec_make(ptr dead_on_unwind noalias noundef writable sret([16 x i8]) align 8 captures(none) dereferenceable(16) %_0, i8 noundef %a, double noundef %b) unnamed_addr #0 !dbg !7 {
            define noundef zeroext i1 @_RNvCsX_1m4flag(i1 noundef zeroext %b) unnamed_addr #1 {
            define { i32, i32 } @_RNvCsX_1m5maybe(i32 noundef %x) unnamed_addr #1 personality ptr @rust_eh_personality !dbg !9 {
            define internal fastcc range(i64 0, 10) i64 @_RNvCsX_1m6helper(ptr align 8 %0, i64 %1) unnamed_addr #0 {
            define align 8 ptr @_RNvCsX_1m5boxed(i32 %d) unnamed_addr #1 {
            define i32 @printf_like(ptr %fmt, ...) #1 {
            declare void @_RNvCsX_1m8external(i32)
            @c_add = unnamed_addr alias i32 (i32, i32), ptr @_RNvCsX_1m3add
            @second = alias i32 (i32, i32), ptr @c_add
            @DATA = constant i32 7
            attributes #0 = { nounwind nonlazybind uwtable }
            attributes #1 = { nonlazybind uwtable }
            """)
        defs = RA.definitions([ll])
        @test sort!(collect(keys(defs))) == ["_RNvCsX_1m4flag", "_RNvCsX_1m5boxed", "_RNvCsX_1m5maybe",
                                             "_RNvCsX_1m6helper", "_RNvCsX_1m8rec_make", "printf_like"]
        d = defs["_RNvCsX_1m8rec_make"]
        @test d.ret_type == "void" && d.nounwind && isempty(d.cc)
        @test [p.type for p in d.params] == ["ptr", "i8", "double"]
        @test [p.name for p in d.params] == ["%_0", "%a", "%b"]
        @test "sret([16 x i8])" in d.params[1].attrs && "align 8" in d.params[1].attrs &&
              "dereferenceable(16)" in d.params[1].attrs
        d = defs["_RNvCsX_1m4flag"]
        @test d.ret_type == "i1" && "zeroext" in d.ret_attrs && "zeroext" in d.params[1].attrs && !d.nounwind
        d = defs["_RNvCsX_1m5maybe"]
        @test d.ret_type == "{ i32, i32 }" && d.personality == "rust_eh_personality"
        d = defs["_RNvCsX_1m6helper"]
        @test d.cc == "fastcc" && "internal" in d.linkage && d.ret_type == "i64" && d.nounwind
        @test d.ret_attrs == ["range(i64 0, 10)"]
        @test defs["_RNvCsX_1m5boxed"].ret_attrs == ["align 8"] && defs["_RNvCsX_1m5boxed"].ret_type == "ptr"
        @test defs["printf_like"].variadic && length(defs["printf_like"].params) == 1

        al = RA.aliases([ll])
        @test al == Dict("c_add" => "_RNvCsX_1m3add", "second" => "c_add")
        # An alias ends at a definition, through other aliases, or nowhere.
        @test RA.alias_target("second", Dict("_RNvCsX_1m3add" => 1), al) == "_RNvCsX_1m3add"
        @test RA.alias_target("second", defs, al) === nothing
    end
end

@testset "Rust ABI: the surface is the crate's own exported functions" begin
    for opt in (0, 3)
        x = rust_extract(opt)
        @test length(x.symbols) == 31
        @test isempty(x.skipped)
        @test issorted(x.symbols)
        # Every one is exported, and is the crate's by its path or is unmangled.
        @test all(s -> s in x.exported, x.symbols)
        @test sort!([s for s in x.symbols if !startswith(s, "_R")]) == ["c_add", "cpoint_norm"]
        @test all(s -> RA.v0_root(s).crate == "lean", [s for s in x.symbols if startswith(s, "_R")])
        # The fixture's export list also has names the IR does not define (the
        # standard library's): they are not the crate's, whatever they are.
        @test count(s -> startswith(s, "_R") && RA.v0_root(s).crate != "lean", x.exported) >= 10
    end
    # A definition that is not exported is not API, and neither is one rooted elsewhere.
    x = rust_extract(0)
    add = only(s for s in x.symbols if endswith(s, "4lean3add"))
    (symbols, _, _) = RA.surface(x.defs, setdiff(x.exported, [add]), "lean")
    @test !(add in symbols) && length(symbols) == 30
    (symbols, _, foreign) = RA.surface(x.defs, x.exported, "other_crate")
    @test symbols == ["c_add", "cpoint_norm"] && foreign == 29
end

@testset "Rust ABI: the join, unoptimized and optimized" begin
    for opt in (0, 3)
        x = rust_extract(opt)
        row(tail) = rust_row(x, tail)
        callable = [s for s in x.symbols if x.rows[s]["refusal"] === nothing]
        @test length(callable) == 29

        # Scalars cross as themselves.
        @test passes(row("4lean3add")) == [("a", "direct", [1]), ("b", "direct", [2])]
        @test row("4lean3add")["return"]["pass"] == "direct"
        @test row("4lean4flag")["ir"]["parameters"][1]["type"] == "i1"
        @test "zeroext" in row("4lean4flag")["ir"]["parameters"][1]["attrs"]
        @test row("4lean4wide")["ir"]["return"]["type"] == "i128"
        @test passes(row("4lean9char_code")) == [("c", "direct", [1])]

        # A record over a word goes by pointer to a copy and comes back through
        # a hidden pointer, which is IR parameter 1.
        r = row("4lean8rec_make")
        @test r["return"]["pass"] == "indirect" && r["return"]["ir"] == [1] && r["return"]["type"] == "lean::Rec"
        @test passes(r) == [("a", "direct", [2]), ("b", "direct", [3]), ("c", "direct", [4]), ("d", "direct", [5])]
        @test passes(row("4lean7rec_sum")) == [("r", "indirect", [1])]
        # A reference to the same record is one pointer too, and is NOT a copy:
        # only DWARF's type tells the two apart.
        @test passes(row("4lean11rec_sum_ref")) == [("r", "direct", [1])]
        @test row("4lean11rec_sum_ref")["parameters"][1]["type"] == "&lean::Rec"
        @test row("4lean7rec_sum")["ir"]["parameters"][1]["type"] == row("4lean11rec_sum_ref")["ir"]["parameters"][1]["type"] == "ptr"

        # Two scalars are two arguments, and a pair on the way back.
        @test passes(row("4lean6f2_dot")) == [("a", "pair", [1, 2]), ("b", "pair", [3, 4])]
        @test row("4lean7f2_make")["return"]["pass"] == "pair"
        @test row("4lean10two16_swap")["ir"]["return"]["type"] == "{ i16, i16 }"
        @test row("4lean4pair")["return"]["pass"] == "pair"
        @test row("4lean5maybe")["return"]["pass"] == "pair" &&
              row("4lean5maybe")["return"]["type"] == "core::option::Option<i32>"
        # Fat pointers.
        @test passes(row("4lean9slice_sum")) == [("xs", "pair", [1, 2])]
        @test passes(row("4lean7str_len")) == [("s", "pair", [1, 2])]
        @test passes(row("4lean8dyn_area")) == [("a", "pair", [1, 2])]
        @test row("4lean5hello")["return"]["pass"] == "pair"

        # Three bytes are one integer of three bytes; a zero-sized value is nothing.
        @test passes(row("4lean6b3_rot")) == [("x", "cast", [1])] && row("4lean6b3_rot")["return"]["pass"] == "cast"
        @test row("4lean6b3_rot")["ir"]["return"]["type"] == "i24"
        z = row("4lean1z")
        @test [(p["pass"], p["ir"]) for p in z["parameters"]] == [("ignore", Int[]), ("direct", [1])]
        @test z["return"]["pass"] == "ignore"

        # Enums: a fieldless one is its integer, one with data goes by pointer,
        # and `Option<&T>` is the pointer itself.
        @test passes(row("4lean10level_next")) == [("l", "direct", [1])]
        @test row("4lean4rect")["return"]["pass"] == "indirect"
        @test passes(row("4lean9opt_ref_d")) == [("r", "direct", [1])]

        # Methods: `self` is the first parameter, and the owner is the type.
        push = row("3Acc4push")
        @test push["owner"] == "lean::Acc" && push["name"] == "push" && push["scope"] == ["lean", "Acc"]
        @test passes(push) == [("self", "direct", [1]), ("x", "direct", [2])]
        @test push["parameters"][1]["type"] == "&mut lean::Acc" && push["return"]["pass"] == "ignore"
        @test row("3Acc3new")["return"]["pass"] == "pair"
        @test row("4Area4area")["parameters"][1]["type"] == "&lean::Shape"

        # What cannot unwind says so only where the compiler proved it.
        @test row("4lean11checked_div")["nounwind"] == false
        @test row("4lean3add")["nounwind"] == (opt == 3)
    end
end

@testset "Rust ABI: a function that does not match is refused" begin
    for opt in (0, 3)
        x = rust_extract(opt)
        # `#[track_caller]`: one IR parameter DWARF does not list.
        located = rust_row(x, "4lean7located")
        @test located["refusal"] !== nothing && occursin("track_caller", located["refusal"])
        @test !haskey(located["parameters"][1], "pass") && length(located["ir"]["parameters"]) == 2
        # The crate's C API takes a record the way C does, as one `{ double, double }`.
        norm = x.rows["cpoint_norm"]
        @test norm["refusal"] !== nothing && norm["no_mangle"]
        @test norm["ir"]["parameters"][1]["type"] == "{ double, double }"
        @test sort!([s for s in x.symbols if x.rows[s]["refusal"] !== nothing]) ==
              sort!(["cpoint_norm", located["symbol"]])
    end

    x = rust_extract(0)
    sym(tail) = only(s for s in x.symbols if endswith(s, tail))
    join(tail, def) = RA.join_function(sym(tail), def, x.dwarf[sym(tail)], x.types)
    redefine(d; ret_type = d.ret_type, params = d.params, cc = d.cc, variadic = d.variadic) =
        RA.IRDefinition(d.symbol, d.linkage, cc, ret_type, d.ret_attrs, params, variadic, d.nounwind, d.personality)

    # No debug info: nothing is inferred from the IR alone.
    r = RA.join_function(sym("4lean3add"), x.defs[sym("4lean3add")], nothing, x.types)
    @test occursin("no debug info", r["refusal"]) && isempty(r["parameters"])
    # An IR parameter too few, and one too many.
    d = x.defs[sym("4lean3add")]
    @test occursin("do not match", join("4lean3add", redefine(d; params = d.params[1:1]))["refusal"])
    extra = vcat(d.params, [RA.IRParam("i64", String[], "%n")])
    @test occursin("do not match", join("4lean3add", redefine(d; params = extra))["refusal"])
    # A width the type does not have.
    wrong = [RA.IRParam("i64", String[], "%a"), d.params[2]]
    @test join("4lean3add", redefine(d; params = wrong))["refusal"] !== nothing
    @test occursin("does not match", join("4lean3add", redefine(d; ret_type = "i64"))["refusal"])
    # A hidden return pointer for another size than the return type's.
    d = x.defs[sym("4lean8rec_make")]
    bad = vcat([RA.IRParam("ptr", ["sret([24 x i8])", "align 8"], "%_0")], d.params[2:end])
    @test occursin("24 bytes", join("4lean8rec_make", redefine(d; params = bad))["refusal"])
    # A calling convention a caller outside the crate cannot use, and a variadic one.
    d = x.defs[sym("4lean3add")]
    @test occursin("fastcc", join("4lean3add", redefine(d; cc = "fastcc"))["refusal"])
    @test occursin("variadic", join("4lean3add", redefine(d; variadic = true))["refusal"])
    # A by-value copy says how many bytes it is; another count is not that record.
    d = x.defs[sym("4lean7rec_sum")]
    @test join("4lean7rec_sum", d)["refusal"] === nothing
    lied = [RA.IRParam("ptr", ["dereferenceable(40)"], "%r")]
    @test join("4lean7rec_sum", redefine(d; params = lied))["refusal"] !== nothing
    # A type two definitions disagree on has no layout, so nothing crosses by value.
    types = deepcopy(x.types)
    types["lean::Rec"]["conflict"] = true
    r = RA.join_function(sym("4lean7rec_sum"), d, x.dwarf[sym("4lean7rec_sum")], types)
    @test occursin("layout", r["refusal"])
    # ...and a reference to it still does: a pointer needs no layout.
    r = RA.join_function(sym("4lean11rec_sum_ref"), x.defs[sym("4lean11rec_sum_ref")],
                         x.dwarf[sym("4lean11rec_sum_ref")], types)
    @test r["refusal"] === nothing
end

@testset "Rust ABI: more than one way to match is a refusal" begin
    # Two 16-byte records of two pointers each, and three pointers in the IR:
    # the first as a pair and the second by pointer, or the other way round.
    ptr = Dict{String,Any}("kind" => "pointer", "pointer" => "ref", "target" => "u8", "size" => 8)
    two = Dict{String,Any}("kind" => "struct", "size" => 16, "align" => 8, "fields" => Any[
        Dict{String,Any}("name" => "a", "offset" => 0, "type" => "&u8"),
        Dict{String,Any}("name" => "b", "offset" => 8, "type" => "&u8")])
    types = Dict{String,Any}("&u8" => ptr, "T" => two,
                             "f64" => Dict{String,Any}("kind" => "base", "size" => 8, "encoding" => "float"))
    dwarf = Dict{String,Any}("name" => "f", "scope" => String[], "owner" => nothing, "return" => nothing,
                             "parameters" => Any[Dict{String,Any}("name" => "x", "type" => "T"),
                                                 Dict{String,Any}("name" => "y", "type" => "T")])
    def(types...) = RA.IRDefinition("f", String[], "", "void", String[],
                                    [RA.IRParam(t, String[], "%$i") for (i, t) in enumerate(types)],
                                    false, false, "")
    r = RA.join_function("f", def("ptr", "ptr", "ptr"), dwarf, types)
    @test r["refusal"] !== nothing && occursin("more than one way", r["refusal"])
    # Four pointers, or two, have one reading each.
    r = RA.join_function("f", def("ptr", "ptr", "ptr", "ptr"), dwarf, types)
    @test r["refusal"] === nothing && [p["pass"] for p in r["parameters"]] == ["pair", "pair"]
    r = RA.join_function("f", def("ptr", "ptr"), dwarf, types)
    @test r["refusal"] === nothing && [p["pass"] for p in r["parameters"]] == ["indirect", "indirect"]
    # A fat pointer is always its two words, which settles the same three.
    types["T"]["fat"] = "dyn"
    r = RA.join_function("f", def("ptr", "ptr", "ptr"), dwarf, types)
    @test r["refusal"] !== nothing && occursin("do not match", r["refusal"])
end

@testset "Rust ABI: one byte that is only 0 or 1 is one bit" begin
    # rustc passes such a byte as `i1`: a two-variant enum, a `Result<(), E>`
    # with nothing in either arm (what every `fmt` returns), a `bool`. DWARF
    # says "one byte"; the define line says which.
    unit = Dict{String,Any}("kind" => "struct", "size" => 0, "fields" => Any[])
    types = Dict{String,Any}(
        "u8" => Dict{String,Any}("kind" => "base", "size" => 1, "encoding" => "unsigned"),
        "f64" => Dict{String,Any}("kind" => "base", "size" => 8, "encoding" => "float"),
        "bool" => Dict{String,Any}("kind" => "base", "size" => 1, "encoding" => "boolean"),
        "Side" => Dict{String,Any}("kind" => "c_enum", "size" => 1, "enumerators" => Any[Any["L", 0], Any["R", 1]]),
        "Res::Ok" => unit, "Res::Err" => unit,
        "Res" => Dict{String,Any}("kind" => "enum", "size" => 1, "fields" => Any[],
                                  "discriminant" => Dict{String,Any}("offset" => 0, "type" => "u8", "size" => 1),
                                  "variants" => Any[Dict{String,Any}("name" => "Ok", "discr_value" => 0, "type" => "Res::Ok", "offset" => 0),
                                                    Dict{String,Any}("name" => "Err", "discr_value" => 1, "type" => "Res::Err", "offset" => 0)]),
        "Opts" => Dict{String,Any}("kind" => "struct", "size" => 16, "fields" => Any[
            Dict{String,Any}("name" => "scale", "offset" => 0, "type" => "f64"),
            Dict{String,Any}("name" => "on", "offset" => 8, "type" => "bool")]))
    dwarf(ret, params...) = Dict{String,Any}("name" => "f", "scope" => String[], "owner" => nothing, "return" => ret,
                                             "parameters" => Any[Dict{String,Any}("name" => "p$i", "type" => t) for (i, t) in enumerate(params)])
    def(ret, params...) = RA.IRDefinition("f", String[], "", ret, String[],
                                          [RA.IRParam(t, String[], "%$i") for (i, t) in enumerate(params)], false, false, "")
    join(d, w) = RA.join_function("f", d, w, types)
    # The enum, as a parameter and a return.
    r = join(def("i1", "i1"), dwarf("Side", "Side"))
    @test r["refusal"] === nothing && r["return"]["pass"] == "direct" && r["parameters"][1]["pass"] == "direct"
    # ...and as a whole byte, which is how a wider range is passed: both are the type's one byte.
    @test join(def("i8", "i8"), dwarf("Side", "Side"))["refusal"] === nothing
    # `Result<(), E>`: its one byte is the discriminant.
    r = join(def("i1"), dwarf("Res"))
    @test r["refusal"] === nothing && r["return"]["pass"] == "direct"
    # Beside a double, in a pair.
    r = join(def("{ double, i1 }"), dwarf("Opts"))
    @test r["refusal"] === nothing && r["return"]["pass"] == "pair"
    r = join(def("void", "double", "i1"), dwarf(nothing, "Opts"))
    @test r["refusal"] === nothing && r["parameters"][1]["pass"] == "pair"
    # One bit is never a wider value's call, and a wider integer is never a bool's.
    @test join(def("i1"), dwarf("f64"))["refusal"] !== nothing
    @test join(def("i16"), dwarf("Side"))["refusal"] !== nothing
    @test join(def("i8"), dwarf("bool"))["refusal"] !== nothing
end

@testset "Rust ABI: merged functions are aliases" begin
    # Unoptimized, `c_add` is a function of its own; optimized it has `add`'s
    # code, and LLVM keeps the name as an alias with no `define` and no DIE.
    x0 = rust_extract(0)
    @test isempty(x0.aliases) && haskey(x0.defs, "c_add") && haskey(x0.dwarf, "c_add")
    x3 = rust_extract(3)
    add = only(s for s in x3.symbols if endswith(s, "4lean3add"))
    @test x3.aliases == Dict("c_add" => add)
    @test !haskey(x3.defs, "c_add") && !haskey(x3.dwarf, "c_add")
    @test "c_add" in x3.symbols && x3.target["c_add"] == add
    # Its call is its target's.
    r = x3.rows["c_add"]
    @test r["refusal"] === nothing && r["symbol"] == "c_add" && r["no_mangle"]
    @test passes(r) == passes(x3.rows[add]) == [("a", "direct", [1]), ("b", "direct", [2])]
    # Without the alias table the symbol is simply not there, which is the bug this guards.
    (symbols, _, _) = RA.surface(x3.defs, x3.exported, "lean")
    @test !("c_add" in symbols) && length(symbols) == 30
end

@testset "Rust ABI: type shapes" begin
    x = rust_extract(0)
    sh(key) = RA.shape(x.types, key)
    leaves(key) = [(l.offset, l.size, l.class) for l in sh(key).leaves]
    @test leaves("lean::Rec") == [(0, 8, :float), (8, 4, :int), (12, 2, :int), (14, 1, :int)]
    @test leaves("lean::F2") == [(0, 8, :float), (8, 8, :float)]
    @test leaves("&[f64]") == [(0, 8, :ptr), (8, 8, :int)] && sh("&[f64]").fat
    @test leaves("core::option::Option<i32>") == [(0, 4, :int), (4, 4, :int)]
    # The niche: the discriminant IS the pointer, so one word that is either.
    @test leaves("core::option::Option<&lean::Rec>") == [(0, 8, :mixed)]
    @test leaves("lean::Level") == [(0, 1, :int)]
    @test sh("lean::Z").size == 0 && isempty(sh("lean::Z").leaves)
    @test sh("lean::Shape").size == 24 && length(sh("lean::Shape").leaves) == 3
    @test sh(nothing).size == 0                                  # no return type
    @test sh("no::such::Type").size == -1 && sh("no::such::Type").leaves === :unknown
    # An array past what a call could use is not walked element by element.
    types = Dict{String,Any}("u8" => x.types["u8"],
                             "[u8; 4096]" => Dict{String,Any}("kind" => "array", "element" => "u8",
                                                               "count" => 4096, "size" => 4096))
    @test RA.shape(types, "[u8; 4096]").leaves === :many
end
