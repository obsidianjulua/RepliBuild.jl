# What the Rust build writes per type so a value can be dropped (`RustGlue`),
# without a toolchain: which types it asks the compiler about, how it spells
# one for Rust source, the file it writes, what it reads from a failed
# compile, and what it believes of the library that was built.
#
# The compiles themselves are in test_rust_build_live.jl.

using Test
using RepliBuild

isdefined(@__MODULE__, :rust_fixture) || include(joinpath(@__DIR__, "support", "rust_fixture.jl"))

const RG = RepliBuild.RustGlue
const _RGT_RB = RepliBuild.RustBuild

@testset "RustGlue: a type's name, as source can write it" begin
    # The default allocator is always the last argument, and cannot be named.
    @test RG.source_name("alloc::vec::Vec<f64, alloc::alloc::Global>") == "alloc::vec::Vec<f64>"
    @test RG.source_name("alloc::vec::Vec<alloc::boxed::Box<a::B, alloc::alloc::Global>, alloc::alloc::Global>") ==
          "alloc::vec::Vec<alloc::boxed::Box<a::B>>"
    @test RG.source_name("a::Uses<alloc::alloc::Global, u8>") == "a::Uses<alloc::alloc::Global, u8>"
    @test RG.source_name("alloc::string::String") == "alloc::string::String"

    @test RG.paths("core::option::Option<alloc::boxed::Box<dep::Rec>>") == ["core::option::Option", "alloc::boxed::Box", "dep::Rec"]
    @test RG.paths("alloc::boxed::Box<dyn dep::Area>") == ["alloc::boxed::Box", "dep::Area"]
    @test RG.paths("dep::Pair<dep::Rec, dep::Rec>") == ["dep::Pair", "dep::Rec"]              # once each
    @test isempty(RG.paths("(i32, f64)")) && isempty(RG.paths("&[u8]")) && RG.paths("alloc::vec::Vec<&str>") == ["alloc::vec::Vec"]

    @test RG.unnameable("dep::Token") === nothing
    @test RG.unnameable("core::result::Result<i32, alloc::string::String>") === nothing
    @test RG.unnameable("alloc::boxed::Box<[u8], alloc::alloc::Global>") === nothing
    @test RG.unnameable("fn(i32) -> i32") === nothing
    # A closure, an `impl` block, a type local to a function: the compiler's own names.
    @test occursin("no name in source", RG.unnameable("dep::make::{closure_env#0}"))
    @test occursin("no name in source", RG.unnameable("dep::{impl#3}::Local"))
    @test occursin("keyword", RG.unnameable("dep::dyn::Thing"))
    @test RG.unnameable("dep::Pair<(i32, f64") !== nothing && RG.unnameable("dep::Odd>") !== nothing
end

@testset "RustGlue: the paths that may name an item from outside" begin
    @test RG.candidates("dep::Token", "dep") == ["dep::Token"]
    # A private module's item is re-exported from above: the path itself, then with less of its middle.
    @test RG.candidates("dep::inner::Token", "dep") == ["dep::inner::Token", "dep::Token"]
    c = RG.candidates("std::collections::hash::map::HashMap", "dep")
    @test length(c) == 8 && allunique(c)
    @test c[1] == "std::collections::hash::map::HashMap" && c[end] == "std::HashMap"
    @test c[2:4] == ["std::collections::hash::HashMap", "std::collections::map::HashMap", "std::hash::map::HashMap"]
    @test findfirst(==("std::collections::HashMap"), c) < findfirst(==("std::HashMap"), c)
    # Another crate's item may be re-exported by the wrapped one.
    @test RG.candidates("other::Thing", "dep") == ["other::Thing", "dep::other::Thing", "dep::Thing"]
    @test RG.candidates("core::option::Option", "dep") == ["core::option::Option", "core::Option"]
    # A long path is not tried in every way: itself, and its two ends.
    long = RG.candidates("a::b::c::d::e::f::g::T", "a")
    @test long == ["a::b::c::d::e::f::g::T", "a::T"]
    @test length(RG.candidates("x::b::c::d::e::f::T", "dep")) <= 48

    chosen = Dict("dep::inner::Token" => "dep::Token", "alloc::vec::Vec" => "alloc::vec::Vec",
                  "core::option::Option" => "core::option::Option")
    @test RG.spell("alloc::vec::Vec<dep::inner::Token, alloc::alloc::Global>", chosen) == "alloc::vec::Vec<dep::Token>"
    @test RG.spell("core::option::Option<&dep::inner::Token>", chosen) == "core::option::Option<&dep::Token>"
    @test RG.spell("alloc::vec::Vec<dep::hidden::Secret, alloc::alloc::Global>", chosen) === nothing
    @test RG.spell("(i32, f64)", chosen) == "(i32, f64)"
end

@testset "RustGlue: the file" begin
    cands = Dict("dep::inner::Token" => ["dep::inner::Token", "dep::Token"], "a::B" => ["a::B"])
    (text, lines) = RG.probe_text(cands)
    ls = split(text, '\n')
    @test startswith(ls[1], "//") && length(lines) == 3
    for (n, (_, candidate)) in lines
        @test ls[n] == "#[allow(unused_imports)] use $candidate as _;"
    end
    # One path's candidates keep their order down the file: the first that resolves is the first tried.
    @test [lines[n][2] for n in sort!(collect(keys(lines))) if lines[n][1] == "dep::inner::Token"] == cands["dep::inner::Token"]

    rows = [RG.Row(3, "dep::inner::Token", "dep::Token", false),
            RG.Row(7, "alloc::vec::Vec<f64, alloc::alloc::Global>", "alloc::vec::Vec<f64>", true)]
    (text, line_of) = RG.glue_text(rows)
    ls = split(text, '\n')
    @test count(==('\n'), text) == 3 && sort!(collect(keys(line_of))) == [3, 7]
    token = ls[line_of[3]]
    @test startswith(token, "#[no_mangle] pub unsafe extern \"C\" fn __rb_drop_3(p: *mut dep::Token, ")
    # The compiler's answer to "is this type dropped" is the length of an array in the signature.
    @test occursin("_owns: *const [u8; core::mem::needs_drop::<dep::Token>() as usize]) -> usize", token)
    # No two of these functions have the same code: each says which it is.
    @test occursin("{ if p.is_null() { 3 } else { __rb_drop(p) } }", token) && !occursin("__rb_vec", token)
    vec = ls[line_of[7]]
    @test occursin("{ if p.is_null() { 7 } else { __rb_drop(p) } }", vec)
    @test occursin("fn __rb_vec_view_7(v: *const alloc::vec::Vec<f64>, data: *mut *const u8, len: *mut usize)", vec)
    @test occursin("fn __rb_vec_new_7(data: *const u8, len: usize, out: *mut alloc::vec::Vec<f64>)", vec)
    # Nothing to ask is a file of its comment alone, and the same rows are the same text.
    @test RG.glue_text(RG.Row[])[1] == ls[1] * "\n" && RG.glue_text(rows)[1] == text
    @test (RG.drop_symbol(12), RG.vec_view_symbol(12), RG.vec_new_symbol(12)) == ("__rb_drop_12", "__rb_vec_view_12", "__rb_vec_new_12")
end

@testset "RustGlue: a function the library has no symbol for" begin
    # A path as the demangler prints it is an expression source can write, or it is not one.
    @test RG.function_expression("dep::inner::deep") == "dep::inner::deep"
    @test RG.function_expression("<dep::Rec>::dsum") == "<dep::Rec>::dsum"
    @test RG.function_expression("<dep::Rec as core::clone::Clone>::clone") == "<dep::Rec as core::clone::Clone>::clone"
    @test RG.function_expression("<alloc::vec::Vec<dep::Rec, alloc::alloc::Global> as dep::Sum>::total") ==
          "<alloc::vec::Vec<dep::Rec> as dep::Sum>::total"
    @test RG.function_expression("dep::uses::{closure#0}") === nothing
    # Rust lets nothing call a destructor by name.
    @test RG.function_expression("<dep::Guard as core::ops::drop::Drop>::drop") === nothing
    # Its paths are chosen like a type's: the method stays, the type takes its public path.
    chosen = Dict("dep::inner::Token" => "dep::Token", "dep::inner::deep" => "dep::deep", "core::clone::Clone" => "core::clone::Clone")
    @test RG.spell("<dep::inner::Token>::id2", chosen) == "<dep::Token>::id2"
    @test RG.spell("dep::inner::deep", chosen) == "dep::deep"
    @test RG.spell("<dep::inner::Token as core::clone::Clone>::clone", chosen) == "<dep::Token as core::clone::Clone>::clone"

    # One line per function, after the types' lines: naming it makes the compiler build it, and the static holds where.
    rows = [RG.Row(1, "dep::Rec", "dep::Rec", false)]
    fns = [RG.FnRow(4, "dep::small", "inline", "_RNvCsX_3dep5small"), RG.FnRow(9, "dep::largest::<f64>", "instantiate", "")]
    (text, line_of, fn_line_of) = RG.glue_text(rows, fns)
    ls = split(text, '\n')
    @test ls[fn_line_of[4]] == "#[no_mangle] pub static __rb_fn_4: __RbAddr = __RbAddr((dep::small) as *const ());"
    @test ls[fn_line_of[9]] == "#[no_mangle] pub static __rb_fn_9: __RbAddr = __RbAddr((dep::largest::<f64>) as *const ());"
    @test line_of[1] < fn_line_of[4] < fn_line_of[9] && count(==('\n'), text) == 4
    @test RG.glue_text(rows) == (RG.glue_text(rows, RG.FnRow[])[1], line_of)        # the types alone, as before
    @test RG.fn_symbol(4) == "__rb_fn_4"

    # Which function each static points at is the root crate's IR's word, at either optimization level.
    mktempdir() do dir
        ir = joinpath(dir, "root.ll")
        write(ir, """
            @__rb_fn_1 = local_unnamed_addr constant ptr @_RNvCs6MEz9VGGpyc_3dep5small, align 8, !dbg !0
            @__rb_fn_10 = constant ptr @_RINvCs7I8noyAdbh6_3dep7largestdECs5ovRDRVLBmB_6dep_rb, align 8, !dbg !11
            @__rb_other = constant ptr @not_one, align 8
            @__rb_fn_3 = constant i64 7, align 8
            define internal noundef i32 @_RNvCs6MEz9VGGpyc_3dep5small(i32 noundef %x) unnamed_addr #0 {
            """)
        @test RG.address_table([ir]) == Dict("__rb_fn_1" => "_RNvCs6MEz9VGGpyc_3dep5small",
                                             "__rb_fn_10" => "_RINvCs7I8noyAdbh6_3dep7largestdECs5ovRDRVLBmB_6dep_rb")
        @test isempty(RG.address_table(String[]))
    end
end

@testset "RustGlue: what a failed compile said" begin
    msg(level, text, file, line; rendered = "$level: $text") =
        "{\"reason\":\"compiler-message\",\"message\":{\"level\":\"$level\",\"message\":\"$text\",\"rendered\":\"$rendered\"," *
        "\"spans\":[{\"file_name\":\"$file\",\"line_start\":$line,\"is_primary\":true}]}}"
    messages = join([
        msg("error", "module `inner` is private", "rb_glue.rs", 2),
        msg("error", "a second error on one line", "rb_glue.rs", 2),
        msg("error", "unresolved import `other`", "/abs/build/rust/root/rb_glue.rs", 5),
        msg("warning", "unused import", "rb_glue.rs", 3),
        msg("error", "mismatched types", "src/lib.rs", 1; rendered = "error[E0308]: mismatched types"),
        "{\"reason\":\"compiler-message\",\"message\":{\"level\":\"error\",\"message\":\"aborting due to 3 previous errors\",\"rendered\":\"error: aborting\",\"spans\":[]}}",
        "{\"reason\":\"compiler-artifact\",\"target\":{\"name\":\"dep\"}}",
        "   Compiling dep v0.1.0",
        "{ not json"], "\n")
    (bad, other) = RG.error_lines(messages)
    # A line's first error is its reason; a warning is not one; the file is told by its name.
    @test bad == Dict(2 => "module `inner` is private", 5 => "unresolved import `other`")
    @test other == ["error[E0308]: mismatched types"]
    rendered = RG.rendered_errors(messages)
    @test occursin("module `inner` is private", rendered) && occursin("E0308", rendered) && !occursin("unused import", rendered)
    @test RG.error_lines("") == (Dict{Int,String}(), String[]) && RG.rendered_errors("") == ""
end

@testset "RustGlue: what the built library says" begin
    ptr(target) = Dict{String,Any}("kind" => "pointer", "target" => target)
    types = Dict{String,Any}(
        "dep::inner::Token" => Dict{String,Any}("kind" => "struct"), "*mut dep::inner::Token" => ptr("dep::inner::Token"),
        "dep::Plain" => Dict{String,Any}("kind" => "struct"), "*mut dep::Plain" => ptr("dep::Plain"),
        "dep::Other" => Dict{String,Any}("kind" => "struct"), "*mut dep::Other" => ptr("dep::Other"),
        "V" => Dict{String,Any}("kind" => "struct"), "*mut V" => ptr("V"),
        "V7" => Dict{String,Any}("kind" => "struct"), "*mut V7" => ptr("V7"),
        "[u8; 1]" => Dict{String,Any}("kind" => "array", "count" => 1), "*const [u8; 1]" => ptr("[u8; 1]"),
        "[u8; 0]" => Dict{String,Any}("kind" => "array", "count" => 0), "*const [u8; 0]" => ptr("[u8; 0]"),
        "[u8; 2]" => Dict{String,Any}("kind" => "array", "count" => 2), "*const [u8; 2]" => ptr("[u8; 2]"))
    fn(p, o) = Dict{String,Any}("parameters" => Any[Dict{String,Any}("name" => "p", "type" => p),
                                                    Dict{String,Any}("name" => "_owns", "type" => o)])
    dwarf = Dict{String,Any}(
        "__rb_drop_1" => fn("*mut dep::inner::Token", "*const [u8; 1]"),
        "__rb_drop_2" => fn("*mut dep::Plain", "*const [u8; 0]"),
        "__rb_drop_3" => fn("*mut dep::Other", "*const [u8; 1]"),         # the spelling named another type
        "__rb_drop_4" => fn("*mut dep::Plain", "*const [u8; 1]"),         # not exported
        "__rb_drop_6" => fn("*mut V", "*const [u8; 1]"),
        "__rb_drop_7" => fn("*mut V7", "*const [u8; 2]"),                  # an answer that is neither
        "__rb_drop_8" => fn("*mut V", "*const [u8; 1]"))
    exported = Set(["__rb_drop_1", "__rb_drop_2", "__rb_drop_3", "__rb_drop_5", "__rb_drop_6", "__rb_drop_7", "__rb_drop_8",
                    "__rb_vec_view_6", "__rb_vec_new_6", "__rb_vec_view_8"])
    plan = RG.Plan(["a"], [RG.Row(1, "dep::inner::Token", "dep::Token", false), RG.Row(2, "dep::Plain", "dep::Plain", false),
                           RG.Row(3, "dep::Wanted", "dep::Other", false), RG.Row(4, "dep::Gone", "dep::Gone", false),
                           RG.Row(5, "dep::NoDie", "dep::NoDie", false), RG.Row(6, "V", "V", true),
                           RG.Row(7, "V7", "V7", false), RG.Row(8, "V", "V", true)],
                   Dict("dep::hidden::Secret" => "no path names it"))
    plan8 = RG.Plan(plan.wanted, [r for r in plan.rows if r.n != 6], plan.refused)
    (glue, refused) = RG.verify(RG.Plan(plan.wanted, [r for r in plan.rows if r.n != 8], plan.refused), dwarf, types, exported)
    @test glue["dep::inner::Token"] == Dict{String,Any}("drop" => "__rb_drop_1", "owns" => true, "spelling" => "dep::Token")
    @test glue["dep::Plain"]["owns"] === false && glue["dep::Plain"]["drop"] == "__rb_drop_2"
    @test glue["V"]["vec"] == Dict{String,Any}("view" => "__rb_vec_view_6", "new" => "__rb_vec_new_6")
    @test sort!(collect(keys(glue))) == ["V", "dep::Plain", "dep::inner::Token"]
    # A refusal for each row that is not believed, and the plan's own kept.
    @test occursin("names another type (`dep::Other`)", refused["dep::Wanted"])
    @test occursin("is not in the library", refused["dep::Gone"])
    @test occursin("no debug info", refused["dep::NoDie"])
    @test occursin("was not read", refused["V7"])
    @test refused["dep::hidden::Secret"] == "no path names it"
    # A `Vec` whose two functions are not both in the library is dropped like any other type, and not read or built.
    (glue8, _) = RG.verify(plan8, dwarf, types, exported)
    @test glue8["V"]["drop"] == "__rb_drop_8" && !haskey(glue8["V"], "vec")
    @test RG.verify(RG.Plan(), dwarf, types, exported) == (Dict{String,Any}(), Dict{String,String}())
end

@testset "RustGlue: which types the build asks about" begin
    md = RepliBuild._read_json(rust_fixture("owned_o3.metadata.json"))
    types = md["types"]
    need = RG.wanted(md["functions"], types)
    @test issorted(need) && allunique(need) && length(need) >= 15
    # Every one of them has the compiler's answer, or the reason it has none.
    @test Set(need) == union(Set(keys(md["glue"])), Set(keys(md["glue_refused"])))
    for key in ("alloc::string::String", "alloc::vec::Vec<f64, alloc::alloc::Global>",
                "alloc::boxed::Box<rb_fix::Rec, alloc::alloc::Global>", "rb_fix::Named", "rb_fix::Guard", "rb_fix::View",
                "rb_fix::Slot", "rb_fix::inner::Token", "rb_fix::hidden::Secret", "rb_fix::hidden::Sealed", "rb_fix::hidden::Fault",
                "core::result::Result<i32, alloc::string::String>", "core::option::Option<alloc::string::String>",
                "alloc::boxed::Box<dyn rb_fix::Area, alloc::alloc::Global>")
        @test key in need
    end
    # A `Box`'s target, a `Vec`'s element and a reference's target: `Rec` is all three.
    @test "rb_fix::Rec" in need && "alloc::vec::Vec<alloc::string::String, alloc::alloc::Global>" in need
    # Never a scalar, a reference, a fieldless enum, or a tuple (its fields are asked about).
    @test !any(k -> startswith(k, "&") || startswith(k, "*") || startswith(k, "("), need)
    @test !("f64" in need) && !("bool" in need) && !("rb_fix::Side" in need)
    # A function that is not called asks about nothing.
    none = deepcopy(md["functions"])
    foreach(r -> r["refusal"] = "refused", none)
    @test isempty(RG.wanted(none, types))

    @test RG.vec_element(types, "alloc::vec::Vec<f64, alloc::alloc::Global>") == "f64"
    @test RG.vec_element(types, "alloc::vec::Vec<rb_fix::Rec, alloc::alloc::Global>") == "rb_fix::Rec"
    @test RG.vec_element(types, "alloc::string::String") === nothing && RG.vec_element(types, "no::such") === nothing
    @test RG.payload_enum(types["core::option::Option<alloc::string::String>"]) === :option
    @test RG.payload_enum(types["core::result::Result<i32, alloc::string::String>"]) === :result
    @test RG.payload_enum(types["rb_fix::Slot"]) === nothing && RG.payload_enum(types["rb_fix::Rec"]) === nothing

    # The fixture is the compiler's word on each case the generator's tests lean on.
    g = md["glue"]
    @test g["rb_fix::Guard"]["owns"] === true && g["rb_fix::View"]["owns"] === false && g["rb_fix::Rec"]["owns"] === false
    @test g["rb_fix::inner::Token"]["spelling"] == "rb_fix::Token"
    @test g["rb_fix::hidden::Secret"]["by_address"] === true && g["rb_fix::hidden::Fault"]["owns"] === true
    @test occursin("module `hidden` is private", md["glue_refused"]["rb_fix::hidden::Sealed"])

    # A type no path spells is taken from a function a value of it comes out of: what it
    # returns first, then the payload of a `Result` or `Option` it returns.
    fns = md["functions"]
    secret = RG.positions(fns, types, "rb_fix::hidden::Secret")
    @test [(p.path, p.helper) for p in secret] == [("rb_fix::secret", "__rb_ret0")]
    @test secret[1].what == "what `rb_fix::secret` returns"
    fault = RG.positions(fns, types, "rb_fix::hidden::Fault")
    @test [(p.path, p.helper, p.what) for p in fault] == [("rb_fix::parse_byte", "__rb_err1", "the `Err` of what `rb_fix::parse_byte` returns")]
    @test isempty(RG.positions(fns, types, "rb_fix::hidden::Sealed"))           # taken by value, returned by nothing
    @test ("rb_fix::find_word", "__rb_some2") in [(p.path, p.helper) for p in RG.positions(fns, types, "alloc::string::String"; limit = 100)]
    @test length(RG.positions(fns, types, "alloc::string::String")) == 4       # the best few, not all
    @test first(RG.positions(fns, types, "alloc::string::String")).helper[1:8] == "__rb_ret"
    # A function the build refused, or one of too many parameters, is not one to take a type from.
    wide = deepcopy(only(f for f in fns if f["path"] == "rb_fix::secret"))
    wide["parameters"] = [Dict{String,Any}("name" => "a$i", "type" => "u8") for i in 1:RG.MAX_ARITY+1]
    @test isempty(RG.positions([wide], types, "rb_fix::hidden::Secret"))
    wide["parameters"] = Any[]; wide["refusal"] = "refused"
    @test isempty(RG.positions([wide], types, "rb_fix::hidden::Secret"))
end

@testset "RustGlue: a type taken from a function that returns it" begin
    rows = [RG.Row(2, "dep::hidden::Secret", "what `dep::secret` returns", false, "__rb_ret0(dep::secret)"),
            RG.Row(5, "dep::hidden::Fault", "the `Err` of what `dep::parse` returns", false, "__rb_err1(dep::parse)"),
            RG.Row(6, "dep::Named", "dep::Named", false)]
    (text, line_of) = RG.glue_text(rows)
    ls = split(text, '\n')
    @test ls[line_of[2]] == "#[no_mangle] pub static __rb_dropfn_2: __RbAddr = __RbAddr(__rb_ret0(dep::secret) as *const ());"
    @test ls[line_of[5]] == "#[no_mangle] pub static __rb_dropfn_5: __RbAddr = __RbAddr(__rb_err1(dep::parse) as *const ());"
    @test startswith(ls[line_of[6]], "#[no_mangle] pub unsafe extern \"C\" fn __rb_drop_6(p: *mut dep::Named, ")
    @test RG.dropfn_symbol(5) == "__rb_dropfn_5" && RG.Row(1, "a", "a", true).call == ""

    # What the library says: the function the compiler picked is named for whether the type
    # is dropped, and its debug info for which type it is.
    ptr(target) = Dict{String,Any}("kind" => "pointer", "target" => target)
    types = Dict{String,Any}("dep::hidden::Secret" => Dict{String,Any}("kind" => "struct"), "*mut dep::hidden::Secret" => ptr("dep::hidden::Secret"),
                             "dep::hidden::Fault" => Dict{String,Any}("kind" => "enum"), "*mut dep::hidden::Fault" => ptr("dep::hidden::Fault"),
                             "dep::Flat" => Dict{String,Any}("kind" => "struct"), "*mut dep::Flat" => ptr("dep::Flat"))
    fn(p) = Dict{String,Any}("parameters" => Any[Dict{String,Any}("name" => "p", "type" => p), Dict{String,Any}("name" => "_unused", "type" => "usize")])
    dwarf = Dict{String,Any}("_Rowned_secret" => fn("*mut dep::hidden::Secret"), "_Rplain_flat" => fn("*mut dep::Flat"),
                             "_Rowned_fault" => fn("*mut dep::hidden::Fault"))
    demangled = Dict("_Rowned_secret" => "dep_rb::__rb_drop_owned::<dep::hidden::Secret>", "_Rplain_flat" => "dep_rb::__rb_drop_plain::<dep::Flat>",
                     "_Rowned_fault" => "dep_rb::__rb_drop_owned::<dep::hidden::Fault>", "_Rother" => "dep_rb::something_else")
    table = Dict("__rb_dropfn_1" => "_Rowned_secret", "__rb_dropfn_2" => "_Rplain_flat", "__rb_dropfn_3" => "_Rowned_fault",
                 "__rb_dropfn_4" => "_Rother", "__rb_dropfn_6" => "_Rowned_secret")
    exported = Set(["__rb_dropfn_$i" for i in (1, 2, 3, 4, 6)])
    plan = RG.Plan(String[], [RG.Row(1, "dep::hidden::Secret", "what `dep::secret` returns", false, "__rb_ret0(dep::secret)"),
                              RG.Row(2, "dep::Flat", "what `dep::flat` returns", false, "__rb_ret0(dep::flat)"),
                              RG.Row(3, "dep::Wanted", "what `dep::other` returns", false, "__rb_ret0(dep::other)"),
                              RG.Row(4, "dep::Odd", "what `dep::odd` returns", false, "__rb_ret0(dep::odd)"),
                              RG.Row(5, "dep::Gone", "what `dep::gone` returns", false, "__rb_ret0(dep::gone)")],
                   Dict{String,String}())
    (glue, refused) = RG.verify(plan, dwarf, types, exported; table, demangled)
    @test glue["dep::hidden::Secret"] == Dict{String,Any}("drop" => "__rb_dropfn_1", "by_address" => true, "owns" => true,
                                                          "spelling" => "what `dep::secret` returns")
    @test glue["dep::Flat"]["owns"] === false && glue["dep::Flat"]["by_address"] === true
    @test sort!(collect(keys(glue))) == ["dep::Flat", "dep::hidden::Secret"]
    @test occursin("is another type (`dep::hidden::Fault`)", refused["dep::Wanted"])
    @test occursin("was not read", refused["dep::Odd"]) && occursin("is not in the library", refused["dep::Gone"])
end

@testset "RustGlue: the fixture's refusal" begin
    md = RepliBuild._read_json(rust_fixture("owned_o3.metadata.json"))
    @test collect(keys(md["glue_refused"])) == ["rb_fix::hidden::Sealed"]
end

@testset "RustGlue: the file and its record on disk" begin
    RCM = RepliBuild.ConfigurationManager
    mktempdir() do dir
        rc = RCM.RustConfig(joinpath(dir, "crate", "Cargo.toml"), "", "", 0, String[], true, joinpath(dir, "build", "rust", "target"))
        plan = RG.Plan(["dep::Token", "dep::hidden::Secret"], [RG.Row(1, "dep::Token", "dep::Token", false)],
                       Dict("dep::hidden::Secret" => "no path names it"))
        path = _RGT_RB._write_glue(rc, plan; identity = "one")
        @test path == joinpath(_RGT_RB.root_dir(rc), "rb_glue.rs") && read(path, String) == RG.glue_text(plan.rows)[1]
        back = _RGT_RB._read_plan(rc, "one")
        @test back.wanted == plan.wanted && back.refused == plan.refused
        @test [(r.n, r.key, r.spelling, r.vec) for r in back.rows] == [(1, "dep::Token", "dep::Token", false)]
        @test isempty(back.fns) && isempty(back.wanted_fns)
        # The functions' lines are part of the same record.
        both = RG.Plan(plan.wanted, plan.rows, plan.refused, ["inline|_Rsym", "instantiate|dep::f::<u8>"],
                       [RG.FnRow(1, "dep::small", "inline", "_Rsym"), RG.FnRow(2, "dep::f::<u8>", "instantiate", "")],
                       Dict("_Rother" => "no public path names it"))
        _RGT_RB._write_glue(rc, both; identity = "one")
        @test read(path, String) == RG.glue_text(both.rows, both.fns)[1]
        again = _RGT_RB._read_plan(rc, "one")
        @test again.wanted_fns == both.wanted_fns && again.fn_refused == both.fn_refused
        @test [(f.n, f.expr, f.source, f.symbol) for f in again.fns] == [(1, "dep::small", "inline", "_Rsym"), (2, "dep::f::<u8>", "instantiate", "")]
        _RGT_RB._write_glue(rc, plan; identity = "one")
        # Unchanged text is not rewritten: cargo rebuilds what is newer.
        stamp = mtime(path)
        sleep(0.05)
        _RGT_RB._write_glue(rc, plan; identity = "one")
        @test mtime(path) == stamp
        # Another compiler or crate: the lines are not carried over.
        other = _RGT_RB._read_plan(rc, "two")
        @test isempty(other.rows) && isempty(other.wanted) && read(path, String) == RG.glue_text(RG.Row[])[1]
        # A file that is not the one the record describes: the same.
        _RGT_RB._write_glue(rc, plan; identity = "one")
        write(path, read(path, String) * "// edited\n")
        @test isempty(_RGT_RB._read_plan(rc, "one").rows) && !occursin("edited", read(path, String))
    end
    tc = _RGT_RB.Toolchain("cargo", "rustc", "1.0.0", "x86_64-unknown-linux-gnu", "20.0.0", "id")
    crate = _RGT_RB.Crate("rb-fix", "rb_fix", "/p/crate/Cargo.toml", "/p/crate")
    rc(features) = RCM.RustConfig("/p/crate/Cargo.toml", "", "", 0, features, true, "/p/build/rust/target")
    id = _RGT_RB._glue_identity(rc(String[]), crate, tc, "gen")
    @test id == _RGT_RB._glue_identity(rc(String[]), crate, tc, "gen") && length(id) == 64
    @test id != _RGT_RB._glue_identity(rc(["serde"]), crate, tc, "gen") && id != _RGT_RB._glue_identity(rc(String[]), crate, tc, "gen2")
end
