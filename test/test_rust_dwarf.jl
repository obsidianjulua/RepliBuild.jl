# The Rust DWARF pass (`RustDWARF`), without a toolchain: vendored GNU readelf
# dumps of test/fixtures/rust/lean.rs (test_rust_abi.jl's header lists them)
# and small synthetic dumps for the cases a real build does not produce on
# demand: DIEs in the other order, a location expression where an offset
# should be, two types under one name.

using Test
using RepliBuild

const RDW = RepliBuild.RustDWARF

rust_dump(opt) = joinpath(@__DIR__, "fixtures", "rust", "lean_o$opt.dump")
rust_exports(opt) = Set(readlines(joinpath(@__DIR__, "fixtures", "rust", "lean_o$opt.exports")))

@testset "Rust DWARF: constants as binutils prints them" begin
    @test RDW._number("24") == 24
    @test RDW._number("0x10000") == 65536            # four and eight bytes print in hex
    @test RDW._number("1\t(public)") == 1
    @test RDW._number("28\t(Rust)") == 28
    @test RDW._number("-3") == -3
    # A location expression starts with a digit and is not a number.
    @test RDW._number("2 byte block: 23 8 \t(DW_OP_plus_uconst: 8)") === nothing
    @test RDW._number("<0x690>") === nothing
    # A niche discriminant does not fit Int64, and is kept exactly, as text.
    @test RDW._wide_number("0x8000000000000000") == "9223372036854775808"
    @test RDW._wide_number("0x7fffffffffffffff") == typemax(Int)
    @test RDW._ref("<0x690>") == 0x690 && RDW._ref("24") === nothing
    @test RDW._attr_value("(indirect string, offset: 0x487e3): probe.rs") == "probe.rs"
    @test RDW._attr_value("(indexed string: 0x1): x") == "x"
    @test RDW._attr_value("<0xd03>") == "<0xd03>"
    # Only the form's own prefix is cut: a name may hold `): ` itself.
    @test RDW._attr_value("(indirect string, offset: 0x1): f(a): b") == "f(a): b"
end

@testset "Rust DWARF: only the units that name a wanted function are built" begin
    for opt in (0, 3)
        text = read(rust_dump(opt), String)
        @test count("Compilation Unit @ offset", text) == 2        # the standard library's, then the crate's
        add = only(s for s in rust_exports(opt) if endswith(s, "4lean3add"))
        units = RDW.read_units(eachline(rust_dump(opt)), Set([add]))
        @test length(units.roots) == 1
        root = units.roots[1]
        @test root.tag == "compile_unit" && occursin("lean", root.attrs["name"])
        @test root.attrs["language"] == "28\t(Rust)"
        # The other unit comes FIRST in the file and none of its DIEs was kept.
        first_unit = parse(Int, match(r"<0><([0-9a-f]+)>: Abbrev Number: \d+ \(DW_TAG_compile_unit\)", text).captures[1]; base = 16)
        @test first_unit != root.offset && !haskey(units.by_offset, first_unit)
        # Nothing wanted: nothing built.
        none = RDW.read_units(eachline(rust_dump(opt)), Set(["_RNvCsX_5other3fun"]))
        @test isempty(none.roots) && isempty(none.by_offset)
        # A symbol rustc did not mangle is found by its DW_AT_name.
        norm = RDW.read_units(eachline(rust_dump(opt)), Set(["cpoint_norm"]))
        @test length(norm.roots) == 1
    end
end

@testset "Rust DWARF: functions and the types they reach" begin
    for opt in (0, 3)
        wanted = rust_exports(opt)
        units = RDW.read_units(eachline(rust_dump(opt)), wanted)
        (fns, types) = RDW.describe_functions(units, wanted)
        f(tail) = fns[only(s for s in keys(fns) if endswith(s, tail))]
        @test length(fns) == (opt == 0 ? 31 : 30)      # optimized, `c_add` is an alias with no DIE
        @test all(v -> v["defined"], values(fns))

        # Parameters come from the DEFINITION, with their names, in order.
        rm = f("4lean8rec_make")
        @test rm["name"] == "rec_make" && rm["scope"] == ["lean"] && rm["owner"] === nothing
        @test [(p["name"], p["type"]) for p in rm["parameters"]] ==
              [("a", "u8"), ("b", "f64"), ("c", "u16"), ("d", "u32")]
        @test rm["return"] == "lean::Rec"
        # A method's name and scope come from its declaration inside the type.
        push = f("3Acc4push")
        @test push["name"] == "push" && push["owner"] == "lean::Acc" && push["scope"] == ["lean", "Acc"]
        @test [(p["name"], p["type"]) for p in push["parameters"]] == [("self", "&mut lean::Acc"), ("x", "f64")]
        @test push["return"] === nothing
        # A trait impl's method is scoped by its impl block.
        @test f("4Area4area")["scope"][1] == "lean" && startswith(f("4Area4area")["scope"][2], "{impl#")
        # An unnamed parameter still counts.
        @test [p["type"] for p in f("4lean1z")["parameters"]] == ["lean::Z", "i32"]

        # The compiler reordered `Rec`'s fields, and DWARF has where they went.
        rec = types["lean::Rec"]
        @test rec["kind"] == "struct" && rec["size"] == 16 && rec["align"] == 8
        @test Dict(m["name"] => m["offset"] for m in rec["fields"]) == Dict("b" => 0, "d" => 8, "c" => 12, "a" => 14)
        # An enum that carries data keeps its discriminant and each variant's value.
        shape = types["lean::Shape"]
        @test shape["kind"] == "enum" && shape["size"] == 24
        @test shape["discriminant"]["offset"] == 0 && shape["discriminant"]["type"] == "u64" &&
              shape["discriminant"]["size"] == 8
        @test [(v["name"], v["discr_value"], v["type"]) for v in shape["variants"]] ==
              [("Circle", 0, "lean::Shape::Circle"), ("Rect", 1, "lean::Shape::Rect"), ("Empty", 2, "lean::Shape::Empty")]
        @test [(m["name"], m["offset"]) for m in types["lean::Shape::Rect"]["fields"]] == [("w", 8), ("h", 16)]
        # Each `Option<T>` has its own `Some`: the variants are scoped by their enum.
        @test types["core::option::Option<i32>::Some"]["fields"][1]["offset"] == 4
        @test types["core::option::Option<&lean::Rec>::Some"]["fields"][1]["type"] == "&lean::Rec"
        # A niche: `None` is 0 and `Some` is every other value.
        niche = types["core::option::Option<&lean::Rec>"]
        @test [(v["name"], v["default"]) for v in niche["variants"]] == [("None", false), ("Some", true)]
        @test niche["variants"][1]["discr_value"] == 0 && !haskey(niche["variants"][2], "discr_value")
        # A fieldless enum is its enumerators.
        @test types["lean::Level"]["kind"] == "c_enum" && types["lean::Level"]["size"] == 1
        @test types["lean::Level"]["enumerators"] == Any[Any["Low", 0], Any["Mid", 1], Any["High", 2]]
        # Pointers keep what kind they are; fat pointers are named records.
        @test types["&lean::Rec"]["pointer"] == "ref" && types["&lean::Rec"]["target"] == "lean::Rec"
        @test types["&mut lean::Acc"]["pointer"] == "mut_ref"
        @test types["&[f64]"]["fat"] == "slice" && types["&str"]["fat"] == "slice"
        @test types["&dyn lean::Area"]["fat"] == "dyn"
        @test types["(i32, f64)"]["tuple"] === true
        @test types["char"]["encoding"] == "utf" && types["char"]["size"] == 4
        @test types["bool"]["encoding"] == "boolean"
        # An array's size is its elements'.
        @test types["[usize; 4]"]["count"] == 4 && types["[usize; 4]"]["size"] == 32
        # Nothing the crate's functions do not reach is described.
        @test !any(k -> startswith(k, "std::") || startswith(k, "libc::"), keys(types))
        @test !any(t -> get(t, "conflict", false) === true, values(types))
    end
end

# A synthetic unit: a record, a method declared inside it, and the method's
# definition, which names the declaration by offset.
function synthetic_unit(; definition_first::Bool, member_location::String = "8")
    record = """
         <1><20>: Abbrev Number: 2 (DW_TAG_namespace)
            <21>   DW_AT_name        : m
         <2><30>: Abbrev Number: 3 (DW_TAG_structure_type)
            <31>   DW_AT_name        : S
            <32>   DW_AT_byte_size   : 16
            <33>   DW_AT_alignment   : 8
         <3><40>: Abbrev Number: 4 (DW_TAG_member)
            <41>   DW_AT_name        : a
            <42>   DW_AT_type        : <0x90>
            <43>   DW_AT_data_member_location: 0
         <3><48>: Abbrev Number: 4 (DW_TAG_member)
            <49>   DW_AT_name        : b
            <4a>   DW_AT_type        : <0x90>
            <4b>   DW_AT_data_member_location: $member_location
         <3><50>: Abbrev Number: 5 (DW_TAG_subprogram)
            <51>   DW_AT_linkage_name: (indirect string, offset: 0x10): _RNvMCsX_1mNtB2_1S3get
            <52>   DW_AT_name        : (indirect string, offset: 0x20): get
            <53>   DW_AT_type        : <0x90>
            <54>   DW_AT_declaration : 1
         <4><58>: Abbrev Number: 6 (DW_TAG_formal_parameter)
            <59>   DW_AT_type        : <0xa0>
         <4><5c>: Abbrev Number: 0
         <3><5d>: Abbrev Number: 0
         <2><5e>: Abbrev Number: 0
        """
    definition = """
         <1><70>: Abbrev Number: 7 (DW_TAG_subprogram)
            <71>   DW_AT_low_pc      : 0x1000
            <72>   DW_AT_high_pc     : 0x10
            <73>   DW_AT_specification: <0x50>
         <2><78>: Abbrev Number: 8 (DW_TAG_formal_parameter)
            <79>   DW_AT_name        : self
            <7a>   DW_AT_type        : <0xa0>
         <2><7e>: Abbrev Number: 0
        """
    types = """
         <1><90>: Abbrev Number: 9 (DW_TAG_base_type)
            <91>   DW_AT_name        : u64
            <92>   DW_AT_encoding    : 7\t(unsigned)
            <93>   DW_AT_byte_size   : 8
         <1><a0>: Abbrev Number: 10 (DW_TAG_pointer_type)
            <a1>   DW_AT_type        : <0x30>
            <a2>   DW_AT_name        : &m::S
         <1><b0>: Abbrev Number: 0
        """
    return " <0><b>: Abbrev Number: 1 (DW_TAG_compile_unit)\n    <c>   DW_AT_language    : 28\t(Rust)\n" *
           (definition_first ? definition * record : record * definition) * types
end

@testset "Rust DWARF: a definition finds its declaration wherever it is" begin
    wanted = Set(["_RNvMCsX_1mNtB2_1S3get"])
    answers = map((false, true)) do definition_first
        units = RDW.read_units(split(synthetic_unit(; definition_first), '\n'), wanted)
        (fns, types) = RDW.describe_functions(units, wanted)
        f = fns["_RNvMCsX_1mNtB2_1S3get"]
        @test f["defined"] && f["name"] == "get" && f["owner"] == "m::S" && f["scope"] == ["m", "S"]
        @test [(p["name"], p["type"]) for p in f["parameters"]] == [("self", "&m::S")]
        @test f["return"] == "u64"
        @test [(m["name"], m["offset"]) for m in types["m::S"]["fields"]] == [("a", 0), ("b", 8)]
        (f, types)
    end
    # The two orders describe one thing.
    @test answers[1] == answers[2]
end

@testset "Rust DWARF: what is not a constant offset is not an offset" begin
    wanted = Set(["_RNvMCsX_1mNtB2_1S3get"])
    text = synthetic_unit(; definition_first = false,
                          member_location = "2 byte block: 23 8 \t(DW_OP_plus_uconst: 8)")
    (_, types) = RDW.describe_functions(RDW.read_units(split(text, '\n'), wanted), wanted)
    @test types["m::S"]["fields"][2]["offset"] === nothing
    # ...and a record with a member it cannot place has no shape to pass by value.
    @test RepliBuild.RustABI.shape(types, "m::S").leaves === :unknown
end

@testset "Rust DWARF: two different types under one name are marked" begin
    wanted = Set(["_RNvMCsX_1mNtB2_1S3get", "_RNvCsX_1m5other"])
    # A second unit that defines `m::S` with another layout, and a function taking it.
    second = """
         <0><200>: Abbrev Number: 1 (DW_TAG_compile_unit)
            <201>   DW_AT_language    : 28\t(Rust)
         <1><210>: Abbrev Number: 2 (DW_TAG_namespace)
            <211>   DW_AT_name        : m
         <2><220>: Abbrev Number: 3 (DW_TAG_structure_type)
            <221>   DW_AT_name        : S
            <222>   DW_AT_byte_size   : 8
         <3><230>: Abbrev Number: 4 (DW_TAG_member)
            <231>   DW_AT_name        : a
            <232>   DW_AT_type        : <0x290>
            <233>   DW_AT_data_member_location: 0
         <3><238>: Abbrev Number: 0
         <2><240>: Abbrev Number: 5 (DW_TAG_subprogram)
            <241>   DW_AT_low_pc      : 0x2000
            <242>   DW_AT_linkage_name: _RNvCsX_1m5other
            <243>   DW_AT_name        : other
         <3><250>: Abbrev Number: 6 (DW_TAG_formal_parameter)
            <251>   DW_AT_name        : s
            <252>   DW_AT_type        : <0x220>
         <3><258>: Abbrev Number: 0
         <2><259>: Abbrev Number: 0
         <1><290>: Abbrev Number: 9 (DW_TAG_base_type)
            <291>   DW_AT_name        : u64
            <292>   DW_AT_encoding    : 7\t(unsigned)
            <293>   DW_AT_byte_size   : 8
         <1><2a0>: Abbrev Number: 0
        """
    text = synthetic_unit(; definition_first = false) * second
    units = RDW.read_units(split(text, '\n'), wanted)
    @test length(units.roots) == 2
    (fns, types) = RDW.describe_functions(units, wanted)
    @test length(fns) == 2
    @test types["m::S"]["conflict"] === true
    # The same `u64` in both units is one type, not a conflict.
    @test !haskey(types["u64"], "conflict")
    # A conflicted name has no layout: the function taking it by value is refused.
    def = RepliBuild.RustABI.IRDefinition("_RNvCsX_1m5other", String[], "", "void", String[],
                                          [RepliBuild.RustABI.IRParam("i64", String[], "%s")], false, false, "")
    row = RepliBuild.RustABI.join_function("_RNvCsX_1m5other", def, fns["_RNvCsX_1m5other"], types)
    @test row["refusal"] !== nothing && occursin("layout", row["refusal"])
end
