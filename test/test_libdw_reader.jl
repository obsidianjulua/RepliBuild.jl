# test_libdw_reader.jl — the in-process DWARF reader's own pieces.
#
# `LibDW` walks a library's DIE tree through elfutils' libdw and writes it in
# the format `parse_dwarf_dump` reads from GNU readelf, so the parser is the
# same either way. This file holds what needs no library: which reader
# REPLIBUILD_DWARF selects, and the location-expression printer, whose text
# the parser matches for a virtual base's offset. test_libdw_dwarf_live.jl
# holds the two readers to one answer on real libraries.

using Test
using RepliBuild

const _LD = RepliBuild.LibDW
const _LD_C = RepliBuild.Compiler

@testset "REPLIBUILD_DWARF selects the reader, and readelf is the default" begin
    @test withenv(() -> _LD_C._dwarf_reader(), "REPLIBUILD_DWARF" => nothing) === :readelf
    @test withenv(() -> _LD_C._dwarf_reader(), "REPLIBUILD_DWARF" => "") === :readelf
    @test withenv(() -> _LD_C._dwarf_reader(), "REPLIBUILD_DWARF" => "ReadELF") === :readelf
    @test_throws ErrorException withenv(() -> _LD_C._dwarf_reader(), "REPLIBUILD_DWARF" => "dwarfdump")
    if Sys.islinux() && _LD.available()
        @test withenv(() -> _LD_C._dwarf_reader(), "REPLIBUILD_DWARF" => " libdw ") === :libdw
    else
        # Asked for where it cannot work: an error, never the other reader.
        @test_throws ErrorException withenv(() -> _LD_C._dwarf_reader(), "REPLIBUILD_DWARF" => "libdw")
    end
end

@testset "a location expression prints as readelf prints it" begin
    # The offset of a virtual base, as clang writes it: the parser's own
    # pattern must find the constant in this text.
    vbase = UInt8[0x12, 0x06, 0x10, 0x18, 0x1c, 0x06, 0x22]
    text = _LD.expression(vbase)
    @test text == "DW_OP_dup; DW_OP_deref; DW_OP_constu: 24; DW_OP_minus; DW_OP_deref; DW_OP_plus"
    m = match(r"DW_OP_dup[;,].*DW_OP_deref[;,].*(?:DW_OP_constu:?\s*(0x[0-9a-fA-F]+|\d+)|DW_OP_lit(\d+))\s*[;,].*DW_OP_minus[;,].*DW_OP_deref[;,].*DW_OP_plus", text)
    @test m !== nothing && m.captures[1] == "24"
    @test _LD.expression(UInt8[0x12, 0x06, 0x48, 0x1c, 0x06, 0x22]) ==
          "DW_OP_dup; DW_OP_deref; DW_OP_lit24; DW_OP_minus; DW_OP_deref; DW_OP_plus"

    @test _LD.expression(UInt8[0x23, 0x08]) == "DW_OP_plus_uconst: 8"
    @test _LD.expression(UInt8[0x23, 0x80, 0x01]) == "DW_OP_plus_uconst: 128"      # ULEB128
    @test _LD.expression(UInt8[0x91, 0x6c]) == "DW_OP_fbreg: -20"                  # SLEB128
    @test _LD.expression(UInt8[0x11, 0x7f]) == "DW_OP_consts: -1"
    @test _LD.expression(UInt8[0x09, 0xff]) == "DW_OP_const1s: -1"
    @test _LD.expression(UInt8[0x0a, 0x34, 0x12]) == "DW_OP_const2u: 4660"
    @test _LD.expression(UInt8[0x03, 0x10, 0x40, 0, 0, 0, 0, 0, 0]) == "DW_OP_addr: 4010"
    @test _LD.expression(UInt8[0x9c]) == "DW_OP_call_frame_cfa"
    @test _LD.expression(UInt8[0xa3, 0x01, 0x55, 0x9f]) == "DW_OP_entry_value: (DW_OP_reg5 (r5)); DW_OP_stack_value"
    # An operation the printer does not decode ends the text: its operands
    # are unknown, so nothing after it is read as an operation.
    @test _LD.expression(UInt8[0x12, 0xfe, 0x06]) == "DW_OP_dup; DW_OP_unknown: 0xfe"
    @test _LD.expression(UInt8[]) == ""
    # A truncated operand is not read past the block.
    @test _LD.expression(UInt8[0x0c, 0x01]) == ""
end

@testset "the tag and attribute names are the parser's" begin
    # Every name the readelf parser matches must be one this module can print:
    # a spelling missing here is a DIE or an attribute the parser never sees.
    src = read(joinpath(dirname(@__DIR__), "src", "Builder", "Compiler.jl"), String)
    lo = findfirst("function parse_dwarf_dump(", src)
    hi = findnext("\nend\n", src, last(lo))
    # Code only: the parser's comments name spellings it does not match
    # (llvm-dwarfdump's `_parameter`, an attribute that does not exist).
    body = join(filter(l -> !startswith(lstrip(l), "#"), split(src[first(lo):first(hi)], '\n')), "\n")
    tags = Set(m.captures[1] for m in eachmatch(r"DW_TAG_([a-z_]+[a-z])", body))
    attrs = Set(m.captures[1] for m in eachmatch(r"DW_AT_([A-Za-z_]+[A-Za-z])", body))
    @test length(tags) >= 20
    @test length(attrs) >= 25
    known_tags = Set(values(_LD.TAGS))
    known_attrs = Set(values(_LD.ATTRS))
    # `DW_TAG_restrict|volatile|atomic` are spelled through one alternation.
    missing_tags = sort!([t for t in tags if !(t in known_tags) && t != "restrict"])
    missing_attrs = sort!([a for a in attrs if !(a in known_attrs)])
    @test isempty(missing_tags)
    @test isempty(missing_attrs)
    isempty(missing_tags) || println("  not printable tags: ", missing_tags)
    isempty(missing_attrs) || println("  not printable attributes: ", missing_attrs)
    # binutils' spelling, which the parser matches (llvm-dwarfdump's is `_parameter`).
    @test _LD.TAGS[0x2f] == "template_type_param"
    @test _LD.TAGS[0x30] == "template_value_param"
end
