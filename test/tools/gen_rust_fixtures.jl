# gen_rust_fixtures.jl — build test/fixtures/rust/lean.rs with rustc and vendor
# what the Rust front end reads from the result, so its unit tests need no
# toolchain:
#
#   lean_o<N>.dump     the GNU readelf `.debug_info` dump of the crate's own
#                      compile units, and of one unit of the standard library
#                      (which `RustDWARF.read_units` must pass over)
#   lean_o<N>.ir.txt   the `define`, `attributes` and alias lines of the crate's
#                      IR (`.txt`: the repository ignores `*.ll`)
#   lean_o<N>.exports  the names in the library's dynamic symbol table that the
#                      IR defines, and a few it does not
#
# for N = 0 and 3: an unoptimized build describes a function one way (a
# definition DIE with its parameters) and an optimized one another (an abstract
# instance and a concrete one), and the `define` lines carry other attributes.
#
#     julia --project=. test/tools/gen_rust_fixtures.jl
#
# Hand-run; needs rustc and GNU readelf. Refuses to write a fixture that does
# not hold what the tests look for.

using RepliBuild
const T = RepliBuild.Tools

const FIXTURES = normpath(joinpath(@__DIR__, "..", "fixtures", "rust"))
const SOURCE = joinpath(FIXTURES, "lean.rs")
const FOREIGN_UNIT_MAX_LINES = 400

function units_of(dump::String)
    lines = split(dump, '\n')
    starts = [i for (i, l) in enumerate(lines) if occursin("Compilation Unit @ offset", l)]
    isempty(starts) && error("the dump has no compile unit")
    return [lines[a:(b - 1)] for (a, b) in zip(starts, vcat(starts[2:end], length(lines) + 1))]
end

function generate(opt::Int)
    mktempdir() do dir
        cp(SOURCE, joinpath(dir, "lean.rs"))
        rustc = RepliBuild.RustBuild.tool("rustc")
        args = ["--edition", "2021", "--crate-type", "dylib", "--crate-name", "lean",
                "-C", "opt-level=$opt", "-C", "debuginfo=2", "-C", "codegen-units=1",
                "-C", "symbol-mangling-version=v0", "--remap-path-prefix=$dir=/fixture",
                "--emit=llvm-ir,link", "--out-dir", dir]
        # Above opt-level 0 a small `pub fn` keeps its symbol only with
        # incremental compilation on (RustBuild.jl's header).
        opt > 0 && append!(args, ["-C", "incremental=" * joinpath(dir, "inc")])
        run(Cmd(`$rustc $args lean.rs`; dir))
        lib = joinpath(dir, "liblean.so")
        ir = joinpath(dir, "lean.ll")

        defines = [l for l in eachline(ir) if startswith(l, "define ") || startswith(l, "attributes #") ||
                                              (startswith(l, "@") && occursin(" alias ", l))]
        defined = Set(String(m.captures[1]) for m in (match(r"@([-a-zA-Z$._0-9]+)\(", l) for l in defines) if m !== nothing)
        exported = sort!([String(split(l)[3]) for l in eachline(T.tool_cmd(`nm -D --defined-only $lib`))
                          if length(split(l)) == 3])
        aliased = Set(String(m.captures[1]) for m in (match(r"^@([-a-zA-Z$._0-9]+) = ", l) for l in defines) if m !== nothing)
        mine = [s for s in exported if s in defined || s in aliased]
        others = [s for s in exported if !(s in defined) && !(s in aliased) && startswith(s, "_R")][1:20]

        dump = read(T.tool_cmd(`readelf --debug-dump=info $lib`), String)
        us = units_of(dump)
        crate = [u for u in us if any(l -> occursin("DW_AT_linkage_name", l) && occursin("4lean", l), u)]
        foreign = [u for u in us if !(u in crate) && length(u) <= FOREIGN_UNIT_MAX_LINES]
        isempty(crate) && error("opt-level $opt: no compile unit names a function of the crate")
        isempty(foreign) && error("opt-level $opt: no small unit of the standard library to vendor beside the crate's")
        kept = vcat(foreign[1], crate...)        # the foreign unit FIRST: it must be passed over, not just trailed

        # What the tests look for. A fixture without it would pass them vacuously.
        text = join(kept, "\n")
        for needle in ("DW_TAG_variant_part", "DW_AT_discr_value", "3add", "3Acc4push", "DW_AT_specification",
                       "cpoint_norm", "(indirect string")
            occursin(needle, text) || error("opt-level $opt: the dump lacks $(repr(needle))")
        end
        opt > 0 && !occursin("DW_AT_abstract_origin", text) &&
            error("opt-level $opt: the dump has no abstract origin; an optimized build should")
        # Optimized, `c_add` has `add`'s code and LLVM keeps it as an alias with no DIE.
        opt > 0 && !any(l -> startswith(l, "@c_add = ") && occursin(" alias ", l), defines) &&
            error("opt-level $opt: `c_add` is not an alias; the alias fixture is gone")
        opt == 0 && !occursin("c_add", text) && error("opt-level 0: the dump lacks `c_add`")
        length(mine) >= 30 || error("opt-level $opt: only $(length(mine)) exported definitions")
        any(l -> occursin("sret(", l), defines) || error("opt-level $opt: no sret define line")

        stem = joinpath(FIXTURES, "lean_o$opt")
        write(stem * ".dump", join(vcat(["Contents of the .debug_info section:", ""], kept), "\n") * "\n")
        write(stem * ".ir.txt", join(defines, "\n") * "\n")
        write(stem * ".exports", join(sort!(vcat(mine, others)), "\n") * "\n")
        println("lean_o$opt: $(length(mine)) exported definitions, $(length(defines)) IR lines, " *
                "$(length(crate)) crate unit(s) + 1 foreign, $(length(kept)) dump lines, " *
                "$(round(filesize(stem * ".dump") / 1024, digits = 1)) KiB")
    end
end

generate(0)
generate(3)
