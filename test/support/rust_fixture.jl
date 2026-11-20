# Shared by the Rust front end's toolchain-free tests: what
# `RustBuild.extract_metadata` computes, from the vendored files of one build
# of test/fixtures/rust/lean.rs (test/tools/gen_rust_fixtures.jl writes them).

using RepliBuild

const RA = RepliBuild.RustABI
const RD = RepliBuild.RustDWARF

rust_fixture(name) = joinpath(dirname(@__DIR__), "fixtures", "rust", name)

"Everything `RustBuild.extract_metadata` computes, from the vendored files of one build."
function rust_extract(opt::Int)
    ir = [rust_fixture("lean_o$opt.ir.txt")]
    defs = RA.definitions(ir)
    aliases = RA.aliases(ir)
    exported = Set(readlines(rust_fixture("lean_o$opt.exports")))
    (symbols, skipped, foreign) = RA.surface(defs, exported, "lean"; aliases)
    target = Dict(s => RA.alias_target(s, defs, aliases) for s in symbols)
    wanted = union(Set(symbols), Set(values(target)))
    units = RD.read_units(eachline(rust_fixture("lean_o$opt.dump")), wanted)
    (dwarf, types) = RD.describe_functions(units, wanted)
    rows = Dict{String,Any}()
    for s in symbols
        own = get(dwarf, s, nothing)
        rows[s] = RA.join_function(s, defs[target[s]], own === nothing ? get(dwarf, target[s], nothing) : own, types)
    end
    return (; defs, aliases, exported, symbols, skipped, foreign, target, units, dwarf, types, rows)
end

# A function of the crate by the end of its symbol (`4lean3add`): the crate's
# disambiguator in front of it is a hash of the build.
function rust_row(x, tail::AbstractString)
    hits = [s for s in x.symbols if endswith(s, tail)]
    length(hits) == 1 || error("$(length(hits)) symbols end in $tail")
    return x.rows[hits[1]]
end
passes(row) = [(p["name"], p["pass"], p["ir"]) for p in row["parameters"]]

"""
The metadata `RustBuild.extract_metadata` would write for one vendored build:
its rows in path order, each with the Rust path the demangler would print
(spelled here from the DWARF scope, since these tests run no tool).
"""
function rust_metadata(opt::Int)
    x = rust_extract(opt)
    rows = Dict{String,Any}[]
    for s in x.symbols
        row = copy(x.rows[s])
        own = get(x.dwarf, s, nothing)
        row["own_debug_info"] = own !== nothing
        x.target[s] == s || (row["alias_of"] = x.target[s])
        row["path"] = !startswith(s, "_R") ? s :
                      endswith(s, "4Area4area") ? "<lean::Shape as lean::Area>::area" :
                      row["owner"] !== nothing ? "<$(row["owner"])>::$(row["name"])" : "lean::$(row["name"])"
        push!(rows, row)
    end
    sort!(rows; by = r -> (r["path"], r["symbol"]))
    return Dict{String,Any}("language" => "rust", "metadata_version" => "rust-1",
                            "crate" => Dict{String,Any}("name" => "lean", "package" => "lean", "root" => "lean_rb"),
                            "library" => "liblean.so", "library_sha" => "0"^64,
                            "toolchain" => Dict{String,Any}("rustc" => "fixture"),
                            "functions" => rows, "types" => x.types, "skipped" => Any[])
end
