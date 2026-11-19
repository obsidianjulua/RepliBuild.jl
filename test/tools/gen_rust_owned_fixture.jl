# gen_rust_owned_fixture.jl — build test/rust_test/crate with cargo and vendor
# the part of its metadata that is about ownership, so the unit tests of the
# generator and of `RustGlue` need no toolchain:
#
#   test/fixtures/rust/owned_o3.metadata.json
#
# It is `compilation_metadata.json` of an opt-level 3 build, cut to the
# functions named in `KEEP`, the types those reach, and the build's answers
# about those types (`glue`, `glue_refused`). Nothing in it is written by
# hand: the layouts, the niche values and the "does the compiler drop this"
# answers are the compiler's.
#
#     julia --project=. test/tools/gen_rust_owned_fixture.jl
#
# Hand-run; needs cargo, rustc and GNU readelf. Refuses to write a fixture
# that does not hold what the tests look for.

using RepliBuild
using JSON: JSON

const RG = RepliBuild.RustGlue
const CRATE = normpath(joinpath(@__DIR__, "..", "rust_test", "crate"))
const OUT = normpath(joinpath(@__DIR__, "..", "fixtures", "rust", "owned_o3.metadata.json"))

# By Rust path. Each is here for one way a value is owned or handed over.
const KEEP = ["rb_fix::make_greeting", "rb_fix::string_len_owned", "rb_fix::string_ref_len", "rb_fix::shout",
              "rb_fix::squares", "rb_fix::vec_sum", "rb_fix::vec_push", "rb_fix::recs", "rb_fix::words",
              "rb_fix::join_words", "rb_fix::parse_int", "rb_fix::check", "rb_fix::find_word",
              "rb_fix::name_and_len", "rb_fix::boxed", "rb_fix::unbox_d", "rb_fix::box_d", "rb_fix::opt_box",
              "rb_fix::slot_make", "rb_fix::slot_d", "rb_fix::slot_take", "rb_fix::named", "rb_fix::named_len",
              "rb_fix::named_into_name", "rb_fix::token", "rb_fix::token_id", "rb_fix::guard",
              "rb_fix::guard_release", "rb_fix::view", "rb_fix::view_len", "rb_fix::secret", "rb_fix::secret_len",
              "rb_fix::rec_sum_ref", "rb_fix::make_area", "rb_fix::flip", "rb_fix::flag", "rb_fix::compare",
              "rb_fix::ordering_sign", "rb_fix::level_next", "rb_fix::bytes16", "rb_fix::bytes16_sum", "rb_fix::grid",
              "rb_fix::opt_bytes", "rb_fix::recs_total"]

# Every type a description names.
function reach!(seen::Set{String}, types, key)
    (key isa AbstractString && haskey(types, key) && !(key in seen)) || return
    push!(seen, String(key))
    t = types[key]
    for k in ("target", "element", "underlying", "return")
        reach!(seen, types, get(t, k, nothing))
    end
    for k in ("fields", "variants", "params")
        foreach(f -> reach!(seen, types, get(f, "type", nothing)), get(t, k, Any[]))
    end
    haskey(t, "discriminant") && reach!(seen, types, get(t["discriminant"], "type", nothing))
    foreach(p -> reach!(seen, types, p), get(t, "parameters", Any[]))
    return
end

mktempdir() do dir
    cp(CRATE, joinpath(dir, "crate"))
    rm(joinpath(dir, "crate", "target"); force = true, recursive = true)
    toml = joinpath(dir, "replibuild.toml")
    write(toml, """
        [project]
        name = "rb_fix"

        [rust]
        manifest = "crate/Cargo.toml"
        opt_level = 3
        """)
    lib = withenv("REPLIBUILD_JULIA_ENV" => "none", "REPLIBUILD_HOME" => mktempdir()) do
        RepliBuild.build(toml)
    end
    md = RepliBuild._read_json(joinpath(dirname(lib), "compilation_metadata.json"))

    functions = [f for f in md["functions"] if f["path"] in KEEP]
    missing_paths = setdiff(KEEP, [f["path"] for f in functions])
    isempty(missing_paths) || error("the build has no function at: $(join(missing_paths, ", "))")
    all(f -> get(f, "own_debug_info", false) === true, functions) ||
        error("a kept function has no debug info of its own: the fixture would describe another function's types")

    seen = Set{String}()
    for f in functions
        foreach(p -> reach!(seen, md["types"], get(p, "type", nothing)), f["parameters"])
        reach!(seen, md["types"], f["return"]["type"])
        reach!(seen, md["types"], get(f, "owner", nothing))
    end
    types = Dict{String,Any}(k => md["types"][k] for k in seen)
    need = RG.wanted(functions, types)
    glue = Dict{String,Any}(k => v for (k, v) in md["glue"] if k in need)
    refused = Dict{String,Any}(k => v for (k, v) in md["glue_refused"] if k in need)

    # What the tests look for.
    owns(k) = get(get(glue, k, Dict()), "owns", nothing)
    owns("alloc::string::String") === true || error("the build does not say a String is dropped")
    owns("rb_fix::Guard") === true || error("the build does not say Guard (a type with only `impl Drop`) is dropped")
    owns("rb_fix::View") === false || error("the build does not say View (a raw pointer, no drop code) is plain")
    owns("rb_fix::Rec") === false || error("the build does not say Rec is plain")
    get(get(glue, "rb_fix::inner::Token", Dict()), "spelling", "") == "rb_fix::Token" ||
        error("Token was not found under its public path")
    haskey(refused, "rb_fix::hidden::Secret") || error("Secret, which no path names, was not refused")
    haskey(get(glue, "alloc::vec::Vec<f64, alloc::alloc::Global>", Dict()), "vec") || error("Vec<f64> has no functions")
    isempty(setdiff(need, union(keys(glue), keys(refused)))) || error("a wanted type has neither an answer nor a refusal")

    out = Dict{String,Any}(
        "metadata_version" => md["metadata_version"], "language" => "rust", "crate" => md["crate"],
        "library" => md["library"], "library_sha" => "0"^64,
        "toolchain" => Dict{String,Any}("rustc" => md["toolchain"]["rustc"]),
        "functions" => functions, "types" => types, "glue" => glue, "glue_refused" => refused,
        "glue_helpers" => md["glue_helpers"], "skipped" => Any[])
    open(OUT, "w") do io
        JSON.print(io, out, 1)
    end
    println("wrote $OUT: $(length(functions)) functions, $(length(types)) types, $(length(glue)) answers, ",
            "$(length(refused)) refused (rustc $(md["toolchain"]["rustc"]))")
end
