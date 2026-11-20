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
const KEEP = ["rb_fix::add", "rb_fix::make_greeting", "rb_fix::string_len_owned", "rb_fix::string_ref_len", "rb_fix::shout",
              "rb_fix::squares", "rb_fix::vec_sum", "rb_fix::vec_push", "rb_fix::recs", "rb_fix::words",
              "rb_fix::join_words", "rb_fix::parse_int", "rb_fix::check", "rb_fix::find_word",
              "rb_fix::name_and_len", "rb_fix::boxed", "rb_fix::unbox_d", "rb_fix::box_d", "rb_fix::opt_box",
              "rb_fix::slot_make", "rb_fix::slot_d", "rb_fix::slot_take", "rb_fix::named", "rb_fix::named_len",
              "rb_fix::named_into_name", "rb_fix::token", "rb_fix::token_id", "rb_fix::guard",
              "rb_fix::guard_release", "rb_fix::view", "rb_fix::view_len", "rb_fix::secret", "rb_fix::secret_len",
              "rb_fix::rec_sum_ref", "rb_fix::make_area", "rb_fix::flip", "rb_fix::flag", "rb_fix::compare",
              "rb_fix::ordering_sign", "rb_fix::level_next", "rb_fix::bytes16", "rb_fix::bytes16_sum", "rb_fix::grid",
              "rb_fix::opt_bytes", "rb_fix::recs_total",
              # Functions the library has no symbol for, and a destructor, which it has and nothing may call.
              "rb_fix::small", "rb_fix::sum_diff", "rb_fix::divide", "<rb_fix::Named>::rename",
              "<rb_fix::Tag as core::clone::Clone>::clone", "rb_fix::midpoint::<f64>", "<rb_fix::Pair<i32>>::sum",
              "<rb_fix::Guard as core::ops::drop::Drop>::drop",
              # Types no path spells, taken from the functions that return them; and one nothing returns.
              "rb_fix::secret_take", "rb_fix::parse_byte", "rb_fix::fault_text", "rb_fix::sealed_len",
              # Trait objects: taken, returned, and a trait's methods called through one.
              "rb_fix::dyn_area", "rb_fix::dyn_size", "rb_fix::dyn_grow", "rb_fix::boxed_scale",
              "<dyn rb_fix::Area as rb_fix::Area>::area", "<dyn rb_fix::Scale as rb_fix::Scale>::label",
              "<dyn rb_fix::Scale as rb_fix::Scale>::scale", "<rb_fix::F2 as rb_fix::Scale>::size",
              "<rb_fix::Named as rb_fix::Scale>::size", "<f64 as rb_fix::Scale>::size",
              "<rb_fix::Shape as rb_fix::Area>::area", "rb_fix::f2_make",
              # Callbacks: a Rust function pointer, a C one, one taking a record, a generic one, one not carried.
              "rb_fix::apply", "rb_fix::apply_c", "rb_fix::each", "rb_fix::count_if", "rb_fix::map_sum::<fn(f64) -> f64>",
              "rb_fix::text_callback"]

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
        instantiate = ["rb_fix::midpoint::<f64>", "rb_fix::Pair::<i32>::sum", "rb_fix::map_sum::<fn(f64) -> f64>"]
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
    foreach(s -> reach!(seen, md["types"], s["type"]), md["statics"])
    foreach(k -> reach!(seen, md["types"], k), keys(md["callbacks"]))
    for d in keys(md["dyn"]), ref in ("&" * d, "&mut " * d)
        reach!(seen, md["types"], ref)
        foreach(t -> reach!(seen, md["types"], t), keys(md["dyn"][d]["implementors"]))
    end
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
    get(get(glue, "rb_fix::hidden::Secret", Dict()), "by_address", false) === true ||
        error("Secret, which no path names, was not taken from the function that returns it")
    get(get(glue, "rb_fix::hidden::Fault", Dict()), "by_address", false) === true ||
        error("Fault, exported under another name, was not taken from the function that returns it")
    haskey(refused, "rb_fix::hidden::Sealed") || error("Sealed, which no path names and nothing returns, was not refused")
    haskey(get(glue, "alloc::vec::Vec<f64, alloc::alloc::Global>", Dict()), "vec") || error("Vec<f64> has no functions")
    isempty(setdiff(need, union(keys(glue), keys(refused)))) || error("a wanted type has neither an answer nor a refusal")
    by_path = Dict(f["path"] => f for f in functions)
    get(by_path["rb_fix::small"], "instance", "") == "inline" || error("`small`, an inline function, was not compiled in")
    get(by_path["rb_fix::midpoint::<f64>"], "instance", "") == "instantiate" || error("`midpoint::<f64>` was not instantiated")
    haskey(by_path["rb_fix::add"], "address") && error("`add`, which the library exports, is reached by an address")
    length(md["statics"]) >= 5 || error("the build found no statics")
    sort!(collect(keys(md["dyn"]["dyn rb_fix::Scale"]["implementors"]))) == ["f64", "rb_fix::F2", "rb_fix::Named"] ||
        error("the build did not find the three types that implement Scale")
    haskey(md["callbacks"], "fn(i32) -> i32") || error("the build has no function for the callback `fn(i32) -> i32`")
    by_path["rb_fix::apply"]["refusal"] === nothing || error("`apply`, which takes a function pointer, was refused")

    out = Dict{String,Any}(
        "metadata_version" => md["metadata_version"], "language" => "rust", "crate" => md["crate"],
        "library" => md["library"], "library_sha" => "0"^64,
        "toolchain" => Dict{String,Any}("rustc" => md["toolchain"]["rustc"]),
        "functions" => functions, "types" => types, "glue" => glue, "glue_refused" => refused,
        "glue_helpers" => md["glue_helpers"], "instances" => md["instances"], "statics" => md["statics"],
        "dyn" => md["dyn"], "callbacks" => md["callbacks"], "skipped" => Any[])
    open(OUT, "w") do io
        JSON.print(io, out, 1)
    end
    println("wrote $OUT: $(length(functions)) functions, $(length(types)) types, $(length(glue)) answers, ",
            "$(length(refused)) refused (rustc $(md["toolchain"]["rustc"]))")
end
