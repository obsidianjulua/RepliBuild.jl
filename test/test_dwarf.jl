#!/usr/bin/env julia
# DWARF Extraction Tester - Fast feedback on extraction capabilities
# Usage: julia --project=.. test_dwarf.jl

using JSON
using RepliBuild

# ==============================================================================
# COMPILE TEST BINARY
# ==============================================================================

function compile(cpp::String, out::String)
    cmd = `clang++ -g -O0 -std=c++17 -fPIC -shared -o $out $cpp`
    println("⚙ Compiling with -g (DWARF debug info)")
    success(cmd) || error("Compilation failed")
    println("✓ Compiled: $out")
    filesize(out) / 1024 |> x -> println("   Size: $(round(x, digits=1)) KB")
end

# ==============================================================================
# EXTRACT WITH REPLIBUILD (via direct call)
# ==============================================================================

function extract_replibuild(binary::String)
    # Access internal Compiler module
    (output, exitcode) = RepliBuild.BuildBridge.execute("readelf", ["--debug-dump=info", binary])

    if exitcode != 0
        error("readelf failed: $output")
    end

    # Call the internal extraction function directly
    (return_types, struct_defs) = RepliBuild.Compiler.extract_dwarf_return_types(binary)

    println("\n🔍 RepliBuild Extraction:")
    println("   Functions:  $(length(return_types))")
    println("   Types:      $(length(struct_defs))")

    # Count by category
    structs = count(k -> !startswith(k, "__enum__") && haskey(struct_defs[k], "members"), keys(struct_defs))
    enums = count(k -> startswith(k, "__enum__"), keys(struct_defs))

    println("     Structs:  $structs")
    println("     Enums:    $enums")

    return Dict("return_types" => return_types, "struct_defs" => struct_defs)
end

# ==============================================================================
# EXTRACT RAW DWARF WITH TOOLS
# ==============================================================================

function extract_raw_dwarf(binary::String)
    println("\n📖 Raw DWARF Extraction:")

    # Try dwarfdump first (best output)
    dwarfdump_output = nothing
    try
        dwarfdump_output = read(`llvm-dwarfdump --debug-info $binary`, String)
        println("   ✓ llvm-dwarfdump: $(length(dwarfdump_output)) bytes")
    catch
        try
            dwarfdump_output = read(`dwarfdump --debug-info $binary`, String)
            println("   ✓ dwarfdump: $(length(dwarfdump_output)) bytes")
        catch
            println("   ℹ dwarfdump not found (optional)")
        end
    end

    # readelf (standard)
    readelf_output = try
        out = read(`readelf --debug-dump=info $binary`, String)
        println("   ✓ readelf: $(length(out)) bytes")
        out
    catch e
        println("   ✗ readelf failed: $e")
        nothing
    end

    # objdump (alternative)
    objdump_output = try
        out = read(`objdump --dwarf=info $binary`, String)
        println("   ✓ objdump: $(length(out)) bytes")
        out
    catch
        nothing
    end

    return (dwarfdump_output, readelf_output, objdump_output)
end

# ==============================================================================
# ANALYZE DWARF TAGS
# ==============================================================================

function analyze_tags(raw_output::String)
    tags = Dict{String,Int}()

    for line in split(raw_output, '\n')
        m = match(r"DW_TAG_(\w+)", line)
        !isnothing(m) && (tags[m.captures[1]] = get(tags, m.captures[1], 0) + 1)
    end

    return tags
end

# ==============================================================================
# COMPARE TOOLS
# ==============================================================================

function compare_tools(replibuild_data, tags_found)
    println("\n" * "="^70)
    println("COMPARISON: RepliBuild vs Full DWARF")
    println("="^70)

    # Supported tags in RepliBuild (updated with new extractions)
    supported = Set([
        "subprogram", "base_type", "pointer_type", "const_type", "volatile_type",
        "reference_type", "structure_type", "class_type", "enumeration_type",
        "enumerator", "member", "array_type", "subrange_type",
        "subroutine_type", "formal_parameter", "rvalue_reference_type",
        # Newly added
        "typedef", "inheritance", "template_type_parameter", "template_value_parameter",
        "namespace"
    ])

    # High-priority missing (not yet extracted)
    priority = Dict(
        "imported_declaration" => "Using declarations (not critical for FFI)",
        "unspecified_parameters" => "Variadic functions (...)",
        "variable" => "Global variables"
    )

    println("\n🏷  DWARF Tag Coverage:")
    println("-" ^ 70)

    sorted = sort(collect(tags_found), by=x->x[2], rev=true)
    total_instances = sum(values(tags_found))
    supported_instances = sum(count for (tag, count) in tags_found if tag in supported)

    for (tag, count) in sorted
        if count > 5  # Only show significant tags
            pct = round(100 * count / total_instances, digits=1)

            status = if tag in supported
                "✅"
            elseif haskey(priority, tag)
                "🎯"
            else
                "  "
            end

            info = get(priority, tag, "")
            println("  $status  $(rpad(tag, 32))  $(lpad(count, 5))  ($pct%)")
            !isempty(info) && println("      → $info")
        end
    end

    println("-" ^ 70)
    println("Coverage: $supported_instances / $total_instances instances ($(round(100*supported_instances/total_instances, digits=1))%)")
    println("Supported: $(length(filter(t -> t in supported, keys(tags_found)))) / $(length(tags_found)) tag types")

    # Show what we're missing
    missing_priority = [(tag, desc) for (tag, desc) in priority if haskey(tags_found, tag)]
    if !isempty(missing_priority)
        println("\n🎯 High-Priority Missing Features:")
        for (tag, desc) in missing_priority
            println("   • $tag - $desc ($(tags_found[tag]) instances)")
        end
    end
end

# ==============================================================================
# SHOW EXTRACTION SAMPLES
# ==============================================================================

function show_samples(replibuild_data, raw_dwarf)
    println("\n" * "="^70)
    println("EXTRACTION SAMPLES")
    println("="^70)

    # Show a few functions
    println("\n📋 Functions (sample):")
    funcs = collect(replibuild_data["return_types"])
    for (i, (name, info)) in enumerate(funcs[1:min(5, length(funcs))])
        ret = get(info, "julia_type", "?")
        params = get(info, "parameters", [])
        param_types = [get(p, "julia_type", "Any") for p in params]
        println("   $i. $name")
        println("      → $ret($(join(param_types, ", ")))")
    end

    # Show a few structs
    println("\n🏗  Structs (sample):")
    count = 0
    for (name, info) in replibuild_data["struct_defs"]
        startswith(name, "__enum__") && continue
        !haskey(info, "members") && continue

        count += 1
        count > 3 && break

        members = info["members"]
        println("   struct $name {")
        for m in members[1:min(3, length(members))]
            mtype = get(m, "julia_type", "?")
            mname = get(m, "name", "?")
            println("      $mtype $mname;")
        end
        length(members) > 3 && println("      ... $(length(members)-3) more")
        println("   }")
    end

    # Show enums
    println("\n🔢 Enums (sample):")
    count = 0
    for (key, info) in replibuild_data["struct_defs"]
        !startswith(key, "__enum__") && continue

        count += 1
        count > 2 && break

        name = replace(key, "__enum__" => "")
        enums = get(info, "enumerators", [])
        println("   enum $name {")
        for e in enums[1:min(4, length(enums))]
            ename = get(e, "name", "?")
            eval_val = get(e, "value", "?")
            println("      $ename = $eval_val,")
        end
        println("   }")
    end
end

# ==============================================================================
# SAVE COMPARISON FILES
# ==============================================================================

function save_comparison(binary, replibuild_data, dwarfdump, readelf, tags)
    dir = @__DIR__

    # Save RepliBuild extraction
    open(joinpath(dir, "replibuild_extraction.json"), "w") do io
        JSON.print(io, replibuild_data, 2)
    end

    # Save raw DWARF dumps
    !isnothing(dwarfdump) && write(joinpath(dir, "dwarfdump_full.txt"), dwarfdump)
    !isnothing(readelf) && write(joinpath(dir, "readelf_full.txt"), readelf)

    # Save tag comparison
    open(joinpath(dir, "tag_comparison.txt"), "w") do io
        println(io, "DWARF Tag Analysis")
        println(io, "="^70)
        println(io, "\nFormat: <tag_name> <count> <% of total>")
        println(io, "-"^70)

        total = sum(values(tags))
        sorted = sort(collect(tags), by=x->x[2], rev=true)

        for (tag, count) in sorted
            pct = round(100 * count / total, digits=1)
            println(io, "$(rpad(tag, 35)) $(lpad(count, 6))  ($pct%)")
        end
    end

    println("\n💾 Saved Outputs:")
    println("   replibuild_extraction.json  - What RepliBuild extracted")
    println("   tag_comparison.txt          - DWARF tag frequency analysis")
    !isnothing(dwarfdump) && println("   dwarfdump_full.txt          - Full dwarfdump output")
    !isnothing(readelf) && println("   readelf_full.txt            - Full readelf output")
end

# ==============================================================================
# MAIN
# ==============================================================================

function main()
    dir = @__DIR__
    cpp = joinpath(dir, "dwarf_test.cpp")
    binary = joinpath(dir, "dwarf_test.so")

    println("="^70)
    println("DWARF EXTRACTION TEST & COMPARISON")
    println("="^70)

    # Compile with debug info
    println("\n▶ Step 1: Compile with -g")
    compile(cpp, binary)

    # Extract with RepliBuild
    println("\n▶ Step 2: Extract with RepliBuild")
    replibuild_data = extract_replibuild(binary)

    # Extract raw DWARF
    println("\n▶ Step 3: Extract raw DWARF")
    (dwarfdump, readelf, objdump) = extract_raw_dwarf(binary)

    # Analyze tags
    println("\n▶ Step 4: Analyze DWARF tags")
    raw = !isnothing(dwarfdump) ? dwarfdump : readelf
    tags = analyze_tags(raw)
    println("   Found $(length(tags)) unique tag types")

    # Compare
    compare_tools(replibuild_data, tags)

    # Show samples
    show_samples(replibuild_data, raw)

    # Save everything
    save_comparison(binary, replibuild_data, dwarfdump, readelf, tags)

    println("\n" * "="^70)
    println("✅ COMPLETE - Check output files for detailed comparison")
    println("="^70)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
