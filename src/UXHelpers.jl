#!/usr/bin/env julia
# UXHelpers.jl - User Experience Improvements
# Provides: Progress indicators, helpful error messages, formatting utilities

module UXHelpers

using ProgressMeter

export @progress_foreach, @progress_map, HelpfulError, show_helpful_error

"""
Progress-aware foreach that shows a progress bar
"""
macro progress_foreach(desc, iterator, body)
    quote
        items = $(esc(iterator))
        p = Progress(length(items), desc=$(esc(desc)))
        for item in items
            let $(esc(:item)) = item
                $(esc(body))
            end
            next!(p)
        end
    end
end

"""
Progress-aware map that shows a progress bar
"""
function progress_map(f, items; desc="Processing")
    results = similar(items, Any)
    p = Progress(length(items), desc=desc)
    for (i, item) in enumerate(items)
        results[i] = f(item)
        next!(p)
    end
    return results
end

"""
Structured error with helpful information and solutions
"""
struct HelpfulError <: Exception
    title::String
    description::String
    solutions::Vector{String}
    docs_link::String
    original_error::Union{Exception,Nothing}
end

function HelpfulError(title::String, description::String, solutions::Vector{String};
                     docs_link::String="", original_error::Union{Exception,Nothing}=nothing)
    return HelpfulError(title, description, solutions, docs_link, original_error)
end

"""
Display a helpful error message with formatting
"""
function show_helpful_error(io::IO, err::HelpfulError)
    println(io, "\n" * "="^70)
    println(io, "❌ Error: $(err.title)")
    println(io, "="^70)
    println(io)
    println(io, err.description)

    if !isempty(err.solutions)
        println(io, "\n💡 Possible solutions:")
        for (i, solution) in enumerate(err.solutions)
            println(io, "   $i. $solution")
        end
    end

    if !isempty(err.docs_link)
        println(io, "\n📚 Documentation: $(err.docs_link)")
    end

    if err.original_error !== nothing
        println(io, "\n🔍 Original error:")
        println(io, "   $(err.original_error)")
    end

    println(io, "\n" * "="^70)
end

Base.showerror(io::IO, err::HelpfulError) = show_helpful_error(io, err)

"""
Create a helpful LLVM not found error
"""
function llvm_not_found_error()
    return HelpfulError(
        "LLVM Toolchain Not Found",
        """
        RepliBuild requires an LLVM toolchain with clang++ for C++ compilation.
        No LLVM installation was found (tried JLL artifact, in-tree, and system paths).
        """,
        [
            "Install LLVM_full_assert_jll: julia> using Pkg; Pkg.add(\"LLVM_full_assert_jll\")",
            "Install system LLVM: sudo pacman -S clang llvm  (or your package manager)",
            "Manually specify LLVM path in replibuild.toml:\n     [llvm]\n     root = \"/path/to/llvm\""
        ],
        docs_link="https://github.com/user/RepliBuild.jl#llvm-setup"
    )
end

"""
Create a helpful configuration error
"""
function config_validation_error(field::String, issue::String)
    return HelpfulError(
        "Configuration Validation Failed",
        "Invalid value for field '$field': $issue",
        [
            "Check your replibuild.toml configuration file",
            "Run: RepliBuild.generate_default_config() to create a valid template",
            "See example configs in: examples/"
        ],
        docs_link="https://github.com/user/RepliBuild.jl#configuration"
    )
end

"""
Create a helpful compiler error with learning suggestions
"""
function compiler_error(source_file::String, error_output::String)
    return HelpfulError(
        "Compilation Failed",
        "Failed to compile: $source_file",
        [
            "Check the error output below for syntax or semantic errors",
            "Ensure all #include paths are correct in your replibuild.toml [compile] section",
            "Try adding -v to compiler flags for verbose output",
            "RepliBuild's error learning system will auto-fix common issues on retry"
        ],
        docs_link="https://github.com/user/RepliBuild.jl#troubleshooting",
        original_error=ErrorException(error_output)
    )
end

end  # module UXHelpers
