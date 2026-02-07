#!/usr/bin/env julia
# AgentControl.jl - CLI Interface for AI Agents
# Usage: julia scripts/AgentControl.jl <command> [args...]

using Pkg
Pkg.activate(".")

using RepliBuild

function main()
    if length(ARGS) < 1
        println("Usage: julia AgentControl.jl <command>")
        println("Commands:")
        println("  build [toml_path]   - Compile C++ library")
        println("  wrap [toml_path]    - Generate Julia wrappers")
        println("  clean [toml_path]   - Remove build artifacts")
        println("  info [toml_path]    - Show project status")
        println("  discover [path]     - Scan for C++ code")
        println("  regenerate          - Clean, Build, and Wrap")
        exit(1)
    end

    command = ARGS[1]
    toml_path = length(ARGS) > 1 ? ARGS[2] : "replibuild.toml"

    try
        if command == "build"
            RepliBuild.build(toml_path)
        elseif command == "wrap"
            RepliBuild.wrap(toml_path)
        elseif command == "clean"
            RepliBuild.clean(toml_path)
        elseif command == "info"
            RepliBuild.info(toml_path)
        elseif command == "discover"
            path = length(ARGS) > 1 ? ARGS[2] : "."
            RepliBuild.discover(path)
        elseif command == "regenerate"
            println(">> Agent Command: Regenerate All")
            RepliBuild.clean(toml_path)
            RepliBuild.build(toml_path)
            RepliBuild.wrap(toml_path)
        else
            println("Unknown command: $command")
            exit(1)
        end
        
        println("
[AgentControl] Success: $command completed.")
        exit(0)
    catch e
        println("
[AgentControl] Error during $command:")
        showerror(stdout, e, catch_backtrace())
        println()
        exit(1)
    end
end

main()
