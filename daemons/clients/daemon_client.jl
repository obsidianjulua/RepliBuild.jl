#!/usr/bin/env julia
"""
Daemon Client - Send requests to daemon servers

Usage:
    julia daemon_client.jl --server build --command handle_build_request --target mylib
    julia daemon_client.jl --server error --command handle_error --error "undefined reference"
    julia daemon_client.jl --server watcher --command start_watch --path ./src
"""

using DaemonMode

const SERVERS = Dict(
    "build" => 3001,
    "error" => 3002,
    "watcher" => 3003
)

"""
Send a request to a daemon server
"""
function send_request(server::String, func::String, args::Dict)
    if !haskey(SERVERS, server)
        println("Error: Unknown server '$server'")
        println("Available servers: $(join(keys(SERVERS), ", "))")
        return nothing
    end

    port = SERVERS[server]

    try
        # Build expression to call the function with arguments
        expr = :($(Symbol(func))($args))

        # Send to daemon
        result = runexpr(expr, port=port)

        return result

    catch e
        println("Error communicating with daemon on port $port:")
        println(e)
        return nothing
    end
end

"""
Parse command line arguments
"""
function parse_args()
    args = Dict{String, Any}()
    func_args = Dict{String, Any}()

    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]

        if arg == "--server"
            args["server"] = ARGS[i+1]
            i += 2
        elseif arg == "--command"
            args["command"] = ARGS[i+1]
            i += 2
        elseif startswith(arg, "--")
            # Parse as function argument
            key = arg[3:end]
            if i < length(ARGS) && !startswith(ARGS[i+1], "--")
                func_args[key] = ARGS[i+1]
                i += 2
            else
                func_args[key] = true
                i += 1
            end
        else
            i += 1
        end
    end

    args["func_args"] = func_args
    return args
end

"""
Main entry point
"""
function main()
    if length(ARGS) == 0
        println("""
        Daemon Client - Send requests to daemon servers

        Usage:
            julia daemon_client.jl --server <server> --command <function> [--args...]

        Servers:
            build    - Build daemon (port 3001)
            error    - Error handler daemon (port 3002)
            watcher  - File watcher daemon (port 3003)

        Examples:
            # Build a target
            julia daemon_client.jl --server build --command handle_build_request --target mylib

            # Process an error
            julia daemon_client.jl --server error --command handle_error --error "undefined reference"

            # Start watching files
            julia daemon_client.jl --server watcher --command start_watch --path ./src --interval 1.0

            # Check for changes
            julia daemon_client.jl --server watcher --command check_changes --path ./src
        """)
        return
    end

    args = parse_args()

    if !haskey(args, "server") || !haskey(args, "command")
        println("Error: --server and --command are required")
        return
    end

    println("Sending request to $(args["server"]) daemon...")
    result = send_request(args["server"], args["command"], args["func_args"])

    if result !== nothing
        println("\nResult:")
        println(result)
    end
end

# Run if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
