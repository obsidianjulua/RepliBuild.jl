#!/usr/bin/env julia
"""
Reactive Handler - Orchestrates daemon responses to events

This script defines reactive logic: "When X happens, do Y"

Examples:
    - File changed -> trigger rebuild
    - Build error -> send to error handler -> auto-fix if possible
    - Error pattern detected -> update learning database
"""

using DaemonMode

include("../clients/daemon_client.jl")

"""
Event definitions and handlers
"""
const EVENT_HANDLERS = Dict{String, Function}()

"""
Register an event handler
"""
function on_event(event_type::String, handler::Function)
    EVENT_HANDLERS[event_type] = handler
end

"""
Dispatch an event to its handler
"""
function handle_event(event_type::String, data::Dict)
    if haskey(EVENT_HANDLERS, event_type)
        handler = EVENT_HANDLERS[event_type]
        try
            return handler(data)
        catch e
            println("[REACTIVE] Error handling event '$event_type': $e")
            return Dict(:success => false, :error => string(e))
        end
    else
        println("[REACTIVE] No handler for event '$event_type'")
        return Dict(:success => false, :error => "No handler registered")
    end
end

# ============================================================================
# Event Handler Definitions
# ============================================================================

"""
Handler: File Changed -> Rebuild
"""
on_event("file_changed") do data
    println("[REACTIVE] File changed: $(data["file"])")

    # Determine which target to rebuild based on file
    target = get(data, "target", "")
    config = get(data, "config", "replibuild.toml")

    # Send rebuild request to build daemon
    build_result = send_request("build", "handle_build_request", Dict(
        "target" => target,
        "config" => config,
        "force" => true
    ))

    if build_result !== nothing && get(build_result, :success, false)
        println("[REACTIVE] Rebuild successful")
        return build_result
    else
        # Build failed, send error to error handler
        if build_result !== nothing && haskey(build_result, :error)
            handle_event("build_error", merge(data, Dict(
                "error" => build_result[:error]
            )))
        end
        return build_result
    end
end

"""
Handler: Build Error -> Error Analysis -> Auto-fix
"""
on_event("build_error") do data
    println("[REACTIVE] Build error detected")

    error_text = get(data, "error", "")

    # Send to error handler daemon
    error_result = send_request("error", "handle_error", Dict(
        "error" => error_text,
        "context" => data,
        "auto_fix" => get(data, "auto_fix", false)
    ))

    if error_result !== nothing
        println("[REACTIVE] Error analysis complete")

        # If high confidence solution found, optionally auto-fix
        if get(error_result, :confidence, 0.0) > 0.8
            println("[REACTIVE] High confidence solution found ($(error_result[:confidence]))")

            if get(data, "auto_fix", false)
                println("[REACTIVE] Applying auto-fix...")
                # Auto-fix logic would go here
            else
                println("[REACTIVE] Auto-fix disabled. Manual intervention required.")
            end
        end

        return error_result
    end

    return Dict(:success => false, :error => "Error handler failed")
end

"""
Handler: Watch Started -> Continuous Monitoring
"""
on_event("watch_started") do data
    println("[REACTIVE] Starting continuous file monitoring")

    path = get(data, "path", "")
    interval = get(data, "interval", 1.0)
    patterns = get(data, "patterns", ["*.cpp", "*.h"])

    # Start watch on watcher daemon
    watch_result = send_request("watcher", "start_watch", Dict(
        "path" => path,
        "interval" => interval,
        "patterns" => patterns
    ))

    if watch_result !== nothing && get(watch_result, :success, false)
        println("[REACTIVE] Watching $(watch_result[:files_count]) files")

        # Start periodic change checking (in a real system, this would be async)
        while true
            sleep(interval)

            changes = send_request("watcher", "check_changes", Dict(
                "path" => path,
                "patterns" => patterns
            ))

            if changes !== nothing && get(changes, :count, 0) > 0
                println("[REACTIVE] Detected $(changes[:count]) change(s)")

                # Trigger file_changed event for each change
                for change in changes[:changes]
                    handle_event("file_changed", merge(data, change))
                end
            end
        end
    end

    return watch_result
end

"""
Handler: Successful Build -> Update Statistics
"""
on_event("build_success") do data
    println("[REACTIVE] Build succeeded, updating statistics")

    # Could update build time stats, success rates, etc.
    return Dict(:success => true, :logged => true)
end

# ============================================================================
# Main Entry Point
# ============================================================================

"""
CLI for reactive event handling
"""
function main()
    if length(ARGS) == 0
        println("""
        Reactive Handler - Orchestrate daemon responses to events

        Usage:
            julia reactive_handler.jl <event_type> [--key value...]

        Available Events:
            file_changed     - Trigger rebuild on file change
            build_error      - Handle and learn from build errors
            watch_started    - Start continuous file monitoring
            build_success    - Log successful build

        Examples:
            # Simulate file change
            julia reactive_handler.jl file_changed --file src/main.cpp --target mylib

            # Handle build error
            julia reactive_handler.jl build_error --error "undefined reference to foo"

            # Start watching directory
            julia reactive_handler.jl watch_started --path ./src --interval 2.0
        """)
        return
    end

    event_type = ARGS[1]
    data = Dict{String, Any}()

    # Parse additional arguments
    i = 2
    while i <= length(ARGS)
        if startswith(ARGS[i], "--")
            key = ARGS[i][3:end]
            if i < length(ARGS) && !startswith(ARGS[i+1], "--")
                data[key] = ARGS[i+1]
                i += 2
            else
                data[key] = true
                i += 1
            end
        else
            i += 1
        end
    end

    println("Processing event: $event_type")
    result = handle_event(event_type, data)

    println("\nEvent Result:")
    println(result)
end

# Run if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
