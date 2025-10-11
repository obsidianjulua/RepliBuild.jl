#!/usr/bin/env julia
"""
Orchestrator Daemon Server - Manages the full RepliBuild build pipeline

Start with: julia --project=.. orchestrator_daemon.jl
Port: 3004

Coordinates:
- Discovery Daemon (3001) - File scanning, AST, binaries
- Setup Daemon (3002) - Configuration management
- Compilation Daemon (3003) - Parallel builds
- Error Handler Daemon (3002) - Error learning
- Watcher Daemon (3003) - File monitoring

Pipeline: compile(path) → discover → setup → compile → return
"""

using DaemonMode
using Dates

# Add project to load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "src"))

using RepliBuild

const PORT = 3004

# Daemon ports
const DAEMON_PORTS = Dict(
    "discovery" => 3001,
    "setup" => 3002,
    "compilation" => 3003,
    "error" => 3002,
    "watcher" => 3003
)

# Include daemon client utilities
include("../clients/daemon_client.jl")

# ============================================================================
# PIPELINE ORCHESTRATION
# ============================================================================

"""
Send request to a specific daemon
"""
function call_daemon(daemon::String, func::String, args::Dict)
    port = get(DAEMON_PORTS, daemon, 0)

    if port == 0
        return Dict(
            :success => false,
            :error => "Unknown daemon: $daemon"
        )
    end

    try
        # Build expression to call the function with arguments
        expr = :($(Symbol(func))($args))

        # Send to daemon
        result = runexpr(expr, port=port)

        return result

    catch e
        return Dict(
            :success => false,
            :error => "Daemon communication error: $e",
            :daemon => daemon,
            :port => port
        )
    end
end

"""
Check if all required daemons are running
"""
function check_daemons(args::Dict)
    println("[ORCHESTRATOR] Checking daemon status...")

    status = Dict{String, Bool}()

    for (daemon, port) in DAEMON_PORTS
        try
            # Try simple expression
            result = runexpr(:(1 + 1), port=port)
            status[daemon] = true
            println("[ORCHESTRATOR] ✓ $daemon daemon (port $port)")
        catch e
            status[daemon] = false
            println("[ORCHESTRATOR] ✗ $daemon daemon (port $port) - NOT RUNNING")
        end
    end

    all_running = all(values(status))

    return Dict(
        :success => true,
        :all_running => all_running,
        :daemons => status
    )
end

"""
Full build pipeline: discover → setup → compile
"""
function build_project(args::Dict)
    project_path = get(args, "path", pwd())
    force_discovery = get(args, "force_discovery", false)
    force_compile = get(args, "force_compile", false)

    println("="^70)
    println("[ORCHESTRATOR] Starting RepliBuild Build Pipeline")
    println("[ORCHESTRATOR] Project: $project_path")
    println("="^70)

    start_time = time()
    results = Dict{Symbol, Any}()

    try
        # Check daemons are running
        daemon_check = check_daemons(Dict())
        if !daemon_check[:all_running]
            return Dict(
                :success => false,
                :error => "Not all daemons are running",
                :daemon_status => daemon_check[:daemons]
            )
        end

        # Stage 1: Discovery
        println("\n📍 Stage 1: Discovery")
        println("-"^70)

        discovery_result = call_daemon("discovery", "discover_project", Dict(
            "path" => project_path,
            "force" => force_discovery
        ))

        results[:discovery] = discovery_result

        if !discovery_result[:success]
            return Dict(
                :success => false,
                :stage => "discovery",
                :error => discovery_result[:error],
                :results => results
            )
        end

        println("[ORCHESTRATOR] ✓ Discovery complete")
        if get(discovery_result, :cached, false)
            println("[ORCHESTRATOR]   (using cached results)")
        end

        # Stage 2: Setup/Configuration
        println("\n⚙️  Stage 2: Configuration")
        println("-"^70)

        setup_result = call_daemon("setup", "generate_config", Dict(
            "path" => project_path,
            "force" => force_discovery,
            "discovery_results" => discovery_result[:results]
        ))

        results[:setup] = setup_result

        if !setup_result[:success]
            return Dict(
                :success => false,
                :stage => "setup",
                :error => setup_result[:error],
                :results => results
            )
        end

        println("[ORCHESTRATOR] ✓ Configuration ready")

        # Validate configuration
        config_path = joinpath(project_path, "replibuild.toml")
        validate_result = call_daemon("setup", "validate_config", Dict(
            "config" => config_path
        ))

        if !validate_result[:valid]
            println("[ORCHESTRATOR] ⚠️  Configuration validation warnings:")
            for issue in validate_result[:issues]
                println("[ORCHESTRATOR]     • $issue")
            end
        end

        # Stage 3: Compilation
        println("\n🔨 Stage 3: Compilation")
        println("-"^70)

        compile_result = call_daemon("compilation", "compile_full_pipeline", Dict(
            "config" => config_path,
            "force" => force_compile
        ))

        results[:compilation] = compile_result

        if !compile_result[:success]
            # Try error handler if compilation failed
            println("[ORCHESTRATOR] ❌ Compilation failed, consulting error handler...")

            error_result = call_daemon("error", "handle_error", Dict(
                "error" => get(compile_result, :error, "Unknown error"),
                "context" => Dict("project" => project_path),
                "auto_fix" => false
            ))

            results[:error_analysis] = error_result

            return Dict(
                :success => false,
                :stage => "compilation",
                :error => compile_result[:error],
                :error_analysis => error_result,
                :results => results
            )
        end

        println("[ORCHESTRATOR] ✓ Compilation complete")
        println("[ORCHESTRATOR]   Library: $(compile_result[:library_path])")

        if haskey(compile_result, :compile_stats)
            stats = compile_result[:compile_stats]
            println("[ORCHESTRATOR]   Files compiled: $(stats[:total])")
            println("[ORCHESTRATOR]   From cache: $(stats[:cached_count])")
        end

        # Success!
        elapsed = time() - start_time

        println("\n" * "="^70)
        println("[ORCHESTRATOR] ✅ BUILD SUCCESSFUL")
        println("[ORCHESTRATOR] Time: $(round(elapsed, digits=2))s")
        println("[ORCHESTRATOR] Output: $(compile_result[:library_path])")
        println("="^70)

        return Dict(
            :success => true,
            :library_path => compile_result[:library_path],
            :elapsed_time => elapsed,
            :results => results
        )

    catch e
        elapsed = time() - start_time

        println("\n" * "="^70)
        println("[ORCHESTRATOR] ❌ BUILD FAILED")
        println("[ORCHESTRATOR] Time: $(round(elapsed, digits=2))s")
        println("[ORCHESTRATOR] Error: $e")
        println("="^70)

        return Dict(
            :success => false,
            :error => string(e),
            :stacktrace => sprint(showerror, e, catch_backtrace()),
            :elapsed_time => elapsed,
            :results => results
        )
    end
end

"""
Quick compile (assumes discovery already done)
"""
function quick_compile(args::Dict)
    project_path = get(args, "path", pwd())
    config_path = joinpath(project_path, "replibuild.toml")

    println("[ORCHESTRATOR] Quick compile: $project_path")

    if !isfile(config_path)
        return Dict(
            :success => false,
            :error => "No replibuild.toml found. Run full build_project first."
        )
    end

    try
        # Just run compilation stage
        compile_result = call_daemon("compilation", "compile_full_pipeline", Dict(
            "config" => config_path,
            "force" => get(args, "force", false)
        ))

        return compile_result

    catch e
        return Dict(
            :success => false,
            :error => string(e)
        )
    end
end

"""
Incremental build (only rebuild changed files)
"""
function incremental_build(args::Dict)
    project_path = get(args, "path", pwd())

    println("[ORCHESTRATOR] Incremental build: $project_path")

    # Just force=false compile (uses IR cache)
    return quick_compile(Dict(
        "path" => project_path,
        "force" => false
    ))
end

"""
Clean build (force recompilation)
"""
function clean_build(args::Dict)
    project_path = get(args, "path", pwd())

    println("[ORCHESTRATOR] Clean build: $project_path")

    try
        # Clear all caches
        call_daemon("discovery", "clear_caches", Dict())
        call_daemon("setup", "clear_cache", Dict())
        call_daemon("compilation", "clear_caches", Dict())

        # Run full build with force
        return build_project(Dict(
            "path" => project_path,
            "force_discovery" => true,
            "force_compile" => true
        ))

    catch e
        return Dict(
            :success => false,
            :error => string(e)
        )
    end
end

"""
Watch and rebuild on file changes
"""
function watch_and_build(args::Dict)
    project_path = get(args, "path", pwd())
    interval = get(args, "interval", 2.0)

    println("[ORCHESTRATOR] Starting watch mode: $project_path")
    println("[ORCHESTRATOR] Checking for changes every $(interval)s")
    println("[ORCHESTRATOR] Press Ctrl+C to stop")

    try
        # Start watching
        watch_result = call_daemon("watcher", "start_watch", Dict(
            "path" => project_path,
            "interval" => interval,
            "patterns" => ["*.cpp", "*.h", "*.c", "*.hpp"]
        ))

        if !watch_result[:success]
            return watch_result
        end

        println("[ORCHESTRATOR] Watching $(watch_result[:files_count]) files")

        # Poll for changes
        while true
            sleep(interval)

            changes = call_daemon("watcher", "check_changes", Dict(
                "path" => project_path,
                "patterns" => ["*.cpp", "*.h", "*.c", "*.hpp"]
            ))

            if changes[:success] && changes[:count] > 0
                println("\n[ORCHESTRATOR] 🔔 Detected $(changes[:count]) change(s)")

                for change in changes[:changes]
                    println("[ORCHESTRATOR]    $(change[:type]): $(change[:file])")
                end

                println("[ORCHESTRATOR] Triggering incremental rebuild...")

                rebuild_result = incremental_build(Dict("path" => project_path))

                if rebuild_result[:success]
                    println("[ORCHESTRATOR] ✅ Rebuild successful")
                else
                    println("[ORCHESTRATOR] ❌ Rebuild failed: $(rebuild_result[:error])")
                end
            end
        end

    catch e
        if isa(e, InterruptException)
            println("\n[ORCHESTRATOR] Watch mode stopped")
            return Dict(:success => true, :stopped => true)
        else
            return Dict(
                :success => false,
                :error => string(e)
            )
        end
    end
end

"""
Get build statistics from all daemons
"""
function get_stats(args::Dict)
    println("[ORCHESTRATOR] Gathering statistics from all daemons...")

    stats = Dict{String, Any}()

    # Discovery stats
    discovery_stats = call_daemon("discovery", "cache_stats", Dict())
    stats["discovery"] = discovery_stats

    # Setup stats
    setup_stats = call_daemon("setup", "cache_stats", Dict())
    stats["setup"] = setup_stats

    # Compilation stats
    compile_stats = call_daemon("compilation", "cache_stats", Dict())
    stats["compilation"] = compile_stats

    return Dict(
        :success => true,
        :stats => stats
    )
end

# ============================================================================
# MAIN
# ============================================================================

"""
Main daemon server function
"""
function main()
    println("="^70)
    println("RepliBuild Orchestrator Daemon Server")
    println("Port: $PORT")
    println("="^70)
    println()
    println("Managed Daemons:")
    for (name, port) in sort(collect(DAEMON_PORTS), by=x->x[2])
        println("  • $name (port $port)")
    end
    println()
    println("Available Functions:")
    println("  • build_project(path, force_discovery=false, force_compile=false)")
    println("  • quick_compile(path, force=false)")
    println("  • incremental_build(path)")
    println("  • clean_build(path)")
    println("  • watch_and_build(path, interval=2.0)")
    println("  • check_daemons()")
    println("  • get_stats()")
    println()
    println("Ready to orchestrate builds...")
    println("="^70)

    # Start the daemon server
    serve(PORT)
end

# Start the daemon if run directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
