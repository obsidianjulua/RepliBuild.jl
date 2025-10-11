#!/usr/bin/env julia
"""
DaemonManager.jl - Integrated daemon lifecycle management for RepliBuild

Replaces shell script daemon management with native Julia API.
Manages discovery, setup, compilation, and orchestrator daemons.

Usage:
    dm = DaemonManager.start_all()
    DaemonManager.stop_all(dm)
    DaemonManager.status(dm)
"""

module DaemonManager

using Distributed
using Sockets

# Optional DaemonMode support
const DAEMONMODE_AVAILABLE = Ref(false)

function __init__()
    try
        @eval using DaemonMode
        DAEMONMODE_AVAILABLE[] = true
    catch
        @warn "DaemonMode not available. Daemon RPC functionality will be limited."
        DAEMONMODE_AVAILABLE[] = false
    end
end

export DaemonSystem, start_all, stop_all, restart_daemon, status, ensure_running
export call_daemon, wait_for_daemon

# Daemon configuration (production daemons only)
# Removed: error_handler_daemon, watcher_daemon, build_daemon (unused/experimental)
const DAEMON_CONFIG = Dict(
    "discovery" => Dict(
        "port" => 3001,
        "script" => "daemons/servers/discovery_daemon.jl",
        "workers" => 0
    ),
    "setup" => Dict(
        "port" => 3002,
        "script" => "daemons/servers/setup_daemon.jl",
        "workers" => 0
    ),
    "compilation" => Dict(
        "port" => 3003,
        "script" => "daemons/servers/compilation_daemon.jl",
        "workers" => 4
    ),
    "orchestrator" => Dict(
        "port" => 3004,
        "script" => "daemons/servers/orchestrator_daemon.jl",
        "workers" => 0
    )
)

# Daemon system state
mutable struct DaemonInfo
    name::String
    port::Int
    pid::Union{Int, Nothing}
    process::Union{Base.Process, Nothing}
    workers::Int
    status::Symbol  # :stopped, :starting, :running, :error
    started_at::Union{Float64, Nothing}
end

mutable struct DaemonSystem
    daemons::Dict{String, DaemonInfo}
    project_root::String
    log_dir::String
end

"""
    start_all(;project_root=pwd(), log_dir="daemons/logs")

Start all RepliBuild daemons with integrated process management.
Returns a DaemonSystem handle for managing the daemons.
"""
function start_all(;project_root=pwd(), log_dir=joinpath(project_root, "daemons", "logs"))
    println("="^70)
    println("Starting RepliBuild Daemon System")
    println("="^70)

    # Create log directory
    mkpath(log_dir)

    # Initialize daemon system
    system = DaemonSystem(
        Dict{String, DaemonInfo}(),
        project_root,
        log_dir
    )

    # Start daemons in dependency order
    daemon_order = ["discovery", "setup", "compilation", "orchestrator"]

    for daemon_name in daemon_order
        start_daemon(system, daemon_name)
        sleep(2)  # Give daemon time to initialize
    end

    println()
    println("="^70)
    println("All daemons started successfully!")
    println("="^70)

    # Print status
    status(system)

    return system
end

"""
    start_daemon(system::DaemonSystem, name::String)

Start a specific daemon.
"""
function start_daemon(system::DaemonSystem, name::String)
    if !haskey(DAEMON_CONFIG, name)
        error("Unknown daemon: $name")
    end

    config = DAEMON_CONFIG[name]
    script_path = joinpath(system.project_root, config["script"])

    if !isfile(script_path)
        error("Daemon script not found: $script_path")
    end

    println("\nStarting $name daemon (port $(config["port"]))...")

    # Build command
    cmd_args = ["julia", "--project=$(system.project_root)"]

    if config["workers"] > 0
        push!(cmd_args, "-p", string(config["workers"]))
    end

    push!(cmd_args, script_path)

    # Redirect output to log file
    log_file = joinpath(system.log_dir, "$(name).log")
    log_io = open(log_file, "w")

    # Start process
    process = run(pipeline(Cmd(cmd_args), stdout=log_io, stderr=log_io), wait=false)

    # Create daemon info
    info = DaemonInfo(
        name,
        config["port"],
        getpid(process),
        process,
        config["workers"],
        :starting,
        time()
    )

    system.daemons[name] = info

    println("  PID: $(info.pid)")
    println("  Log: $log_file")

    # Wait for daemon to be ready
    if wait_for_daemon(info.port, timeout=10)
        info.status = :running
        println("  ✓ Running")
    else
        info.status = :error
        println("  ✗ Failed to start (timeout)")
    end
end

"""
    wait_for_daemon(port::Int; timeout::Int=10)

Wait for daemon to be ready by attempting to connect to its port.
"""
function wait_for_daemon(port::Int; timeout::Int=10)
    start_time = time()

    while time() - start_time < timeout
        try
            sock = connect("localhost", port)
            close(sock)
            return true
        catch
            sleep(0.5)
        end
    end

    return false
end

"""
    stop_all(system::DaemonSystem)

Stop all running daemons gracefully.
"""
function stop_all(system::DaemonSystem)
    println("\n" * "="^70)
    println("Stopping RepliBuild Daemon System")
    println("="^70)

    # Stop in reverse order
    daemon_order = ["orchestrator", "compilation", "setup", "discovery"]

    for daemon_name in daemon_order
        if haskey(system.daemons, daemon_name)
            stop_daemon(system, daemon_name)
        end
    end

    println("\n" * "="^70)
    println("All daemons stopped")
    println("="^70)
end

"""
    stop_daemon(system::DaemonSystem, name::String)

Stop a specific daemon.
"""
function stop_daemon(system::DaemonSystem, name::String)
    if !haskey(system.daemons, name)
        println("Daemon $name is not running")
        return
    end

    info = system.daemons[name]

    println("\nStopping $name daemon (PID: $(info.pid))...")

    if !isnothing(info.process)
        try
            # Try graceful shutdown first
            kill(info.process, Base.SIGTERM)

            # Wait up to 5 seconds for graceful shutdown
            for i in 1:10
                if !process_running(info.process)
                    println("  ✓ Stopped gracefully")
                    info.status = :stopped
                    delete!(system.daemons, name)
                    return
                end
                sleep(0.5)
            end

            # Force kill if still running
            kill(info.process, Base.SIGKILL)
            println("  ✓ Force killed")

        catch e
            println("  ⚠ Error stopping daemon: $e")
        end

        info.status = :stopped
        delete!(system.daemons, name)
    end
end

"""
    restart_daemon(system::DaemonSystem, name::String)

Restart a specific daemon.
"""
function restart_daemon(system::DaemonSystem, name::String)
    println("Restarting $name daemon...")
    stop_daemon(system, name)
    sleep(1)
    start_daemon(system, name)
end

"""
    status(system::DaemonSystem)

Display status of all daemons.
"""
function status(system::DaemonSystem)
    println("\nDaemon Status:")
    println("-"^70)

    for daemon_name in ["discovery", "setup", "compilation", "orchestrator"]
        if haskey(system.daemons, daemon_name)
            info = system.daemons[daemon_name]
            status_icon = if info.status == :running
                "✓"
            elseif info.status == :starting
                "⋯"
            elseif info.status == :error
                "✗"
            else
                "○"
            end

            uptime = if !isnothing(info.started_at)
                elapsed = time() - info.started_at
                "$(round(Int, elapsed))s"
            else
                "N/A"
            end

            workers_str = info.workers > 0 ? " ($(info.workers) workers)" : ""

            println("  $status_icon $daemon_name: $(info.status) | PID: $(info.pid) | Port: $(info.port)$workers_str | Uptime: $uptime")
        else
            println("  ○ $daemon_name: not running")
        end
    end

    println("-"^70)
end

"""
    ensure_running(system::DaemonSystem)

Check if all daemons are running, restart any that have crashed.
"""
function ensure_running(system::DaemonSystem)
    needs_restart = String[]

    for (name, info) in system.daemons
        if info.status == :running
            # Check if process is actually still alive
            if isnothing(info.process) || !process_running(info.process)
                info.status = :error
                push!(needs_restart, name)
            end
        elseif info.status == :error
            push!(needs_restart, name)
        end
    end

    # Restart crashed daemons
    for name in needs_restart
        println("⚠ Daemon $name has crashed, restarting...")
        restart_daemon(system, name)
    end

    return isempty(needs_restart)
end

"""
    call_daemon(system::DaemonSystem, daemon::String, func::String, args::Dict)

Call a function on a specific daemon (requires DaemonMode integration).
"""
function call_daemon(system::DaemonSystem, daemon::String, func::String, args::Dict)
    if !DAEMONMODE_AVAILABLE[]
        return Dict(
            :success => false,
            :error => "DaemonMode not available. Install it with: Pkg.add(url=\"https://github.com/dmolina/DaemonMode.jl\")"
        )
    end

    if !haskey(system.daemons, daemon)
        return Dict(
            :success => false,
            :error => "Daemon $daemon is not running"
        )
    end

    info = system.daemons[daemon]

    if info.status != :running
        return Dict(
            :success => false,
            :error => "Daemon $daemon is not in running state (status: $(info.status))"
        )
    end

    try
        # Build expression
        expr = Meta.parse("$func($args)")

        # Call daemon via DaemonMode
        result = @eval DaemonMode.runexpr($expr, port=$(info.port))

        return result

    catch e
        return Dict(
            :success => false,
            :error => "Daemon call failed: $e",
            :daemon => daemon,
            :port => info.port
        )
    end
end

"""
    cleanup_stale_pids(project_root::String)

Clean up stale PID files from previous runs.
"""
function cleanup_stale_pids(project_root::String)
    pid_dir = joinpath(project_root, "daemons")

    # Check if directory exists first
    if !isdir(pid_dir)
        return  # No daemon directory, nothing to clean
    end

    for file in readdir(pid_dir)
        if startswith(file, ".daemon_pids.")
            pid_file = joinpath(pid_dir, file)
            try
                rm(pid_file)
            catch
            end
        end
    end
end

end # module DaemonManager
