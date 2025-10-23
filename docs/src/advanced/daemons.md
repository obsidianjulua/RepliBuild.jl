# Daemon System

RepliBuild's optional daemon system accelerates compilation through persistent background processes.

## Overview

The daemon system consists of four components:

1. **Discovery Daemon** - Watches for file changes and tracks dependencies
2. **Setup Daemon** - Manages project configuration and initialization
3. **Compilation Daemon** - Handles C++ compilation with caching
4. **Orchestrator Daemon** - Coordinates workflows between daemons

Benefits:
- **Faster compilation** - Avoid repeated toolchain initialization
- **Incremental builds** - Only recompile changed files
- **Background processing** - Continue working while compiling
- **Dependency tracking** - Automatic rebuild when headers change

## Basic Usage

### Start Daemons

```julia
using RepliBuild

# Start all daemons in current directory
RepliBuild.start_daemons()
```

Output:
```
🚀 Starting RepliBuild daemons...
   ✅ Discovery daemon (PID: 12345)
   ✅ Setup daemon (PID: 12346)
   ✅ Compilation daemon (PID: 12347)
   ✅ Orchestrator daemon (PID: 12348)
✅ All daemons started
```

### Check Status

```julia
RepliBuild.daemon_status()
```

Output:
```
RepliBuild Daemon Status:
  Discovery:    ✅ Running (PID 12345)
  Setup:        ✅ Running (PID 12346)
  Compilation:  ✅ Running (PID 12347)
  Orchestrator: ✅ Running (PID 12348)

Uptime: 5 minutes
Cache hits: 42
Cache misses: 3
```

### Compile with Daemons

```julia
# Compile uses daemons automatically
RepliBuild.compile()
```

First compilation (cold cache):
```
🔨 Compiling with daemon system...
   Compiling src/main.cpp... 2.3s
   Compiling src/utils.cpp... 1.8s
   Linking... 0.5s
✅ Complete (4.6s)
```

Second compilation (warm cache):
```
🔨 Compiling with daemon system...
   Using cached src/main.cpp ✅
   Recompiling src/utils.cpp... 1.8s
   Linking... 0.5s
✅ Complete (2.3s) [50% cached]
```

### Stop Daemons

```julia
RepliBuild.stop_daemons()
```

Output:
```
🛑 Stopping RepliBuild daemons...
   ✅ Discovery daemon stopped
   ✅ Setup daemon stopped
   ✅ Compilation daemon stopped
   ✅ Orchestrator daemon stopped
✅ All daemons stopped
```

## Advanced Features

### Project-Specific Daemons

```julia
# Start daemons for specific project
RepliBuild.start_daemons(project_root="/path/to/project")

# Each project can have its own daemon set
```

### Ensure Daemons Running

```julia
# Check and restart crashed daemons
if !RepliBuild.ensure_daemons()
    println("Warning: Some daemons failed to restart")
end
```

### Daemon Configuration

Create `.replibuild/daemon_config.toml` in project:

```toml
[daemons]
# Enable/disable specific daemons
enable_discovery = true
enable_setup = true
enable_compilation = true
enable_orchestrator = true

# Daemon settings
[daemons.discovery]
watch_interval = 1000  # milliseconds
ignored_patterns = ["*.tmp", "*.swp"]

[daemons.compilation]
max_cache_size = 1024  # MB
cache_ttl = 86400      # seconds (24 hours)
parallel_jobs = 0      # 0 = auto

[daemons.orchestrator]
queue_size = 100
timeout = 300  # seconds
```

## Daemon Details

### Discovery Daemon

Monitors project for changes:

- Watches source files (`.cpp`, `.h`)
- Tracks dependencies between files
- Invalidates cache on changes
- Triggers recompilation when needed

**Port:** 9001

### Setup Daemon

Manages project configuration:

- Loads and validates `replibuild.toml`
- Resolves module dependencies
- Prepares compiler environment
- Caches configuration

**Port:** 9002

### Compilation Daemon

Handles compilation:

- Maintains warm LLVM/Clang instance
- Caches compiled object files
- Performs incremental builds
- Manages parallel compilation

**Port:** 9003

### Orchestrator Daemon

Coordinates workflows:

- Manages build queue
- Schedules tasks across daemons
- Handles inter-daemon communication
- Reports status and progress

**Port:** 9004

## Communication Protocol

Daemons use TCP sockets for communication:

```julia
# Internal RepliBuild communication (simplified)
using Sockets

# Connect to compilation daemon
socket = connect("localhost", 9003)

# Send compilation request
request = Dict(
    "action" => "compile",
    "file" => "src/main.cpp",
    "flags" => ["-O2", "-std=c++17"]
)
write(socket, JSON.json(request))

# Receive response
response = JSON.parse(readline(socket))
close(socket)
```

## Performance Comparison

### Without Daemons

```
First build:  15.2s
Second build: 14.8s (no cache)
Third build:  15.1s (no cache)
```

### With Daemons

```
First build:  12.3s (daemon startup + compile)
Second build: 2.1s  (cache hit)
Third build:  2.0s  (cache hit)

After changing one file:
Incremental:  3.5s  (recompile one file + link)
```

**Speedup:** Up to 7x faster for incremental builds

## Cache Management

### View Cache Statistics

```julia
using RepliBuild

# Get cache info
cache_dir = RepliBuild.get_cache_dir()
cache_size = sum(filesize(joinpath(cache_dir, f)) for f in readdir(cache_dir))

println("Cache directory: $cache_dir")
println("Cache size: $(round(cache_size / 1024 / 1024, digits=2)) MB")
```

### Clear Cache

```julia
# Clear all cache
cache_dir = RepliBuild.get_cache_dir()
rm(cache_dir, recursive=true, force=true)
RepliBuild.initialize_directories()

# Restart daemons
RepliBuild.stop_daemons()
RepliBuild.start_daemons()
```

### Cache Location

- **Linux/macOS:** `~/.replibuild/cache/`
- **Windows:** `%LOCALAPPDATA%/RepliBuild/cache/`

## Troubleshooting

### Daemon Won't Start

**Check ports:**
```bash
# Linux
netstat -tulpn | grep 900[1-4]

# macOS
lsof -i :9001-9004
```

**Solution:** Kill conflicting processes or change ports

### Daemon Crashes

**Check logs:**
```bash
cat ~/.replibuild/daemon_logs/compilation.log
```

**Restart:**
```julia
RepliBuild.stop_daemons()
RepliBuild.start_daemons()
```

### Stale PID Files

**Clean up:**
```julia
# RepliBuild automatically cleans stale PIDs
RepliBuild.start_daemons()

# Or manually
rm(joinpath(RepliBuild.get_replibuild_dir(), "pids"), recursive=true)
```

### Cache Issues

**Symptoms:**
- Outdated binaries
- Wrong compilation results

**Solution:**
```julia
# Clear cache and rebuild
cache = RepliBuild.get_cache_dir()
rm(cache, recursive=true)
RepliBuild.compile()
```

## Best Practices

### 1. Start Daemons for Active Development

```julia
# At start of work session
RepliBuild.start_daemons()

# ... develop and compile frequently ...

# At end of session
RepliBuild.stop_daemons()
```

### 2. Use Separate Daemon Sets for Multiple Projects

```julia
# Terminal 1 - Project A
cd("project_a")
RepliBuild.start_daemons()

# Terminal 2 - Project B
cd("project_b")
RepliBuild.start_daemons()

# Each has independent caches
```

### 3. Monitor Cache Size

```julia
# Periodically check cache
cache_size = # calculate size
if cache_size > 5 * 1024^3  # 5 GB
    println("Cache too large, clearing...")
    # clear old cache
end
```

### 4. Disable Daemons for CI/CD

```julia
# In CI environments, don't use daemons
# Just compile directly
RepliBuild.compile()
```

## Integration with Development Workflow

### With VSCode

Create `.vscode/tasks.json`:

```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Start RepliBuild Daemons",
            "type": "julia",
            "command": "using RepliBuild; RepliBuild.start_daemons()"
        },
        {
            "label": "Compile with RepliBuild",
            "type": "julia",
            "command": "using RepliBuild; RepliBuild.compile()"
        },
        {
            "label": "Stop RepliBuild Daemons",
            "type": "julia",
            "command": "using RepliBuild; RepliBuild.stop_daemons()"
        }
    ]
}
```

### With Make

```makefile
.PHONY: start-daemons stop-daemons compile

start-daemons:
\tjulia -e 'using RepliBuild; RepliBuild.start_daemons()'

compile:
\tjulia -e 'using RepliBuild; RepliBuild.compile()'

stop-daemons:
\tjulia -e 'using RepliBuild; RepliBuild.stop_daemons()'
```

## Security Considerations

- Daemons listen only on `localhost`
- No remote connections accepted
- PID files prevent unauthorized access
- Cache is user-private

## Next Steps

- **[Error Learning](error-learning.md)**: Intelligent error handling
- **[LLVM Toolchain](llvm-toolchain.md)**: Toolchain management
- **[Advanced API](../api/advanced.md)**: Daemon API reference
