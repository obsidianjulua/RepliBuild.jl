#!/usr/bin/env julia
# Templates.jl - RepliBuild Project Seed (Self-Destructing Plant File)
#
# USAGE: Copy this file anywhere, run it, and it:
#   1. Creates proper RepliBuild directory structure
#   2. Creates replibuild.toml with UUID (project marker)
#   3. ERASES ITSELF
#
# After planting, the RepliBuild toolchain handles everything:
#   - julia -e 'using RepliBuild; RepliBuild.discover()'  # Scans, walks AST, generates config
#   - julia -e 'using RepliBuild; RepliBuild.compile()'   # Builds the project

module Templates

include("ConfigurationManager.jl")
using .ConfigurationManager

"""
    plant(target_dir::String=pwd(), project_name::String=""; unsafe::Bool=false)

Self-destructing project seed - creates RepliBuild structure and erases itself.

# Safety
- Includes all safety checks from initialize_project()
- Will NOT self-destruct if safety checks fail
- Requires unsafe=true to bypass safety checks
"""
function plant(target_dir::String=pwd(), project_name::String=""; unsafe::Bool=false)
    # SAFETY CHECKS
    if !unsafe
        safety_check_target_dir(target_dir)
    end

    println("🌱 RepliBuild Project Seed - Planting in: $target_dir")
    println("="^70)

    println("\n📁 Creating directory structure...")
    create_structure(target_dir)

    println("\n📝 Creating replibuild.toml with UUID...")
    create_initial_config(target_dir, project_name, force=false)

    println("\n🔥 Self-destructing...")
    self_destruct()

    println("\n✅ RepliBuild project structure planted!")
    println("\n⚠️  IMPORTANT: RepliBuild is scoped to: $target_dir")
    println("   Files outside this directory will NOT be affected.")
    println("\n📂 Next steps:")
    println("   1. julia -e 'using RepliBuild; RepliBuild.discover(\"$target_dir\")'")
    println("   2. julia -e 'using RepliBuild; RepliBuild.compile()'")
    println("="^70)
end

"""
    initialize_project(target_dir::String, project_name::String=""; force::Bool=false, unsafe::Bool=false)

Initialize a RepliBuild project without self-destructing (for permanent use).

# Safety Features
- Refuses to initialize in system directories (/usr, /etc, /bin, /lib, /var, /sys, /proc, /home)
- Refuses to initialize in home directory root
- Refuses to initialize in directories with many existing files (>100 by default)
- Requires `unsafe=true` flag to bypass safety checks
- Requires `force=true` to overwrite existing replibuild.toml

# Arguments
- `target_dir`: Directory to initialize
- `project_name`: Optional project name (default: directory name)
- `force`: Allow overwriting existing replibuild.toml
- `unsafe`: DANGER - Bypass safety checks (use with extreme caution)
"""
function initialize_project(target_dir::String, project_name::String=""; force::Bool=false, unsafe::Bool=false)
    # SAFETY CHECKS
    if !unsafe
        safety_check_target_dir(target_dir)
    else
        @warn "⚠️  UNSAFE MODE: Safety checks bypassed! You are responsible for any damage."
    end

    println("🌱 Initializing RepliBuild project: $target_dir")
    println("="^70)

    create_structure(target_dir)
    config = create_initial_config(target_dir, project_name, force=force)

    println("\n✅ RepliBuild project initialized!")
    println("📦 Project: $(config.project_name)")
    println("🔑 UUID: $(config.project_uuid)")
    println("📁 Root: $(config.project_root)")
    println("\n⚠️  SAFETY NOTICE:")
    println("   RepliBuild will ONLY process files within: $(config.project_root)")
    println("   Discovery is scoped to this directory and its subdirectories.")
    println("\n📂 Next steps:")
    println("   1. Add C++ files to src/")
    println("   2. Add headers to include/")
    println("   3. Run: julia -e 'using RepliBuild; RepliBuild.discover(\"$target_dir\")'")
    println("="^70)

    return config
end

function create_structure(root_dir::String)
    structure = [
        "src", "include", "lib", "bin", "julia",
        "build", "build/ir", "build/linked", "build/obj",
        ".replibuild_cache", "test", "docs"
    ]

    for dir in structure
        dir_path = joinpath(root_dir, dir)
        if !isdir(dir_path)
            mkpath(dir_path)
            println("  ✅ Created: $dir/")
        else
            println("  ⏭️  Exists:  $dir/")
        end
    end
end

function safety_check_target_dir(target_dir::String)
    abs_target = abspath(target_dir)

    # CRITICAL: Prevent system directory initialization
    dangerous_dirs = [
        "/usr", "/etc", "/bin", "/sbin", "/lib", "/lib64",
        "/var", "/sys", "/proc", "/dev", "/boot", "/opt",
        "/home", "/root"
    ]

    for danger in dangerous_dirs
        if abs_target == danger || startswith(abs_target, danger * "/")
            error("""
            ❌ SAFETY ERROR: Cannot initialize RepliBuild project in system directory!
            Target: $abs_target

            System directories are protected to prevent accidental compilation of system files.

            Safe locations:
            - Your user directory: $(homedir())/my_project
            - Dedicated workspace: /data/projects/my_project
            - Temporary location: /tmp/my_project

            If you REALLY need to do this (you probably don't), use unsafe=true flag.
            """)
        end
    end

    # Prevent home directory root
    if abs_target == homedir()
        error("""
        ❌ SAFETY ERROR: Cannot initialize RepliBuild project in home directory root!
        Target: $abs_target

        Create a subdirectory instead:
        - $(homedir())/projects/my_project
        - $(homedir())/code/my_project

        Use unsafe=true to bypass (not recommended).
        """)
    end

    # Check for excessive existing files (might be wrong directory)
    if isdir(abs_target)
        file_count = 0
        try
            for (root, dirs, files) in walkdir(abs_target)
                file_count += length(files)
                if file_count > 100
                    @warn """
                    ⚠️  WARNING: Target directory has many existing files ($file_count+)
                    Target: $abs_target

                    This might not be the right directory for a RepliBuild project.
                    RepliBuild will scan and potentially compile files in this directory.

                    Recommended: Use a clean directory for RepliBuild projects.

                    To proceed anyway, use unsafe=true flag.
                    """ maxlog=1

                    print("\nContinue initialization? [y/N]: ")
                    response = readline()
                    if !startswith(lowercase(strip(response)), "y")
                        error("Initialization cancelled by user")
                    end
                    break
                end
            end
        catch
            # If we can't scan, it's probably safe to continue
        end
    end
end

function create_initial_config(root_dir::String, project_name::String=""; force::Bool=false)
    config_path = joinpath(root_dir, "replibuild.toml")

    if isfile(config_path) && !force
        error("""
        ❌ ERROR: replibuild.toml already exists at: $config_path

        This directory is already a RepliBuild project.
        Use force=true to overwrite the existing configuration.

        Warning: This will reset all project settings!
        """)
    elseif isfile(config_path)
        println("  ⚠️  Overwriting existing replibuild.toml (force=true)")
    end

    # Determine project name
    name = isempty(project_name) ? basename(abspath(root_dir)) : project_name

    # Create default config with UUID
    config = ConfigurationManager.create_default_config(config_path)
    config.project_name = name
    config.project_root = abspath(root_dir)

    ConfigurationManager.save_config(config)

    println("  ✅ Created: replibuild.toml (UUID: $(config.project_uuid))")

    return config
end

function self_destruct()
    try
        rm(@__FILE__)
        println("  💥 Templates.jl erased")
    catch e
        @warn "Could not self-destruct: $e"
    end
end

export plant, initialize_project

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    # Parse command line args for project name
    project_name = length(ARGS) > 0 ? ARGS[1] : ""
    Templates.plant(pwd(), project_name)
end
