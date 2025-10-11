#!/usr/bin/env julia
# ProjectWizard.jl - Template-based interactive project builder
# User interacts with normal API, wizard just helps set things up

module ProjectWizard

using TOML

export create_project_interactive, available_templates, use_template

# Available project templates
const TEMPLATES = Dict(
    "simple_lib" => Dict(
        "name" => "Simple C++ Library",
        "description" => "Single library with Julia bindings",
        "type" => :library,
        "example" => """
        // src/math.cpp
        extern "C" {
            int add(int a, int b) { return a + b; }
            double multiply(double x, double y) { return x * y; }
        }
        """
    ),

    "executable" => Dict(
        "name" => "C++ Executable",
        "description" => "Standalone executable (no Julia bindings)",
        "type" => :executable,
        "example" => """
        // src/main.cpp
        #include <iostream>
        int main(int argc, char** argv) {
            std::cout << "Hello RepliBuild!" << std::endl;
            return 0;
        }
        """
    ),

    "lib_and_exe" => Dict(
        "name" => "Library + Executable (Multi-stage)",
        "description" => "Build .so first, then link executable against it",
        "type" => :multi_stage,
        "example" => """
        # Stage 1: Build library
        # src/lib/mylib.cpp
        extern "C" void my_function() { /* ... */ }

        # Stage 2: Build executable that uses the library
        # src/main.cpp
        extern "C" void my_function();
        int main() { my_function(); return 0; }
        """
    ),

    "cmake_import" => Dict(
        "name" => "Import from CMake",
        "description" => "Import existing CMake project",
        "type" => :cmake,
        "example" => """
        # Point to existing CMakeLists.txt
        RepliBuild.import_cmake("path/to/CMakeLists.txt")
        RepliBuild.compile()
        """
    ),

    "external_libs" => Dict(
        "name" => "Project with External Libraries",
        "description" => "Uses pkg-config to find system libraries (sqlite, zlib, etc.)",
        "type" => :library,
        "auto_detect" => true,
        "example" => """
        // src/database.cpp
        #include <sqlite3.h>
        extern "C" int open_db(const char* path) {
            sqlite3* db;
            return sqlite3_open(path, &db);
        }
        """
    )
)

"""
    available_templates()

Show all available project templates with descriptions.
"""
function available_templates()
    println("📦 Available RepliBuild Project Templates:")
    println("="^60)

    for (key, template) in sort(collect(TEMPLATES), by=x->x[1])
        println("\n$(key)")
        println("  $(template["name"])")
        println("  → $(template["description"])")
    end

    println("\n" * "="^60)
    println("Usage: RepliBuild.use_template(\"template_name\", \"my_project\")")
end

"""
    use_template(template_name::String, project_name::String="my_project")

Create a new project from a template.
"""
function use_template(template_name::String, project_name::String="my_project")
    if !haskey(TEMPLATES, template_name)
        @error "Unknown template: $template_name"
        println("Available templates:")
        available_templates()
        return
    end

    template = TEMPLATES[template_name]
    println("🚀 Creating project: $project_name")
    println("📋 Template: $(template["name"])")
    println("="^60)

    # Create project structure
    project_dir = abspath(project_name)

    if isdir(project_dir)
        print("⚠️  Directory exists. Overwrite? (y/N): ")
        response = readline()
        if !startswith(lowercase(strip(response)), "y")
            println("Cancelled.")
            return
        end
    end

    # Call appropriate setup based on template type
    if template["type"] == :library
        setup_library_template(project_dir, project_name, template)
    elseif template["type"] == :executable
        setup_executable_template(project_dir, project_name, template)
    elseif template["type"] == :multi_stage
        setup_multistage_template(project_dir, project_name, template)
    elseif template["type"] == :cmake
        setup_cmake_template(project_dir, project_name, template)
    end

    # Print next steps
    print_next_steps(project_dir, template)
end

"""
Setup simple library template
"""
function setup_library_template(project_dir, project_name, template)
    mkpath(project_dir)
    mkpath(joinpath(project_dir, "src"))
    mkpath(joinpath(project_dir, "include"))

    # Create example source
    example_src = joinpath(project_dir, "src", "example.cpp")
    write(example_src, template["example"])

    # Create replibuild.toml
    create_library_config(project_dir, project_name, template)

    println("✅ Library project created!")
    println("📁 Structure:")
    println("  $project_dir/")
    println("    ├── src/example.cpp")
    println("    ├── include/")
    println("    └── replibuild.toml")
end

"""
Setup executable template
"""
function setup_executable_template(project_dir, project_name, template)
    mkpath(project_dir)
    mkpath(joinpath(project_dir, "src"))

    # Create main.cpp
    main_cpp = joinpath(project_dir, "src", "main.cpp")
    write(main_cpp, template["example"])

    # Create replibuild.toml for executable
    create_executable_config(project_dir, project_name)

    println("✅ Executable project created!")
    println("📁 Structure:")
    println("  $project_dir/")
    println("    ├── src/main.cpp")
    println("    └── replibuild.toml")
end

"""
Setup multi-stage template
"""
function setup_multistage_template(project_dir, project_name, template)
    mkpath(project_dir)
    mkpath(joinpath(project_dir, "src", "lib"))
    mkpath(joinpath(project_dir, "src", "app"))

    # Create library source
    lib_src = joinpath(project_dir, "src", "lib", "mylib.cpp")
    write(lib_src, """
extern "C" {
    #include <stdio.h>

    void greet(const char* name) {
        printf("Hello, %s!\\n", name);
    }
}
""")

    # Create executable source
    app_src = joinpath(project_dir, "src", "app", "main.cpp")
    write(app_src, """
extern "C" void greet(const char* name);

int main() {
    greet("RepliBuild");
    return 0;
}
""")

    # Create build script (shows the normal API workflow)
    build_script = joinpath(project_dir, "build.jl")
    write(build_script, """
using RepliBuild

# Stage 1: Build the library
println("Stage 1: Building library...")
lib_config = \"\"\"
[project]
name = "$project_name"

[paths]
source = "src/lib"
output = "build"

[compile]
libraries = []
\"\"\"
write("lib_config.toml", lib_config)
lib_compiler = RepliBuild.LLVMJuliaCompiler("lib_config.toml")
RepliBuild.compile_project(lib_compiler)

# Stage 2: Build the executable
println("\\nStage 2: Building executable...")
exe_config = \"\"\"
[project]
name = "$project_name"

[paths]
source = "src/app"
output = "build"

[compile]
lib_dirs = ["build"]
libraries = ["$project_name"]  # Link against the library from stage 1
\"\"\"
write("exe_config.toml", exe_config)

# Compile executable
using RepliBuild.LLVMake
exe_compiler = RepliBuild.LLVMJuliaCompiler("exe_config.toml")
cpp_files = RepliBuild.LLVMake.find_cpp_files("src/app")
ir_files = RepliBuild.LLVMake.compile_to_ir(exe_compiler, cpp_files)
final_ir = RepliBuild.LLVMake.optimize_and_link_ir(exe_compiler, ir_files, "$project_name")
exe_path = RepliBuild.LLVMake.compile_ir_to_executable(exe_compiler, final_ir, "$project_name",
                                                        link_libs=["$project_name"])

println("\\n✅ Build complete!")
println("Run: ./build/$project_name")
""")

    println("✅ Multi-stage project created!")
    println("📁 Structure:")
    println("  $project_dir/")
    println("    ├── src/")
    println("    │   ├── lib/mylib.cpp     (shared library)")
    println("    │   └── app/main.cpp      (executable)")
    println("    └── build.jl              (build script)")
end

"""
Create library config
"""
function create_library_config(project_dir, project_name, template)
    config = """
    [project]
    name = "$project_name"

    [paths]
    source = "src"
    include = "include"
    output = "julia"
    build = "build"

    [compile]
    include_dirs = ["include"]
    libraries = []

    [bindings]
    style = "simple"
    """

    # Auto-detect external libraries if requested
    if get(template, "auto_detect", false)
        config *= """

        # Note: Run discovery to auto-detect external libraries:
        # julia> using RepliBuild
        # julia> RepliBuild.discover("$project_dir")
        """
    end

    write(joinpath(project_dir, "replibuild.toml"), config)
end

"""
Create executable config
"""
function create_executable_config(project_dir, project_name)
    config = """
    [project]
    name = "$project_name"

    [paths]
    source = "src"
    output = "build"
    build = "build"

    [compile]
    libraries = []

    # Note: This builds an executable, not Julia bindings
    # Use compile_ir_to_executable() after compile_to_ir()
    """

    write(joinpath(project_dir, "replibuild.toml"), config)
end

"""
Setup CMake import template
"""
function setup_cmake_template(project_dir, project_name, template)
    mkpath(project_dir)

    readme = joinpath(project_dir, "README.md")
    write(readme, """
    # Import CMake Project

    ## Step 1: Import CMake
    ```julia
    using RepliBuild
    RepliBuild.import_cmake("path/to/CMakeLists.txt")
    ```

    ## Step 2: Compile
    ```julia
    RepliBuild.compile()
    ```

    ## That's it!
    RepliBuild will generate `replibuild.toml` from your CMakeLists.txt.
    """)

    println("✅ CMake import template created!")
    println("📄 See README.md for instructions")
end

"""
Print next steps for the user
"""
function print_next_steps(project_dir, template)
    println("\n" * "="^60)
    println("📚 Next Steps:")
    println("="^60)

    if template["type"] == :library
        println("""
        1. Edit source files in src/
        2. Run discovery:
           julia> using RepliBuild
           julia> RepliBuild.discover("$project_dir")

        3. Compile to Julia:
           julia> RepliBuild.compile()

        4. Use in Julia:
           julia> include("julia/$(basename(project_dir)).jl")
        """)
    elseif template["type"] == :executable
        println("""
        1. Edit src/main.cpp
        2. Build executable:
           julia> using RepliBuild, RepliBuild.LLVMake
           julia> compiler = LLVMJuliaCompiler("$project_dir/replibuild.toml")
           julia> cpp_files = find_cpp_files("$project_dir/src")
           julia> ir_files = compile_to_ir(compiler, cpp_files)
           julia> final_ir = optimize_and_link_ir(compiler, ir_files, "$(basename(project_dir))")
           julia> compile_ir_to_executable(compiler, final_ir, "$(basename(project_dir))")

        3. Run:
           ./build/$(basename(project_dir))
        """)
    elseif template["type"] == :multi_stage
        println("""
        1. Edit library code: src/lib/
        2. Edit executable code: src/app/
        3. Run the build script:
           julia build.jl

        4. Run the executable:
           ./build/$(basename(project_dir))
        """)
    elseif template["type"] == :cmake
        println("""
        1. Point to your CMakeLists.txt
        2. Import and compile (see README.md)
        """)
    end

    println("\n💡 For help: RepliBuild.help()")
end

"""
    create_project_interactive()

Interactive project creation wizard (asks questions).
"""
function create_project_interactive()
    println("🧙 RepliBuild Project Wizard")
    println("="^60)

    # Ask for project type
    println("\nWhat type of project?")
    println("1. C++ Library → Julia bindings (most common)")
    println("2. C++ Executable (no Julia)")
    println("3. Library + Executable (multi-stage)")
    println("4. Import from CMake")
    println("5. Project with external libraries (sqlite, zlib, etc.)")
    print("\nChoice (1-5): ")

    choice = strip(readline())

    template_map = Dict(
        "1" => "simple_lib",
        "2" => "executable",
        "3" => "lib_and_exe",
        "4" => "cmake_import",
        "5" => "external_libs"
    )

    if !haskey(template_map, choice)
        println("Invalid choice. Cancelled.")
        return
    end

    # Ask for project name
    print("\nProject name: ")
    project_name = String(strip(readline()))

    if isempty(project_name)
        println("Invalid name. Cancelled.")
        return
    end

    # Use template
    use_template(template_map[choice], project_name)
end

end # module ProjectWizard
