# windows_build.jl — how to run build.sh on Windows from outside MSYS2.
#
# Plain functions with no module around them: `deps/build.jl` includes this file
# into a bare build process (RepliBuild is not loadable there), and
# EnvironmentDoctor includes it for its advice. One copy of the command for both.
#
# build.sh needs the CLANG64 toolchain and MLIR on PATH, which is what an MSYS2
# CLANG64 shell gives. MSYS2's own bash builds that PATH from MSYSTEM when it
# starts as a login shell, so `bash -l` with MSYSTEM=CLANG64 is that shell, from
# any process: Pkg.build, PowerShell, or a Julia REPL. build.sh finds its folder
# from BASH_SOURCE, so the working directory does not matter.

"""MSYS2's install folder, `/`-separated: `MSYS2_ROOT`, default `C:/msys64`."""
msys2_root() = replace(get(ENV, "MSYS2_ROOT", "C:/msys64"), '\\' => '/')

"""MSYS2's bash, which turns into a CLANG64 shell under `MSYSTEM=CLANG64 -l`."""
msys2_bash(root::AbstractString=msys2_root()) = root * "/usr/bin/bash.exe"

"""`build.sh` in `mlir_dir`, written with `/`: bash strips backslashes."""
jlcs_build_script(mlir_dir::AbstractString) =
    rstrip(replace(String(mlir_dir), '\\' => '/'), '/') * "/build.sh"

"""
    jlcs_msys2_cmd(mlir_dir; root) -> Cmd

Runs `build.sh` in a CLANG64 login shell. Julia quotes the arguments, so a
folder with spaces needs nothing extra.
"""
jlcs_msys2_cmd(mlir_dir::AbstractString; root::AbstractString=msys2_root()) =
    addenv(`$(msys2_bash(root)) -l $(jlcs_build_script(mlir_dir))`, "MSYSTEM" => "CLANG64")

"""
    jlcs_msys2_line(mlir_dir; root) -> String

The same command as one line to paste into PowerShell. MSYS2's `env.exe` sets
MSYSTEM for bash alone, so the user's session keeps its environment. PowerShell
strips the single quotes and re-quotes a path with spaces in double quotes, which
MSYS2 reads. A path holding a `'` does not survive that trip; `Pkg.build`, which
runs `jlcs_msys2_cmd`, does.
"""
jlcs_msys2_line(mlir_dir::AbstractString; root::AbstractString=msys2_root()) =
    "$root/usr/bin/env.exe MSYSTEM=CLANG64 /usr/bin/bash -l '$(jlcs_build_script(mlir_dir))'"
