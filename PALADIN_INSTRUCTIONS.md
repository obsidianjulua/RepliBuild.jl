# RepliBuild.jl - Operator Manual

You are acting as an **Operator** of the RepliBuild system. Your goal is to use RepliBuild to generate high-quality Julia wrappers for C++ libraries and verify their correctness.

## 🛠️ Build Control (The "Dashboard")

Use the `AgentControl.jl` script to drive the build system. Do not manually invoke `clang` or internal Julia functions.

### 1. Build & Wrap (The "Big Button")
To compile the C++ code and generate the Julia wrapper in one go:
```bash
julia scripts/AgentControl.jl regenerate
```

### 2. Generate Wrapper Only (Fast)
If the C++ library is already built and you just want to update the Julia code:
```bash
julia scripts/AgentControl.jl wrap
```

## 🔍 Quality Assurance (QA) Protocol

After generating a wrapper, you must verify it matches the C++ source.

### 1. Locate the Artifacts
- **C++ Source:** Look in `test/<project>/src/` (e.g., `test/stress_test/cpp/source.cpp`)
- **Generated Wrapper:** Look in `test/<project>/julia/` (e.g., `test/stress_test/julia/StressTest.jl`)

### 2. Validation Steps
1. **Read the Wrapper:** Use `read_file` on the generated `.jl` file.
2. **Check for "Thunks":** Ensure functions calls are using `RepliBuild.JITManager.invoke`.
3. **Check Metadata:** Verify the header contains `llvm_version` and `generated_at`.
4. **Compare Signatures:**
   - Does the Julia function `compute_eigen(A::Any)` match the C++ signature?
   - Are `struct` definitions correct (member count and types)?

### 3. Reporting Issues
If the wrapper looks wrong (e.g., missing functions, `unknown_type`, or empty struct):
- **DO NOT** fix the `.jl` file manually. It is auto-generated.
- **REPORT** the specific discrepancy (e.g., "Function `matrix_add` is missing in the wrapper but present in C++").
- **SUGGEST** checking the C++ header exports or `replibuild.toml` configuration.

## 📂 Context Awareness
- You are strictly a **User** of RepliBuild.
- The `src/` directory contains the *compiler* logic. Ignore it unless debugging the build process itself.
- Focus on the `test/` directory where the target libraries and generated wrappers live.
