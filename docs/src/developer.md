# Developer documentation

These pages are for people changing RepliBuild, not for people wrapping a
library. The user manual is the rest of this site, starting at
[Home](index.md).

The architecture essays below are conserved from the previous manual. They are
deliberately not on the user path.

## Architecture

| Page | What it is |
|------|------------|
| [ABI marshalling as compiler IR](mlir.md) | The JLCS dialect: why marshalling is compiled, the thunk contract, the op reference, SysV lowering, gdb inside generated MLIR |
| [The inheritance ABI](inheritance-abi.md) | Mental model for MI and virtual inheritance: upcast helpers, class-local vcall |
| [Internals & dispatch](internals.md) | Pipeline stages, the three calling paths, module-by-module map |
| [Tier 1 (experimental)](tier1.md) | Per-function `llvmcall` slices, and why whole-module `enable_lto` stays off |

## In the repository, not on this site

| Path | What it is |
|------|------------|
| [`docs/updates/`](https://github.com/obsidianjulua/RepliBuild.jl/tree/main/docs/updates) | Dated engineering notes (MI/vcall, virtual inheritance, audits, release write-ups) |
| [`CHANGELOG.md`](https://github.com/obsidianjulua/RepliBuild.jl/blob/main/CHANGELOG.md) | Full version history |
| [`CLAUDE.md`](https://github.com/obsidianjulua/RepliBuild.jl/blob/main/CLAUDE.md) | Contributor operating notes: API, ingest doctrine, Hub rules, known-unbuilt ledger |

## Tests

```bash
julia --project=. test/runtests.jl     # CI suite, no C++ toolchain
julia --project=. test/devtests.jl     # full integration, needs the C++ bucket
```

Dialect and ABI truth lives in `test/test_mlir_templates.jl`,
`test/test_jlcs_invariants.jl`, `test/test_jlcs_producers.jl`,
`test/test_struct_abi.jl`, `test/mi_test/`, `test/vi_test/`.

## Debugging a generated thunk

```julia
D = RepliBuild.Debug
D.thunks("path/to/project")
D.walk("path/to/project", "some_symbol_thunk")    # dialect + asm
```

`ENV["REPLIBUILD_JIT_OBJDUMP"] = "1"` must be set **before** the wrapper loads
for the disassembly path. gdb inside the emitted MLIR:
[Debugging a thunk](mlir.md#9.-Debugging-a-thunk).
