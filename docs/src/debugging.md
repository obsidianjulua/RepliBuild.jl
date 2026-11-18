# Debug it

Sooner or later a call returns the wrong value, or the process crashes inside
the library. With a well-known library that is rare. With your own code, or a
library nobody has wrapped before, it is the normal course of work, and it is
where RepliBuild gives you something other binding tools do not: **every layer
between your Julia call and the library's code can be read and stepped
through.**

There are three layers, and each has a source file a debugger can show:

| Layer | Its source | Why you can see it |
|:------|:-----------|:-------------------|
| The library | Its own `.c` and `.cpp` files | RepliBuild compiled it, with debug information, and kept it |
| The call stub (C++ only) | A generated `.mlir` file in the project's `.debug/` folder | RepliBuild writes the stub's source to disk and tells the debugger about it |
| The wrapper | `julia/<Name>.jl` | It is ordinary generated Julia |

## Start with what needs no debugger

**A crash names the line.** When the library crashes, Julia's crash report
includes the library's source file and line, because the library carries its
debug information:

```
[995976] signal 11 (1): Segmentation fault
in expression starting at crash.jl:2
ml_normalize at /home/me/mathlib/src/mathlib.c:7
ml_normalize at /home/me/mathlib/julia/Mathlib.jl:431
```

The first line is where the C code died. The second is the wrapper function
that called it.

**Read what the wrapper does.** Each function's docstring gives its Julia
signature, the C types of its arguments, and its symbol name in the library:

```julia
?ml_normalize
@less ml_normalize(Ref(ml_vec2(1.0, 2.0)))     # the generated function itself
```

**Compare sizes.** If a struct's fields look shifted, compare what the wrapper
believes with what the library's own compiler says:

```julia
M.struct_size(:ml_vec2)              # 16
M.member_offset(:ml_vec2, :y)        # 8
sizeof(M.ml_vec2)                    # must equal struct_size
```

**See what changed.** `wrap` keeps the previous module as `<Name>.jl.prev`.
After a change to the TOML or an upgrade, `diff julia/Mathlib.jl.prev
julia/Mathlib.jl` shows exactly which functions and types moved.

**Read `wrap`'s report.** It lists what it could not do: structs wrapped as raw
bytes, functions it refused to call (`ABI Safety Trap`), constructors and
methods it added or skipped, names it withheld.

## Step through the library with gdb

The library RepliBuild builds has full debug information. Run Julia under
`gdb`, break on a library function by its name, and you are in its source,
with its arguments and variables:

```bash
gdb -ex 'handle SIGSEGV nostop noprint pass' \
    -ex 'set breakpoint pending on' \
    -ex 'break ml_normalize' \
    -ex run \
    --args julia --project=. script.jl
```

```
Thread 1 "julia" hit Breakpoint 1, ml_normalize (v=0x7fffe0479c50) at mathlib/src/mathlib.c:7
7	ml_status ml_normalize(ml_vec2 *v) { double n = sqrt(ml_dot(*v, *v)); …
(gdb) print *v
$1 = {x = 3, y = 4}
(gdb) finish
Value returned is $2 = ML_OK
```

`print *v` shows what actually arrived from Julia, which settles most "wrong
value" questions at once: either the argument is already wrong on arrival, or
the library computed something you did not expect.

The two `-ex` settings at the front are required:

- `handle SIGSEGV nostop noprint pass`: Julia uses that signal for its own
  purposes during normal running, and gdb would otherwise stop on each one.
- `set breakpoint pending on`: the library is not loaded when gdb starts, so
  the breakpoint has to wait for it.

For a C++ method, break on its C++ name: `break shapes::Rect::area`.

**Build for debugging.** An optimized build removes variables and merges
lines. While you are chasing a bug, build without optimization:

```toml
[compile]
flags = ["-O0", "-fPIC"]

[link]
optimization_level = "0"
```

Put `-O2` and `"2"` back afterwards.

## Step through a C++ call

A C++ function is not called directly. RepliBuild generates a small stub for
it that loads the arguments, finds the right method (through the object's
virtual table when the method is virtual), makes the call, and catches a C++
exception. When a C++ call misbehaves, the question is usually whether the
stub passed the arguments the way the method expects.

You can look. The stubs have a source file too, and gdb stops in it.

Every stub is named after its function's symbol, with `_thunk` at the end.
The symbol is in the function's docstring (`?shapes_Rect_area`, "Mangled
symbol"). Break on the stub and on the method:

```bash
gdb -ex 'handle SIGSEGV nostop noprint pass' \
    -ex 'set breakpoint pending on' \
    -ex 'break _ZNK6shapes4Rect4areaEv_thunk' \
    -ex 'break shapes::Rect::area' \
    -ex run \
    --args julia --project=. script.jl
```

```
Thread 1 "julia" hit Breakpoint 1, _ZNK6shapes4Rect4areaEv_thunk () at jlcs_4eff28487aee4fc6.mlir:217
217	  %val_ptr_1 = "jlcs.get_field"(%args_ptr) {fieldOffset = 0 : i64} : (!llvm.ptr) -> !llvm.ptr
(gdb) continue

Thread 1 "julia" hit Breakpoint 2, shapes::Rect::area (this=0x29e3af0) at shapes/src/shapes.cpp:8
8	double Rect::area() const { return w_ * h_; }
(gdb) bt
#0  shapes::Rect::area (this=0x29e3af0) at shapes/src/shapes.cpp:8
#1  _ZNK6shapes4Rect4areaEv_thunk () at jlcs_4eff28487aee4fc6.mlir:219
#2  _mlir_ciface__ZNK6shapes4Rect4areaEv_thunk () at jlcs_4eff28487aee4fc6.mlir:216
(gdb) print *this
$1 = {<shapes::Shape> = {_vptr$Shape = 0x7fffc0c25d50 <vtable for shapes::Rect+16>, name_ = "rect"}, w_ = 2, h_ = 3}
```

The backtrace runs from your C++ method, through the stub at the exact line
that made the call, back toward Julia. `print *this` shows the object the
method received: if `this` is wrong here, the pointer you passed from Julia
was wrong; if it is right and the result is still wrong, the problem is in the
C++.

`step` and `next` work in the stub. You do not need to understand every word
of a stub's line to use it: each line does one thing, and they read top to
bottom as *get the argument, load it, call, return*.

## Read a stub without running anything

`RepliBuild.Debug` shows a stub's source next to the machine code it became,
from files on disk. No debugger, and no process stopped at the right moment.

Set one environment variable **before** the wrapper is loaded, and run
anything that loads it:

```julia
ENV["REPLIBUILD_JIT_OBJDUMP"] = "1"
using RepliBuild
include("shapes/julia/Shapes.jl")
```

Then, in that session or any later one:

```julia
D = RepliBuild.Debug

D.thunks("shapes")                 # every stub's name
D.walk("shapes", "_ZNK6shapes4Rect4areaEv_thunk")
```

```
═══ MLIR ═══════════════════════════════════════════════════════════════
func.func @_ZNK6shapes4Rect4areaEv_thunk(%args_ptr: !llvm.ptr) -> f64 … {
  %val_ptr_1 = "jlcs.get_field"(%args_ptr) {fieldOffset = 0 : i64} : (!llvm.ptr) -> !llvm.ptr
  %val_1 = llvm.load %val_ptr_1 : !llvm.ptr -> !llvm.ptr
  %ret_val = "jlcs.vcall"(%val_1) { class_name = @Rect, vtable_offset = 0 : i64, slot = 2 : i64, … } : (!llvm.ptr) -> f64
  return %ret_val : f64
}

═══ EMITTED ════════════════════════════════════════════════════════════
0000000000000740 <_ZNK6shapes4Rect4areaEv_thunk>:
  %val_ptr_1 = "jlcs.get_field"(%args_ptr) {fieldOffset = 0 : i64} …
     745:	48 8b 07             	mov    (%rdi),%rax
  %val_1 = llvm.load %val_ptr_1 : !llvm.ptr -> !llvm.ptr
     748:	48 8b 38             	mov    (%rax),%rdi
  %ret_val = "jlcs.vcall"(%val_1) { class_name = @Rect, … slot = 2 … }
     74b:	48 8b 07             	mov    (%rdi),%rax
     74e:	48 8b 40 10          	mov    0x10(%rax),%rax
     75a:	ff d0                	call   *%rax
```

Read it as: the object pointer goes into the first argument register, its
virtual table is loaded, entry 2 of the table (offset `0x10`) is called. If a
virtual call lands in the wrong method, this is where you see which entry was
used. If a struct argument arrives scrambled, this is where you see which
registers it was put in.

| Function | Gives |
|:---------|:------|
| `D.thunks(path)` | The names of all the stubs |
| `D.walk(path, name)` | One stub: its source, then its machine code with the source interleaved |
| `D.mlir_body(path, name)` | Only the stub's source |
| `D.disassemble(path; symbol = name)` | Only the machine code. `source = false` for bare instructions |
| `D.mlir_sources(path)` | The generated source files, newest first |
| `D.has_object(path)` | Whether machine code was captured |

`path` is the project folder, or anything inside it: the library file, the
`.debug` folder, `"."` when you are standing in the project.

Two things to know:

- The stub **source** is always written, to `.debug/mlir/` in the project,
  each time the wrapper prepares its stubs. The **machine code** is kept only
  when `REPLIBUILD_JIT_OBJDUMP` was set before the wrapper loaded. Setting it
  afterwards is too late for that session, and `walk` says so.
- A wrapper built with
  [`aot_thunks = true`](performance.md#C-stubs-compiled-ahead-of-time) can be
  debugged the same way. Its stubs are in `lib<name>_thunks.so`, with the same
  names and the same source: the gdb session above works unchanged, and
  `D.thunks` and `D.mlir_body` work too. Only `D.walk` and `D.disassemble`
  need the machine code captured at load, which such a wrapper never does; for
  those, turn `aot_thunks` off, `wrap`, and load once with the variable set.

## Which copy of a library was called

When a call seems to reach the wrong code entirely, check which library file
each name came from. On Linux:

```bash
LD_DEBUG=bindings julia --project=. script.jl 2>&1 | grep ml_normalize
```

`M.LIBRARY_PATH` is the file the wrapper opens. For two wrappers that link
each other, see [Several libraries](libraries.md#Two-recipes,-one-using-the-other).

## When it is RepliBuild's mistake

If the library receives arguments that differ from what you passed, with types
the docstring says are right, that is a bug in RepliBuild and worth reporting.
The most useful report contains what this page produces: the function's C or
C++ declaration, the docstring of its wrapper function, and for C++ the output
of `D.walk` for its stub.

Until it is fixed, leave the function out with
[`[wrap] exclude_symbols`](toml.md#Leaving-things-out) so the rest of the
wrapper is usable.
