//===- JLCSPasses.cpp - JLCS dialect lowering passes --------------------===//
//
// Lowering passes for the JLCS dialect to LLVM dialect
//
//===----------------------------------------------------------------------===//

#include "JLCSDialect.h"
#include "JLCSTypes.h"
#include "JLCSLoweringUtils.h"
#include "JLCSOps.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::jlcs;

namespace {

// The C++ exception PERSONALITY. A separate axis from the struct ABI selected
// by `kHostAbi` further down, and not derivable from it.
//
// The Itanium C++ ABI unwinds with DWARF CFI and calls its personality routine
// `__gxx_personality_v0`. mingw-w64 on x86-64 unwinds with SEH and calls it
// `__gxx_personality_seh0` — and that is not a spelling preference: the C++
// runtime there EXPORTS seh0 and does not export v0 at all. A thunk emitting v0
// therefore fails to link with exactly one undefined symbol and no other
// symptom, which is a long way from the word "exceptions".
//
// Checked against the toolchain rather than assumed: `clang++` for
// x86_64-w64-windows-gnu emits `personality ptr @__gxx_personality_seh0`, and
// v0 appears in no libc++/libc++abi/libunwind archive shipped there.
//
// Keyed on the host for the same reason `kHostAbi` is: this library feeds an
// in-process JIT and the AOT thunks built beside it, so the target IS the host.
// JLCS_FORCE_ABI_WIN64 deliberately does NOT change it — that flag exists to
// diff Win64 struct lowering from a Linux box, such a library cannot run
// anything regardless, and giving it a personality the Linux runtime lacks
// would break that build for nothing.
//
// Declared here, above every use, rather than beside kHostAbi: the first use is
// in the vcall lowering far above that point.
#if defined(_WIN32)
static constexpr const char *kCxxPersonality = "__gxx_personality_seh0";
#else
static constexpr const char *kCxxPersonality = "__gxx_personality_v0";
#endif

//===----------------------------------------------------------------------===//
// GetFieldOp Lowering
//===----------------------------------------------------------------------===//

struct GetFieldOpLowering : public ConversionPattern {
    GetFieldOpLowering(LLVMTypeConverter& typeConverter, MLIRContext* ctx)
        : ConversionPattern(typeConverter, GetFieldOp::getOperationName(), 1,
              ctx)
    {
    }

    LogicalResult
    matchAndRewrite(Operation* op, ArrayRef<Value> operands,
        ConversionPatternRewriter& rewriter) const override
    {
        auto getFieldOp = cast<GetFieldOp>(op);
        Location loc = getFieldOp.getLoc();

        // Get operands and attributes
        GetFieldOp::Adaptor adaptor(operands);
        Value structPtr = adaptor.getStructValue();
        int64_t byteOffset = getFieldOp.getFieldOffset();
        Type resultType = getFieldOp.getResult().getType();

        // Use helper to load the field
        Value loadedVal = getStructField(loc, rewriter, structPtr, byteOffset, resultType);

        // Replace the original operation
        rewriter.replaceOp(op, loadedVal);
        return success();
    }
};

//===----------------------------------------------------------------------===//
// SetFieldOp Lowering
//===----------------------------------------------------------------------===//

struct SetFieldOpLowering : public ConversionPattern {
    SetFieldOpLowering(LLVMTypeConverter& typeConverter, MLIRContext* ctx)
        : ConversionPattern(typeConverter, SetFieldOp::getOperationName(), 1,
              ctx)
    {
    }

    LogicalResult
    matchAndRewrite(Operation* op, ArrayRef<Value> operands,
        ConversionPatternRewriter& rewriter) const override
    {
        auto setFieldOp = cast<SetFieldOp>(op);
        Location loc = setFieldOp.getLoc();

        // Get operands and attributes
        SetFieldOp::Adaptor adaptor(operands);
        Value structPtr = adaptor.getStructValue();
        Value newValue = adaptor.getNewValue();
        int64_t byteOffset = setFieldOp.getFieldOffset();

        // Use helper to store the field
        setStructField(loc, rewriter, structPtr, byteOffset, newValue);

        // Erase the original operation (no results)
        rewriter.eraseOp(op);
        return success();
    }
};

//===----------------------------------------------------------------------===//
// VirtualCallOp Lowering
//===----------------------------------------------------------------------===//

/// Get or declare an external llvm.func in the module (shared by the vcall
/// EH path; TryCallOpLowering carries its own identical member helper).
static LLVM::LLVMFuncOp getOrInsertLLVMFn(ModuleOp module,
                                          ConversionPatternRewriter& rewriter,
                                          StringRef name,
                                          LLVM::LLVMFunctionType fnType)
{
    if (auto fn = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
        return fn;
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    return LLVM::LLVMFuncOp::create(rewriter, module.getLoc(), name, fnType);
}

struct VirtualCallOpLowering : public ConversionPattern {
    VirtualCallOpLowering(LLVMTypeConverter& typeConverter, MLIRContext* ctx)
        : ConversionPattern(typeConverter, VirtualCallOp::getOperationName(), 1,
              ctx)
    {
    }

    LogicalResult
    matchAndRewrite(Operation* op, ArrayRef<Value> operands,
        ConversionPatternRewriter& rewriter) const override
    {
        auto vcallOp = cast<VirtualCallOp>(op);
        Location loc = vcallOp.getLoc();

        // Get attributes
        int64_t vtableOffset = vcallOp.getVtableOffset();
        int64_t thisOffset = vcallOp.getThisOffset();
        int64_t slot = vcallOp.getSlot();
        VirtualCallOp::Adaptor adaptor(operands);
        ValueRange args = adaptor.getArgs();

        if (args.empty()) {
            return op->emitError("VirtualCallOp requires at least object pointer argument");
        }

        Value objPtr = args[0]; // First arg is always the object pointer
        auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
        auto i64Type = rewriter.getI64Type();

        // Step 1: Load vtable pointer from object
        // vtable_ptr = *(objPtr + vtableOffset)
        // vtableOffset is relative to the ORIGINAL pointer — for a secondary
        // base it already points at that base's vptr, so no pre-adjustment.
        Value vtablePtr = getStructField(loc, rewriter, objPtr, vtableOffset, ptrType);

        // Step 1b: `this` adjustment. A method dispatched through a
        // non-primary base expects `this` to point at that base's subobject,
        // not the derived object's start; entries in a secondary vtable are
        // compiled against the base-relative `this`. Adjust args[0] by
        // this_offset bytes (i8 GEP) before the call. this_offset == 0 (the
        // default, and every single-inheritance case) emits nothing.
        Value adjustedThis = objPtr;
        if (thisOffset != 0) {
            Value thisOffVal = arith::ConstantIntOp::create(rewriter, loc, thisOffset, 64);
            adjustedThis = LLVM::GEPOp::create(rewriter,
                loc, ptrType, rewriter.getI8Type(), objPtr,
                ArrayRef<LLVM::GEPArg>({ thisOffVal }));
        }

        // Step 2: Index into vtable to get function pointer
        // func_ptr = vtable[slot]
        Value slotVal = arith::ConstantIntOp::create(rewriter, loc, slot, 64);
        Value funcPtrAddr = LLVM::GEPOp::create(rewriter, 
            loc, ptrType, ptrType, vtablePtr,
            ArrayRef<LLVM::GEPArg>({ slotVal }));

        Value funcPtr = LLVM::LoadOp::create(rewriter, loc, ptrType, funcPtrAddr);

        // Step 3: Indirect call through the function pointer.
        //
        // llvm.call carries the AttrSizedOperandSegments trait with TWO operand
        // groups — callee_operands and op_bundle_operands. For an indirect call the
        // callee pointer is the *first element of callee_operands*, so the correct
        // operandSegmentSizes is {1 + nArgs, 0}. The previous hand-rolled
        // OperationState set a 3-element {1, nArgs, 0} (mistaking args for their own
        // segment) and omitted var_callee_type; during LLVM IR translation
        // OperandRange::split read the malformed 3-entry array against a 2-segment op
        // and walked off the end (SIGSEGV in translateModuleToLLVMIR). Use the
        // dedicated indirect-call builder, which sets both correctly.
        Type resultLLVMType;
        if (vcallOp.getResult()) {
            resultLLVMType = typeConverter->convertType(vcallOp.getResult().getType());
            if (!resultLLVMType) {
                return op->emitError("VirtualCallOp: Could not convert result type");
            }
        } else {
            resultLLVMType = LLVM::LLVMVoidType::get(rewriter.getContext());
        }

        // The callee's LLVM function type, derived from the (already-converted)
        // argument values. args[0] is the object pointer (`this`).
        SmallVector<Type, 4> paramTypes;
        for (Value a : args)
            paramTypes.push_back(a.getType());
        auto calleeType = LLVM::LLVMFunctionType::get(resultLLVMType, paramTypes,
                                                      /*isVarArg=*/false);

        // Indirect-call operand list: function pointer first, then the call
        // args with the (possibly) adjusted `this` in place of args[0].
        SmallVector<Value, 5> callOperands;
        callOperands.push_back(funcPtr);
        callOperands.push_back(adjustedThis);
        callOperands.append(args.begin() + 1, args.end());

        if (!vcallOp.getMayThrow()
#ifdef _WIN32
            || true
#endif
        ) {
            // Plain indirect call — the pre-existing lowering, unchanged.
            // Windows: no in-JIT landing pad; see TryCallOpLowering.
            auto callOp = LLVM::CallOp::create(rewriter, loc, calleeType, callOperands);

            if (vcallOp.getResult()) {
                rewriter.replaceOp(op, callOp.getResult());
            } else {
                rewriter.eraseOp(op);
            }
            return success();
        }

        // --- may_throw: indirect invoke + landing pad -------------------
        // Same sentinel-continue EH model as TryCallOpLowering: a thrown C++
        // exception is caught, recorded via jlcs_catch_current_exception(),
        // and the call yields a zero sentinel. No sret/packed coercion here
        // (see op description) — producers only set may_throw on
        // scalar/pointer signatures.
        auto moduleOp = op->getParentOfType<ModuleOp>();
        auto voidType = LLVM::LLVMVoidType::get(rewriter.getContext());
        auto i32Type = rewriter.getI32Type();

        auto personalityFnType = LLVM::LLVMFunctionType::get(i32Type, {}, true);
        getOrInsertLLVMFn(moduleOp, rewriter, kCxxPersonality, personalityFnType);
        getOrInsertLLVMFn(moduleOp, rewriter, "__cxa_begin_catch",
            LLVM::LLVMFunctionType::get(ptrType, {ptrType}, false));
        getOrInsertLLVMFn(moduleOp, rewriter, "__cxa_end_catch",
            LLVM::LLVMFunctionType::get(voidType, {}, false));
        getOrInsertLLVMFn(moduleOp, rewriter, "jlcs_catch_current_exception",
            LLVM::LLVMFunctionType::get(ptrType, {}, false));

        if (auto llvmFunc = op->getParentOfType<LLVM::LLVMFuncOp>()) {
            llvmFunc.setPersonalityAttr(
                FlatSymbolRefAttr::get(rewriter.getContext(), kCxxPersonality));
            setUwtableAsync(llvmFunc);
        } else if (auto funcOp = op->getParentOfType<func::FuncOp>()) {
            funcOp->setAttr("llvm.personality",
                FlatSymbolRefAttr::get(rewriter.getContext(), kCxxPersonality));
        }

        // Result slot, zero-initialized for the exception path. Created
        // before the block split so it dominates both branches.
        SmallVector<Type, 1> callResultTypes;
        bool hasResult = static_cast<bool>(vcallOp.getResult());
        Value resultSlot;
        if (hasResult) {
            callResultTypes.push_back(resultLLVMType);
            Value one = arith::ConstantIntOp::create(rewriter, loc, 1, 64);
            resultSlot = LLVM::AllocaOp::create(rewriter, loc, ptrType, resultLLVMType, one);
            Value zero = LLVM::ZeroOp::create(rewriter, loc, resultLLVMType);
            LLVM::StoreOp::create(rewriter, loc, zero, resultSlot);
        }

        Block *currentBlock = rewriter.getInsertionBlock();
        auto *parentRegion = currentBlock->getParent();
        Block *mergeBlock = rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());

        Block *invokeOkBlock = new Block();
        parentRegion->getBlocks().insertAfter(currentBlock->getIterator(), invokeOkBlock);
        Block *catchBlock = new Block();
        parentRegion->getBlocks().insertAfter(invokeOkBlock->getIterator(), catchBlock);

        // Indirect invoke: null callee symbol + var_callee_type, function
        // pointer as the first callee operand. Built via the ODS builder so
        // operandSegmentSizes is computed correctly (the hand-rolled
        // OperationState variant of this once SIGSEGV'd for plain calls).
        rewriter.setInsertionPointToEnd(currentBlock);
        auto invokeOp = LLVM::InvokeOp::create(rewriter, loc,
            TypeRange(callResultTypes),
            /*var_callee_type=*/TypeAttr::get(calleeType),
            /*callee=*/FlatSymbolRefAttr{},
            /*callee_operands=*/ValueRange(callOperands),
            /*arg_attrs=*/ArrayAttr{}, /*res_attrs=*/ArrayAttr{},
            /*normalDestOperands=*/ValueRange{},
            /*unwindDestOperands=*/ValueRange{},
            /*branch_weights=*/DenseI32ArrayAttr{},
            LLVM::CConvAttr::get(rewriter.getContext(), LLVM::CConv::C),
            /*op_bundle_operands=*/ArrayRef<ValueRange>{},
            /*op_bundle_tags=*/ArrayAttr{},
            invokeOkBlock, catchBlock);

        rewriter.setInsertionPointToEnd(invokeOkBlock);
        if (hasResult) {
            LLVM::StoreOp::create(rewriter, loc, invokeOp->getResult(0), resultSlot);
        }
        LLVM::BrOp::create(rewriter, loc, ValueRange{}, mergeBlock);

        rewriter.setInsertionPointToEnd(catchBlock);
        auto lpStructType = LLVM::LLVMStructType::getLiteral(
            rewriter.getContext(), {ptrType, i32Type}, false);
        Value nullPtr = LLVM::ZeroOp::create(rewriter, loc, ptrType);
        auto landingPad = LLVM::LandingpadOp::create(rewriter,
            loc, lpStructType, /*cleanup=*/false, ValueRange{nullPtr});
        Value exnPtr = LLVM::ExtractValueOp::create(rewriter, loc, ptrType,
            landingPad, ArrayRef<int64_t>{0});
        LLVM::CallOp::create(rewriter, loc, TypeRange{ptrType},
            "__cxa_begin_catch", ValueRange{exnPtr});
        LLVM::CallOp::create(rewriter, loc, TypeRange{ptrType},
            "jlcs_catch_current_exception", ValueRange{});
        LLVM::CallOp::create(rewriter, loc, TypeRange{},
            "__cxa_end_catch", ValueRange{});
        LLVM::BrOp::create(rewriter, loc, ValueRange{}, mergeBlock);

        rewriter.setInsertionPointToStart(mergeBlock);
        if (hasResult) {
            Value result = LLVM::LoadOp::create(rewriter, loc, resultLLVMType, resultSlot);
            rewriter.replaceOp(op, result);
        } else {
            rewriter.eraseOp(op);
        }

        return success();
    }
};

//===----------------------------------------------------------------------===//
// LoadArrayElementOp Lowering
//===----------------------------------------------------------------------===//

struct LoadArrayElementOpLowering : public ConversionPattern {
    LoadArrayElementOpLowering(LLVMTypeConverter& typeConverter, MLIRContext* ctx)
        : ConversionPattern(typeConverter,
              LoadArrayElementOp::getOperationName(), 1, ctx)
    {
    }

    LogicalResult
    matchAndRewrite(Operation* op, ArrayRef<Value> operands,
        ConversionPatternRewriter& rewriter) const override
    {
        auto loadOp = cast<LoadArrayElementOp>(op);
        Location loc = loadOp.getLoc();

        LoadArrayElementOp::Adaptor adaptor(operands);
        Value viewPtr = adaptor.getView();
        ValueRange indices = adaptor.getIndices();

        auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
        auto i64Type = rewriter.getI64Type();

        // 1. Load the base data pointer from the ArrayView struct (offset 0)
        Value dataPtr = getStructField(loc, rewriter, viewPtr, 0, ptrType);

        // 2. Load the strides pointer (offset 16)
        Value stridesPtr = getStructField(loc, rewriter, viewPtr, 16, ptrType);

        // 3. Calculate the total offset: Offset = Sum(index_i * stride_i)
        Value totalOffset = arith::ConstantIntOp::create(rewriter, loc, 0, 64);

        for (size_t i = 0; i < indices.size(); ++i) {
            Value index = indices[i];

            // GEP to the i-th stride in the strides array
            Value strideIndex = arith::ConstantIntOp::create(rewriter, loc, i, 64);
            Value strideAddr = LLVM::GEPOp::create(rewriter, 
                loc, ptrType, i64Type, stridesPtr,
                ArrayRef<LLVM::GEPArg>({ strideIndex }));

            // Load the i-th stride value
            Value stride = LLVM::LoadOp::create(rewriter, loc, i64Type, strideAddr);

            // Calculate: index * stride
            Value elementOffset = arith::MulIOp::create(rewriter, loc, index, stride);

            // Accumulate: totalOffset += elementOffset
            totalOffset = arith::AddIOp::create(rewriter, loc, totalOffset, elementOffset);
        }

        // 4. Calculate the final address (GEP on the base data pointer)
        Type elemType = loadOp.getResult().getType();
        Value finalAddr = LLVM::GEPOp::create(rewriter, 
            loc, ptrType, elemType, dataPtr,
            ArrayRef<LLVM::GEPArg>({ totalOffset }));

        // 5. Load the element
        Value result = LLVM::LoadOp::create(rewriter, loc, elemType, finalAddr);

        rewriter.replaceOp(op, result);
        return success();
    }
};

//===----------------------------------------------------------------------===//
// StoreArrayElementOp Lowering
//===----------------------------------------------------------------------===//

struct StoreArrayElementOpLowering : public ConversionPattern {
    StoreArrayElementOpLowering(LLVMTypeConverter& typeConverter, MLIRContext* ctx)
        : ConversionPattern(typeConverter,
              StoreArrayElementOp::getOperationName(), 1, ctx)
    {
    }

    LogicalResult
    matchAndRewrite(Operation* op, ArrayRef<Value> operands,
        ConversionPatternRewriter& rewriter) const override
    {
        auto storeOp = cast<StoreArrayElementOp>(op);
        Location loc = storeOp.getLoc();

        StoreArrayElementOp::Adaptor adaptor(operands);
        Value value = adaptor.getValue();
        Value viewPtr = adaptor.getView();
        ValueRange indices = adaptor.getIndices();

        auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
        auto i64Type = rewriter.getI64Type();

        // 1. Load the base data pointer (offset 0)
        Value dataPtr = getStructField(loc, rewriter, viewPtr, 0, ptrType);

        // 2. Load the strides pointer (offset 16)
        Value stridesPtr = getStructField(loc, rewriter, viewPtr, 16, ptrType);

        // 3. Calculate the total offset
        Value totalOffset = arith::ConstantIntOp::create(rewriter, loc, 0, 64);

        for (size_t i = 0; i < indices.size(); ++i) {
            Value index = indices[i];
            Value strideIndex = arith::ConstantIntOp::create(rewriter, loc, i, 64);
            Value strideAddr = LLVM::GEPOp::create(rewriter, 
                loc, ptrType, i64Type, stridesPtr,
                ArrayRef<LLVM::GEPArg>({ strideIndex }));

            Value stride = LLVM::LoadOp::create(rewriter, loc, i64Type, strideAddr);
            Value elementOffset = arith::MulIOp::create(rewriter, loc, index, stride);
            totalOffset = arith::AddIOp::create(rewriter, loc, totalOffset, elementOffset);
        }

        // 4. Calculate the final address
        Value finalAddr = LLVM::GEPOp::create(rewriter, 
            loc, ptrType, value.getType(), dataPtr,
            ArrayRef<LLVM::GEPArg>({ totalOffset }));

        // 5. Store the value
        LLVM::StoreOp::create(rewriter, loc, value, finalAddr);

        rewriter.eraseOp(op);
        return success();
    }
};

//===----------------------------------------------------------------------===//
// TypeInfoOp Lowering
//===----------------------------------------------------------------------===//

struct TypeInfoOpLowering : public ConversionPattern {
    TypeInfoOpLowering(LLVMTypeConverter& typeConverter, MLIRContext* ctx)
        : ConversionPattern(typeConverter, TypeInfoOp::getOperationName(), 1,
              ctx)
    {
    }

    LogicalResult
    matchAndRewrite(Operation* op, ArrayRef<Value> operands,
        ConversionPatternRewriter& rewriter) const override
    {
        // TypeInfoOp is metadata, safe to erase during lowering to LLVM
        rewriter.eraseOp(op);
        return success();
    }
};

//===----------------------------------------------------------------------===//
// Shared ABI helpers
//===----------------------------------------------------------------------===//

// Calculate the packed (no-padding) size of a type in bits.
// Used by FFECallOp and TryCallOp lowering for ABI coercion decisions.
static uint64_t getPackedSizeInBits(Type type) {
    if (auto structType = dyn_cast<LLVM::LLVMStructType>(type)) {
        uint64_t sum = 0;
        for (Type elem : structType.getBody()) {
            sum += getPackedSizeInBits(elem);
        }
        return sum;
    }
    if (auto arrType = dyn_cast<LLVM::LLVMArrayType>(type)) {
        return arrType.getNumElements() * getPackedSizeInBits(arrType.getElementType());
    }
    if (type.isIntOrFloat()) {
        return type.getIntOrFloatBitWidth();
    }
    if (isa<LLVM::LLVMPointerType>(type)) {
        return 64; // Assume 64-bit pointers
    }
    return 0; // Unknown
}

// --- Struct ABI classification ----------------------------------------------
//
// Two x86-64 conventions, one derivation. The host IS the target: this pass
// feeds an in-process JIT (and the AOT thunks built beside it), so the ABI is
// picked at compile time from the host and never varies at run time.
//
// Both rule sets are compiled on every host — only the default differs — so
// the inactive one cannot bit-rot unnoticed the way a #ifdef'd-out branch does.

enum class AbiTarget { SysV, Win64 };

// -DJLCS_FORCE_ABI_WIN64 builds the Win64 rules into a Linux libJLCS. The
// resulting library CANNOT run anything — the JIT would emit Win64 calls into a
// SysV process — but the pass itself is IR→IR and touches no OS API, so it can
// still be driven on Linux and its output diffed against
// `clang --target=x86_64-w64-windows-gnu`. That is the only way to exercise the
// Win64 LOWERING (not just the decision table) without a Windows host, so build
// it to a scratch path and never over the working libJLCS.so.
#if defined(JLCS_FORCE_ABI_WIN64)
static constexpr AbiTarget kHostAbi = AbiTarget::Win64;
#elif defined(_WIN32)
static constexpr AbiTarget kHostAbi = AbiTarget::Win64;
#elif defined(__x86_64__) || defined(_M_X64)
static constexpr AbiTarget kHostAbi = AbiTarget::SysV;
#else
// Silently applying x86-64 struct classification to AArch64 AAPCS (or anything
// else) miscompiles every aggregate call with no diagnostic. Fail at build.
#error "JLCS struct ABI classification supports x86-64 only (SysV or Win64)."
#endif

// `regs` is the coerced scalar per argument register the aggregate occupies:
// up to two on SysV (one per eightbyte), exactly one on Win64.
//
// The three fields after it are set only from the generator's statement of the
// record's SysV class (`SysVFacts` below), never from a body:
//   * `ignore`: no class at all (a record of padding): no operand is passed
//     and a return is void;
//   * `loNone`: the low eightbyte is padding, so `regs` holds the high one
//     alone and it travels from offset 8 (clang's `getDirect(HighPart, 8)`);
//   * `memAlign`: the record's alignment for a byval or sret slot (0: unknown,
//     so take the body's).
struct StructABI {
    bool memoryClass = true;              // true → sret / indirect-pointer path
    SmallVector<Type, 2> regs;
    bool ignore = false;
    bool loNone = false;
    unsigned memAlign = 0;
};

// --- The generator's statement of a record's SysV class ----------------------
//
// FunctionGen classifies every record it passes or returns by value from the
// record's DWARF facts (`sysv_record_abi`: members at their offsets, bitfields
// by bit range, nested records, unions, long double, _Complex, __int128, the
// record's alignment) and states the verdict on the call:
//
//   sysv_args = ["", "regs:int,sse", "mem:16", "ignore", ...]   one per operand
//   sysv_ret  = "regs:none,sse"                                  for a record result
//
//   "mem:<align>"     MEMORY class: a byval or sret slot of that alignment (≥ 8)
//   "regs:<c>[,<c>]"  one class per eightbyte: int (an i64), sse (an f64), or
//                     none for a low eightbyte of padding, which C passes nowhere
//   "ignore"          no class (only padding): no operand, a void return
//
// A body cannot say these things: StructGen's pad arrays are bytes like a
// field's, a tiled body spells a nested record or a union as bytes, a bitfield
// sits in its unit's whole type, and nothing in a body says `aligned(16)`.
// Classifying from the body put `{float; double}` in RDI and XMM0, a packed
// `{char; {short; float}}` in registers, `#pragma pack(1) {char; int : 31}` in
// memory, and gave `__int128` records an 8-aligned slot (test_sysv_abi_oracle,
// 2026-10-24). An operand or result with no statement is still classified from
// its body, which is what hand-written IR (test_struct_abi.jl) relies on. The
// statement is SysV's; the Win64 rules never read it.
struct SysVFacts {
    bool stated = false;
    bool memory = false;
    bool ignore = false;
    bool loNone = false;
    unsigned align = 8;
    SmallVector<Type, 2> regs;
};

static bool parseSysVFacts(StringRef s, MLIRContext* ctx, SysVFacts& out, std::string& err) {
    out = SysVFacts();
    if (s.empty())
        return true;
    out.stated = true;
    if (s == "ignore") {
        out.ignore = true;
        return true;
    }
    if (s.consume_front("mem:")) {
        unsigned a = 0;
        if (s.getAsInteger(10, a) || a < 8 || !llvm::isPowerOf2_32(a)) {
            err = ("a byval/sret alignment must be a power of two of at least 8: `mem:" + s + "`").str();
            return false;
        }
        out.memory = true;
        out.align = a;
        return true;
    }
    if (s.consume_front("regs:")) {
        SmallVector<StringRef, 2> parts;
        s.split(parts, ',');
        if (parts.empty() || parts.size() > 2) {
            err = ("a register-class record has one or two eightbytes: `regs:" + s + "`").str();
            return false;
        }
        for (auto [i, c] : llvm::enumerate(parts)) {
            if (c == "int") {
                out.regs.push_back(IntegerType::get(ctx, 64));
            } else if (c == "sse") {
                out.regs.push_back(Float64Type::get(ctx));
            } else if (c == "none" && i == 0 && parts.size() == 2) {
                out.loNone = true;
            } else {
                err = ("an eightbyte class is int, sse, or none in the low eightbyte of two: `regs:" + s + "`").str();
                return false;
            }
        }
        return true;
    }
    err = ("not a SysV record class (`mem:<align>`, `regs:<class>[,<class>]`, `ignore`): `" + s + "`").str();
    return false;
}

// What a jlcs.ffe_call / jlcs.try_call states about its record crossings.
// `args` has one entry per operand of the op (unstated where it says nothing).
struct CallFacts {
    SysVFacts ret;
    SmallVector<SysVFacts, 4> args;
    std::string error;
};

static CallFacts readCallFacts(Operation* op, unsigned numOperands) {
    CallFacts f;
    MLIRContext* ctx = op->getContext();
    f.args.resize(numOperands);
    if (auto ret = op->getAttrOfType<StringAttr>("sysv_ret")) {
        if (!parseSysVFacts(ret.getValue(), ctx, f.ret, f.error))
            f.error = "sysv_ret: " + f.error;
    }
    if (auto arr = op->getAttrOfType<ArrayAttr>("sysv_args")) {
        if (arr.size() != numOperands) {
            f.error = ("sysv_args lists " + Twine(arr.size()) + " operands, the call has " +
                       Twine(numOperands)).str();
            return f;
        }
        for (auto [i, a] : llvm::enumerate(arr)) {
            auto str = dyn_cast<StringAttr>(a);
            if (!str) {
                f.error = "sysv_args: every entry is a string";
                return f;
            }
            std::string err;
            if (!parseSysVFacts(str.getValue(), ctx, f.args[i], err)) {
                f.error = ("sysv_args[" + Twine(i) + "]: " + err).str();
                return f;
            }
        }
    }
    return f;
}

static uint64_t abiAlign(Type t) {
    if (auto st = dyn_cast<LLVM::LLVMStructType>(t)) {
        if (st.isPacked())
            return 1;
        uint64_t a = 1;
        if (!st.isOpaque())
            for (Type e : st.getBody())
                a = std::max(a, abiAlign(e));
        return a;
    }
    if (auto ar = dyn_cast<LLVM::LLVMArrayType>(t))
        return abiAlign(ar.getElementType());
    if (isa<LLVM::LLVMPointerType>(t))
        return 8;
    if (t.isIntOrFloat()) {
        uint64_t b = (t.getIntOrFloatBitWidth() + 7) / 8;
        return std::min<uint64_t>(std::max<uint64_t>(b, 1), 8);
    }
    return 8;
}

static uint64_t abiSize(Type t, unsigned depth = 0) {
    if (depth > 32)
        return 1 << 20; // malformed recursive type: force MEMORY class
    if (auto st = dyn_cast<LLVM::LLVMStructType>(t)) {
        if (st.isOpaque())
            return 0;
        uint64_t cur = 0;
        for (Type e : st.getBody()) {
            if (!st.isPacked())
                cur = llvm::alignTo(cur, abiAlign(e));
            cur += abiSize(e, depth + 1);
        }
        if (!st.isPacked())
            cur = llvm::alignTo(cur, abiAlign(st));
        return cur;
    }
    if (auto ar = dyn_cast<LLVM::LLVMArrayType>(t))
        return ar.getNumElements() * abiSize(ar.getElementType(), depth + 1);
    if (isa<LLVM::LLVMPointerType>(t))
        return 8;
    if (t.isIntOrFloat())
        return (t.getIntOrFloatBitWidth() + 7) / 8;
    return 0;
}

struct AbiLeaf {
    uint64_t offset;
    uint64_t size;
    bool isFP;
};

static void collectAbiLeaves(Type t, uint64_t base, SmallVectorImpl<AbiLeaf>& out) {
    if (auto st = dyn_cast<LLVM::LLVMStructType>(t)) {
        uint64_t cur = 0;
        for (Type e : st.getBody()) {
            if (!st.isPacked())
                cur = llvm::alignTo(cur, abiAlign(e));
            collectAbiLeaves(e, base + cur, out);
            cur += abiSize(e);
        }
        return;
    }
    if (auto ar = dyn_cast<LLVM::LLVMArrayType>(t)) {
        uint64_t esz = abiSize(ar.getElementType());
        for (uint64_t i = 0; i < ar.getNumElements(); ++i)
            collectAbiLeaves(ar.getElementType(), base + i * esz, out);
        return;
    }
    uint64_t sz = abiSize(t);
    if (sz)
        out.push_back({base, sz, isa<FloatType>(t)});
}

// x86-64 System V.
//
// A struct is MEMORY class (hidden sret pointer for returns) only when it is
// larger than 16 bytes or contains a misaligned field. Small structs with
// naturally aligned fields travel in registers, one per eightbyte: INTEGER
// eightbytes in GPRs, SSE (all-float) eightbytes in XMMs. "Packed" in the
// generator's sense merely means padding-free, NOT MEMORY class — an 8-byte
// {ptr} handle (pugixml xml_node) is returned in RAX by native code, and
// forcing sret shifted every argument by one register (the callee read the
// sret slot as `this`; found live 2026-07-18).
//
// LLVM's own return lowering for a literal struct assigns one register PER
// ELEMENT, which diverges from SysV's one-per-eightbyte whenever two fields
// share an eightbyte ({i32,i32} → SysV packs both in RAX). So register-class
// structs are coerced through memory to one scalar per eightbyte (i64 / f64),
// exactly clang's coercion, and reconstructed by a typed load on the way out.
static StructABI classifySysVStruct(LLVM::LLVMStructType s, MLIRContext* ctx) {
    StructABI r;
    uint64_t size = abiSize(s);
    if (size == 0 || size > 16)
        return r;
    SmallVector<AbiLeaf, 8> leaves;
    collectAbiLeaves(s, 0, leaves);
    bool anyInt[2] = {false, false}, anyFP[2] = {false, false};
    for (const AbiLeaf& lf : leaves) {
        uint64_t natural = std::min<uint64_t>(std::max<uint64_t>(lf.size, 1), 8);
        if (lf.offset % natural != 0)
            return r; // genuinely misaligned (attribute-packed) field → MEMORY
        uint64_t first = lf.offset / 8;
        uint64_t last = (lf.offset + lf.size - 1) / 8;
        for (uint64_t eb = first; eb <= last && eb < 2; ++eb)
            (lf.isFP ? anyFP : anyInt)[eb] = true;
    }
    r.memoryClass = false;
    unsigned nEB = (size + 7) / 8;
    for (unsigned i = 0; i < nEB; ++i) {
        bool sse = anyFP[i] && !anyInt[i];
        r.regs.push_back(sse ? (Type)Float64Type::get(ctx)
                             : (Type)IntegerType::get(ctx, 64));
    }
    return r;
}

// Microsoft x64 (what mingw-w64 GCC/clang target — GCC implements Win64 here,
// it does not carry SysV onto Windows).
//
// Not a variation on SysV; a different and much narrower algorithm:
//
//   * SIZE IS THE ONLY CRITERION. An aggregate travels in one register iff its
//     size is exactly 1, 2, 4 or 8 bytes. Everything else — 3, 5, 6, 7, and
//     anything above 8, INCLUDING the 9..16-byte band SysV splits across two
//     registers — is indirect. There is no eightbyte concept to split into.
//   * NO SSE CLASS FOR AGGREGATES. Only scalar float/double parameters reach
//     XMM registers; a struct never does. `{float,float}` is 8 bytes, so SysV
//     sends it in XMM0 and Win64 sends it in RCX as an i64. Coercing it to f64
//     here would compile clean and hand the callee the wrong register file —
//     the same silent class as the 2026-07-18 sret shift, so: integers only.
//   * NO MISALIGNMENT ESCAPE. An attribute-packed 4-byte struct is still
//     register class; the SysV alignment probe has no counterpart.
//   * EXACT-WIDTH COERCION. clang coerces to iN of the struct's own size, not
//     always i64. That matters most on return: loading i64 out of RAX when the
//     callee only wrote EAX yields garbage in the high half, and the following
//     store would then write 8 bytes into a 4-byte slot.
//
// Shared gap with the SysV path: neither models the C++ "non-trivial for the
// purposes of calls" rule (copy ctor / dtor / virtuals force indirect under
// both conventions). By this point the type is an LLVM struct with no C++
// semantics attached, so the information is already gone.
static StructABI classifyWin64Struct(LLVM::LLVMStructType s, MLIRContext* ctx) {
    StructABI r;
    uint64_t size = abiSize(s);
    if (size != 1 && size != 2 && size != 4 && size != 8)
        return r;                  // includes opaque (size 0) → indirect
    r.memoryClass = false;
    r.regs.push_back(IntegerType::get(ctx, static_cast<unsigned>(size * 8)));
    return r;
}

// The generator's statement, as a StructABI.
static StructABI classifyFromFacts(const SysVFacts& f) {
    StructABI r;
    if (f.ignore) {
        r.memoryClass = false;
        r.ignore = true;
        return r;
    }
    if (f.memory) {
        r.memoryClass = true;
        r.memAlign = f.align;
        return r;
    }
    r.memoryClass = false;
    r.loNone = f.loNone;
    r.regs = f.regs;
    return r;
}

// SysV takes the generator's statement when there is one and the body
// otherwise; Win64 crosses a record by size alone and never reads it.
static StructABI classifyStruct(LLVM::LLVMStructType s, MLIRContext* ctx,
    AbiTarget target, const SysVFacts& facts = SysVFacts())
{
    if (target == AbiTarget::Win64)
        return classifyWin64Struct(s, ctx);
    return facts.stated ? classifyFromFacts(facts) : classifySysVStruct(s, ctx);
}

// Alloca element type sized/aligned for coercion. On SysV the coerced eightbyte
// scalars round the slot up to a multiple of 8 with 8-byte alignment; on Win64
// the single register is iN of the struct's own size, so the slot fits it
// exactly. Either way, storing the struct and loading the register scalars (or
// vice versa) never reads or writes past the slot.
static Type coercionSlotType(const StructABI& abi, MLIRContext* ctx) {
    // A record whose low eightbyte is padding: one register, read from and
    // written to offset 8 of a 16-byte slot.
    if (abi.loNone)
        return LLVM::LLVMStructType::getLiteral(ctx, {IntegerType::get(ctx, 64), abi.regs[0]});
    if (abi.regs.size() == 1)
        return abi.regs[0];
    return LLVM::LLVMStructType::getLiteral(ctx, abi.regs);
}

// What the call passes or returns for a register-class record: the scalars.
static Type coercedType(const StructABI& abi, MLIRContext* ctx) {
    if (abi.regs.size() == 1)
        return abi.regs[0];
    return LLVM::LLVMStructType::getLiteral(ctx, abi.regs);
}

// The byte in the coercion slot where the coerced scalars sit.
static int32_t coercionOffset(const StructABI& abi) {
    return abi.loNone ? 8 : 0;
}

// Stack slots that stand in for a MEMORY-class argument or return value are
// addressed by native code, so they must carry the alignment that code assumes.
// Clang never emits an sret/byval slot below 8 on x86-64, and a struct we
// failed to model field-by-field degrades to `!llvm.array<N x i8>`, whose
// natural alignment is 1 — leaving the slot alignment implicit would hand the
// callee an underaligned buffer for exactly the types we understand least.
//
// Win64 raises that floor to 16, because the x64 convention requires the
// caller-allocated temporary backing an indirect aggregate to be 16-byte
// aligned. Note this is deliberately ABOVE what clang stamps on the callee
// declaration (it emits the type's natural alignment there — `align 4` for a
// 12-byte struct); the floor is ours, and over-aligning a caller-owned slot is
// always safe while under-aligning it is not.
static unsigned memorySlotAlign(Type t, AbiTarget target) {
    uint64_t floor = target == AbiTarget::Win64 ? 16 : 8;
    return static_cast<unsigned>(std::max<uint64_t>(floor, abiAlign(t)));
}

// The same, for a record whose SysV class the generator stated: the record's
// own alignment from DWARF (`aligned(16)`, `__int128`, `long double` give 16,
// which no body of ours says), still never below the floor.
static unsigned memorySlotAlign(Type t, AbiTarget target, const StructABI& abi) {
    unsigned a = memorySlotAlign(t, target);
    if (target == AbiTarget::SysV && abi.memAlign)
        a = std::max(a, abi.memAlign);
    return a;
}

// --- Shared SysV call shape ------------------------------------------------
//
// ffe_call and try_call marshal their operands identically; only the
// terminator differs (llvm.call vs llvm.invoke). The two used to carry
// independent copies of this logic, which is how a fix to one could silently
// miss the other. One derivation.

struct CallShape {
    // Carried on the shape rather than passed alongside it: byValArgAttrs and
    // retargetCallee have to size their slots the same way buildCallShape did,
    // and a separate parameter is a second place to get it wrong.
    AbiTarget target = kHostAbi;
    bool needsSret = false;
    unsigned sretAlign = 0;
    Type sretStructType;
    Value sretSlot;
    // A record return the generator stated as `ignore`: the callee returns
    // void and the result is the record's zero.
    bool retIgnored = false;
    Type origRetStructType;   // non-null → register-class (or ignored) struct return
    Type coercedRetType;      // what the call returns for a register-class record
    Type retSlotType;         // the slot the coerced value is stored to and the record loaded from
    int32_t retSlotOffset = 0; // where in that slot the coerced value goes (8 when the low eightbyte is padding)
    SmallVector<Value, 4> args;
    SmallVector<Type, 4> argTypes;
    // Every by-value struct that takes the `llvm.byval` attribute: its
    // argument index, pointee type and slot alignment. SysV only — see the
    // indirect-argument comment.
    struct ByVal {
        unsigned index;
        Type pointee;
        unsigned align;
    };
    SmallVector<ByVal, 2> byval;
};

static CallShape buildCallShape(ConversionPatternRewriter& rewriter, Location loc,
    ArrayRef<Type> resultTypeVec, ValueRange args, const CallFacts& facts,
    AbiTarget target = kHostAbi)
{
    CallShape s;
    s.target = target;
    MLIRContext* ctx = rewriter.getContext();
    Type ptrType = LLVM::LLVMPointerType::get(ctx);
    Value one = arith::ConstantIntOp::create(rewriter, loc, 1, 64);

    // Return classification: MEMORY class uses sret; register-class structs
    // are coerced to one scalar per register (SysV: per eightbyte, up to two;
    // Win64: exactly one, iN of the struct's size). Under SysV the class is
    // the generator's statement where it made one (`sysv_ret`).
    if (!resultTypeVec.empty()) {
        if (auto retStructType = dyn_cast<LLVM::LLVMStructType>(resultTypeVec[0])) {
            auto abi = classifyStruct(retStructType, ctx, target, facts.ret);
            if (abi.ignore) {
                s.retIgnored = true;
                s.origRetStructType = retStructType;
            } else if (abi.memoryClass) {
                s.needsSret = true;
                s.sretStructType = retStructType;
                s.sretAlign = memorySlotAlign(retStructType, target, abi);
                s.sretSlot = LLVM::AllocaOp::create(rewriter, loc, ptrType, retStructType,
                    one, s.sretAlign);
            } else {
                s.origRetStructType = retStructType;
                s.coercedRetType = coercedType(abi, ctx);
                s.retSlotType = coercionSlotType(abi, ctx);
                s.retSlotOffset = coercionOffset(abi);
            }
        }
    }

    // If sret, the return pointer is the first argument. On both SysV and
    // Win64 that pointer occupies an integer argument register.
    if (s.needsSret) {
        s.args.push_back(s.sretSlot);
        s.argTypes.push_back(ptrType);
    }

    // Free integer / SSE argument registers. A register-class struct that does
    // not fit in what remains is passed entirely in memory. Splitting it
    // anyway puts the leftover eightbytes on the stack as scalars, which is
    // not the convention: the callee looks for one memory operand. SysV has
    // 6 GPR + 8 XMM; Win64 has 4 + 4, and its aggregates are already a single
    // integer whose spill LLVM gets right, so the "does not fit → memory"
    // rule is SysV only.
    int gprLeft = target == AbiTarget::Win64 ? 4 : 6;
    int xmmLeft = target == AbiTarget::Win64 ? 4 : 8;
    if (s.needsSret && gprLeft > 0)
        --gprLeft;
    auto noteScalar = [&](Type t) {
        if (isa<LLVM::LLVMPointerType>(t) || t.isIntOrIndex()) {
            if (gprLeft > 0) --gprLeft;
        } else if (t.isIntOrFloat() && !t.isIntOrIndex()) {
            if (xmmLeft > 0) --xmmLeft;
        }
    };

    for (auto [k, arg] : llvm::enumerate(args)) {
        Type argType = arg.getType();

        if (auto structType = dyn_cast<LLVM::LLVMStructType>(argType)) {
            auto abi = classifyStruct(structType, ctx, target,
                k < facts.args.size() ? facts.args[k] : SysVFacts());
            // A record of padding only: C passes it nowhere, and passing a
            // register for it would shift every later argument.
            if (abi.ignore)
                continue;
            bool asRegs = !abi.memoryClass;
            if (asRegs && target == AbiTarget::SysV) {
                int needG = 0, needX = 0;
                for (Type rt : abi.regs) {
                    if (rt.isIntOrFloat() && !rt.isIntOrIndex())
                        ++needX;
                    else
                        ++needG;
                }
                if (needG > gprLeft || needX > xmmLeft)
                    asRegs = false; // whole struct on the stack; no registers
            }
            if (asRegs) {
                // Register-class by-value struct: pass one scalar per register
                // (through memory, matching clang's coercion). The 8-byte GEP
                // stride is the SysV eightbyte layout; Win64 yields a single
                // register, so the stride is never exercised there. A record
                // whose low eightbyte is padding passes its high one alone,
                // read from offset 8.
                Value slot = LLVM::AllocaOp::create(rewriter, loc, ptrType,
                    coercionSlotType(abi, ctx), one);
                LLVM::StoreOp::create(rewriter, loc, arg, slot);
                int32_t base = coercionOffset(abi);
                for (unsigned i = 0; i < abi.regs.size(); ++i) {
                    Value p = slot;
                    int32_t off = base + static_cast<int32_t>(8 * i);
                    if (off > 0) {
                        p = LLVM::GEPOp::create(rewriter, loc, ptrType,
                            rewriter.getI8Type(), slot, ArrayRef<LLVM::GEPArg>{off});
                    }
                    Value v = LLVM::LoadOp::create(rewriter, loc, abi.regs[i], p);
                    s.args.push_back(v);
                    s.argTypes.push_back(abi.regs[i]);
                    noteScalar(abi.regs[i]);
                }
                continue;
            }

            // Indirect by-value struct. Same IR shape on both targets — alloca,
            // store, pass the pointer — but the attribute differs, and it is
            // not cosmetic.
            //
            // SysV wants a CALLER-OWNED STACK COPY in the outgoing argument
            // area, not a reference to one. `llvm.byval` on a pointer parameter
            // is precisely that, and precisely what clang emits. Both of the
            // shapes this replaces were wrong:
            //   • packed structs were passed as a BARE pointer — by reference,
            //     so the callee read an address where it expected bytes;
            //   • non-packed ones were passed as a first-class LLVM struct
            //     value, which the backend splits per ELEMENT across registers
            //     and stack. That is LLVM's own aggregate convention, not
            //     SysV's, so every subsequent argument shifted too.
            // llama.cpp's `llama_model_load_from_file(path, llama_model_params)`
            // — 72 bytes, MEMORY class — segfaulted on the second (2026-08-05).
            //
            // Win64 does NOT use byval: the convention is a pointer to a
            // caller-allocated temporary, and clang's WinX86_64ABIInfo emits
            // getIndirect(..., /*ByVal=*/false) for exactly this case. The
            // alloca+store above ALREADY IS that temporary, so the pointer is
            // complete on its own; adding byval would ask the backend to make a
            // second copy into an outgoing area the convention does not have.
            unsigned align = memorySlotAlign(argType, target, abi);
            Value stackSlot = LLVM::AllocaOp::create(rewriter, loc, ptrType, argType,
                one, align);
            LLVM::StoreOp::create(rewriter, loc, arg, stackSlot);
            if (target == AbiTarget::SysV)
                s.byval.push_back({static_cast<unsigned>(s.args.size()), argType, align});
            s.args.push_back(stackSlot);
            s.argTypes.push_back(ptrType);
            continue;
        }

        s.args.push_back(arg);
        s.argTypes.push_back(arg.getType());
        noteScalar(arg.getType());
    }

    return s;
}

// The call's result types: none under sret or an ignored record return, the
// coerced scalars for a register-class record, the declared type otherwise.
static SmallVector<Type, 1> callResultTypesOf(const CallShape& s, ArrayRef<Type> resultTypeVec) {
    SmallVector<Type, 1> out;
    if (s.needsSret || s.retIgnored)
        return out;
    out.assign(resultTypeVec.begin(), resultTypeVec.end());
    if (s.coercedRetType)
        out[0] = s.coercedRetType;
    return out;
}

// The value the jlcs op's result becomes once the call has returned
// `callResults`: the sret slot read back, the record's zero for an ignored
// return, the coerced scalars re-read as the record through a slot, or the
// call's own result. `ffe_call` and `try_call`'s plain-call branch use this;
// try_call's invoke path stores the coerced value into its own slot first.
static Value materializeCallResult(ConversionPatternRewriter& rewriter, Location loc,
    const CallShape& s, ValueRange callResults, Value one)
{
    if (s.needsSret)
        return LLVM::LoadOp::create(rewriter, loc, s.sretStructType, s.sretSlot);
    if (s.retIgnored)
        return LLVM::ZeroOp::create(rewriter, loc, s.origRetStructType);
    if (!s.coercedRetType)
        return callResults.empty() ? Value() : callResults[0];
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Value slot = LLVM::AllocaOp::create(rewriter, loc, ptrType, s.retSlotType, one);
    Value at = slot;
    if (s.retSlotOffset)
        at = LLVM::GEPOp::create(rewriter, loc, ptrType, rewriter.getI8Type(), slot,
            ArrayRef<LLVM::GEPArg>{s.retSlotOffset});
    LLVM::StoreOp::create(rewriter, loc, callResults[0], at);
    return LLVM::LoadOp::create(rewriter, loc, s.origRetStructType, slot);
}

// `llvm.byval` has to travel on the call site AND on the declaration: the
// call site is what the backend lowers, and the declaration is what the
// verifier checks the call site against.
static ArrayAttr byValArgAttrs(ConversionPatternRewriter& rewriter, const CallShape& s) {
    if (s.byval.empty())
        return nullptr;
    SmallVector<Attribute> attrs(s.args.size(), rewriter.getDictionaryAttr({}));
    for (const CallShape::ByVal& b : s.byval) {
        attrs[b.index] = rewriter.getDictionaryAttr({
            rewriter.getNamedAttr(LLVM::LLVMDialect::getByValAttrName(),
                                  TypeAttr::get(b.pointee)),
            rewriter.getNamedAttr(LLVM::LLVMDialect::getAlignAttrName(),
                                  rewriter.getI64IntegerAttr(b.align))});
    }
    return rewriter.getArrayAttr(attrs);
}

// Rewrite the external declaration to the coerced signature and stamp byval
// onto its pointer parameters.
static void retargetCallee(ConversionPatternRewriter& rewriter, ModuleOp moduleOp,
    StringRef callee, const CallShape& s, ArrayRef<Type> callResultTypes)
{
    if (auto calleeFn = moduleOp.lookupSymbol<func::FuncOp>(callee)) {
        calleeFn.setFunctionType(
            FunctionType::get(rewriter.getContext(), s.argTypes, callResultTypes));
        for (const CallShape::ByVal& b : s.byval) {
            calleeFn.setArgAttr(b.index, LLVM::LLVMDialect::getByValAttrName(),
                                TypeAttr::get(b.pointee));
            calleeFn.setArgAttr(b.index, LLVM::LLVMDialect::getAlignAttrName(),
                                rewriter.getI64IntegerAttr(b.align));
        }
    } else if (auto llvmCalleeFn = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(callee)) {
        llvmCalleeFn.setFunctionType(LLVM::LLVMFunctionType::get(
            callResultTypes.empty() ? (Type)LLVM::LLVMVoidType::get(rewriter.getContext())
                                    : callResultTypes[0],
            s.argTypes, /*isVarArg=*/false));
        for (const CallShape::ByVal& b : s.byval) {
            llvmCalleeFn.setArgAttr(b.index, LLVM::LLVMDialect::getByValAttrName(),
                                    TypeAttr::get(b.pointee));
            llvmCalleeFn.setArgAttr(b.index, LLVM::LLVMDialect::getAlignAttrName(),
                                    rewriter.getI64IntegerAttr(b.align));
        }
    }
}

//===----------------------------------------------------------------------===//
// FFECallOp Lowering
//===----------------------------------------------------------------------===//

struct FFECallOpLowering : public ConversionPattern {
    FFECallOpLowering(LLVMTypeConverter& typeConverter, MLIRContext* ctx)
        : ConversionPattern(typeConverter, FFECallOp::getOperationName(), 1,
              ctx)
    {
    }

    LogicalResult
    matchAndRewrite(Operation* op, ArrayRef<Value> operands,
        ConversionPatternRewriter& rewriter) const override
    {
        auto ffeCallOp = cast<FFECallOp>(op);
        Location loc = ffeCallOp.getLoc();
        FFECallOp::Adaptor adaptor(operands);
        ValueRange args = adaptor.getArgs();

        // Get the callee attribute
        auto calleeAttr = op->getAttrOfType<FlatSymbolRefAttr>("callee");
        if (!calleeAttr) {
            return op->emitError("ffe_call requires a 'callee' symbol reference attribute");
        }

        // Determine result types
        SmallVector<Type, 1> resultTypeVec;
        for (Type t : ffeCallOp.getResults().getTypes()) {
            Type converted = typeConverter->convertType(t);
            if (!converted) {
                return op->emitError("FFECallOp: Could not convert result type");
            }
            resultTypeVec.push_back(converted);
        }

        // ABI coercion: match the convention the external C function was
        // compiled with — sret for MEMORY-class returns, one scalar per
        // argument register for register-class aggregates, and a caller-owned
        // stack copy for indirect by-value arguments. Target is the host
        // (SysV or Win64); see the classification block. Shared with try_call.
        // Under SysV each record's class is what the generator stated on the
        // op from the record's DWARF facts (`sysv_args`, `sysv_ret`).
        CallFacts facts = readCallFacts(op, args.size());
        if (!facts.error.empty())
            return op->emitError("ffe_call: ") << facts.error;
        Value one = arith::ConstantIntOp::create(rewriter, loc, 1, 64);
        CallShape shape = buildCallShape(rewriter, loc, resultTypeVec, args, facts);

        // The call returns void under sret or an ignored record return, the
        // coerced scalars for a register-class record, else the declared type.
        SmallVector<Type, 1> callResultTypes = callResultTypesOf(shape, resultTypeVec);

        retargetCallee(rewriter, op->getParentOfType<ModuleOp>(),
                       calleeAttr.getValue(), shape, callResultTypes);

        // Direct named call
        auto callOp = LLVM::CallOp::create(rewriter,
            loc, callResultTypes, calleeAttr.getValue(),
            ValueRange(shape.args));
        if (auto argAttrs = byValArgAttrs(rewriter, shape))
            callOp.setArgAttrsAttr(argAttrs);

        // Replace the op
        if (resultTypeVec.empty()) {
            rewriter.eraseOp(op);
        } else {
            rewriter.replaceOp(op, materializeCallResult(rewriter, loc, shape, callOp.getResults(), one));
        }

        return success();
    }
};

//===----------------------------------------------------------------------===//
// TryCallOp Lowering (Exception-safe FFE Call)
//===----------------------------------------------------------------------===//

struct TryCallOpLowering : public ConversionPattern {
    TryCallOpLowering(LLVMTypeConverter& typeConverter, MLIRContext* ctx)
        : ConversionPattern(typeConverter, TryCallOp::getOperationName(), 1,
              ctx)
    {
    }

    /// Get or declare an external function in the module
    LLVM::LLVMFuncOp getOrInsertFunction(ModuleOp module,
                                          ConversionPatternRewriter& rewriter,
                                          StringRef name,
                                          LLVM::LLVMFunctionType fnType) const {
        if (auto fn = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
            return fn;
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(module.getBody());
        return LLVM::LLVMFuncOp::create(rewriter, module.getLoc(), name, fnType);
    }

    LogicalResult
    matchAndRewrite(Operation* op, ArrayRef<Value> operands,
        ConversionPatternRewriter& rewriter) const override
    {
        auto tryCallOp = cast<TryCallOp>(op);
        Location loc = tryCallOp.getLoc();
        TryCallOp::Adaptor adaptor(operands);
        ValueRange args = adaptor.getArgs();

        auto calleeAttr = op->getAttrOfType<FlatSymbolRefAttr>("callee");
        if (!calleeAttr) {
            return op->emitError("try_call requires a 'callee' symbol reference attribute");
        }

        auto moduleOp = op->getParentOfType<ModuleOp>();
        auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
        auto voidType = LLVM::LLVMVoidType::get(rewriter.getContext());
        auto i8Type = rewriter.getI8Type();
        auto i32Type = rewriter.getI32Type();
        auto i64Type = rewriter.getI64Type();
        Value one = arith::ConstantIntOp::create(rewriter, loc, 1, 64);

        // --- ABI coercion (same logic as FFECallOpLowering) ---
        SmallVector<Type, 1> resultTypeVec;
        for (Type t : tryCallOp.getResults().getTypes()) {
            Type converted = typeConverter->convertType(t);
            if (!converted)
                return op->emitError("TryCallOp: Could not convert result type");
            resultTypeVec.push_back(converted);
        }

        CallFacts facts = readCallFacts(op, args.size());
        if (!facts.error.empty())
            return op->emitError("try_call: ") << facts.error;
        CallShape shape = buildCallShape(rewriter, loc, resultTypeVec, args, facts);
        bool needsSret = shape.needsSret;
        Type sretStructType = shape.sretStructType;
        Value sretSlot = shape.sretSlot;
        Type origRetStructType = shape.origRetStructType;
        Type coercedRetType = shape.coercedRetType;
        SmallVector<Value, 4>& coercedArgs = shape.args;

        SmallVector<Type, 1> callResultTypes = callResultTypesOf(shape, resultTypeVec);

        retargetCallee(rewriter, moduleOp, calleeAttr.getValue(), shape, callResultTypes);

#ifdef _WIN32
        // In-JIT landing pads need a Win64 EXCEPTION_ROUTINE at an RVA in the
        // JIT image. RuntimeDyld's ADDR32NB stub for __gxx_personality_seh0
        // is not a valid SEH handler: Windows jumps into .xdata and AVs in a
        // nested-exception loop. Catch in libJLCS (jlcs_guard_*) instead.
        // This call still gets .pdata unwind codes so the throw can walk out.
        auto callOp = LLVM::CallOp::create(rewriter, loc, callResultTypes,
            calleeAttr.getValue(), ValueRange(coercedArgs));
        if (auto argAttrs = byValArgAttrs(rewriter, shape))
            callOp.setArgAttrsAttr(argAttrs);
        if (resultTypeVec.empty()) {
            rewriter.eraseOp(op);
        } else {
            rewriter.replaceOp(op, materializeCallResult(rewriter, loc, shape, callOp.getResults(), one));
        }
        return success();
#endif

        // --- Ensure EH helper functions are declared ---

        // __gxx_personality_v0
        auto personalityFnType = LLVM::LLVMFunctionType::get(i32Type, {}, true);
        getOrInsertFunction(moduleOp, rewriter, kCxxPersonality, personalityFnType);

        // __cxa_begin_catch(void*) -> void*
        auto cxaBeginType = LLVM::LLVMFunctionType::get(ptrType, {ptrType}, false);
        getOrInsertFunction(moduleOp, rewriter, "__cxa_begin_catch", cxaBeginType);

        // __cxa_end_catch() -> void
        auto cxaEndType = LLVM::LLVMFunctionType::get(voidType, {}, false);
        getOrInsertFunction(moduleOp, rewriter, "__cxa_end_catch", cxaEndType);

        // jlcs_set_pending_exception(const char*) -> void
        auto setPendingType = LLVM::LLVMFunctionType::get(voidType, {ptrType}, false);
        getOrInsertFunction(moduleOp, rewriter, "jlcs_set_pending_exception", setPendingType);

        // jlcs_catch_current_exception() -> const char*
        auto catchCurrentType = LLVM::LLVMFunctionType::get(ptrType, {}, false);
        getOrInsertFunction(moduleOp, rewriter, "jlcs_catch_current_exception", catchCurrentType);

        // --- Set personality function on parent function ---
        // Must handle both func.func (pre-lowering) and llvm.func (post-lowering)
        if (auto llvmFunc = op->getParentOfType<LLVM::LLVMFuncOp>()) {
            llvmFunc.setPersonalityAttr(FlatSymbolRefAttr::get(rewriter.getContext(), kCxxPersonality));
            setUwtableAsync(llvmFunc);
        } else if (auto funcOp = op->getParentOfType<func::FuncOp>()) {
            // Set as a generic attribute that FuncToLLVM will carry through
            funcOp->setAttr("llvm.personality",
                FlatSymbolRefAttr::get(rewriter.getContext(), kCxxPersonality));
        }

        // --- Emit invoke + landing pad ---
        // Block structure:
        //   currentBlock: ... alloca resultSlot ... invoke → invokeOkBlock / catchBlock
        //   invokeOkBlock: store invoke result → resultSlot, br → mergeBlock
        //   catchBlock: landingpad, __cxa_begin_catch, jlcs_catch_current_exception,
        //               __cxa_end_catch, br → mergeBlock
        //   mergeBlock: load resultSlot → replacement value, [remaining ops from split]

        // Allocate result storage (initialized to zero for the exception path).
        // For a register-class record the slot is the coercion slot, sized
        // for the record, and the coerced scalars go at `retSlotOffset` (8
        // when the record's low eightbyte is padding and only the high one
        // comes back); the record is then read from the slot's start.
        Value resultSlot;
        Value resultStore;
        Type resultType;
        if (!callResultTypes.empty()) {
            resultType = callResultTypes[0];
            Type slotType = coercedRetType ? shape.retSlotType : resultType;
            resultSlot = LLVM::AllocaOp::create(rewriter, loc, ptrType, slotType, one);
            Value zero = LLVM::ZeroOp::create(rewriter, loc, slotType);
            LLVM::StoreOp::create(rewriter, loc, zero, resultSlot);
            resultStore = resultSlot;
            if (coercedRetType && shape.retSlotOffset)
                resultStore = LLVM::GEPOp::create(rewriter, loc, ptrType, i8Type, resultSlot,
                    ArrayRef<LLVM::GEPArg>{shape.retSlotOffset});
        }

        Block *currentBlock = rewriter.getInsertionBlock();
        auto *parentRegion = currentBlock->getParent();

        // Split: ops after try_call go to mergeBlock
        Block *mergeBlock = rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());

        // Create invokeOkBlock (single predecessor: invoke normal dest)
        Block *invokeOkBlock = new Block();
        parentRegion->getBlocks().insertAfter(currentBlock->getIterator(), invokeOkBlock);

        // Create catchBlock
        Block *catchBlock = new Block();
        parentRegion->getBlocks().insertAfter(invokeOkBlock->getIterator(), catchBlock);

        // --- currentBlock: emit invoke ---
        rewriter.setInsertionPointToEnd(currentBlock);

        auto invokeOp = LLVM::InvokeOp::create(rewriter, 
            loc,
            callResultTypes,
            calleeAttr,
            ValueRange(coercedArgs),
            invokeOkBlock,   // normal dest
            ValueRange{},
            catchBlock,       // unwind dest
            ValueRange{});
        if (auto argAttrs = byValArgAttrs(rewriter, shape))
            invokeOp.setArgAttrsAttr(argAttrs);

        // --- invokeOkBlock: store result, branch to mergeBlock ---
        rewriter.setInsertionPointToEnd(invokeOkBlock);
        if (!callResultTypes.empty()) {
            LLVM::StoreOp::create(rewriter, loc, invokeOp.getResult(), resultStore);
        }
        if (needsSret) {
            // sret result is already in sretSlot from the invoke ABI
        }
        LLVM::BrOp::create(rewriter, loc, ValueRange{}, mergeBlock);

        // --- catchBlock: landing pad + exception handling ---
        rewriter.setInsertionPointToEnd(catchBlock);

        auto lpStructType = LLVM::LLVMStructType::getLiteral(
            rewriter.getContext(), {ptrType, i32Type}, false);

        Value nullPtr = LLVM::ZeroOp::create(rewriter, loc, ptrType);
        auto landingPad = LLVM::LandingpadOp::create(rewriter, 
            loc, lpStructType, /*cleanup=*/false, ValueRange{nullPtr});

        Value exnPtr = LLVM::ExtractValueOp::create(rewriter, loc, ptrType, landingPad, ArrayRef<int64_t>{0});

        LLVM::CallOp::create(rewriter, loc, TypeRange{ptrType},
            "__cxa_begin_catch", ValueRange{exnPtr});
        LLVM::CallOp::create(rewriter, loc, TypeRange{ptrType},
            "jlcs_catch_current_exception", ValueRange{});
        LLVM::CallOp::create(rewriter, loc, TypeRange{},
            "__cxa_end_catch", ValueRange{});

        // Branch to mergeBlock (resultSlot still has zero sentinel)
        LLVM::BrOp::create(rewriter, loc, ValueRange{}, mergeBlock);

        // --- mergeBlock: load result, replace try_call op ---
        rewriter.setInsertionPointToStart(mergeBlock);

        if (needsSret) {
            Value result = LLVM::LoadOp::create(rewriter, loc, sretStructType, sretSlot);
            rewriter.replaceOp(op, result);
        } else if (shape.retIgnored) {
            // A record of padding: the callee returned nothing.
            rewriter.replaceOp(op, LLVM::ZeroOp::create(rewriter, loc, origRetStructType));
        } else if (!resultTypeVec.empty()) {
            // For a register-class struct return, resultSlot holds the coerced
            // eightbyte scalars; reload it as the original struct type.
            Type loadType = coercedRetType ? origRetStructType : resultType;
            Value result = LLVM::LoadOp::create(rewriter, loc, loadType, resultSlot);
            rewriter.replaceOp(op, result);
        } else {
            rewriter.eraseOp(op);
        }

        return success();
    }
};

//===----------------------------------------------------------------------===//
// ConstructorCallOp Lowering
//===----------------------------------------------------------------------===//

struct ConstructorCallOpLowering : public ConversionPattern {
    ConstructorCallOpLowering(LLVMTypeConverter& typeConverter, MLIRContext* ctx)
        : ConversionPattern(typeConverter,
              ConstructorCallOp::getOperationName(), 1, ctx)
    {
    }

    LogicalResult
    matchAndRewrite(Operation* op, ArrayRef<Value> operands,
        ConversionPatternRewriter& rewriter) const override
    {
        auto ctorOp = cast<ConstructorCallOp>(op);
        Location loc = ctorOp.getLoc();
        ConstructorCallOp::Adaptor adaptor(operands);

        auto calleeAttr = ctorOp.getCalleeAttr();
        ValueRange args = adaptor.getArgs();

        if (args.empty()) {
            return op->emitError("ctor_call requires at least the object pointer argument");
        }

        // Direct call to the constructor symbol — void return
        LLVM::CallOp::create(rewriter, 
            loc, TypeRange(), calleeAttr.getValue(), ValueRange(args));

        rewriter.eraseOp(op);
        return success();
    }
};

//===----------------------------------------------------------------------===//
// DestructorCallOp Lowering
//===----------------------------------------------------------------------===//

struct DestructorCallOpLowering : public ConversionPattern {
    DestructorCallOpLowering(LLVMTypeConverter& typeConverter, MLIRContext* ctx)
        : ConversionPattern(typeConverter,
              DestructorCallOp::getOperationName(), 1, ctx)
    {
    }

    LogicalResult
    matchAndRewrite(Operation* op, ArrayRef<Value> operands,
        ConversionPatternRewriter& rewriter) const override
    {
        auto dtorOp = cast<DestructorCallOp>(op);
        Location loc = dtorOp.getLoc();
        DestructorCallOp::Adaptor adaptor(operands);

        auto calleeAttr = dtorOp.getCalleeAttr();
        Value objPtr = adaptor.getObjPtr();

        if (!dtorOp.getMayThrow()
#ifdef _WIN32
            || true
#endif
        ) {
            // Direct call to the destructor symbol — void return, single arg.
            // Windows: no in-JIT landing pad; see TryCallOpLowering.
            LLVM::CallOp::create(rewriter,
                loc, TypeRange(), calleeAttr.getValue(), ValueRange({objPtr}));

            rewriter.eraseOp(op);
            return success();
        }

        // --- may_throw: direct invoke + landing pad ----------------------
        // Same sentinel-continue EH model as TryCallOpLowering and the vcall
        // EH path, and simpler than both: a destructor has no result, so there
        // is no sentinel slot to allocate and nothing to coerce. The producer
        // sets may_throw under the same rule that would have picked try_call,
        // so routing a destructor thunk through this op cannot silently drop
        // the landing pad it used to get.
        auto moduleOp = op->getParentOfType<ModuleOp>();
        auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
        auto voidType = LLVM::LLVMVoidType::get(rewriter.getContext());
        auto i32Type = rewriter.getI32Type();

        getOrInsertLLVMFn(moduleOp, rewriter, kCxxPersonality,
            LLVM::LLVMFunctionType::get(i32Type, {}, true));
        getOrInsertLLVMFn(moduleOp, rewriter, "__cxa_begin_catch",
            LLVM::LLVMFunctionType::get(ptrType, {ptrType}, false));
        getOrInsertLLVMFn(moduleOp, rewriter, "__cxa_end_catch",
            LLVM::LLVMFunctionType::get(voidType, {}, false));
        getOrInsertLLVMFn(moduleOp, rewriter, "jlcs_catch_current_exception",
            LLVM::LLVMFunctionType::get(ptrType, {}, false));

        if (auto llvmFunc = op->getParentOfType<LLVM::LLVMFuncOp>()) {
            llvmFunc.setPersonalityAttr(
                FlatSymbolRefAttr::get(rewriter.getContext(), kCxxPersonality));
            setUwtableAsync(llvmFunc);
        } else if (auto funcOp = op->getParentOfType<func::FuncOp>()) {
            funcOp->setAttr("llvm.personality",
                FlatSymbolRefAttr::get(rewriter.getContext(), kCxxPersonality));
        }

        Block* currentBlock = rewriter.getInsertionBlock();
        auto* parentRegion = currentBlock->getParent();
        Block* mergeBlock = rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());

        Block* invokeOkBlock = new Block();
        parentRegion->getBlocks().insertAfter(currentBlock->getIterator(), invokeOkBlock);
        Block* catchBlock = new Block();
        parentRegion->getBlocks().insertAfter(invokeOkBlock->getIterator(), catchBlock);

        rewriter.setInsertionPointToEnd(currentBlock);
        LLVM::InvokeOp::create(rewriter, loc,
            /*resultTypes=*/TypeRange{},
            calleeAttr,
            ValueRange({objPtr}),
            invokeOkBlock, /*normalDestOperands=*/ValueRange{},
            catchBlock, /*unwindDestOperands=*/ValueRange{});

        rewriter.setInsertionPointToEnd(invokeOkBlock);
        LLVM::BrOp::create(rewriter, loc, ValueRange{}, mergeBlock);

        rewriter.setInsertionPointToEnd(catchBlock);
        auto lpStructType = LLVM::LLVMStructType::getLiteral(
            rewriter.getContext(), {ptrType, i32Type}, false);
        Value nullPtr = LLVM::ZeroOp::create(rewriter, loc, ptrType);
        auto landingPad = LLVM::LandingpadOp::create(rewriter,
            loc, lpStructType, /*cleanup=*/false, ValueRange{nullPtr});
        Value exnPtr = LLVM::ExtractValueOp::create(rewriter, loc, ptrType,
            landingPad, ArrayRef<int64_t>{0});
        LLVM::CallOp::create(rewriter, loc, TypeRange{ptrType},
            "__cxa_begin_catch", ValueRange{exnPtr});
        LLVM::CallOp::create(rewriter, loc, TypeRange{ptrType},
            "jlcs_catch_current_exception", ValueRange{});
        LLVM::CallOp::create(rewriter, loc, TypeRange{},
            "__cxa_end_catch", ValueRange{});
        LLVM::BrOp::create(rewriter, loc, ValueRange{}, mergeBlock);

        rewriter.setInsertionPointToStart(mergeBlock);
        rewriter.eraseOp(op);
        return success();
    }
};

//===----------------------------------------------------------------------===//
// YieldOp Lowering
//===----------------------------------------------------------------------===//

struct YieldOpLowering : public ConversionPattern {
    YieldOpLowering(LLVMTypeConverter& typeConverter, MLIRContext* ctx)
        : ConversionPattern(typeConverter,
              YieldOp::getOperationName(), 1, ctx)
    {
    }

    LogicalResult
    matchAndRewrite(Operation* op, ArrayRef<Value> operands,
        ConversionPatternRewriter& rewriter) const override
    {
        // Yield is a scope terminator — erase during lowering
        rewriter.eraseOp(op);
        return success();
    }
};

//===----------------------------------------------------------------------===//
// ScopeOp Lowering
//===----------------------------------------------------------------------===//

struct ScopeOpLowering : public ConversionPattern {
    ScopeOpLowering(LLVMTypeConverter& typeConverter, MLIRContext* ctx)
        : ConversionPattern(typeConverter,
              ScopeOp::getOperationName(), 1, ctx)
    {
    }

    /// Check if any op in the body is a try_call (needs exception-safe RAII)
    bool bodyContainsTryCall(Region& bodyRegion) const {
        for (auto& op : bodyRegion.front()) {
            if (isa<TryCallOp>(&op))
                return true;
        }
        return false;
    }

    /// Emit destructor calls in reverse order for a given set of managed pointers
    void emitDestructors(Location loc, ConversionPatternRewriter& rewriter,
                         ValueRange managedPtrs, ArrayAttr destructors) const {
        for (int i = (int)destructors.size() - 1; i >= 0; --i) {
            auto dtorRef = cast<FlatSymbolRefAttr>(destructors[i]);
            Value ptr = managedPtrs[i];
            LLVM::CallOp::create(rewriter, 
                loc, TypeRange(), dtorRef.getValue(), ValueRange({ptr}));
        }
    }

    LogicalResult
    matchAndRewrite(Operation* op, ArrayRef<Value> operands,
        ConversionPatternRewriter& rewriter) const override
    {
        auto scopeOp = cast<ScopeOp>(op);
        Location loc = scopeOp.getLoc();
        ScopeOp::Adaptor adaptor(operands);

        // Get the body block
        Region& bodyRegion = scopeOp.getBody();
        Block& bodyBlock = bodyRegion.front();

        ValueRange managedPtrs = adaptor.getManagedPtrs();
        ArrayAttr destructors = scopeOp.getDestructors();

        // Erase the yield terminator before inlining
        Operation* terminator = bodyBlock.getTerminator();
        if (terminator)
            rewriter.eraseOp(terminator);

        // Inline body ops before the scope op in the parent block
        rewriter.inlineBlockBefore(&bodyBlock, op);

        // Normal path: destructor calls in reverse order (C++ destruction semantics)
        emitDestructors(loc, rewriter, managedPtrs, destructors);

        // Erase the scope op
        rewriter.eraseOp(op);
        return success();
    }
};

//===----------------------------------------------------------------------===//
// MarshalArgOp Lowering
//===----------------------------------------------------------------------===//

struct MarshalArgOpLowering : public ConversionPattern {
    MarshalArgOpLowering(LLVMTypeConverter& typeConverter, MLIRContext* ctx)
        : ConversionPattern(typeConverter, MarshalArgOp::getOperationName(), 1,
              ctx)
    {
    }

    LogicalResult
    matchAndRewrite(Operation* op, ArrayRef<Value> operands,
        ConversionPatternRewriter& rewriter) const override
    {
        auto marshalOp = cast<MarshalArgOp>(op);
        Location loc = marshalOp.getLoc();

        MarshalArgOp::Adaptor adaptor(operands);
        Value srcPtr = adaptor.getSrcPtr();

        // Get attributes
        ArrayAttr memberTypes = marshalOp.getMemberTypes();
        ArrayAttr juliaOffsets = marshalOp.getJuliaOffsets();

        // Result type is the packed struct
        Type packedType = typeConverter->convertType(marshalOp.getResult().getType());
        if (!packedType) {
            return op->emitError("MarshalArgOp: Could not convert result type");
        }

        auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

        // 1. Start with undef packed struct
        Value result = LLVM::UndefOp::create(rewriter, loc, packedType);

        // 2. For each member: GEP into Julia pointer, load, insert into packed struct
        for (size_t i = 0; i < memberTypes.size(); ++i) {
            // Get Julia byte offset for this member
            int64_t offset = cast<IntegerAttr>(juliaOffsets[i]).getInt();

            // Get member type from the memberTypes attribute
            Type memberType = cast<TypeAttr>(memberTypes[i]).getValue();
            // Convert through the type converter in case it's a dialect type
            Type convertedMemberType = typeConverter->convertType(memberType);
            if (!convertedMemberType)
                convertedMemberType = memberType;

            // GEP to field address at Julia byte offset (i8 element type)
            Value offsetVal = arith::ConstantIntOp::create(rewriter, loc, offset, 64);
            Value fieldPtr = LLVM::GEPOp::create(rewriter,
                loc, ptrType, rewriter.getI8Type(), srcPtr,
                ArrayRef<LLVM::GEPArg>({offsetVal}));

            // Load with alignment=1 (unaligned — Julia may pad differently than C)
            Value fieldVal = LLVM::LoadOp::create(rewriter, loc, convertedMemberType, fieldPtr,
                /*alignment=*/1);

            // Insert into packed struct at position [i]
            result = LLVM::InsertValueOp::create(rewriter, loc, packedType, result, fieldVal,
                ArrayRef<int64_t>{static_cast<int64_t>(i)});
        }

        rewriter.replaceOp(op, result);
        return success();
    }
};

//===----------------------------------------------------------------------===//
// MarshalRetOp Lowering
//===----------------------------------------------------------------------===//

struct MarshalRetOpLowering : public ConversionPattern {
    MarshalRetOpLowering(LLVMTypeConverter& typeConverter, MLIRContext* ctx)
        : ConversionPattern(typeConverter, MarshalRetOp::getOperationName(), 1,
              ctx)
    {
    }

    LogicalResult
    matchAndRewrite(Operation* op, ArrayRef<Value> operands,
        ConversionPatternRewriter& rewriter) const override
    {
        auto marshalOp = cast<MarshalRetOp>(op);
        Location loc = marshalOp.getLoc();

        MarshalRetOp::Adaptor adaptor(operands);
        Value packedValue = adaptor.getPackedValue();

        int64_t numMembers = marshalOp.getNumMembers();

        // Get the packed input type (already converted by type converter)
        Type packedType = packedValue.getType();
        auto packedStructType = dyn_cast<LLVM::LLVMStructType>(packedType);
        if (!packedStructType) {
            return op->emitError("MarshalRetOp: input must be an LLVM struct type");
        }

        // Get the aligned output type
        Type alignedType = typeConverter->convertType(marshalOp.getResult().getType());
        if (!alignedType) {
            return op->emitError("MarshalRetOp: Could not convert result type");
        }

        // 1. Start with undef aligned struct
        Value result = LLVM::UndefOp::create(rewriter, loc, alignedType);

        // 2. For each member: extract from packed, insert into aligned
        for (int64_t i = 0; i < numMembers; ++i) {
            // Extract from packed struct at position [i]
            Type memberType = packedStructType.getBody()[i];
            Value field = LLVM::ExtractValueOp::create(rewriter, loc, memberType,
                packedValue, ArrayRef<int64_t>{i});

            // Insert into aligned struct at same position [i]
            result = LLVM::InsertValueOp::create(rewriter, loc, alignedType, result, field,
                ArrayRef<int64_t>{i});
        }

        rewriter.replaceOp(op, result);
        return success();
    }
};

//===----------------------------------------------------------------------===//
// Lower JLCS to LLVM Pass
//===----------------------------------------------------------------------===//

struct LowerJLCSToLLVMPass
    : public PassWrapper<LowerJLCSToLLVMPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerJLCSToLLVMPass)

    StringRef getArgument() const override { return "jlcs-lower-to-llvm"; }
    StringRef getDescription() const override
    {
        return "Lowers the JLCS dialect to the LLVM dialect.";
    }

    void runOnOperation() override
    {
        LLVMTypeConverter typeConverter(&getContext());

        // Register CStructType conversion for packed struct support
        typeConverter.addConversion([&](CStructType type) -> Type {
            SmallVector<Type> llvmFields;
            for (Type fieldType : type.getFieldTypes()) {
                llvmFields.push_back(typeConverter.convertType(fieldType));
            }
            return LLVM::LLVMStructType::getLiteral(&getContext(), llvmFields, type.getIsPacked());
        });

        ConversionTarget target(getContext());

        // Define illegal ops (source dialect)
        target.addIllegalOp<GetFieldOp, SetFieldOp, VirtualCallOp,
            LoadArrayElementOp, StoreArrayElementOp, TypeInfoOp, FFECallOp,
            TryCallOp, ConstructorCallOp, DestructorCallOp,
            ScopeOp, YieldOp, MarshalArgOp, MarshalRetOp>();

        // Define legal dialects (target dialects)
        target.addLegalDialect<LLVM::LLVMDialect, arith::ArithDialect>();
        // func.func ops with CStructType signatures can't be handled by the
        // standalone ConvertFuncToLLVMPass (default type converter). Mark them
        // as dynamically illegal so this pass converts them with our custom
        // type converter that knows how to lower CStructType.
        target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
            return typeConverter.isSignatureLegal(op.getFunctionType());
        });

        // Add rewrite patterns
        RewritePatternSet patterns(&getContext());
        // Include func→LLVM patterns so our type converter handles func.func
        // ops whose signatures contain CStructType.
        populateFuncToLLVMConversionPatterns(typeConverter, patterns);
        patterns.add<GetFieldOpLowering>(typeConverter, &getContext());
        patterns.add<SetFieldOpLowering>(typeConverter, &getContext());
        patterns.add<VirtualCallOpLowering>(typeConverter, &getContext());
        patterns.add<LoadArrayElementOpLowering>(typeConverter, &getContext());
        patterns.add<StoreArrayElementOpLowering>(typeConverter, &getContext());
        patterns.add<TypeInfoOpLowering>(typeConverter, &getContext());
        patterns.add<FFECallOpLowering>(typeConverter, &getContext());
        patterns.add<TryCallOpLowering>(typeConverter, &getContext());
        patterns.add<ConstructorCallOpLowering>(typeConverter, &getContext());
        patterns.add<DestructorCallOpLowering>(typeConverter, &getContext());
        patterns.add<YieldOpLowering>(typeConverter, &getContext());
        patterns.add<ScopeOpLowering>(typeConverter, &getContext());
        patterns.add<MarshalArgOpLowering>(typeConverter, &getContext());
        patterns.add<MarshalRetOpLowering>(typeConverter, &getContext());

        // Execute the conversion
        if (failed(applyPartialConversion(getOperation(), target,
                std::move(patterns))))
            signalPassFailure();

        // Post-conversion fixup, two INDEPENDENT jobs. Runs after
        // func.func → llvm.func conversion, so both attributes can be set on
        // the LLVM function.
        //
        // 1. uwtable, on EVERY thunk with a body. It used to be set only inside
        //    the three may_throw lowerings, i.e. as a side effect of emitting a
        //    landing pad — but an unwind table is what lets a throw LEAVE this
        //    frame at all, which is a separate question from who catches it.
        //    Windows already proves they are separate: it lowers every
        //    may_throw to a plain call and catches in libJLCS's jlcs_guard_*,
        //    with .pdata the only reason the throw can reach that frame.
        //    Hoisting it also restores gdb `bt` through non-throwing thunks
        //    (backtraces need CFI; stepping does not — that is .debug_line).
        //
        //    NOT sufficient on its own to replace the landing pads here. On
        //    Linux, JITManager calls the thunk DIRECTLY — the jlcs_guard_*
        //    wrapper is Sys.iswindows()-gated (JITManager.jl) — so there is no
        //    C++ frame between Julia and the thunk to catch. Deleting the
        //    landing pads with only this in place unwinds past Julia frames
        //    into std::terminate (measured: SIGABRT, "terminate called after
        //    throwing an instance of 'std::runtime_error'"). Un-gating the
        //    guard call path is the prerequisite for that cleanup.
        //
        // 2. personality, only where a landing pad actually exists. An invoke
        //    is what requires one; attaching a personality the host runtime may
        //    not export is how the seh0/v0 split bites.
        getOperation().walk([&](LLVM::LLVMFuncOp funcOp) {
            if (!funcOp.isExternal())
                setUwtableAsync(funcOp);

            bool hasInvoke = false;
            funcOp.walk([&](LLVM::InvokeOp) { hasInvoke = true; });
            if (hasInvoke && !funcOp.getPersonalityAttr()) {
                auto personalityRef = FlatSymbolRefAttr::get(
                    &getContext(), kCxxPersonality);
                funcOp.setPersonalityAttr(personalityRef);
            }
        });
    }
};

//===----------------------------------------------------------------------===//
// SysV call-shape probe (test seam)
//===----------------------------------------------------------------------===//
//
// What buildCallShape makes of each jlcs.ffe_call and jlcs.try_call under the
// x86-64 SysV rules, on any host. test/test_sysv_abi_oracle.jl diffs it against
// LLVM's own ABI lowering library (JLCSAbiOracle.cpp). It calls the real
// buildCallShape with an explicit AbiTarget::SysV, so it shares every line of
// the decision with the lowering instead of restating it. The pass above is
// untouched and still takes kHostAbi.
//
// A conversion is the only way to get the ConversionPatternRewriter that
// buildCallShape takes. The IR it emits on the way (allocas, loads) is thrown
// away with the clone the probe runs on.

// One line per call: the callee, the return, then one field per argument.
//   return:   void | ignore | scalar:<type> | sret:<slot align> | regs:[none,]<t>[,<t>]
//   argument: scalar:<type> | ignore | byval:<align> | regs:[none,]<t>[,<t>]
// `regs` lists the scalar per eightbyte, i64 for INTEGER and f64 for SSE, after
// `none` when the low eightbyte is padding and only the high one is passed.
// `ignore` is a record of padding alone, passed nowhere and returned as void.
// `unmodelled:` replaces the fields when an operand or result is not an LLVM
// type yet: the real pass converts those first, and this probe does not.
// `error:` replaces them when the call's `sysv_args`/`sysv_ret` do not parse.
//
// A probed op is marked, not replaced, which is what makes it legal: the
// conversion then never needs a value of its result type.
static constexpr llvm::StringLiteral kSysVProbedAttr = "jlcs.sysv_probed";

struct SysVCallShapeProbe : public ConversionPattern {
    SysVCallShapeProbe(LLVMTypeConverter& typeConverter, MLIRContext* ctx,
        StringRef opName, std::string& out)
        : ConversionPattern(typeConverter, opName, 1, ctx), out(out)
    {
    }

    LogicalResult
    matchAndRewrite(Operation* op, ArrayRef<Value> operands,
        ConversionPatternRewriter& rewriter) const override
    {
        MLIRContext* ctx = rewriter.getContext();
        llvm::raw_string_ostream os(out);
        auto calleeAttr = op->getAttrOfType<FlatSymbolRefAttr>("callee");
        os << (calleeAttr ? calleeAttr.getValue() : StringRef("?"));

        // Both ops take exactly their call arguments as operands.
        SmallVector<Type, 1> resultTypeVec;
        for (Type t : op->getResultTypes())
            resultTypeVec.push_back(typeConverter->convertType(t));
        bool modelled = resultTypeVec.size() <= 1 &&
            llvm::all_of(resultTypeVec, [](Type t) { return t && LLVM::isCompatibleType(t); }) &&
            llvm::all_of(operands, [](Value v) { return LLVM::isCompatibleType(v.getType()); });

        CallFacts facts = readCallFacts(op, operands.size());
        if (!modelled) {
            os << "\tunmodelled:operand or result is not an LLVM type\n";
        } else if (!facts.error.empty()) {
            os << "\terror:" << facts.error << '\n';
        } else {
            CallShape s = buildCallShape(rewriter, op->getLoc(), resultTypeVec,
                ValueRange(operands), facts, AbiTarget::SysV);

            // `regs:none,<t>` is a record whose low eightbyte is padding: only
            // the high one is passed. `ignore` is a record of padding alone.
            auto regsOf = [&](const StructABI& abi, ArrayRef<Type> regs) {
                os << "regs:";
                if (abi.loNone)
                    os << "none,";
                llvm::interleave(regs, os, ",");
            };

            os << '\t';
            if (resultTypeVec.empty()) {
                os << "void";
            } else if (s.needsSret) {
                os << "sret:" << s.sretAlign;
            } else if (s.retIgnored) {
                os << "ignore";
            } else if (s.coercedRetType) {
                StructABI abi = classifyStruct(cast<LLVM::LLVMStructType>(resultTypeVec[0]), ctx,
                    s.target, facts.ret);
                regsOf(abi, abi.regs);
            } else {
                os << "scalar:" << resultTypeVec[0];
            }

            // Walk the flattened argument list the shape built, one original
            // operand at a time. A by-value struct is one byval pointer,
            // classifyStruct's registers, or nothing (ignored), which is what
            // buildCallShape pushed for it.
            unsigned j = s.needsSret ? 1 : 0;
            for (auto [k, arg] : llvm::enumerate(operands)) {
                os << '\t';
                auto st = dyn_cast<LLVM::LLVMStructType>(arg.getType());
                if (!st) {
                    os << "scalar:" << arg.getType();
                    ++j;
                    continue;
                }
                auto byval = llvm::find_if(s.byval,
                    [&](const CallShape::ByVal& b) { return b.index == j; });
                if (byval != s.byval.end()) {
                    os << "byval:" << byval->align;
                    ++j;
                    continue;
                }
                StructABI abi = classifyStruct(st, ctx, s.target, facts.args[k]);
                if (abi.ignore) {
                    os << "ignore";
                    continue;
                }
                if (abi.memoryClass || j + abi.regs.size() > s.argTypes.size()) {
                    os << "inconsistent";
                    j = s.argTypes.size() + 1;
                    break;
                }
                regsOf(abi, ArrayRef<Type>(s.argTypes).slice(j, abi.regs.size()));
                j += abi.regs.size();
            }
            if (j != s.argTypes.size())
                os << "\tinconsistent:" << j << "/" << s.argTypes.size();
            os << '\n';
        }

        rewriter.modifyOpInPlace(op, [&] {
            op->setAttr(kSysVProbedAttr, rewriter.getUnitAttr());
        });
        return success();
    }

    std::string& out;
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

namespace mlir {
namespace jlcs {

std::unique_ptr<Pass> createLowerJLCSToLLVMPass()
{
    return std::make_unique<LowerJLCSToLLVMPass>();
}

std::string describeSysVCallShapes(ModuleOp module)
{
    MLIRContext* ctx = module.getContext();
    OwningOpRef<ModuleOp> clone(cast<ModuleOp>(module->clone()));
    std::string out;

    LLVMTypeConverter typeConverter(ctx);
    ConversionTarget target(*ctx);
    target.addDynamicallyLegalOp<FFECallOp, TryCallOp>(
        [](Operation* op) { return op->hasAttr(kSysVProbedAttr); });
    target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });

    RewritePatternSet patterns(ctx);
    patterns.add<SysVCallShapeProbe>(typeConverter, ctx,
        FFECallOp::getOperationName(), out);
    patterns.add<SysVCallShapeProbe>(typeConverter, ctx,
        TryCallOp::getOperationName(), out);
    if (failed(applyPartialConversion(*clone, target, std::move(patterns))))
        out += "?\terror:the probe conversion failed\n";
    return out;
}

} // namespace jlcs
} // namespace mlir
