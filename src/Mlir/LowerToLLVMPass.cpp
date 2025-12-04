// Pseudocode: LowerToLLVMPass.cpp

#include "/home/grim/Desktop/Projects/RepliBuild.jl/src/Mlir/JLCSDialect.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::jlcs;

namespace {
// Define the pass that will host our rewriting patterns
struct LowerJLCSToLLVMPass : public PassWrapper<LowerJLCSToLLVMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerJLCSToLLVMPass)

  StringRef getArgument() const override { return "jlcs-lower-to-llvm"; }
  StringRef getDescription() const override { return "Lowers the JLCS dialect to the LLVM dialect."; }

  void runOnOperation() override {
    LLVMTypeConverter typeConverter(&getContext());
    ConversionTarget target(getContext());

    // 1. Define what is ILLEGAL (Source Dialect Ops)
    target.addIllegalOp<GetFieldOp, SetFieldOp, VirtualCallOp>();

    // 2. Define what is LEGAL (Target Dialect Ops)
    target.addLegalDialect<LLVM::LLVMDialect, arith::ArithmeticDialect>();

    // 3. Define the Rewrite Patterns
    RewritePatternSet patterns(&getContext());
    patterns.add<GetFieldOpLowering>(&getContext()); // Our custom pattern
    patterns.add<VirtualCallOpLowering>(&getContext()); // Virtual call lowering
    // patterns.add<SetFieldOpLowering>(&getContext()); // The set_field pattern

    // 4. Execute the Conversion
    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // end anonymous namespace

// Pass registration utility (usually in another file or header)
std::unique_ptr<Pass> createLowerJLCSToLLVMPass() {
  return std::make_unique<LowerJLCSToLLVMPass>();
}

// Pseudocode: The GetFieldOpLowering Pattern (within the LowerToLLVMPass.cpp file)

struct GetFieldOpLowering : public ConversionPattern {
  GetFieldOpLowering(MLIRContext *ctx)
      : ConversionPattern(GetFieldOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    auto getFieldOp = cast<GetFieldOp>(op);
    Location loc = getFieldOp.getLoc();

    // 1. Get the operands and target type
    Value structPtr = getFieldOp.getStructValue(); // The struct address
    int64_t offset = getFieldOp.getFieldOffset();   // The byte offset (e.g., 8)
    Type resultType = getFieldOp.getResult().getType();

    // 2. Create the byte offset constant
    // The offset is treated as a 64-bit integer index into a byte array (i8*).
    Value offsetVal = rewriter.create<arith::ConstantIntOp>(loc, offset, 64);

    // 3. Calculate the address (Pointer Arithmetic)
    // We treat 'structPtr' as a pointer to bytes (i8*) and GEP it by the offset.
    // This gives us the address of the field as an i8* pointer.
    Value fieldAddrI8 = rewriter.create<LLVM::GEPOp>(
        loc,
        LLVM::LLVMPointerType::get(IntegerType::get(rewriter.getContext(), 8)), // i8*
        structPtr,
        ArrayRef<Value>({offsetVal})
    );

    // 4. Bitcast the i8* address to the correct pointer type (e.g., i32*)
    Type targetLLVMPtrType = LLVM::LLVMPointerType::get(resultType);
    Value finalAddr = rewriter.create<LLVM::BitcastOp>(loc, targetLLVMPtrType, fieldAddrI8);

    // 5. Load the value from the final address
    Value loadedVal = rewriter.create<LLVM::LoadOp>(loc, finalAddr);

    // 6. Replace the original jlcs.get_field operation with the loaded value
    rewriter.replaceOp(op, loadedVal);

    return success();
  }
};

// Pseudocode: The VirtualCallOpLowering Pattern
// Lowers jlcs.vcall to LLVM IR: load vptr, index vtable, call function

struct VirtualCallOpLowering : public ConversionPattern {
  VirtualCallOpLowering(MLIRContext *ctx)
      : ConversionPattern(VirtualCallOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    auto vcallOp = cast<VirtualCallOp>(op);
    Location loc = vcallOp.getLoc();

    // Get attributes
    int64_t vtableOffset = vcallOp.getVtableOffset();
    int64_t slot = vcallOp.getSlot();
    ValueRange args = vcallOp.getArgs();

    if (args.empty()) {
      return op->emitError("VirtualCallOp requires at least object pointer argument");
    }

    Value objPtr = args[0];  // First arg is always the object pointer
    Type resultType = vcallOp.getResult().getType();

    // Create pointer type for vtable operations
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i64Type = rewriter.getI64Type();

    // Step 1: Load vtable pointer from object
    // vtable_ptr = *(objPtr + vtableOffset)
    Value vtableOffsetVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(vtableOffset));

    Value vtablePtrAddr = rewriter.create<LLVM::GEPOp>(
        loc, ptrType, rewriter.getI8Type(), objPtr,
        ArrayRef<LLVM::GEPArg>({vtableOffsetVal}));

    Value vtablePtr = rewriter.create<LLVM::LoadOp>(loc, ptrType, vtablePtrAddr);

    // Step 2: Index into vtable to get function pointer
    // func_ptr = vtable[slot]
    Value slotVal = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(slot));

    Value funcPtrAddr = rewriter.create<LLVM::GEPOp>(
        loc, ptrType, ptrType, vtablePtr,
        ArrayRef<LLVM::GEPArg>({slotVal}));

    Value funcPtr = rewriter.create<LLVM::LoadOp>(loc, ptrType, funcPtrAddr);

    // Step 3: Call the function pointer with arguments
    SmallVector<Value, 4> callArgs(args.begin(), args.end());

    Value result = rewriter.create<LLVM::CallOp>(
        loc, resultType, funcPtr, callArgs).getResult();

    // Step 4: Replace the jlcs.vcall with the call result
    rewriter.replaceOp(op, result);

    return success();
  }
};
