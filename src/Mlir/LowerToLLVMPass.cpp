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
    target.addIllegalOp<GetFieldOp, SetFieldOp>();

    // 2. Define what is LEGAL (Target Dialect Ops)
    target.addLegalDialect<LLVM::LLVMDialect, arith::ArithmeticDialect>();

    // 3. Define the Rewrite Patterns
    RewritePatternSet patterns(&getContext());
    patterns.add<GetFieldOpLowering>(&getContext()); // Our custom pattern
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
