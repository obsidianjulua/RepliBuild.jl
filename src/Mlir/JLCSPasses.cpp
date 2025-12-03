// src/Mlir/JLCSPasses.cpp (Conceptual Pass Registration File)

#include "JLCSDialect/JLCSDialect.h"
#include "JLCSDialect/JLCSOps.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::jlcs;

namespace {
// Forward declaration of the rewriting pattern
class GetFieldOpLowering;

// Define the Pass itself, which runs on the entire ModuleOp
struct LowerJLCSToLLVMPass : public PassWrapper<LowerJLCSToLLVMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerJLCSToLLVMPass)

  StringRef getArgument() const override { return "jlcs-lower-to-llvm"; }
  StringRef getDescription() const override { return "Lowers the JLCS dialect to the LLVM dialect."; }

  void runOnOperation() override {
    // 1. Setup for Conversion
    LLVMTypeConverter typeConverter(&getContext());
    ConversionTarget target(getContext());

    // 2. Mark our high-level JLCS operations as ILLEGAL
    target.addIllegalOp<GetFieldOp, SetFieldOp>();
    
    // 3. Mark the LLVM dialect and base dialects as LEGAL
    target.addLegalDialect<LLVM::LLVMDialect, arith::ArithmeticDialect>();

    // 4. Define and add the Rewriting Patterns
    RewritePatternSet patterns(&getContext());
    patterns.add<GetFieldOpLowering>(typeConverter, &getContext());

    // 5. Execute the Conversion
    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};
} // end anonymous namespace

// Implementation of the generic rewriting pattern (from previous discussion)
class GetFieldOpLowering : public mlir::LLVMOpLowering<GetFieldOp> {
public:
  using mlir::LLVMOpLowering<GetFieldOp>::LLVMOpLowering;

  LogicalResult matchAndRewrite(GetFieldOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // 1. Get the operands and target type
    Value structPtr = adaptor.getStructValue();
    int64_t offset = op.getFieldOffset();
    Type resultType = op.getResult().getType();

    // 2. Create the byte offset constant
    Value offsetVal = rewriter.create<arith::ConstantIntOp>(loc, offset, 64);
    
    // 3. Calculate the address (Gep using i8* as the pointer type)
    Type i8PtrType = LLVM::LLVMPointerType::get(IntegerType::get(getContext(), 8));
    Value fieldAddrI8 = rewriter.create<LLVM::GEPOp>(loc, i8PtrType, structPtr, 
                                                     ArrayRef<Value>({offsetVal}));
    
    // 4. Bitcast the i8* address to the correct pointer type (e.g., i32*)
    Type targetLLVMPtrType = LLVM::LLVMPointerType::get(resultType);
    Value finalAddr = rewriter.create<LLVM::BitcastOp>(loc, targetLLVMPtrType, fieldAddrI8);

    // 5. Load the value from the final address
    Value loadedVal = rewriter.create<LLVM::LoadOp>(loc, finalAddr);
    
    // 6. Replace the original jlcs.get_field operation
    rewriter.replaceOp(op, loadedVal);
    return success();
  }
};

// Function to create the pass instance (exported for the Julia side to use)
std::unique_ptr<Pass> createLowerJLCSToLLVMPass() {
  return std::make_unique<LowerJLCSToLLVMPass>();
}
