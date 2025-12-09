//===- JLCSPasses.cpp - JLCS dialect lowering passes --------------------===//
//
// Lowering passes for the JLCS dialect to LLVM dialect
//
//===----------------------------------------------------------------------===//

#include "JLCSDialect.h"
#include "JLCSLoweringUtils.h"
#include "JLCSOps.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::jlcs;

namespace {

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
        Value vtablePtr = getStructField(loc, rewriter, objPtr, vtableOffset, ptrType);

        // Step 2: Index into vtable to get function pointer
        // func_ptr = vtable[slot]
        Value slotVal = rewriter.create<arith::ConstantIntOp>(loc, slot, 64);
        Value funcPtrAddr = rewriter.create<LLVM::GEPOp>(
            loc, ptrType, ptrType, vtablePtr,
            ArrayRef<LLVM::GEPArg>({ slotVal }));

        Value funcPtr = rewriter.create<LLVM::LoadOp>(loc, ptrType, funcPtrAddr);

        // Step 3: Call the function pointer with arguments (indirect call)
        SmallVector<Value, 4> callArgs(args.begin(), args.end());

        // Determine result types
        SmallVector<Type, 1> resultTypeVec;
        if (vcallOp.getResult()) {
            resultTypeVec.push_back(vcallOp.getResult().getType());
        }

        // LLVM 21 API: Build CallOp manually with OperationState for indirect calls
        // First arg should be the function pointer
        SmallVector<Value> allOperands;
        allOperands.push_back(funcPtr);
        allOperands.append(callArgs.begin(), callArgs.end());

        OperationState state(loc, LLVM::CallOp::getOperationName());
        state.addOperands(allOperands);
        state.addTypes(resultTypeVec);

        // Add required attributes for indirect call
        state.addAttribute("callee", FlatSymbolRefAttr());  // empty for indirect

        Operation *callOp = rewriter.create(state);

        // Replace the jlcs.vcall with the call result
        if (!resultTypeVec.empty()) {
            rewriter.replaceOp(op, callOp->getResult(0));
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
        Value totalOffset = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);

        for (size_t i = 0; i < indices.size(); ++i) {
            Value index = indices[i];

            // GEP to the i-th stride in the strides array
            Value strideIndex = rewriter.create<arith::ConstantIntOp>(loc, i, 64);
            Value strideAddr = rewriter.create<LLVM::GEPOp>(
                loc, ptrType, i64Type, stridesPtr,
                ArrayRef<LLVM::GEPArg>({ strideIndex }));

            // Load the i-th stride value
            Value stride = rewriter.create<LLVM::LoadOp>(loc, i64Type, strideAddr);

            // Calculate: index * stride
            Value elementOffset = rewriter.create<arith::MulIOp>(loc, index, stride);

            // Accumulate: totalOffset += elementOffset
            totalOffset = rewriter.create<arith::AddIOp>(loc, totalOffset, elementOffset);
        }

        // 4. Calculate the final address (GEP on the base data pointer)
        Type elemType = loadOp.getResult().getType();
        Value finalAddr = rewriter.create<LLVM::GEPOp>(
            loc, ptrType, elemType, dataPtr,
            ArrayRef<LLVM::GEPArg>({ totalOffset }));

        // 5. Load the element
        Value result = rewriter.create<LLVM::LoadOp>(loc, elemType, finalAddr);

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
        Value totalOffset = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);

        for (size_t i = 0; i < indices.size(); ++i) {
            Value index = indices[i];
            Value strideIndex = rewriter.create<arith::ConstantIntOp>(loc, i, 64);
            Value strideAddr = rewriter.create<LLVM::GEPOp>(
                loc, ptrType, i64Type, stridesPtr,
                ArrayRef<LLVM::GEPArg>({ strideIndex }));

            Value stride = rewriter.create<LLVM::LoadOp>(loc, i64Type, strideAddr);
            Value elementOffset = rewriter.create<arith::MulIOp>(loc, index, stride);
            totalOffset = rewriter.create<arith::AddIOp>(loc, totalOffset, elementOffset);
        }

        // 4. Calculate the final address
        Value finalAddr = rewriter.create<LLVM::GEPOp>(
            loc, ptrType, value.getType(), dataPtr,
            ArrayRef<LLVM::GEPArg>({ totalOffset }));

        // 5. Store the value
        rewriter.create<LLVM::StoreOp>(loc, value, finalAddr);

        rewriter.eraseOp(op);
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
        ConversionTarget target(getContext());

        // Define illegal ops (source dialect)
        target.addIllegalOp<GetFieldOp, SetFieldOp, VirtualCallOp,
            LoadArrayElementOp, StoreArrayElementOp>();

        // Define legal dialects (target dialects)
        target.addLegalDialect<LLVM::LLVMDialect, arith::ArithDialect>();

        // Add rewrite patterns
        RewritePatternSet patterns(&getContext());
        patterns.add<GetFieldOpLowering>(typeConverter, &getContext());
        patterns.add<SetFieldOpLowering>(typeConverter, &getContext());
        patterns.add<VirtualCallOpLowering>(typeConverter, &getContext());
        patterns.add<LoadArrayElementOpLowering>(typeConverter, &getContext());
        patterns.add<StoreArrayElementOpLowering>(typeConverter, &getContext());

        // Execute the conversion
        if (failed(applyPartialConversion(getOperation(), target,
                std::move(patterns))))
            signalPassFailure();
    }
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

} // namespace jlcs
} // namespace mlir
