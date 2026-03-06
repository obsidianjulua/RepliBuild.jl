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
            Type converted = typeConverter->convertType(vcallOp.getResult().getType());
            if (!converted) {
                return op->emitError("VirtualCallOp: Could not convert result type");
            }
            resultTypeVec.push_back(converted);
        }

        // LLVM 21 API: Build CallOp manually with OperationState for indirect calls
        // First arg should be the function pointer
        SmallVector<Value> allOperands;
        allOperands.push_back(funcPtr);
        allOperands.append(callArgs.begin(), callArgs.end());

        OperationState state(loc, LLVM::CallOp::getOperationName());
        state.addOperands(allOperands);
        state.addTypes(resultTypeVec);
        
        // Add attributes for segment sizes (indirect call: callee_operands=1, args=N, op_bundle=0)
        int32_t nArgs = (int32_t)callArgs.size();
        state.addAttribute("operandSegmentSizes", 
            rewriter.getDenseI32ArrayAttr({1, nArgs, 0}));

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
// FFECallOp Lowering
//===----------------------------------------------------------------------===//

struct FFECallOpLowering : public ConversionPattern {
    FFECallOpLowering(LLVMTypeConverter& typeConverter, MLIRContext* ctx)
        : ConversionPattern(typeConverter, FFECallOp::getOperationName(), 1,
              ctx)
    {
    }

    // Helper to calculate size of a struct type (simple sum for packed)
    uint64_t getPackedSizeInBits(Type type) const {
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

        // ABI Coercion for packed structs:
        // Clang on x86_64 SysV ABI passes packed structs via byval pointer
        // and returns them via sret (hidden first pointer argument).
        // We must match this convention when calling the external C function.

        Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
        Value one = rewriter.create<arith::ConstantIntOp>(loc, 1, 64);

        // Check if return type needs sret (packed struct, or large struct > 16 bytes)
        bool needsSret = false;
        Type sretStructType;
        Value sretSlot;
        if (!resultTypeVec.empty()) {
            if (auto retStructType = dyn_cast<LLVM::LLVMStructType>(resultTypeVec[0])) {
                if (retStructType.isPacked()) {
                    needsSret = true;
                    sretStructType = retStructType;
                    sretSlot = rewriter.create<LLVM::AllocaOp>(loc, ptrType, sretStructType, one);
                } else {
                    // x86_64 SysV ABI: non-packed structs > 16 bytes use sret
                    uint64_t sizeBits = getPackedSizeInBits(retStructType);
                    if (sizeBits > 128) {
                        needsSret = true;
                        sretStructType = retStructType;
                        sretSlot = rewriter.create<LLVM::AllocaOp>(loc, ptrType, sretStructType, one);
                    }
                }
            }
        }

        // Build coerced argument list
        SmallVector<Value, 4> coercedArgs;
        SmallVector<Type, 4> coercedArgTypes;

        // If sret, the return pointer is the first argument
        if (needsSret) {
            coercedArgs.push_back(sretSlot);
            coercedArgTypes.push_back(ptrType);
        }

        for (Value arg : args) {
            Type argType = arg.getType();

            // Packed struct args: pass by pointer (byval semantics)
            if (auto structType = dyn_cast<LLVM::LLVMStructType>(argType)) {
                if (structType.isPacked()) {
                    // Alloca + store, then pass pointer
                    Value stackSlot = rewriter.create<LLVM::AllocaOp>(loc, ptrType, argType, one);
                    rewriter.create<LLVM::StoreOp>(loc, arg, stackSlot);
                    coercedArgs.push_back(stackSlot);
                    coercedArgTypes.push_back(ptrType);
                    continue;
                }
            }

            coercedArgs.push_back(arg);
            coercedArgTypes.push_back(arg.getType());
        }

        // Determine call result types (void if using sret)
        SmallVector<Type, 1> callResultTypes;
        if (!needsSret) {
            callResultTypes = resultTypeVec;
        }
        // If sret, the call returns void; result is loaded from sret pointer

        // Update the external function declaration's signature to match coerced types.
        {
            auto moduleOp = op->getParentOfType<ModuleOp>();
            if (auto calleeFn = moduleOp.lookupSymbol<func::FuncOp>(calleeAttr.getValue())) {
                auto newFuncType = FunctionType::get(rewriter.getContext(), coercedArgTypes, callResultTypes);
                calleeFn.setFunctionType(newFuncType);
            }
        }

        // Direct named call
        auto callOp = rewriter.create<LLVM::CallOp>(
            loc, callResultTypes, calleeAttr.getValue(),
            ValueRange(coercedArgs));

        // Replace the op
        if (needsSret) {
            // Load the result from the sret pointer
            Value result = rewriter.create<LLVM::LoadOp>(loc, sretStructType, sretSlot);
            rewriter.replaceOp(op, result);
        } else if (!resultTypeVec.empty()) {
            rewriter.replaceOp(op, callOp.getResults());
        } else {
            rewriter.eraseOp(op);
        }

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
            LoadArrayElementOp, StoreArrayElementOp, TypeInfoOp, FFECallOp>();

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
