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
        if (type.isInteger(1) || type.isInteger(8) || type.isInteger(16) || type.isInteger(32) || type.isInteger(64)) {
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

        // ABI Coercion: Convert small packed structs to integers
        SmallVector<Value, 4> coercedArgs;
        
        for (Value arg : args) {
            Type argType = arg.getType();
            bool coerced = false;
            
            // Check if it's a packed struct that is small enough to be coerced (<= 64 bits)
            // This mimics x86_64 SysV ABI for small structs passed in registers
            if (auto structType = dyn_cast<LLVM::LLVMStructType>(argType)) {
                if (structType.isPacked()) {
                    uint64_t bits = getPackedSizeInBits(structType);
                    if (bits > 0 && bits <= 64) {
                        // Coerce to integer!
                        // 1. Alloca stack slot
                        Value one = rewriter.create<arith::ConstantIntOp>(loc, 1, 64);
                        Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
                        Value stackSlot = rewriter.create<LLVM::AllocaOp>(loc, ptrType, argType, one);
                        
                        // 2. Store struct to stack
                        rewriter.create<LLVM::StoreOp>(loc, arg, stackSlot);
                        
                        // 3. Load as integer
                        // We use the same pointer (opaque), but load with integer type
                        Type intType = rewriter.getIntegerType(bits);
                        Value intVal = rewriter.create<LLVM::LoadOp>(loc, intType, stackSlot);
                        
                        coercedArgs.push_back(intVal);
                        coerced = true;
                    }
                }
            }
            
            if (!coerced) {
                coercedArgs.push_back(arg);
            }
        }

        // Create LLVM::CallOp with coerced arguments
        // We use an indirect call strategy to avoid signature mismatch with the module symbol
        // 1. Get pointer to function
        Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
        Value funcPtr = rewriter.create<LLVM::AddressOfOp>(loc, ptrType, calleeAttr.getValue());
        
        // 2. Create CallOp (indirect)
        SmallVector<Value> callOperands;
        callOperands.push_back(funcPtr); // Callee
        callOperands.append(coercedArgs.begin(), coercedArgs.end());
        
        OperationState state(loc, LLVM::CallOp::getOperationName());
        state.addOperands(callOperands);
        state.addTypes(resultTypeVec);
        
        // Add attributes for segment sizes
        int32_t nArgs = (int32_t)coercedArgs.size();
        NamedAttribute opSeg = rewriter.getNamedAttr("operandSegmentSizes", 
            rewriter.getDenseI32ArrayAttr({1, nArgs, 0}));
        NamedAttribute opBundle = rewriter.getNamedAttr("op_bundle_sizes", 
            rewriter.getDenseI32ArrayAttr({}));
        
        SmallVector<NamedAttribute, 2> attrs;
        attrs.push_back(opSeg);
        attrs.push_back(opBundle);

        // Create CallOp using generic builder
        Operation *callOp = rewriter.create<LLVM::CallOp>(
            loc, resultTypeVec, ValueRange(callOperands), attrs);

        // Replace the op
        if (!resultTypeVec.empty()) {
            rewriter.replaceOp(op, callOp->getResults());
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

        // Add rewrite patterns
        RewritePatternSet patterns(&getContext());
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
