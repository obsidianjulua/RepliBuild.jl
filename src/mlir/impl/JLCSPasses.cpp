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
        Value slotVal = arith::ConstantIntOp::create(rewriter, loc, slot, 64);
        Value funcPtrAddr = LLVM::GEPOp::create(rewriter, 
            loc, ptrType, ptrType, vtablePtr,
            ArrayRef<LLVM::GEPArg>({ slotVal }));

        Value funcPtr = LLVM::LoadOp::create(rewriter, loc, ptrType, funcPtrAddr);

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

        // ABI Coercion for packed structs:
        // Clang on x86_64 SysV ABI passes packed structs via byval pointer
        // and returns them via sret (hidden first pointer argument).
        // We must match this convention when calling the external C function.

        Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
        Value one = arith::ConstantIntOp::create(rewriter, loc, 1, 64);

        // Check if return type needs sret (packed struct, or large struct > 16 bytes)
        bool needsSret = false;
        Type sretStructType;
        Value sretSlot;
        if (!resultTypeVec.empty()) {
            if (auto retStructType = dyn_cast<LLVM::LLVMStructType>(resultTypeVec[0])) {
                if (retStructType.isPacked()) {
                    needsSret = true;
                    sretStructType = retStructType;
                    sretSlot = LLVM::AllocaOp::create(rewriter, loc, ptrType, sretStructType, one);
                } else {
                    // x86_64 SysV ABI: non-packed structs > 16 bytes use sret
                    uint64_t sizeBits = getPackedSizeInBits(retStructType);
                    if (sizeBits > 128) {
                        needsSret = true;
                        sretStructType = retStructType;
                        sretSlot = LLVM::AllocaOp::create(rewriter, loc, ptrType, sretStructType, one);
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
                    Value stackSlot = LLVM::AllocaOp::create(rewriter, loc, ptrType, argType, one);
                    LLVM::StoreOp::create(rewriter, loc, arg, stackSlot);
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
            } else if (auto llvmCalleeFn = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(calleeAttr.getValue())) {
                auto newFuncType = LLVM::LLVMFunctionType::get(
                    callResultTypes.empty() ? LLVM::LLVMVoidType::get(rewriter.getContext()) : callResultTypes[0],
                    coercedArgTypes,
                    false // isVarArg
                );
                llvmCalleeFn.setFunctionType(newFuncType);
            }
        }

        // Direct named call
        auto callOp = LLVM::CallOp::create(rewriter, 
            loc, callResultTypes, calleeAttr.getValue(),
            ValueRange(coercedArgs));

        // Replace the op
        if (needsSret) {
            // Load the result from the sret pointer
            Value result = LLVM::LoadOp::create(rewriter, loc, sretStructType, sretSlot);
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

        bool needsSret = false;
        Type sretStructType;
        Value sretSlot;
        if (!resultTypeVec.empty()) {
            if (auto retStructType = dyn_cast<LLVM::LLVMStructType>(resultTypeVec[0])) {
                if (retStructType.isPacked()) {
                    needsSret = true;
                    sretStructType = retStructType;
                    sretSlot = LLVM::AllocaOp::create(rewriter, loc, ptrType, sretStructType, one);
                } else {
                    uint64_t sizeBits = getPackedSizeInBits(retStructType);
                    if (sizeBits > 128) {
                        needsSret = true;
                        sretStructType = retStructType;
                        sretSlot = LLVM::AllocaOp::create(rewriter, loc, ptrType, sretStructType, one);
                    }
                }
            }
        }

        SmallVector<Value, 4> coercedArgs;
        SmallVector<Type, 4> coercedArgTypes;

        if (needsSret) {
            coercedArgs.push_back(sretSlot);
            coercedArgTypes.push_back(ptrType);
        }

        for (Value arg : args) {
            Type argType = arg.getType();
            if (auto structType = dyn_cast<LLVM::LLVMStructType>(argType)) {
                if (structType.isPacked()) {
                    Value stackSlot = LLVM::AllocaOp::create(rewriter, loc, ptrType, argType, one);
                    LLVM::StoreOp::create(rewriter, loc, arg, stackSlot);
                    coercedArgs.push_back(stackSlot);
                    coercedArgTypes.push_back(ptrType);
                    continue;
                }
            }
            coercedArgs.push_back(arg);
            coercedArgTypes.push_back(arg.getType());
        }

        SmallVector<Type, 1> callResultTypes;
        if (!needsSret) {
            callResultTypes = resultTypeVec;
        }

        // Update external function declaration signature
        {
            if (auto calleeFn = moduleOp.lookupSymbol<func::FuncOp>(calleeAttr.getValue())) {
                auto newFuncType = FunctionType::get(rewriter.getContext(), coercedArgTypes, callResultTypes);
                calleeFn.setFunctionType(newFuncType);
            } else if (auto llvmCalleeFn = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(calleeAttr.getValue())) {
                auto newFuncType = LLVM::LLVMFunctionType::get(
                    callResultTypes.empty() ? voidType : callResultTypes[0],
                    coercedArgTypes, false);
                llvmCalleeFn.setFunctionType(newFuncType);
            }
        }

        // --- Ensure EH helper functions are declared ---

        // __gxx_personality_v0
        auto personalityFnType = LLVM::LLVMFunctionType::get(i32Type, {}, true);
        getOrInsertFunction(moduleOp, rewriter, "__gxx_personality_v0", personalityFnType);

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
            llvmFunc.setPersonalityAttr(FlatSymbolRefAttr::get(rewriter.getContext(), "__gxx_personality_v0"));
        } else if (auto funcOp = op->getParentOfType<func::FuncOp>()) {
            // Set as a generic attribute that FuncToLLVM will carry through
            funcOp->setAttr("llvm.personality",
                FlatSymbolRefAttr::get(rewriter.getContext(), "__gxx_personality_v0"));
        }

        // --- Emit invoke + landing pad ---
        // Block structure:
        //   currentBlock: ... alloca resultSlot ... invoke → invokeOkBlock / catchBlock
        //   invokeOkBlock: store invoke result → resultSlot, br → mergeBlock
        //   catchBlock: landingpad, __cxa_begin_catch, jlcs_catch_current_exception,
        //               __cxa_end_catch, br → mergeBlock
        //   mergeBlock: load resultSlot → replacement value, [remaining ops from split]

        // Allocate result storage (initialized to zero for the exception path)
        Value resultSlot;
        Type resultType;
        if (!callResultTypes.empty()) {
            resultType = callResultTypes[0];
            resultSlot = LLVM::AllocaOp::create(rewriter, loc, ptrType, resultType, one);
            Value zero = LLVM::ZeroOp::create(rewriter, loc, resultType);
            LLVM::StoreOp::create(rewriter, loc, zero, resultSlot);
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

        // --- invokeOkBlock: store result, branch to mergeBlock ---
        rewriter.setInsertionPointToEnd(invokeOkBlock);
        if (!callResultTypes.empty()) {
            LLVM::StoreOp::create(rewriter, loc, invokeOp.getResult(), resultSlot);
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
        } else if (!resultTypeVec.empty()) {
            Value result = LLVM::LoadOp::create(rewriter, loc, resultType, resultSlot);
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

        // Direct call to the destructor symbol — void return, single arg
        LLVM::CallOp::create(rewriter, 
            loc, TypeRange(), calleeAttr.getValue(), ValueRange({objPtr}));

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
            ScopeOp, YieldOp>();

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

        // Execute the conversion
        if (failed(applyPartialConversion(getOperation(), target,
                std::move(patterns))))
            signalPassFailure();

        // Post-conversion fixup: set personality function on any llvm.func
        // that contains llvm.invoke ops (needed for landing pads).
        // This runs after func.func → llvm.func conversion, so personality
        // can be properly set on the LLVM function.
        getOperation().walk([&](LLVM::LLVMFuncOp funcOp) {
            bool hasInvoke = false;
            funcOp.walk([&](LLVM::InvokeOp) { hasInvoke = true; });
            if (hasInvoke && !funcOp.getPersonalityAttr()) {
                auto personalityRef = FlatSymbolRefAttr::get(
                    &getContext(), "__gxx_personality_v0");
                funcOp.setPersonalityAttr(personalityRef);
            }
        });
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
