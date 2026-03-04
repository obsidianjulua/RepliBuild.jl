; ModuleID = '/home/john/Desktop/Projects/RepliBuild.jl/test/callback_test/src/callbacks.cpp'
source_filename = "/home/john/Desktop/Projects/RepliBuild.jl/test/callback_test/src/callbacks.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; Function Attrs: mustprogress noinline optnone sspstrong uwtable
define i32 @execute_binary_op(ptr noundef %0, i32 noundef %1, i32 noundef %2) #0 !dbg !11 {
  %4 = alloca i32, align 4
  %5 = alloca ptr, align 8
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  store ptr %0, ptr %5, align 8
    #dbg_declare(ptr %5, !22, !DIExpression(), !23)
  store i32 %1, ptr %6, align 4
    #dbg_declare(ptr %6, !24, !DIExpression(), !25)
  store i32 %2, ptr %7, align 4
    #dbg_declare(ptr %7, !26, !DIExpression(), !27)
  %8 = load ptr, ptr %5, align 8, !dbg !28
  %9 = icmp ne ptr %8, null, !dbg !28
  br i1 %9, label %11, label %10, !dbg !30

10:                                               ; preds = %3
  store i32 0, ptr %4, align 4, !dbg !31
  br label %16, !dbg !31

11:                                               ; preds = %3
  %12 = load ptr, ptr %5, align 8, !dbg !32
  %13 = load i32, ptr %6, align 4, !dbg !33
  %14 = load i32, ptr %7, align 4, !dbg !34
  %15 = call noundef i32 %12(i32 noundef %13, i32 noundef %14), !dbg !32
  store i32 %15, ptr %4, align 4, !dbg !35
  br label %16, !dbg !35

16:                                               ; preds = %11, %10
  %17 = load i32, ptr %4, align 4, !dbg !36
  ret i32 %17, !dbg !36
}

; Function Attrs: mustprogress noinline optnone sspstrong uwtable
define void @simulate_work(i32 noundef %0, ptr noundef %1) #0 !dbg !37 {
  %3 = alloca i32, align 4
  %4 = alloca ptr, align 8
  %5 = alloca i32, align 4
  %6 = alloca float, align 4
  store i32 %0, ptr %3, align 4
    #dbg_declare(ptr %3, !44, !DIExpression(), !45)
  store ptr %1, ptr %4, align 8
    #dbg_declare(ptr %4, !46, !DIExpression(), !47)
  %7 = load ptr, ptr %4, align 8, !dbg !48
  %8 = icmp ne ptr %7, null, !dbg !48
  br i1 %8, label %10, label %9, !dbg !50

9:                                                ; preds = %2
  br label %26, !dbg !51

10:                                               ; preds = %2
    #dbg_declare(ptr %5, !52, !DIExpression(), !54)
  store i32 1, ptr %5, align 4, !dbg !54
  br label %11, !dbg !55

11:                                               ; preds = %23, %10
  %12 = load i32, ptr %5, align 4, !dbg !56
  %13 = load i32, ptr %3, align 4, !dbg !58
  %14 = icmp sle i32 %12, %13, !dbg !59
  br i1 %14, label %15, label %26, !dbg !60

15:                                               ; preds = %11
    #dbg_declare(ptr %6, !61, !DIExpression(), !63)
  %16 = load i32, ptr %5, align 4, !dbg !64
  %17 = sitofp i32 %16 to float, !dbg !64
  %18 = load i32, ptr %3, align 4, !dbg !65
  %19 = sitofp i32 %18 to float, !dbg !65
  %20 = fdiv float %17, %19, !dbg !66
  store float %20, ptr %6, align 4, !dbg !63
  %21 = load ptr, ptr %4, align 8, !dbg !67
  %22 = load float, ptr %6, align 4, !dbg !68
  call void %21(float noundef %22), !dbg !67
  br label %23, !dbg !69

23:                                               ; preds = %15
  %24 = load i32, ptr %5, align 4, !dbg !70
  %25 = add nsw i32 %24, 1, !dbg !70
  store i32 %25, ptr %5, align 4, !dbg !70
  br label %11, !dbg !71, !llvm.loop !72

26:                                               ; preds = %9, %11
  ret void, !dbg !75
}

attributes #0 = { mustprogress noinline optnone sspstrong uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!4, !5, !6, !7, !8, !9}
!llvm.ident = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 21.1.8", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "/home/john/Desktop/Projects/RepliBuild.jl/test/callback_test/src/callbacks.cpp", directory: "/home/john/Desktop/Projects/RepliBuild.jl/test/callback_test", checksumkind: CSK_MD5, checksum: "b849aed653b1f139bce803ecef7f2d77")
!2 = !{!3}
!3 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!4 = !{i32 7, !"Dwarf Version", i32 5}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = !{i32 1, !"wchar_size", i32 4}
!7 = !{i32 8, !"PIC Level", i32 2}
!8 = !{i32 7, !"uwtable", i32 2}
!9 = !{i32 7, !"frame-pointer", i32 2}
!10 = !{!"clang version 21.1.8"}
!11 = distinct !DISubprogram(name: "execute_binary_op", scope: !12, file: !12, line: 3, type: !13, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !21)
!12 = !DIFile(filename: "src/callbacks.cpp", directory: "/home/john/Desktop/Projects/RepliBuild.jl/test/callback_test", checksumkind: CSK_MD5, checksum: "b849aed653b1f139bce803ecef7f2d77")
!13 = !DISubroutineType(types: !14)
!14 = !{!15, !16, !15, !15}
!15 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!16 = !DIDerivedType(tag: DW_TAG_typedef, name: "BinaryOp", file: !17, line: 9, baseType: !18)
!17 = !DIFile(filename: "include/callbacks.h", directory: "/home/john/Desktop/Projects/RepliBuild.jl/test/callback_test", checksumkind: CSK_MD5, checksum: "30a1c0161192f5654f5536c596fa3455")
!18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !19, size: 64)
!19 = !DISubroutineType(types: !20)
!20 = !{!15, !15, !15}
!21 = !{}
!22 = !DILocalVariable(name: "op", arg: 1, scope: !11, file: !12, line: 3, type: !16)
!23 = !DILocation(line: 3, column: 32, scope: !11)
!24 = !DILocalVariable(name: "a", arg: 2, scope: !11, file: !12, line: 3, type: !15)
!25 = !DILocation(line: 3, column: 40, scope: !11)
!26 = !DILocalVariable(name: "b", arg: 3, scope: !11, file: !12, line: 3, type: !15)
!27 = !DILocation(line: 3, column: 47, scope: !11)
!28 = !DILocation(line: 4, column: 10, scope: !29)
!29 = distinct !DILexicalBlock(scope: !11, file: !12, line: 4, column: 9)
!30 = !DILocation(line: 4, column: 9, scope: !29)
!31 = !DILocation(line: 4, column: 14, scope: !29)
!32 = !DILocation(line: 7, column: 12, scope: !11)
!33 = !DILocation(line: 7, column: 15, scope: !11)
!34 = !DILocation(line: 7, column: 18, scope: !11)
!35 = !DILocation(line: 7, column: 5, scope: !11)
!36 = !DILocation(line: 8, column: 1, scope: !11)
!37 = distinct !DISubprogram(name: "simulate_work", scope: !12, file: !12, line: 10, type: !38, scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !21)
!38 = !DISubroutineType(types: !39)
!39 = !{null, !15, !40}
!40 = !DIDerivedType(tag: DW_TAG_typedef, name: "ProgressCallback", file: !17, line: 10, baseType: !41)
!41 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !42, size: 64)
!42 = !DISubroutineType(types: !43)
!43 = !{null, !3}
!44 = !DILocalVariable(name: "iterations", arg: 1, scope: !37, file: !12, line: 10, type: !15)
!45 = !DILocation(line: 10, column: 24, scope: !37)
!46 = !DILocalVariable(name: "cb", arg: 2, scope: !37, file: !12, line: 10, type: !40)
!47 = !DILocation(line: 10, column: 53, scope: !37)
!48 = !DILocation(line: 11, column: 10, scope: !49)
!49 = distinct !DILexicalBlock(scope: !37, file: !12, line: 11, column: 9)
!50 = !DILocation(line: 11, column: 9, scope: !49)
!51 = !DILocation(line: 11, column: 14, scope: !49)
!52 = !DILocalVariable(name: "i", scope: !53, file: !12, line: 14, type: !15)
!53 = distinct !DILexicalBlock(scope: !37, file: !12, line: 14, column: 5)
!54 = !DILocation(line: 14, column: 14, scope: !53)
!55 = !DILocation(line: 14, column: 10, scope: !53)
!56 = !DILocation(line: 14, column: 21, scope: !57)
!57 = distinct !DILexicalBlock(scope: !53, file: !12, line: 14, column: 5)
!58 = !DILocation(line: 14, column: 26, scope: !57)
!59 = !DILocation(line: 14, column: 23, scope: !57)
!60 = !DILocation(line: 14, column: 5, scope: !53)
!61 = !DILocalVariable(name: "progress", scope: !62, file: !12, line: 15, type: !3)
!62 = distinct !DILexicalBlock(scope: !57, file: !12, line: 14, column: 43)
!63 = !DILocation(line: 15, column: 15, scope: !62)
!64 = !DILocation(line: 15, column: 33, scope: !62)
!65 = !DILocation(line: 15, column: 37, scope: !62)
!66 = !DILocation(line: 15, column: 35, scope: !62)
!67 = !DILocation(line: 16, column: 9, scope: !62)
!68 = !DILocation(line: 16, column: 12, scope: !62)
!69 = !DILocation(line: 17, column: 5, scope: !62)
!70 = !DILocation(line: 14, column: 38, scope: !57)
!71 = !DILocation(line: 14, column: 5, scope: !57)
!72 = distinct !{!72, !60, !73, !74}
!73 = !DILocation(line: 17, column: 5, scope: !53)
!74 = !{!"llvm.loop.mustprogress"}
!75 = !DILocation(line: 18, column: 1, scope: !37)
