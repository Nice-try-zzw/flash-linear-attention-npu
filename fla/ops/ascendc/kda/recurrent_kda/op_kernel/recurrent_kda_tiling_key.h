// Copyright (c) 2026 Tianjin University, Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#pragma once

#include "ascendc/host_api/tiling/template_argument.h"

// A6（DAV_9201）静态 UB 引擎的编译期实例。A6 的 V=128 与 V=256 都由 arch92 引擎
// 处理（V=256 在引擎内按 DIM 切片），因此只生成 V/K 两种 State 布局与常用融合档
// 共四档；COMPAT 留给 A2/A3 与 A5 的公共 kernel。
#define RKDA_TPL_COMPAT 0
#define RKDA_TPL_V_FIRST 1
#define RKDA_TPL_K_FIRST 2
#define RKDA_TPL_V_FIRST_FUSED 3
#define RKDA_TPL_K_FIRST_FUSED 4

ASCENDC_TPL_ARGS_DECL(RecurrentKda,
    ASCENDC_TPL_UINT_DECL(KERNEL_MODE, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST,
        RKDA_TPL_COMPAT, RKDA_TPL_V_FIRST, RKDA_TPL_K_FIRST, RKDA_TPL_V_FIRST_FUSED, RKDA_TPL_K_FIRST_FUSED));

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ != 9201)
// 非 A6 的 device 编译只需要 COMPAT：tiling 在这两类平台上恒定下发 COMPAT，
// 其余四档不会生成，避免无谓的模板实例化。host 侧不定义 __NPU_ARCH__，仍按
// 全量枚举参与 GET_TPL_TILING_KEY 的编码。
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
        ASCENDC_TPL_UINT_SEL(KERNEL_MODE, ASCENDC_TPL_UI_LIST, RKDA_TPL_COMPAT)));
#else
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
        ASCENDC_TPL_UINT_SEL(KERNEL_MODE, ASCENDC_TPL_UI_LIST,
            RKDA_TPL_V_FIRST, RKDA_TPL_K_FIRST, RKDA_TPL_V_FIRST_FUSED, RKDA_TPL_K_FIRST_FUSED)));
#endif
