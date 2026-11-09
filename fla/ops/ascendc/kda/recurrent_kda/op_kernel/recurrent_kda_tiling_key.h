// Copyright (c) 2026 Tianjin University, Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#pragma once

#include "ascendc/host_api/tiling/template_argument.h"

// A6（DAV_9201）静态 UB 引擎的编译期实例。COMPAT 走兼容实现（A2/A3 长窗口、
// A5 V=256），其余四档走 arch92 的逐 token VF 引擎；V/K 两种 State 布局各自
// 再分通用与常用融合两种，避免枚举全部布尔组合造成编译膨胀。
#define RKDA_TPL_COMPAT 0
#define RKDA_TPL_V_FIRST 1
#define RKDA_TPL_K_FIRST 2
#define RKDA_TPL_V_FIRST_FUSED 3
#define RKDA_TPL_K_FIRST_FUSED 4

ASCENDC_TPL_ARGS_DECL(RecurrentKda,
    ASCENDC_TPL_UINT_DECL(KERNEL_MODE, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST,
        RKDA_TPL_COMPAT, RKDA_TPL_V_FIRST, RKDA_TPL_K_FIRST, RKDA_TPL_V_FIRST_FUSED, RKDA_TPL_K_FIRST_FUSED));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
        ASCENDC_TPL_UINT_SEL(KERNEL_MODE, ASCENDC_TPL_UI_LIST,
            RKDA_TPL_COMPAT, RKDA_TPL_V_FIRST, RKDA_TPL_K_FIRST, RKDA_TPL_V_FIRST_FUSED, RKDA_TPL_K_FIRST_FUSED)));
