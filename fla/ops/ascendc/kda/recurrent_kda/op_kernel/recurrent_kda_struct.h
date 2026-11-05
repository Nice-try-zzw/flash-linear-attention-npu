/**
 * Copyright (c) 2025-2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file recurrent_kda_struct.h
 * \brief Plain tiling struct shared by aclnn tiling and fast kernel launch.
 */

#ifndef RECURRENT_KDA_STRUCT_H
#define RECURRENT_KDA_STRUCT_H

#include <cstdint>

namespace RecurrentKda {

// 静态 UB 路径按架构固定参数：A5(DAV_3510) 用 248KB 可用 UB、8 token 窗口、3 个 State bank；
// A6(ascend960dt, DAV_9201) 用 384KB、16 token 窗口、4 个 State bank。
constexpr uint32_t RKDA_A5_WINDOW_TOKENS = 8;
constexpr uint32_t RKDA_A5_BANK_COUNT = 3;
constexpr uint32_t RKDA_A6_WINDOW_TOKENS = 16;
constexpr uint32_t RKDA_A6_BANK_COUNT = 4;

// A6 静态 VF 配置，由 host tiling 一次算好；bank/槽的运行时归属仍只由 kernel 决定。
enum RecurrentKdaVfFlags : uint32_t {
    RKDA_VF_NORM = 1U << 0,
    RKDA_VF_RAW_GATE = 1U << 1,
    RKDA_VF_BIAS = 1U << 2,
    RKDA_VF_SIGMOID = 1U << 3,
    RKDA_VF_NEGATIVE = 1U << 4,
    RKDA_VF_SAFE = 1U << 5,
    RKDA_VF_WRITE_STATE = 1U << 6,
    RKDA_VF_INDEXED = 1U << 7,
    RKDA_VF_FUSED = 1U << 8,
};

#pragma pack(push, 8)
struct alignas(8) RecurrentKdaTilingData {
    uint32_t vectorCoreNum;
    uint32_t ubCalSize;
    uint32_t ubRestBytes;
    uint32_t t;
    uint32_t seqLen;
    uint32_t nk;
    uint32_t dk;
    uint32_t nv;
    uint32_t dv;
    uint32_t sBlockNum;
    uint32_t ssmStateStride;
    uint32_t b;
    uint32_t vStep;
    uint32_t stateOutBufferNum;
    uint32_t attnOutBufferNum;
    float scale;
    float lowerBound;
    uint32_t layout;
    uint32_t hasSsmStateIndices;
    uint32_t hasALog;
    uint32_t hasDtBias;
    uint32_t hasAcceptedTokens;
    uint32_t useQkL2norm;
    uint32_t useGateInKernel;
    uint32_t useBetaSigmoid;
    uint32_t allowNegEigval;
    uint32_t safeGate;
    uint32_t stateVFirst;
    uint32_t outputFinalState;
    uint32_t inplaceFinalState;
    uint32_t hasCuSeqlens;
    uint32_t gateDtype;
    uint32_t betaDtype;
    uint32_t cuSeqlensDtype;
    uint32_t ssmStateIndicesDtype;
    uint32_t acceptedTokensDtype;
    uint64_t stateInStride0;
    uint64_t stateInStride1;
    uint64_t stateInStride2;
    uint64_t stateInStride3;
    uint64_t stateOutStride0;
    uint64_t stateOutStride1;
    uint64_t stateOutStride2;
    uint64_t stateOutStride3;
    uint64_t queryTokenStride;
    uint64_t queryHeadStride;
    uint64_t keyTokenStride;
    uint64_t keyHeadStride;
    uint64_t valueTokenStride;
    uint64_t valueHeadStride;
    uint32_t windowSize; // A6 V=128 静态路径的 token 窗口；其余路径忽略。
    uint64_t queryBatchStride;
    uint64_t keyBatchStride;
    uint64_t valueBatchStride;
    uint32_t vfFlags;   // A6 V=128 静态 VF 的功能位；其余路径为 0。
    uint32_t headRatio; // HV/H，A6 静态路径据此把 value head 映射到 k/q head。
};
#pragma pack(pop)

} // namespace RecurrentKda

#endif // RECURRENT_KDA_STRUCT_H
