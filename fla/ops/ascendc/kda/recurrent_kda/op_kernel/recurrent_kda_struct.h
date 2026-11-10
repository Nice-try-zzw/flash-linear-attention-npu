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

// A6（DAV_9201）静态 UB 引擎的布局常量。kernel 与 host tiling 共用这一份定义，
// 避免两侧各自维护同一组数字后漂移。
constexpr int64_t RKDA_A6_DIM = 128;
constexpr int64_t RKDA_A6_TOKEN_TILE = 32;
constexpr int64_t RKDA_A6_INPUT_BUFFERS = 2;
constexpr int64_t RKDA_A6_PARAMETER_BUFFERS = 2;
// FP32 State bank 为 64 KiB，沿用 3 份；BF16 bank 为 32 KiB，多留一份用于初态预取。
constexpr int64_t RKDA_A6_FP32_STATE_BANKS = 3;
constexpr int64_t RKDA_A6_BF16_STATE_BANKS = 4;

// A6 静态 UB 用量：State bank + 输入/输出双槽 + dt_bias 双槽 + VF scratch。
constexpr int64_t RkdaA6UbBytes(int64_t stateBanks, int64_t tokenTile, int64_t stateBytes)
{
    const int64_t bankBytes = RKDA_A6_DIM * RKDA_A6_DIM * stateBytes;
    const int64_t slotBytes = 4 * tokenTile * RKDA_A6_DIM * 2 + // q/k/v/out bf16
                              tokenTile * RKDA_A6_DIM * 4 +     // gate fp32
                              tokenTile * 32;                   // beta 与标量
    return stateBanks * bankBytes + RKDA_A6_INPUT_BUFFERS * slotBytes +
           RKDA_A6_PARAMETER_BUFFERS * RKDA_A6_DIM * 4 +
           RKDA_A6_DIM * 4 +  // prediction scratch
           RKDA_A6_DIM * 4;   // V-first out 的 FP32 staging
}

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
    // BSND 的 batch 跨度：A6 静态 UB 引擎按 (token/seqLen)*batchStride 定位 batch，
    // 因此 batch 维可以与 token 维之间存在间隔；A2/A3 与 A5 不读取该字段。
    uint64_t queryBatchStride;
    uint64_t keyBatchStride;
    uint64_t valueBatchStride;
};
#pragma pack(pop)

} // namespace RecurrentKda

#endif // RECURRENT_KDA_STRUCT_H
