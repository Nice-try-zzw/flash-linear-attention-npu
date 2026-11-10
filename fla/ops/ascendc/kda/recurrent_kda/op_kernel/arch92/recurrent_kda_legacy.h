/**
 * Copyright (c) 2025-2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file recurrent_kda_legacy.h
 * \brief A6 的 RegBase 兼容路径（`V!=128` 或未命中静态 UB 引擎的组合）：
 *        所有逐元素/归约计算改为 RegBase，不保留 membase 向量实现。
 *        输入缓冲按 MAX_MTP 个 token 的窗口分配，段长超过窗口时按窗口顺序递推，
 *        因此 varlen/dense 的每段长度都不受 MAX_MTP 限制。
 */

#ifndef __RECURRENT_KDA_ARCH92_LEGACY_H_
#define __RECURRENT_KDA_ARCH92_LEGACY_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "../recurrent_kda_tiling_data.h"
#include "recurrent_kda_regbase.h"

namespace RecurrentKda {

using namespace matmul;
using namespace AscendC;
using namespace AscendC::MicroAPI;
// 输入缓冲使用双槽，让 MTE2 搬下一个窗口与 VF 消费当前窗口在窗口间流水重叠。
// host CalcFixedUbBytes 对 A6 已按 rawBufferNum=2 核算预算；本路径（非 V=128 静态引擎）
// 走 legacy 5 步流程，vStep 由 FinalizeVStepFromUb 从剩余 UB 里切出。
constexpr uint64_t BUFFER_NUM = 2;
constexpr uint64_t INPUT_BUFFER_NUM = 2;
constexpr uint32_t MAX_OUT_BUFFER_NUM = 2;
// 单次搬入的 token 数上限。段长超过该值时按窗口顺序递推，不是段长限制。
constexpr uint64_t MAX_MTP = 8;
constexpr uint64_t BF16_NUM_PER_BLOCK = 16;
constexpr uint64_t FP32_NUM_PER_BLOCK = 8;
constexpr uint16_t V_LENGTH = VECTOR_REG_WIDTH / sizeof(float);
constexpr uint16_t TWO_V_LENGTH = 2 * V_LENGTH;
constexpr uint64_t INVALID_STATE_SLOT = static_cast<uint64_t>(-1);
constexpr uint32_t STATE_TRANSPOSE_BLOCK = 16;
constexpr uint32_t DATA_BLOCK_BYTES = 32;

struct RKDALegacyInitParams {
    GM_ADDR query;
    GM_ADDR key;
    GM_ADDR value;
    GM_ADDR gate;
    GM_ADDR beta;
    GM_ADDR initState;
    GM_ADDR cuSeqlens;
    GM_ADDR ssmStateIndices;
    GM_ADDR aLog;
    GM_ADDR dtBias;
    GM_ADDR numAcceptedTokens;
    GM_ADDR attnOut;
    GM_ADDR finalState;
};

__simd_vf__ inline void DecayMatVecReduce128Regbase(
    __ubuf__ float *stateAddr, __ubuf__ float *gateAddr, __ubuf__ float *vecAddr,
    __ubuf__ float *dstAddr, uint16_t rowNum, uint32_t alignK)
{
    RegTensor<float> state0;
    RegTensor<float> state1;
    RegTensor<float> gate0;
    RegTensor<float> gate1;
    RegTensor<float> vec0;
    RegTensor<float> vec1;
    RegTensor<float> product0;
    RegTensor<float> product1;
    RegTensor<float> sum;
    MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
    DataCopy<float, LoadDist::DIST_DINTLV_B32>(gate0, gate1, gateAddr);
    DataCopy<float, LoadDist::DIST_DINTLV_B32>(vec0, vec1, vecAddr);
    for (uint16_t i = 0; i < rowNum; ++i) {
        DataCopy<float, LoadDist::DIST_DINTLV_B32>(state0, state1, stateAddr + i * alignK);
        Mul(state0, state0, gate0, pregFull);
        Mul(state1, state1, gate1, pregFull);
        Mul(product0, state0, vec0, pregFull);
        Mul(product1, state1, vec1, pregFull);
        Add(product0, product0, product1, pregFull);
        ReduceSum(sum, product0, pregFull);
        DataCopy<float, StoreDist::DIST_INTLV_B32>(
            stateAddr + i * alignK, state0, state1, pregFull);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstAddr + i, sum, pregFull);
    }
}

template <bool useSigmoid, bool allowNeg>
__simd_vf__ inline void ProcessDeltaKQReduce128Regbase(
    __ubuf__ float *dotAddr, __ubuf__ float *vAddr, __ubuf__ float *betaAddr,
    __ubuf__ float *kAddr, __ubuf__ float *stateAddr, __ubuf__ float *qAddr,
    __ubuf__ float *attnAddr, uint16_t rowNum, uint32_t alignK)
{
    RegTensor<float> delta;
    RegTensor<float> value;
    RegTensor<float> beta;
    RegTensor<float> k0;
    RegTensor<float> k1;
    RegTensor<float> q0;
    RegTensor<float> q1;
    RegTensor<float> state0;
    RegTensor<float> state1;
    RegTensor<float> update;
    RegTensor<float> dot0;
    RegTensor<float> dot1;
    RegTensor<float> sum;
    RegTensor<float> one;
    MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
    DataCopy<float, LoadDist::DIST_BRC_B32>(beta, betaAddr);
    if constexpr (useSigmoid) {
        Muls(beta, beta, -1.0f, pregFull);
        Exp(beta, beta, pregFull);
        Adds(beta, beta, 1.0f, pregFull);
        Duplicate(one, 1.0f);
        Div(beta, one, beta, pregFull);
        if constexpr (allowNeg) {
            Muls(beta, beta, 2.0f, pregFull);
        }
    }
    DataCopy<float, LoadDist::DIST_DINTLV_B32>(k0, k1, kAddr);
    DataCopy<float, LoadDist::DIST_DINTLV_B32>(q0, q1, qAddr);
    for (uint16_t i = 0; i < rowNum; ++i) {
        DataCopy<float, LoadDist::DIST_BRC_B32>(delta, dotAddr + i);
        DataCopy<float, LoadDist::DIST_BRC_B32>(value, vAddr + i);
        Sub(delta, value, delta, pregFull);
        Mul(delta, delta, beta, pregFull);
        DataCopy<float, LoadDist::DIST_DINTLV_B32>(state0, state1, stateAddr + i * alignK);
        Mul(update, delta, k0, pregFull);
        Add(state0, state0, update, pregFull);
        Mul(dot0, state0, q0, pregFull);

        Mul(update, delta, k1, pregFull);
        Add(state1, state1, update, pregFull);
        Mul(dot1, state1, q1, pregFull);

        Add(dot0, dot0, dot1, pregFull);
        ReduceSum(sum, dot0, pregFull);
        DataCopy<float, StoreDist::DIST_INTLV_B32>(
            stateAddr + i * alignK, state0, state1, pregFull);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(attnAddr + i, sum, pregFull);
    }
}

template <typename inType, typename outType, typename stateType>
class RKDALegacy {
public:
    __aicore__ inline explicit RKDALegacy(const RecurrentKdaTilingData *tilingData)
    {
        B_ = tilingData->b;
        T_ = tilingData->t;
        seqLen_ = tilingData->seqLen;
        NK_ = tilingData->nk;
        realK_ = tilingData->dk;
        NV_ = tilingData->nv;
        realV_ = tilingData->dv;
        stateCapacity_ = tilingData->sBlockNum;
        queryTokenStride_ = tilingData->queryTokenStride;
        queryHeadStride_ = tilingData->queryHeadStride;
        keyTokenStride_ = tilingData->keyTokenStride;
        keyHeadStride_ = tilingData->keyHeadStride;
        valueTokenStride_ = tilingData->valueTokenStride;
        valueHeadStride_ = tilingData->valueHeadStride;
        ssmStateStride_ = tilingData->ssmStateStride;
        stateInStride0_ = tilingData->stateInStride0;
        stateInStride1_ = tilingData->stateInStride1;
        stateInStride2_ = tilingData->stateInStride2;
        stateInStride3_ = tilingData->stateInStride3;
        stateOutStride0_ = tilingData->stateOutStride0;
        stateOutStride1_ = tilingData->stateOutStride1;
        stateOutStride2_ = tilingData->stateOutStride2;
        stateOutStride3_ = tilingData->stateOutStride3;
        scale_ = tilingData->scale;
        lowerBound_ = tilingData->lowerBound;
        hasCuSeqlens_ = (tilingData->hasCuSeqlens == 1);
        hasSsmStateIndices_ = (tilingData->hasSsmStateIndices == 1);
        hasAcceptedTokens_ = (tilingData->hasAcceptedTokens == 1);
        hasALog_ = (tilingData->hasALog == 1);
        hasDtBias_ = (tilingData->hasDtBias == 1);
        useQkL2norm_ = (tilingData->useQkL2norm == 1);
        useGateInKernel_ = (tilingData->useGateInKernel == 1);
        useBetaSigmoid_ = (tilingData->useBetaSigmoid == 1);
        allowNegEigval_ = (tilingData->allowNegEigval == 1);
        safeGate_ = (tilingData->safeGate == 1);
        stateVFirst_ = (tilingData->stateVFirst == 1);
        shouldStoreState_ = (tilingData->inplaceFinalState == 1 || tilingData->outputFinalState == 1);
        gateDtype_ = tilingData->gateDtype;
        betaDtype_ = tilingData->betaDtype;
        cuSeqlensDtype_ = tilingData->cuSeqlensDtype;
        ssmStateIndicesDtype_ = tilingData->ssmStateIndicesDtype;
        acceptedTokensDtype_ = tilingData->acceptedTokensDtype;
        vStep_ = tilingData->vStep;
        stateOutBufferNum_ = (tilingData->stateOutBufferNum == MAX_OUT_BUFFER_NUM) ? MAX_OUT_BUFFER_NUM : BUFFER_NUM;
        attnOutBufferNum_ = (tilingData->attnOutBufferNum == MAX_OUT_BUFFER_NUM) ? MAX_OUT_BUFFER_NUM : BUFFER_NUM;
        restUbSize_ = tilingData->ubRestBytes;
        alignK_ = Ceil(tilingData->dk, BF16_NUM_PER_BLOCK) * BF16_NUM_PER_BLOCK;
        alignV_ = Ceil(tilingData->dv, BF16_NUM_PER_BLOCK) * BF16_NUM_PER_BLOCK;
        gateExpA_ = 1.0f;
        eventMte2ToVInitialized_ = false;
        eventVToMte2Initialized_ = false;
        eventVToSInitialized_ = false;
        eventVToMte3Initialized_ = false;
        eventMte3ToVInitialized_ = false;
    }

    __aicore__ inline void Init(const RKDALegacyInitParams &initParams, TPipe *pipe)
    {
        uint64_t blockDim = GetBlockNum();
        blockIdx = GetBlockIdx();
        if (blockIdx >= blockDim) {
            return;
        }
        pipe_ = pipe;
        SetGlobalTensors(initParams);
        InitLocalBuffers();
    }

    __aicore__ inline void SetGlobalTensors(const RKDALegacyInitParams &initParams)
    {
        queryGm_.SetGlobalBuffer((__gm__ inType *)initParams.query);
        keyGm_.SetGlobalBuffer((__gm__ inType *)initParams.key);
        valueGm_.SetGlobalBuffer((__gm__ inType *)initParams.value);
        gateFloatGm_.SetGlobalBuffer((__gm__ float *)initParams.gate);
        gateBf16Gm_.SetGlobalBuffer((__gm__ bfloat16_t *)initParams.gate);
        gateFp16Gm_.SetGlobalBuffer((__gm__ half *)initParams.gate);
        betaFloatGm_.SetGlobalBuffer((__gm__ float *)initParams.beta);
        betaBf16Gm_.SetGlobalBuffer((__gm__ bfloat16_t *)initParams.beta);
        betaFp16Gm_.SetGlobalBuffer((__gm__ half *)initParams.beta);
        initStateGm_.SetGlobalBuffer((__gm__ stateType *)initParams.initState);
        cuSeqlensInt32Gm_.SetGlobalBuffer((__gm__ int32_t *)initParams.cuSeqlens);
        cuSeqlensInt64Gm_.SetGlobalBuffer((__gm__ int64_t *)initParams.cuSeqlens);
        ssmStateIndicesInt32Gm_.SetGlobalBuffer((__gm__ int32_t *)initParams.ssmStateIndices);
        ssmStateIndicesInt64Gm_.SetGlobalBuffer((__gm__ int64_t *)initParams.ssmStateIndices);
        aLogGm_.SetGlobalBuffer((__gm__ float *)initParams.aLog);
        dtBiasGm_.SetGlobalBuffer((__gm__ float *)initParams.dtBias);
        numAcceptedTokensInt32Gm_.SetGlobalBuffer((__gm__ int32_t *)initParams.numAcceptedTokens);
        numAcceptedTokensInt64Gm_.SetGlobalBuffer((__gm__ int64_t *)initParams.numAcceptedTokens);
        finalStateGm_.SetGlobalBuffer((__gm__ stateType *)initParams.finalState);
        attnOutGm_.SetGlobalBuffer((__gm__ outType *)initParams.attnOut);
    }

    __aicore__ inline void InitLocalBuffers()
    {
        uint32_t cubeSize = alignK_ * vStep_ * sizeof(float);
        uint32_t singleVSize = vStep_ * sizeof(float);
        uint32_t vSize = MAX_MTP * alignV_ * sizeof(float);
        uint32_t kSize = MAX_MTP * alignK_ * sizeof(float);
        // FP16/BF16 beta rows expand to 16 FP32 values after Cast.
        uint32_t betaUbSize = MAX_MTP * BF16_NUM_PER_BLOCK * sizeof(float);
        pipe_->InitBuffer(qInQueue_, BUFFER_NUM, MAX_MTP * alignK_ * sizeof(inType));
        pipe_->InitBuffer(kInQueue_, BUFFER_NUM, MAX_MTP * alignK_ * sizeof(inType));
        pipe_->InitBuffer(vInQueue_, BUFFER_NUM, MAX_MTP * alignV_ * sizeof(inType));
        pipe_->InitBuffer(gateInQueue_, BUFFER_NUM, MAX_MTP * alignK_ * sizeof(float));
        pipe_->InitBuffer(betaInQueue_, BUFFER_NUM, betaUbSize);
        pipe_->InitBuffer(stateInQueue_, INPUT_BUFFER_NUM, alignK_ * vStep_ * sizeof(stateType));
        pipe_->InitBuffer(stateOutQueue_, stateOutBufferNum_, alignK_ * vStep_ * sizeof(stateType));
        if (!stateVFirst_) {
            pipe_->InitBuffer(stateTransposeBuf_, alignK_ * vStep_ * sizeof(stateType));
        }
        pipe_->InitBuffer(attnOutQueue_, attnOutBufferNum_, vStep_ * sizeof(outType));
        pipe_->InitBuffer(tmpBuff, restUbSize_);
        pipe_->InitBuffer(scalarBuf_, 64);

        uint32_t buffOffset = 0;
        // 共享工作区（不随 slot 切换）：delta/attn 跨窗口原地续接。
        deltaInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(vStep_), buffOffset);
        buffOffset += singleVSize;
        attnInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(vStep_), buffOffset);
        buffOffset += singleVSize;

        // ===== slot 0 输入缓冲 =====
        // 顺序：v→q→k→state→gateBias→beta→gate。state 放在 k 之后，避免与 v 产生 bank conflict
        // （Compute 中 VF 会同时访问 vInUb 和 stateInUb；若 state 放在 v 之前，起点 bank 差一个
        //  bank stride，两个大 buffer 很容易在同一 VF cycle 落到同 bank）。
        vInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(MAX_MTP * alignV_), buffOffset);
        buffOffset += vSize;
        qInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(MAX_MTP * alignK_), buffOffset);
        buffOffset += kSize;
        kInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(MAX_MTP * alignK_), buffOffset);
        buffOffset += kSize;

        // state + gateBias：跨窗口原地续接，不随 slot 切换（只分配一份）。
        stateInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(alignK_ * vStep_), buffOffset);
        buffOffset += cubeSize;
        gateBiasInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(alignK_), buffOffset);
        buffOffset += static_cast<uint32_t>(alignK_ * sizeof(float));

        betaInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(betaUbSize / sizeof(float)), buffOffset);
        buffOffset += betaUbSize;
        gateInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(MAX_MTP * alignK_), buffOffset);
        buffOffset += kSize;

        // ===== slot 1 输入缓冲（紧接 slot 0 之后，buffOffset 自然推进，绝无地址重叠） =====
        // stateInUb / gateBiasInUb 共享 slot 0，不分配第二份。
        vInUb2 = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(MAX_MTP * alignV_), buffOffset);
        buffOffset += vSize;
        qInUb2 = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(MAX_MTP * alignK_), buffOffset);
        buffOffset += kSize;
        kInUb2 = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(MAX_MTP * alignK_), buffOffset);
        buffOffset += kSize;
        betaInUb2 = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(betaUbSize / sizeof(float)), buffOffset);
        buffOffset += betaUbSize;
        gateInUb2 = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(MAX_MTP * alignK_), buffOffset);
        buffOffset += kSize;
    }

    __aicore__ inline void SyncMte2ToV()
    {
        if (!eventMte2ToVInitialized_) {
            eventIdMte2ToV_ = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
            eventMte2ToVInitialized_ = true;
        }
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV_);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV_);
    }

    __aicore__ inline void SyncVToMte2()
    {
        if (!eventVToMte2Initialized_) {
            eventIdVToMte2_ = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
            eventVToMte2Initialized_ = true;
        }
        SetFlag<HardEvent::V_MTE2>(eventIdVToMte2_);
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2_);
    }

    __aicore__ inline void SyncVToS()
    {
        if (!eventVToSInitialized_) {
            eventIdVToS_ = GetTPipePtr()->FetchEventID(HardEvent::V_S);
            eventVToSInitialized_ = true;
        }
        SetFlag<HardEvent::V_S>(eventIdVToS_);
        WaitFlag<HardEvent::V_S>(eventIdVToS_);
    }

    __aicore__ inline void SyncVToMte3()
    {
        if (!eventVToMte3Initialized_) {
            eventIdVToMte3_ = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
            eventVToMte3Initialized_ = true;
        }
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3_);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3_);
    }

    __aicore__ inline void SyncMte3ToV()
    {
        if (!eventMte3ToVInitialized_) {
            eventIdMte3ToV_ = GetTPipePtr()->FetchEventID(HardEvent::MTE3_V);
            eventMte3ToVInitialized_ = true;
        }
        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV_);
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV_);
    }

    __aicore__ inline void ReleaseEvents()
    {
        if (eventMte2ToVInitialized_) {
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIdMte2ToV_);
            eventMte2ToVInitialized_ = false;
        }
        if (eventVToMte2Initialized_) {
            GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte2_);
            eventVToMte2Initialized_ = false;
        }
        if (eventVToSInitialized_) {
            GetTPipePtr()->ReleaseEventID<HardEvent::V_S>(eventIdVToS_);
            eventVToSInitialized_ = false;
        }
        if (eventVToMte3Initialized_) {
            GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(eventIdVToMte3_);
            eventVToMte3Initialized_ = false;
        }
        if (eventMte3ToVInitialized_) {
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(eventIdMte3ToV_);
            eventMte3ToVInitialized_ = false;
        }
    }

    __aicore__ inline void Process()
    {
        if (!ValidateCuSeqlens()) {
            ReleaseEvents();
            return;
        }
        uint64_t taskIdx = blockIdx;
        uint64_t taskNum = B_ * NV_;
        uint64_t taskStride = GetBlockNum();
        RKDATaskInfo currentTask{};
        RKDATaskInfo nextTask{};
        bool taskValid = true;
        bool hasCurrentTask = PrepareNextTask(taskIdx, taskNum, taskStride, currentTask, taskValid);
        bool hasNextTask = PrepareNextTask(taskIdx, taskNum, taskStride, nextTask, taskValid);
        if (!taskValid) {
            ReleaseEvents();
            return;
        }
        bool statePrefetched = false;
        while (hasCurrentTask) {
            statePrefetched = ProcessHead(
                currentTask.batch, currentTask.seq0, currentTask.seq1,
                currentTask.head, currentTask.stateSlot, statePrefetched,
                hasNextTask, nextTask.stateSlot, nextTask.head);
            if (!hasNextTask) {
                break;
            }
            currentTask = nextTask;
            hasNextTask = PrepareNextTask(taskIdx, taskNum, taskStride, nextTask, taskValid);
            if (!taskValid) {
                ReleaseEvents();
                return;
            }
        }
        ReleaseEvents();
    }

private:
    struct RKDATaskInfo {
        uint64_t batch;
        uint64_t head;
        uint64_t stateSlot;
        int64_t seq0;
        int64_t seq1;
    };

    __aicore__ inline bool PrepareNextTask(uint64_t &taskIdx, uint64_t taskNum, uint64_t taskStride,
                                           RKDATaskInfo &task, bool &taskValid)
    {
        taskValid = true;
        while (taskIdx < taskNum) {
            uint64_t currentTaskIdx = taskIdx;
            taskIdx += taskStride;
            uint64_t batch = currentTaskIdx / NV_;
            int64_t seq0 = SequenceStart(batch);
            int64_t seq1 = SequenceEnd(batch);
            int64_t seqLen64 = seq1 - seq0;
            if (seqLen64 == 0) {
                continue;
            }
            // 段长本身不受 MAX_MTP 限制（按窗口递推）；这里只防御 speculative 的
            // ssm_state_indices 元数据：每个 token 的槽位必须落在该序列的 stride 内。
            if (hasSsmStateIndices_ && ssmStateStride_ > 0 && seqLen64 > ssmStateStride_) {
                continue;
            }
            int64_t seqLen = seqLen64;
            if (!ValidateStateSlots(batch, seq0, seqLen)) {
                taskValid = false;
                return false;
            }
            uint64_t stateSlot = ResolveInitialStateSlot(batch, seq0);
            if (stateSlot == INVALID_STATE_SLOT) {
                taskValid = false;
                return false;
            }
            task.batch = batch;
            task.head = currentTaskIdx % NV_;
            task.stateSlot = stateSlot;
            task.seq0 = seq0;
            task.seq1 = seq1;
            return true;
        }
        return false;
    }

    // 只校验元数据的结构合法性（起点为 0、单调不减、不超过 T）。段长由 ProcessHead
    // 按窗口递推，不再受输入窗口大小约束。
    __aicore__ inline bool ValidateCuSeqlens() const
    {
        if (!hasCuSeqlens_) {
            return true;
        }
        int64_t seq0 = LoadCuSeqlens(0);
        if (seq0 != 0) {
            return false;
        }
        for (uint64_t i = 0; i < B_; i++) {
            int64_t seq1 = LoadCuSeqlens(i + 1);
            if (seq1 < seq0 || seq1 > static_cast<int64_t>(T_)) {
                return false;
            }
            seq0 = seq1;
        }
        return seq0 <= static_cast<int64_t>(T_);
    }

    __aicore__ inline int64_t LoadCuSeqlens(uint64_t index) const
    {
        return cuSeqlensDtype_ == 0 ? static_cast<int64_t>(cuSeqlensInt32Gm_.GetValue(index)) :
                                     cuSeqlensInt64Gm_.GetValue(index);
    }

    __aicore__ inline int64_t SequenceStart(uint64_t batchIdx) const
    {
        return hasCuSeqlens_ ? LoadCuSeqlens(batchIdx) : static_cast<int64_t>(batchIdx * seqLen_);
    }

    __aicore__ inline int64_t SequenceEnd(uint64_t batchIdx) const
    {
        return hasCuSeqlens_ ? LoadCuSeqlens(batchIdx + 1) :
                               static_cast<int64_t>((batchIdx + 1) * seqLen_);
    }

    __aicore__ inline int64_t LoadSsmStateIndex(uint64_t index) const
    {
        return ssmStateIndicesDtype_ == 0 ? static_cast<int64_t>(ssmStateIndicesInt32Gm_.GetValue(index)) :
                                           ssmStateIndicesInt64Gm_.GetValue(index);
    }

    __aicore__ inline int64_t LoadAcceptedTokens(uint64_t index) const
    {
        return acceptedTokensDtype_ == 0 ? static_cast<int64_t>(numAcceptedTokensInt32Gm_.GetValue(index)) :
                                          numAcceptedTokensInt64Gm_.GetValue(index);
    }

    __aicore__ inline uint64_t StateMetadataOffset(uint64_t batchIdx, int64_t seq0, int64_t tokenIdx) const
    {
        if (ssmStateStride_ == 0) {
            return static_cast<uint64_t>(tokenIdx);
        }
        return batchIdx * ssmStateStride_ + static_cast<uint64_t>(tokenIdx - seq0);
    }

    __aicore__ inline uint64_t LoadStateSlot(uint64_t batchIdx, int64_t seq0, int64_t tokenIdx) const
    {
        int64_t stateSlot = LoadSsmStateIndex(StateMetadataOffset(batchIdx, seq0, tokenIdx));
        if (stateSlot < 0 || stateSlot >= static_cast<int64_t>(stateCapacity_)) {
            return INVALID_STATE_SLOT;
        }
        return static_cast<uint64_t>(stateSlot);
    }

    __aicore__ inline bool ValidateStateSlots(uint64_t batchIdx, int64_t seq0, int64_t seqLen) const
    {
        if (!hasSsmStateIndices_) {
            return batchIdx < stateCapacity_;
        }
        if (hasAcceptedTokens_) {
            int64_t acceptedTokenNum = LoadAcceptedTokens(batchIdx);
            if (acceptedTokenNum <= 0 || acceptedTokenNum > seqLen) {
                return false;
            }
        }
        for (int64_t step = 0; step < seqLen; ++step) {
            if (LoadStateSlot(batchIdx, seq0, seq0 + step) == INVALID_STATE_SLOT) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline uint64_t ResolveInitialStateSlot(uint64_t batchIdx, int64_t seq0) const
    {
        if (!hasSsmStateIndices_) {
            return batchIdx;
        }
        int64_t tokenIdx = seq0;
        if (hasAcceptedTokens_) {
            tokenIdx = seq0 + LoadAcceptedTokens(batchIdx) - 1;
        }
        return LoadStateSlot(batchIdx, seq0, tokenIdx);
    }

    template <typename dataType>
    __aicore__ inline void CopyVectorIn(LocalTensor<dataType> &dst, GlobalTensor<dataType> &src, uint64_t offset,
                                        uint64_t count)
    {
        uint64_t rowBytes = count * sizeof(dataType);
        if (rowBytes >= 32 && rowBytes % 32 == 0) {
            DataCopy(dst, src[offset], static_cast<uint32_t>(count));
        } else {
            DataCopyParams params{1, static_cast<uint16_t>(rowBytes), 0, 0};
            DataCopyPadParams padParams{false, 0, 0, 0};
            DataCopyPad(dst, src[offset], params, padParams);
        }
    }

    __aicore__ inline float ReadFloat(GlobalTensor<float> &tensor, uint64_t offset)
    {
        LocalTensor<float> scalar = scalarBuf_.Get<float>();
        DataCopyParams params{1, static_cast<uint16_t>(sizeof(float)), 0, 0};
        DataCopyPadParams padParams{false, 0, 0, 0};
        DataCopyPad(scalar, tensor[offset], params, padParams);
        SyncMte2ToV();
        ScalarCopyVF(reinterpret_cast<__ubuf__ float *>(scalar.GetPhyAddr()),
                     reinterpret_cast<__ubuf__ float *>(scalar.GetPhyAddr()));
        SyncVToS();
        __ubuf__ float *ptr = (__ubuf__ float *)scalar.GetPhyAddr();
        return ptr[0];
    }

    __aicore__ inline float ExpScalar(float x)
    {
        LocalTensor<float> scalar = scalarBuf_.Get<float>();
        ScalarExpVF(reinterpret_cast<__ubuf__ float *>(scalar.GetPhyAddr()), x);
        SyncVToS();
        __ubuf__ float *ptr = (__ubuf__ float *)scalar.GetPhyAddr();
        return ptr[0];
    }

    // 扁平 FP32 向量 -> 输出 dtype（out 一个 V 切片是扁平向量，不是二维行块）。
    template <typename dstType>
    __simd_vf__ inline void PackFlatVF(__ubuf__ dstType *dst, __ubuf__ float *src, uint32_t count)
    {
        MaskReg full = CreateMask<float, MaskPattern::ALL>();
        for (uint32_t i = 0; i < count; i += V_LENGTH) {
            uint32_t remain = count - i;
            MaskReg mask = full;
            if (remain < static_cast<uint32_t>(V_LENGTH)) {
                mask = UpdateMask<float>(remain);
            }
            RegTensor<float> reg;
            RegBase::Load(reg, src + i, mask);
            RegBase::Store(dst + i, reg, mask);
        }
    }

    // 标量参数读取/计算也不再走 membase：单 lane 的 copy/exp 在寄存器内完成。
    __simd_vf__ inline void ScalarCopyVF(__ubuf__ float *dst, __ubuf__ float *src)
    {
        uint32_t one = 1;
        MaskReg mask = UpdateMask<float>(one);
        RegTensor<float> reg;
        RegBase::Load(reg, src, mask);
        RegBase::Store(dst, reg, mask);
    }

    __simd_vf__ inline void ScalarExpVF(__ubuf__ float *dst, float x)
    {
        uint32_t one = 1;
        MaskReg mask = UpdateMask<float>(one);
        RegTensor<float> reg;
        Duplicate(reg, x);
        Exp(reg, reg, mask);
        RegBase::Store(dst, reg, mask);
    }

    // 扁平小向量（beta）也同样走寄存器内转换。
    template <typename srcType>
    __simd_vf__ inline void UnpackFlatVF(__ubuf__ float *dst, __ubuf__ srcType *src, uint32_t count)
    {
        MaskReg full = CreateMask<float, MaskPattern::ALL>();
        for (uint32_t i = 0; i < count; i += V_LENGTH) {
            uint32_t remain = count - i;
            MaskReg mask = full;
            if (remain < static_cast<uint32_t>(V_LENGTH)) {
                mask = UpdateMask<float>(remain);
            }
            RegTensor<float> reg;
            RegBase::Load(reg, src + i, mask);
            RegBase::Store(dst + i, reg, mask);
        }
    }

    // 低精度 -> FP32：寄存器内 unpack/cast，避免整块 membase Cast。
    template <typename srcType, uint32_t COLS>
    __simd_vf__ inline void UnpackRowsVF(__ubuf__ float *dst, __ubuf__ srcType *src, uint32_t rows)
    {
        MaskReg mask = CreateMask<float, MaskPattern::ALL>();
        constexpr uint32_t SEGS = COLS / static_cast<uint32_t>(V_LENGTH);
        for (uint32_t row = 0; row < rows; ++row) {
            const uint32_t base = row * COLS;
            for (uint32_t seg = 0; seg < SEGS; ++seg) {
                RegTensor<float> reg;
                RegBase::Load(reg, src + base + seg * V_LENGTH, mask);
                RegBase::Store(dst + base + seg * V_LENGTH, reg, mask);
            }
        }
    }

    // FP32 -> 输出/State dtype：寄存器内 cast + pack。
    template <typename dstType, uint32_t COLS>
    __simd_vf__ inline void PackRowsVF(__ubuf__ dstType *dst, __ubuf__ float *src, uint32_t rows)
    {
        MaskReg mask = CreateMask<float, MaskPattern::ALL>();
        constexpr uint32_t SEGS = COLS / static_cast<uint32_t>(V_LENGTH);
        for (uint32_t row = 0; row < rows; ++row) {
            const uint32_t base = row * COLS;
            for (uint32_t seg = 0; seg < SEGS; ++seg) {
                RegTensor<float> reg;
                RegBase::Load(reg, src + base + seg * V_LENGTH, mask);
                RegBase::Store(dst + base + seg * V_LENGTH, reg, mask);
            }
        }
    }

    // K 恒为 128：整块 q/k 的 L2 norm 与 q 的 scale 都在寄存器内完成，
    // 取代原来逐行 ReduceSum→GetValue→Muls（每行两次 V→S 同步）。
    __simd_vf__ inline void NormScaleVF(__ubuf__ float *q, __ubuf__ float *k, uint32_t rows,
                                       float scale, bool normalize)
    {
        MaskReg mask = CreateMask<float, MaskPattern::ALL>();
        for (uint32_t row = 0; row < rows; ++row) {
            const uint32_t base = row * TWO_V_LENGTH;
            RegTensor<float> lo, hi;
            RegBase::Load(lo, q + base, mask);
            RegBase::Load(hi, q + base + V_LENGTH, mask);
            if (normalize) {
                RegBase::Normalize(lo, hi, mask);
            }
            Muls(lo, lo, scale, mask);
            Muls(hi, hi, scale, mask);
            RegBase::Store(q + base, lo, mask);
            RegBase::Store(q + base + V_LENGTH, hi, mask);
            RegBase::Load(lo, k + base, mask);
            RegBase::Load(hi, k + base + V_LENGTH, mask);
            if (normalize) {
                RegBase::Normalize(lo, hi, mask);
            }
            RegBase::Store(k + base, lo, mask);
            RegBase::Store(k + base + V_LENGTH, hi, mask);
        }
    }

    // raw gate（A_log/dt_bias/safe_gate）同样在寄存器内完成，逐行处理 tile。
    __simd_vf__ inline void GateTransformVF(__ubuf__ float *g, __ubuf__ float *bias, uint32_t rows,
                                           float expA, float lower, bool safe, bool hasBias)
    {
        MaskReg mask = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> biasLo, biasHi;
        if (hasBias) {
            RegBase::Load(biasLo, bias, mask);
            RegBase::Load(biasHi, bias + V_LENGTH, mask);
        }
        for (uint32_t row = 0; row < rows; ++row) {
            const uint32_t base = row * TWO_V_LENGTH;
            RegTensor<float> lo, hi;
            RegBase::Load(lo, g + base, mask);
            RegBase::Load(hi, g + base + V_LENGTH, mask);
            if (hasBias) {
                Add(lo, lo, biasLo, mask);
                Add(hi, hi, biasHi, mask);
            }
            if (safe) {
                Muls(lo, lo, expA, mask);
                Muls(hi, hi, expA, mask);
                RegBase::Sigmoid(lo, mask);
                RegBase::Sigmoid(hi, mask);
                Muls(lo, lo, lower, mask);
                Muls(hi, hi, lower, mask);
            } else {
                Exp(lo, lo, mask);
                Exp(hi, hi, mask);
                Adds(lo, lo, 1.0f, mask);
                Adds(hi, hi, 1.0f, mask);
                Ln(lo, lo, mask);
                Ln(hi, hi, mask);
                Muls(lo, lo, -expA, mask);
                Muls(hi, hi, -expA, mask);
            }
            RegBase::Store(g + base, lo, mask);
            RegBase::Store(g + base + V_LENGTH, hi, mask);
        }
    }

    // alpha = exp(g)：整块 gate 的指数也在寄存器内完成。
    __simd_vf__ inline void GateExpVF(__ubuf__ float *g, uint32_t rows)
    {
        MaskReg mask = CreateMask<float, MaskPattern::ALL>();
        for (uint32_t row = 0; row < rows; ++row) {
            const uint32_t base = row * TWO_V_LENGTH;
            RegTensor<float> lo, hi;
            RegBase::Load(lo, g + base, mask);
            RegBase::Load(hi, g + base + V_LENGTH, mask);
            Exp(lo, lo, mask);
            Exp(hi, hi, mask);
            RegBase::Store(g + base, lo, mask);
            RegBase::Store(g + base + V_LENGTH, hi, mask);
        }
    }

    // 与 token 无关的 gate 参数（A_log 的 exp 与 dt_bias）每个 head 只准备一次，
    // 窗口/ v-tile 循环内复用，避免每次搬入都重复标量读取。
    __aicore__ inline void PrepareGateParams(uint64_t head)
    {
        if (!useGateInKernel_) {
            return;
        }
        gateExpA_ = hasALog_ ? ExpScalar(ReadFloat(aLogGm_, head)) : 1.0f;
        if (hasDtBias_) {
            // 上一个 task 的 GateTransformVF 仍可能在读 gateBiasInUb，覆盖前先等 V 排空。
            SyncVToMte2();
            CopyVectorIn(gateBiasInUb, dtBiasGm_, head * realK_, realK_);
            SyncMte2ToV();
        }
    }

    __aicore__ inline void ApplyGateInKernel(int32_t rows, uint32_t slot)
    {
        GateTransformVF(reinterpret_cast<__ubuf__ float *>(GateSlot(slot).GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(gateBiasInUb.GetPhyAddr()),
                        static_cast<uint32_t>(rows), gateExpA_, lowerBound_, safeGate_, hasDtBias_);
    }

    template <typename gateType>
    __aicore__ inline void CopyInGate(uint64_t gateOffset, int32_t seqLen, uint32_t slot)
    {
        LocalTensor<gateType> gateLocal = gateInQueue_.AllocTensor<gateType>();
        DataCopyExtParams gateInParams{static_cast<uint16_t>(seqLen),
                                       static_cast<uint32_t>(realK_ * sizeof(gateType)),
                                       static_cast<uint32_t>((NV_ - 1) * realK_ * sizeof(gateType)), 0, 0};
        DataCopyPadExtParams<gateType> gatePadParams{
            true, 0, static_cast<uint8_t>(alignK_ - realK_), static_cast<gateType>(0)};
        if constexpr (std::is_same<gateType, float32_t>()) {
            DataCopyPad(gateLocal, gateFloatGm_[gateOffset], gateInParams, gatePadParams);
        } else if constexpr (std::is_same<gateType, bfloat16_t>()) {
            DataCopyPad(gateLocal, gateBf16Gm_[gateOffset], gateInParams, gatePadParams);
        } else {
            DataCopyPad(gateLocal, gateFp16Gm_[gateOffset], gateInParams, gatePadParams);
        }
        gateInQueue_.EnQue<gateType>(gateLocal);
        gateLocal = gateInQueue_.DeQue<gateType>();
        UnpackRowsVF<gateType, 128>(reinterpret_cast<__ubuf__ float *>(GateSlot(slot).GetPhyAddr()),
                                    reinterpret_cast<__ubuf__ gateType *>(gateLocal.GetPhyAddr()),
                                    static_cast<uint32_t>(seqLen));
        gateInQueue_.FreeTensor(gateLocal);
    }

    // 只搬入当前窗口 [windowStart, windowStart + windowLen) 的 Q/K/V/gate，
    // 供本 v-tile 内该窗口的逐 token 递推使用。
    __aicore__ inline void CopyInQKVGate(int64_t windowStart, int32_t windowLen, uint64_t head_i, uint32_t slot)
    {
        uint64_t qkHead = head_i / (NV_ / NK_);
        uint64_t qOffset = static_cast<uint64_t>(windowStart) * queryTokenStride_ + qkHead * queryHeadStride_;
        uint64_t kOffset = static_cast<uint64_t>(windowStart) * keyTokenStride_ + qkHead * keyHeadStride_;
        uint64_t vOffset = static_cast<uint64_t>(windowStart) * valueTokenStride_ + head_i * valueHeadStride_;
        uint64_t gateOffset = (static_cast<uint64_t>(windowStart) * NV_ + head_i) * realK_;
        LocalTensor<inType> qLocal = qInQueue_.AllocTensor<inType>();
        LocalTensor<inType> kLocal = kInQueue_.AllocTensor<inType>();
        LocalTensor<inType> vLocal = vInQueue_.AllocTensor<inType>();

        DataCopyExtParams qInParams{static_cast<uint16_t>(windowLen), static_cast<uint32_t>(realK_ * sizeof(inType)),
                                    static_cast<uint32_t>((queryTokenStride_ - realK_) * sizeof(inType)), 0, 0};
        DataCopyExtParams kInParams{static_cast<uint16_t>(windowLen), static_cast<uint32_t>(realK_ * sizeof(inType)),
                                    static_cast<uint32_t>((keyTokenStride_ - realK_) * sizeof(inType)), 0, 0};
        DataCopyExtParams vInParams{static_cast<uint16_t>(windowLen), static_cast<uint32_t>(realV_ * sizeof(inType)),
                                    static_cast<uint32_t>((valueTokenStride_ - realV_) * sizeof(inType)), 0, 0};
        DataCopyPadExtParams<inType> qkPadParams{true, 0, static_cast<uint8_t>(alignK_ - realK_), 0};
        DataCopyPadExtParams<inType> vPadParams{true, 0, static_cast<uint8_t>(alignV_ - realV_), 0};

        DataCopyPad(qLocal, queryGm_[qOffset], qInParams, qkPadParams);
        DataCopyPad(kLocal, keyGm_[kOffset], kInParams, qkPadParams);
        DataCopyPad(vLocal, valueGm_[vOffset], vInParams, vPadParams);
        qInQueue_.EnQue<inType>(qLocal);
        kInQueue_.EnQue<inType>(kLocal);
        vInQueue_.EnQue<inType>(vLocal);
        if (gateDtype_ == 0) {
            CopyInGate<float>(gateOffset, windowLen, slot);
        } else if (gateDtype_ == 1) {
            CopyInGate<bfloat16_t>(gateOffset, windowLen, slot);
        } else {
            CopyInGate<half>(gateOffset, windowLen, slot);
        }

        qLocal = qInQueue_.DeQue<inType>();
        kLocal = kInQueue_.DeQue<inType>();
        vLocal = vInQueue_.DeQue<inType>();
        UnpackRowsVF<inType, 128>(reinterpret_cast<__ubuf__ float *>(QSlot(slot).GetPhyAddr()),
                                  reinterpret_cast<__ubuf__ inType *>(qLocal.GetPhyAddr()),
                                  static_cast<uint32_t>(windowLen));
        UnpackRowsVF<inType, 128>(reinterpret_cast<__ubuf__ float *>(KSlot(slot).GetPhyAddr()),
                                  reinterpret_cast<__ubuf__ inType *>(kLocal.GetPhyAddr()),
                                  static_cast<uint32_t>(windowLen));
        UnpackRowsVF<inType, 256>(reinterpret_cast<__ubuf__ float *>(VSlot(slot).GetPhyAddr()),
                                  reinterpret_cast<__ubuf__ inType *>(vLocal.GetPhyAddr()),
                                  static_cast<uint32_t>(windowLen));
        NormScaleVF(reinterpret_cast<__ubuf__ float *>(QSlot(slot).GetPhyAddr()),
                    reinterpret_cast<__ubuf__ float *>(KSlot(slot).GetPhyAddr()),
                    static_cast<uint32_t>(windowLen), scale_, useQkL2norm_);
        if (useGateInKernel_) {
            ApplyGateInKernel(windowLen, slot);
        }
        GateExpVF(reinterpret_cast<__ubuf__ float *>(GateSlot(slot).GetPhyAddr()), static_cast<uint32_t>(windowLen));
        AscendC::PipeBarrier<PIPE_V>();

        qInQueue_.FreeTensor(qLocal);
        kInQueue_.FreeTensor(kLocal);
        vInQueue_.FreeTensor(vLocal);
    }

    // ===== 流水线拆分：QKVGate =====
    // Start：对 q/k/v/gate 四个 queue 异步发起 MTE2 DataCopyPad + EnQue，不 DeQue 不 VF 操作。
    // End：DeQue 全部四个 tensor + VF UnpackRowsVF/NormScaleVF/GateTransform/GateExp + PipeBarrier。

    template <typename gateType>
    __aicore__ inline void CopyInGateStart(uint64_t gateOffset, int32_t seqLen)
    {
        LocalTensor<gateType> gateLocal = gateInQueue_.AllocTensor<gateType>();
        DataCopyExtParams gateInParams{static_cast<uint16_t>(seqLen),
                                       static_cast<uint32_t>(realK_ * sizeof(gateType)),
                                       static_cast<uint32_t>((NV_ - 1) * realK_ * sizeof(gateType)), 0, 0};
        DataCopyPadExtParams<gateType> gatePadParams{
            true, 0, static_cast<uint8_t>(alignK_ - realK_), static_cast<gateType>(0)};
        if constexpr (std::is_same<gateType, float32_t>()) {
            DataCopyPad(gateLocal, gateFloatGm_[gateOffset], gateInParams, gatePadParams);
        } else if constexpr (std::is_same<gateType, bfloat16_t>()) {
            DataCopyPad(gateLocal, gateBf16Gm_[gateOffset], gateInParams, gatePadParams);
        } else {
            DataCopyPad(gateLocal, gateFp16Gm_[gateOffset], gateInParams, gatePadParams);
        }
        gateInQueue_.EnQue<gateType>(gateLocal);
    }

    __aicore__ inline void CopyInQKVGateStart(int64_t windowStart, int32_t windowLen, uint64_t head_i)
    {
        uint64_t qkHead = head_i / (NV_ / NK_);
        uint64_t qOffset = static_cast<uint64_t>(windowStart) * queryTokenStride_ + qkHead * queryHeadStride_;
        uint64_t kOffset = static_cast<uint64_t>(windowStart) * keyTokenStride_ + qkHead * keyHeadStride_;
        uint64_t vOffset = static_cast<uint64_t>(windowStart) * valueTokenStride_ + head_i * valueHeadStride_;
        uint64_t gateOffset = (static_cast<uint64_t>(windowStart) * NV_ + head_i) * realK_;
        LocalTensor<inType> qLocal = qInQueue_.AllocTensor<inType>();
        LocalTensor<inType> kLocal = kInQueue_.AllocTensor<inType>();
        LocalTensor<inType> vLocal = vInQueue_.AllocTensor<inType>();

        DataCopyExtParams qInParams{static_cast<uint16_t>(windowLen), static_cast<uint32_t>(realK_ * sizeof(inType)),
                                    static_cast<uint32_t>((queryTokenStride_ - realK_) * sizeof(inType)), 0, 0};
        DataCopyExtParams kInParams{static_cast<uint16_t>(windowLen), static_cast<uint32_t>(realK_ * sizeof(inType)),
                                    static_cast<uint32_t>((keyTokenStride_ - realK_) * sizeof(inType)), 0, 0};
        DataCopyExtParams vInParams{static_cast<uint16_t>(windowLen), static_cast<uint32_t>(realV_ * sizeof(inType)),
                                    static_cast<uint32_t>((valueTokenStride_ - realV_) * sizeof(inType)), 0, 0};
        DataCopyPadExtParams<inType> qkPadParams{true, 0, static_cast<uint8_t>(alignK_ - realK_), 0};
        DataCopyPadExtParams<inType> vPadParams{true, 0, static_cast<uint8_t>(alignV_ - realV_), 0};

        DataCopyPad(qLocal, queryGm_[qOffset], qInParams, qkPadParams);
        DataCopyPad(kLocal, keyGm_[kOffset], kInParams, qkPadParams);
        DataCopyPad(vLocal, valueGm_[vOffset], vInParams, vPadParams);
        qInQueue_.EnQue<inType>(qLocal);
        kInQueue_.EnQue<inType>(kLocal);
        vInQueue_.EnQue<inType>(vLocal);
        // Gate queue：单独发起，不 DeQue 不 Free。
        if (gateDtype_ == 0) {
            CopyInGateStart<float>(gateOffset, windowLen);
        } else if (gateDtype_ == 1) {
            CopyInGateStart<bfloat16_t>(gateOffset, windowLen);
        } else {
            CopyInGateStart<half>(gateOffset, windowLen);
        }
    }

    template <typename gateType>
    __aicore__ inline void CopyInGateEnd(uint32_t windowLen, uint32_t slot)
    {
        LocalTensor<gateType> gateLocal = gateInQueue_.DeQue<gateType>();
        UnpackRowsVF<gateType, 128>(reinterpret_cast<__ubuf__ float *>(GateSlot(slot).GetPhyAddr()),
                                    reinterpret_cast<__ubuf__ gateType *>(gateLocal.GetPhyAddr()),
                                    static_cast<uint32_t>(windowLen));
        gateInQueue_.FreeTensor(gateLocal);
    }

    __aicore__ inline void CopyInQKVGateEnd(int32_t windowLen, uint64_t head_i, uint32_t slot)
    {
        // 先完成 gate queue：DeQue + VF UnpackRowsVF(gate) + FreeTensor
        if (gateDtype_ == 0) {
            CopyInGateEnd<float>(windowLen, slot);
        } else if (gateDtype_ == 1) {
            CopyInGateEnd<bfloat16_t>(windowLen, slot);
        } else {
            CopyInGateEnd<half>(windowLen, slot);
        }

        // 再完成 q/k/v queue：DeQue + VF UnpackRowsVF × 3 + NormScaleVF + GateTransformVF + GateExpVF + PipeBarrier
        LocalTensor<inType> qLocal = qInQueue_.DeQue<inType>();
        LocalTensor<inType> kLocal = kInQueue_.DeQue<inType>();
        LocalTensor<inType> vLocal = vInQueue_.DeQue<inType>();
        UnpackRowsVF<inType, 128>(reinterpret_cast<__ubuf__ float *>(QSlot(slot).GetPhyAddr()),
                                  reinterpret_cast<__ubuf__ inType *>(qLocal.GetPhyAddr()),
                                  static_cast<uint32_t>(windowLen));
        UnpackRowsVF<inType, 128>(reinterpret_cast<__ubuf__ float *>(KSlot(slot).GetPhyAddr()),
                                  reinterpret_cast<__ubuf__ inType *>(kLocal.GetPhyAddr()),
                                  static_cast<uint32_t>(windowLen));
        UnpackRowsVF<inType, 256>(reinterpret_cast<__ubuf__ float *>(VSlot(slot).GetPhyAddr()),
                                  reinterpret_cast<__ubuf__ inType *>(vLocal.GetPhyAddr()),
                                  static_cast<uint32_t>(windowLen));
        NormScaleVF(reinterpret_cast<__ubuf__ float *>(QSlot(slot).GetPhyAddr()),
                    reinterpret_cast<__ubuf__ float *>(KSlot(slot).GetPhyAddr()),
                    static_cast<uint32_t>(windowLen), scale_, useQkL2norm_);
        if (useGateInKernel_) {
            ApplyGateInKernel(windowLen, slot);
        }
        GateExpVF(reinterpret_cast<__ubuf__ float *>(GateSlot(slot).GetPhyAddr()), static_cast<uint32_t>(windowLen));
        AscendC::PipeBarrier<PIPE_V>();

        qInQueue_.FreeTensor(qLocal);
        kInQueue_.FreeTensor(kLocal);
        vInQueue_.FreeTensor(vLocal);
    }

    template <typename transType>
    __aicore__ inline void TransposeStateMatrixTyped(const LocalTensor<transType> &dst,
                                                      const LocalTensor<transType> &src,
                                                      uint32_t rowCount, uint32_t columnCount,
                                                      uint32_t srcRowStride, uint32_t dstRowStride)
    {
        constexpr uint32_t elementsPerBlock = DATA_BLOCK_BYTES / sizeof(transType);
        uint64_t dstList[STATE_TRANSPOSE_BLOCK];
        uint64_t srcList[STATE_TRANSPOSE_BLOCK];
        uint64_t dstAddr = reinterpret_cast<uint64_t>(dst.GetPhyAddr());
        uint64_t srcAddr = reinterpret_cast<uint64_t>(src.GetPhyAddr());
        uint16_t repeatTimes = static_cast<uint16_t>(columnCount / elementsPerBlock);
        TransDataTo5HDParams transposeParams{
            false, false, static_cast<uint8_t>(repeatTimes),
            static_cast<uint16_t>(repeatTimes > 1 ? dstRowStride : 0),
            static_cast<uint16_t>(repeatTimes > 1 ? 1 : 0)};
        for (uint32_t rowBlock = 0; rowBlock < rowCount; rowBlock += STATE_TRANSPOSE_BLOCK) {
            for (uint32_t i = 0; i < STATE_TRANSPOSE_BLOCK; ++i) {
                srcList[i] = srcAddr + (rowBlock + i) * srcRowStride * sizeof(transType);
                if constexpr (sizeof(transType) == sizeof(float)) {
                    dstList[i] = dstAddr +
                        (rowBlock + (i / 2) * dstRowStride + (i % 2) * elementsPerBlock) * sizeof(transType);
                } else {
                    dstList[i] = dstAddr + (rowBlock + i * dstRowStride) * sizeof(transType);
                }
            }
            TransDataTo5HD<transType>(dstList, srcList, transposeParams);
        }
    }

    __aicore__ inline void TransposeKFirstToVFirst(const LocalTensor<stateType> &dst,
                                                   const LocalTensor<stateType> &src,
                                                   uint32_t curSingleV)
    {
        if constexpr (std::is_same<stateType, float32_t>()) {
            TransposeStateMatrixTyped<float>(dst, src, alignK_, curSingleV, vStep_, alignK_);
        } else {
            TransposeStateMatrixTyped<uint16_t>(
                dst.template ReinterpretCast<uint16_t>(), src.template ReinterpretCast<uint16_t>(),
                alignK_, curSingleV, vStep_, alignK_);
        }
    }

    __aicore__ inline void TransposeVFirstToKFirst(const LocalTensor<stateType> &dst,
                                                   const LocalTensor<stateType> &src,
                                                   uint32_t curSingleV)
    {
        if constexpr (std::is_same<stateType, float32_t>()) {
            TransposeStateMatrixTyped<float>(dst, src, curSingleV, alignK_, alignK_, vStep_);
        } else {
            TransposeStateMatrixTyped<uint16_t>(
                dst.template ReinterpretCast<uint16_t>(), src.template ReinterpretCast<uint16_t>(),
                curSingleV, alignK_, alignK_, vStep_);
        }
    }

    __aicore__ inline void PrefetchState(uint64_t stateSlot, uint64_t head, uint64_t vOffset,
                                          uint32_t curSingleV)
    {
        LocalTensor<stateType> stateLocal = stateInQueue_.AllocTensor<stateType>();
        if (stateVFirst_) {
            uint64_t stateOffset =
                stateInStride0_ * stateSlot + stateInStride1_ * head + stateInStride2_ * vOffset;
            DataCopyExtParams stateInParams{static_cast<uint16_t>(curSingleV),
                                            static_cast<uint16_t>(realK_ * sizeof(stateType)), 0, 0, 0};
            DataCopyPadExtParams<stateType> padParams{true, 0, static_cast<uint8_t>(alignK_ - realK_), 0};
            DataCopyPad(stateLocal, initStateGm_[stateOffset], stateInParams, padParams);
        } else {
            uint64_t stateOffset = stateInStride0_ * stateSlot + stateInStride1_ * head +
                                   stateInStride3_ * vOffset;
            int64_t srcStride = static_cast<int64_t>(
                (stateInStride2_ - curSingleV) * sizeof(stateType));
            int64_t dstStride = static_cast<int64_t>(
                (vStep_ - curSingleV) * sizeof(stateType) / DATA_BLOCK_BYTES);
            DataCopyExtParams stateInParams{static_cast<uint16_t>(realK_),
                                            static_cast<uint32_t>(curSingleV * sizeof(stateType)),
                                            srcStride, dstStride, 0};
            DataCopyPadExtParams<stateType> padParams{false, 0, 0, 0};
            DataCopyPad(stateLocal, initStateGm_[stateOffset], stateInParams, padParams);
        }
        stateInQueue_.EnQue<stateType>(stateLocal);
    }

    __aicore__ inline void LoadPrefetchedState(uint32_t curSingleV)
    {
        LocalTensor<stateType> stateLocal = stateInQueue_.DeQue<stateType>();
        if (stateVFirst_) {
            if constexpr (std::is_same<stateType, float32_t>()) {
                DataCopy(stateInUb, stateLocal, alignK_ * curSingleV);
            } else {
                UnpackRowsVF<stateType, 128>(reinterpret_cast<__ubuf__ float *>(stateInUb.GetPhyAddr()),
                                         reinterpret_cast<__ubuf__ stateType *>(stateLocal.GetPhyAddr()),
                                         curSingleV);
            }
        } else {
            LocalTensor<stateType> stateTransposeLocal = stateTransposeBuf_.Get<stateType>();
            TransposeKFirstToVFirst(stateTransposeLocal, stateLocal, curSingleV);
            PipeBarrier<PIPE_V>();
            if constexpr (std::is_same<stateType, float32_t>()) {
                UnpackRowsVF<stateType, 128>(reinterpret_cast<__ubuf__ float *>(stateInUb.GetPhyAddr()),
                                         reinterpret_cast<__ubuf__ stateType *>(stateTransposeLocal.GetPhyAddr()),
                                         curSingleV);
            } else {
                UnpackRowsVF<stateType, 128>(reinterpret_cast<__ubuf__ float *>(stateInUb.GetPhyAddr()),
                                         reinterpret_cast<__ubuf__ stateType *>(stateTransposeLocal.GetPhyAddr()),
                                         curSingleV);
            }
        }
        stateInQueue_.FreeTensor(stateLocal);
    }

    __aicore__ inline void DecayMatVecReduce128(LocalTensor<float> &stateTensor,
                                                const LocalTensor<float> &gateTensor,
                                                const LocalTensor<float> &vecTensor,
                                                LocalTensor<float> &dstTensor, uint32_t rows)
    {
        __ubuf__ float *stateAddr = (__ubuf__ float *)stateTensor.GetPhyAddr();
        __ubuf__ float *gateAddr = (__ubuf__ float *)gateTensor.GetPhyAddr();
        __ubuf__ float *vecAddr = (__ubuf__ float *)vecTensor.GetPhyAddr();
        __ubuf__ float *dstAddr = (__ubuf__ float *)dstTensor.GetPhyAddr();
        uint16_t rowNum = static_cast<uint16_t>(rows);
        DecayMatVecReduce128Regbase(stateAddr, gateAddr, vecAddr, dstAddr, rowNum, alignK_);
    }

    template <bool useSigmoid, bool allowNeg>
    __aicore__ inline void ProcessDeltaKQReduce128(const LocalTensor<float> &dotTensor,
                                                   const LocalTensor<float> &vTensor,
                                                   const LocalTensor<float> &betaTensor,
                                                   uint64_t betaOffset,
                                                   const LocalTensor<float> &kTensor,
                                                   LocalTensor<float> &stateTensor,
                                                   const LocalTensor<float> &qTensor,
                                                   LocalTensor<float> &attnTensor, uint32_t rows)
    {
        __ubuf__ float *dotAddr = (__ubuf__ float *)dotTensor.GetPhyAddr();
        __ubuf__ float *vAddr = (__ubuf__ float *)vTensor.GetPhyAddr();
        uint64_t betaRowStride =
            betaDtype_ == 0 ? FP32_NUM_PER_BLOCK : BF16_NUM_PER_BLOCK;
        __ubuf__ float *betaAddr = (__ubuf__ float *)betaTensor.GetPhyAddr() + betaOffset * betaRowStride;
        __ubuf__ float *kAddr = (__ubuf__ float *)kTensor.GetPhyAddr();
        __ubuf__ float *stateAddr = (__ubuf__ float *)stateTensor.GetPhyAddr();
        __ubuf__ float *qAddr = (__ubuf__ float *)qTensor.GetPhyAddr();
        __ubuf__ float *attnAddr = (__ubuf__ float *)attnTensor.GetPhyAddr();
        uint16_t rowNum = static_cast<uint16_t>(rows);
        ProcessDeltaKQReduce128Regbase<useSigmoid, allowNeg>(dotAddr, vAddr, betaAddr, kAddr, stateAddr, qAddr, attnAddr, rowNum, alignK_);
    }

    __aicore__ inline void ProcessDeltaKQReduce128Dispatch(const LocalTensor<float> &dotTensor,
                                                           const LocalTensor<float> &vTensor,
                                                           const LocalTensor<float> &betaTensor,
                                                           uint64_t betaOffset,
                                                           const LocalTensor<float> &kTensor,
                                                           LocalTensor<float> &stateTensor,
                                                           const LocalTensor<float> &qTensor,
                                                           LocalTensor<float> &attnTensor, uint32_t rows)
    {
        if (useBetaSigmoid_) {
            if (allowNegEigval_) {
                ProcessDeltaKQReduce128<true, true>(
                    dotTensor, vTensor, betaTensor, betaOffset, kTensor,
                    stateTensor, qTensor, attnTensor, rows);
            } else {
                ProcessDeltaKQReduce128<true, false>(
                    dotTensor, vTensor, betaTensor, betaOffset, kTensor,
                    stateTensor, qTensor, attnTensor, rows);
            }
        } else {
            ProcessDeltaKQReduce128<false, false>(
                dotTensor, vTensor, betaTensor, betaOffset, kTensor,
                stateTensor, qTensor, attnTensor, rows);
        }
    }

    __aicore__ inline void Compute(uint32_t curSingleV, uint64_t curQKOffset,
                                   uint64_t curVOffset, uint64_t betaOffset, uint32_t slot)
    {
        // K 恒为 128：decay/S@k 与 delta/outer/S@q 都走 regbase 融合 VF。
        DecayMatVecReduce128(stateInUb, GateSlot(slot)[curQKOffset],
                             KSlot(slot)[curQKOffset], deltaInUb, curSingleV);
        AscendC::PipeBarrier<PIPE_V>();
        ProcessDeltaKQReduce128Dispatch(deltaInUb, VSlot(slot)[curVOffset],
                                        BetaSlot(slot), betaOffset, KSlot(slot)[curQKOffset], stateInUb,
                                        QSlot(slot)[curQKOffset], attnInUb, curSingleV);

        LocalTensor<outType> attnOutLocal = attnOutQueue_.AllocTensor<outType>();
        if (shouldStoreState_) {
            LocalTensor<stateType> stateOutLocal = stateOutQueue_.AllocTensor<stateType>();
            if constexpr (std::is_same<stateType, float32_t>()) {
                DataCopy(stateOutLocal, stateInUb, alignK_ * curSingleV);
            } else {
                PackRowsVF<stateType, 128>(reinterpret_cast<__ubuf__ stateType *>(stateOutLocal.GetPhyAddr()),
                                           reinterpret_cast<__ubuf__ float *>(stateInUb.GetPhyAddr()),
                                           curSingleV);
            }
            stateOutQueue_.EnQue<stateType>(stateOutLocal);
        }
        PackFlatVF<outType>(reinterpret_cast<__ubuf__ outType *>(attnOutLocal.GetPhyAddr()),
                            reinterpret_cast<__ubuf__ float *>(attnInUb.GetPhyAddr()), curSingleV);
        attnOutQueue_.EnQue<outType>(attnOutLocal);
    }

    __aicore__ inline void CopyOutAttn(uint64_t attnOffset, uint32_t curSingleV)
    {
        LocalTensor<outType> attnLocal = attnOutQueue_.DeQue<outType>();
        DataCopyParams attnOutParams{1, static_cast<uint16_t>(curSingleV * sizeof(outType)), 0, 0};
        DataCopyPad(attnOutGm_[attnOffset], attnLocal, attnOutParams);
        attnOutQueue_.FreeTensor(attnLocal);
    }

    __aicore__ inline void CopyOutState(uint64_t stateSlot, uint64_t head, uint64_t vOffset,
                                         uint32_t curSingleV)
    {
        LocalTensor<stateType> stateOutLocal = stateOutQueue_.DeQue<stateType>();
        if (stateVFirst_) {
            uint64_t stateOffset =
                stateOutStride0_ * stateSlot + stateOutStride1_ * head + stateOutStride2_ * vOffset;
            DataCopyParams stateOutParams{static_cast<uint16_t>(curSingleV),
                                          static_cast<uint16_t>(realK_ * sizeof(stateType)), 0, 0};
            DataCopyPad(finalStateGm_[stateOffset], stateOutLocal, stateOutParams);
        } else {
            LocalTensor<stateType> stateTransposeLocal = stateTransposeBuf_.Get<stateType>();
            TransposeVFirstToKFirst(stateTransposeLocal, stateOutLocal, curSingleV);
            SyncVToMte3();
            uint64_t stateOffset = stateOutStride0_ * stateSlot + stateOutStride1_ * head +
                                   stateOutStride3_ * vOffset;
            int64_t srcStride = static_cast<int64_t>(
                (vStep_ - curSingleV) * sizeof(stateType) / DATA_BLOCK_BYTES);
            int64_t dstStride = static_cast<int64_t>(
                (stateOutStride2_ - curSingleV) * sizeof(stateType));
            DataCopyExtParams stateOutParams{static_cast<uint16_t>(realK_),
                                             static_cast<uint32_t>(curSingleV * sizeof(stateType)),
                                             srcStride, dstStride, 0};
            DataCopyPad(finalStateGm_[stateOffset], stateTransposeLocal, stateOutParams);
            SyncMte3ToV();
        }
        stateOutQueue_.FreeTensor(stateOutLocal);
    }

    __aicore__ inline LocalTensor<float> &BetaSlot(uint32_t slot)
    {
        return slot == 0 ? betaInUb : betaInUb2;
    }
    __aicore__ inline LocalTensor<float> &QSlot(uint32_t slot)
    {
        return slot == 0 ? qInUb : qInUb2;
    }
    __aicore__ inline LocalTensor<float> &KSlot(uint32_t slot)
    {
        return slot == 0 ? kInUb : kInUb2;
    }
    __aicore__ inline LocalTensor<float> &VSlot(uint32_t slot)
    {
        return slot == 0 ? vInUb : vInUb2;
    }
    __aicore__ inline LocalTensor<float> &GateSlot(uint32_t slot)
    {
        return slot == 0 ? gateInUb : gateInUb2;
    }

    template <typename betaType>
    __aicore__ inline void CopyInBetaTyped(int64_t seq0, int64_t seq1, uint64_t head, uint32_t slot)
    {
        int64_t seqLen = seq1 - seq0;
        constexpr uint64_t betaRowStride = 32 / sizeof(betaType);
        uint64_t betaCount = static_cast<uint64_t>(seqLen) * betaRowStride;
        LocalTensor<betaType> betaLocal = betaInQueue_.AllocTensor<betaType>();
        DataCopyExtParams betaInParams{
            static_cast<uint16_t>(seqLen), static_cast<uint32_t>(sizeof(betaType)),
            static_cast<uint32_t>((NV_ - 1) * sizeof(betaType)), 0, 0};
        DataCopyPadExtParams<betaType> betaPadParams{
            true, 0, static_cast<uint8_t>(betaRowStride - 1), static_cast<betaType>(0)};
        uint64_t betaOffset = static_cast<uint64_t>(seq0) * NV_ + head;
        if constexpr (std::is_same<betaType, float32_t>()) {
            DataCopyPad(betaLocal, betaFloatGm_[betaOffset], betaInParams, betaPadParams);
        } else if constexpr (std::is_same<betaType, bfloat16_t>()) {
            DataCopyPad(betaLocal, betaBf16Gm_[betaOffset], betaInParams, betaPadParams);
        } else {
            DataCopyPad(betaLocal, betaFp16Gm_[betaOffset], betaInParams, betaPadParams);
        }
        betaInQueue_.EnQue<betaType>(betaLocal);
        betaLocal = betaInQueue_.DeQue<betaType>();
        UnpackFlatVF<betaType>(reinterpret_cast<__ubuf__ float *>(BetaSlot(slot).GetPhyAddr()),
                               reinterpret_cast<__ubuf__ betaType *>(betaLocal.GetPhyAddr()),
                               static_cast<uint32_t>(betaCount));
        betaInQueue_.FreeTensor(betaLocal);
    }

    __aicore__ inline void CopyInBeta(int64_t seq0, int64_t seq1, uint64_t head, uint32_t slot)
    {
        if (betaDtype_ == 0) {
            CopyInBetaTyped<float>(seq0, seq1, head, slot);
        } else if (betaDtype_ == 1) {
            CopyInBetaTyped<bfloat16_t>(seq0, seq1, head, slot);
        } else {
            CopyInBetaTyped<half>(seq0, seq1, head, slot);
        }
    }

    // ===== 流水线拆分 =====
    // 异步发起 MTE2 DataCopyPad + EnQue（不 DeQue，MTE2 在后台搬）。
    // 紧接着用 CopyInBetaEnd 阻塞完成 + VF UnpackFlatVF 到目标 slot UB。
    template <typename betaType>
    __aicore__ inline void CopyInBetaStartTyped(int64_t seq0, int64_t seq1, uint64_t head)
    {
        int64_t seqLen = seq1 - seq0;
        constexpr uint64_t betaRowStride = 32 / sizeof(betaType);
        (void)betaRowStride;
        LocalTensor<betaType> betaLocal = betaInQueue_.AllocTensor<betaType>();
        DataCopyExtParams betaInParams{
            static_cast<uint16_t>(seqLen), static_cast<uint32_t>(sizeof(betaType)),
            static_cast<uint32_t>((NV_ - 1) * sizeof(betaType)), 0, 0};
        DataCopyPadExtParams<betaType> betaPadParams{
            true, 0, static_cast<uint8_t>(betaRowStride - 1), static_cast<betaType>(0)};
        uint64_t betaOffset = static_cast<uint64_t>(seq0) * NV_ + head;
        if constexpr (std::is_same<betaType, float32_t>()) {
            DataCopyPad(betaLocal, betaFloatGm_[betaOffset], betaInParams, betaPadParams);
        } else if constexpr (std::is_same<betaType, bfloat16_t>()) {
            DataCopyPad(betaLocal, betaBf16Gm_[betaOffset], betaInParams, betaPadParams);
        } else {
            DataCopyPad(betaLocal, betaFp16Gm_[betaOffset], betaInParams, betaPadParams);
        }
        betaInQueue_.EnQue<betaType>(betaLocal);
    }

    __aicore__ inline void CopyInBetaStart(int64_t seq0, int64_t seq1, uint64_t head)
    {
        if (betaDtype_ == 0) {
            CopyInBetaStartTyped<float>(seq0, seq1, head);
        } else if (betaDtype_ == 1) {
            CopyInBetaStartTyped<bfloat16_t>(seq0, seq1, head);
        } else {
            CopyInBetaStartTyped<half>(seq0, seq1, head);
        }
    }

    template <typename betaType>
    __aicore__ inline void CopyInBetaEndTyped(int64_t seq1, int64_t seq0, uint32_t slot)
    {
        int64_t seqLen = seq1 - seq0;
        constexpr uint64_t betaRowStride = 32 / sizeof(betaType);
        uint64_t betaCount = static_cast<uint64_t>(seqLen) * betaRowStride;
        LocalTensor<betaType> betaLocal = betaInQueue_.DeQue<betaType>();
        UnpackFlatVF<betaType>(reinterpret_cast<__ubuf__ float *>(BetaSlot(slot).GetPhyAddr()),
                               reinterpret_cast<__ubuf__ betaType *>(betaLocal.GetPhyAddr()),
                               static_cast<uint32_t>(betaCount));
        betaInQueue_.FreeTensor(betaLocal);
    }

    __aicore__ inline void CopyInBetaEnd(int64_t seq1, int64_t seq0, uint32_t slot)
    {
        if (betaDtype_ == 0) {
            CopyInBetaEndTyped<float>(seq1, seq0, slot);
        } else if (betaDtype_ == 1) {
            CopyInBetaEndTyped<bfloat16_t>(seq1, seq0, slot);
        } else {
            CopyInBetaEndTyped<half>(seq1, seq0, slot);
        }
    }

    __aicore__ inline uint64_t StateSlotForToken(uint64_t batchIdx, int64_t seq0, int64_t tokenIdx) const
    {
        if (hasSsmStateIndices_) {
            return LoadStateSlot(batchIdx, seq0, tokenIdx);
        }
        return batchIdx;
    }

    __aicore__ inline bool ProcessHead(uint64_t batchIdx, int64_t seq0, int64_t seq1,
                                      uint64_t head_i, uint64_t stateSlot, bool statePrefetched,
                                      bool hasNextTask, uint64_t nextStateSlot, uint64_t nextHead)
    {
        if (realV_ == 0) {
            return false;
        }
        PrepareGateParams(head_i);
        uint64_t nextVOffset = statePrefetched ? vStep_ : 0;
        uint32_t queuedBufferNum = statePrefetched ? 1 : 0;
        for (uint32_t bufferIdx = queuedBufferNum;
             bufferIdx < INPUT_BUFFER_NUM && nextVOffset < realV_; ++bufferIdx) {
            uint32_t nextSingleV =
                nextVOffset + vStep_ > realV_ ? realV_ - nextVOffset : vStep_;
            PrefetchState(stateSlot, head_i, nextVOffset, nextSingleV);
            nextVOffset += vStep_;
        }
        bool nextStatePrefetched = false;
        for (uint64_t v_i = 0; v_i < realV_; v_i += vStep_) {
            uint32_t curSingleV = v_i + vStep_ > realV_ ? realV_ - v_i : vStep_;
            LoadPrefetchedState(curSingleV);
            if (nextVOffset < realV_) {
                uint32_t nextSingleV =
                    nextVOffset + vStep_ > realV_ ? realV_ - nextVOffset : vStep_;
                PrefetchState(stateSlot, head_i, nextVOffset, nextSingleV);
                nextVOffset += vStep_;
            } else if (hasNextTask && !nextStatePrefetched) {
                uint32_t nextSingleV = realV_ > vStep_ ? vStep_ : realV_;
                PrefetchState(nextStateSlot, nextHead, 0, nextSingleV);
                nextStatePrefetched = true;
            }
            uint64_t pendingAttnOffset = 0;
            uint64_t pendingStateSlot = 0;
            bool hasPendingAttn = false;
            bool hasPendingState = false;
            // 段长超过输入窗口时按窗口顺序递推：state 在同一个 v-tile 内跨窗口原地续接，
            // 只有 Q/K/V/gate/beta 的窗口缓冲被重新搬入。双槽 ping-pong 切 slot，为后续
            // 真流水重叠做准备（当前先做同步基线验证）。
            int64_t windowStart = seq0;
            uint32_t slot = 0;
            while (windowStart < seq1) {
                const int64_t remaining = seq1 - windowStart;
                const int32_t windowLen = static_cast<int32_t>(
                    remaining < static_cast<int64_t>(MAX_MTP) ? remaining : static_cast<int64_t>(MAX_MTP));
                const int64_t windowEnd = windowStart + windowLen;
                CopyInBeta(windowStart, windowEnd, head_i, slot);
                CopyInQKVGate(windowStart, windowLen, head_i, slot);
                for (int64_t seq_i = windowStart; seq_i < windowEnd; seq_i++) {
                    const uint64_t localIdx = static_cast<uint64_t>(seq_i - windowStart);
                    uint64_t betaOffset = localIdx;
                    uint64_t curQKOffset = localIdx * alignK_;
                    uint64_t curVOffset = localIdx * alignV_ + v_i;
                    uint64_t attnOffset = (static_cast<uint64_t>(seq_i) * NV_ + head_i) * realV_ + v_i;
                    uint64_t curStateSlot = StateSlotForToken(batchIdx, seq0, seq_i);
                    uint64_t curStateOutSlot = curStateSlot;
                    Compute(curSingleV, curQKOffset, curVOffset, betaOffset, slot);
                    if (attnOutBufferNum_ == BUFFER_NUM) {
                        CopyOutAttn(attnOffset, curSingleV);
                    } else {
                        if (hasPendingAttn) {
                            CopyOutAttn(pendingAttnOffset, curSingleV);
                        }
                        pendingAttnOffset = attnOffset;
                        hasPendingAttn = true;
                    }
                    if (shouldStoreState_) {
                        if (stateOutBufferNum_ == BUFFER_NUM) {
                            CopyOutState(curStateOutSlot, head_i, v_i, curSingleV);
                        } else {
                            if (hasPendingState) {
                                CopyOutState(pendingStateSlot, head_i, v_i, curSingleV);
                            }
                            pendingStateSlot = curStateOutSlot;
                            hasPendingState = true;
                        }
                    }
                }
                windowStart = windowEnd;
                slot ^= 1;
            }
            if (hasPendingAttn) {
                CopyOutAttn(pendingAttnOffset, curSingleV);
            }
            if (hasPendingState) {
                CopyOutState(pendingStateSlot, head_i, v_i, curSingleV);
            }
        }
        return nextStatePrefetched;
     }

private:
    GlobalTensor<inType> queryGm_;
    GlobalTensor<inType> keyGm_;
    GlobalTensor<inType> valueGm_;
    GlobalTensor<float> gateFloatGm_;
    GlobalTensor<bfloat16_t> gateBf16Gm_;
    GlobalTensor<half> gateFp16Gm_;
    GlobalTensor<float> betaFloatGm_;
    GlobalTensor<bfloat16_t> betaBf16Gm_;
    GlobalTensor<half> betaFp16Gm_;
    GlobalTensor<stateType> initStateGm_;
    GlobalTensor<int32_t> cuSeqlensInt32Gm_;
    GlobalTensor<int64_t> cuSeqlensInt64Gm_;
    GlobalTensor<int32_t> ssmStateIndicesInt32Gm_;
    GlobalTensor<int64_t> ssmStateIndicesInt64Gm_;
    GlobalTensor<float> aLogGm_;
    GlobalTensor<float> dtBiasGm_;
    GlobalTensor<int32_t> numAcceptedTokensInt32Gm_;
    GlobalTensor<int64_t> numAcceptedTokensInt64Gm_;
    GlobalTensor<stateType> finalStateGm_;
    GlobalTensor<outType> attnOutGm_;
    TPipe *pipe_;
    TQue<QuePosition::VECIN, 1> qInQueue_;
    TQue<QuePosition::VECIN, 1> kInQueue_;
    TQue<QuePosition::VECIN, 1> vInQueue_;
    TQue<QuePosition::VECIN, 1> gateInQueue_;
    TQue<QuePosition::VECIN, 1> betaInQueue_;
    TQue<QuePosition::VECIN, INPUT_BUFFER_NUM> stateInQueue_;
    TQue<QuePosition::VECOUT, MAX_OUT_BUFFER_NUM> attnOutQueue_;
    TQue<QuePosition::VECOUT, MAX_OUT_BUFFER_NUM> stateOutQueue_;
    TBuf<TPosition::VECCALC> tmpBuff;
    TBuf<TPosition::VECCALC> stateTransposeBuf_;
    TBuf<TPosition::VECCALC> scalarBuf_;
    LocalTensor<float> qInUb;
    LocalTensor<float> kInUb;
    LocalTensor<float> vInUb;
    LocalTensor<float> gateInUb;
    LocalTensor<float> betaInUb;
    // 窗口 ping-pong 双槽：slot 1 与 slot 0 同容量，让 MTE2 搬下一个窗口与 VF 消费当前窗口重叠。
    LocalTensor<float> qInUb2;
    LocalTensor<float> kInUb2;
    LocalTensor<float> vInUb2;
    LocalTensor<float> gateInUb2;
    LocalTensor<float> betaInUb2;
    LocalTensor<float> deltaInUb;
    LocalTensor<float> gateBiasInUb;
    LocalTensor<float> attnInUb;
    LocalTensor<float> stateInUb;
    float gateExpA_;
    TEventID eventIdMte2ToV_;
    TEventID eventIdVToMte2_;
    TEventID eventIdVToS_;
    TEventID eventIdVToMte3_;
    TEventID eventIdMte3ToV_;
    bool eventMte2ToVInitialized_;
    bool eventVToMte2Initialized_;
    bool eventVToSInitialized_;
    bool eventVToMte3Initialized_;
    bool eventMte3ToVInitialized_;
    uint32_t B_;
    uint32_t T_;
    uint32_t seqLen_;
    uint32_t NK_;
    uint32_t alignK_;
    uint32_t realK_;
    uint32_t NV_;
    uint32_t alignV_;
    uint32_t realV_;
    uint32_t stateCapacity_;
    uint32_t ssmStateStride_;
    uint64_t queryTokenStride_;
    uint64_t queryHeadStride_;
    uint64_t keyTokenStride_;
    uint64_t keyHeadStride_;
    uint64_t valueTokenStride_;
    uint64_t valueHeadStride_;
    uint64_t stateInStride0_;
    uint64_t stateInStride1_;
    uint64_t stateInStride2_;
    uint64_t stateInStride3_;
    uint64_t stateOutStride0_;
    uint64_t stateOutStride1_;
    uint64_t stateOutStride2_;
    uint64_t stateOutStride3_;
    uint32_t vStep_;
    uint32_t stateOutBufferNum_;
    uint32_t attnOutBufferNum_;
    uint32_t restUbSize_;
    uint32_t gateDtype_;
    uint32_t betaDtype_;
    uint32_t cuSeqlensDtype_;
    uint32_t ssmStateIndicesDtype_;
    uint32_t acceptedTokensDtype_;
    bool hasCuSeqlens_;
    bool hasSsmStateIndices_;
    bool hasAcceptedTokens_;
    bool hasALog_;
    bool hasDtBias_;
    bool useQkL2norm_;
    bool useGateInKernel_;
    bool useBetaSigmoid_;
    bool allowNegEigval_;
    bool safeGate_;
    bool stateVFirst_;
    bool shouldStoreState_;
    float scale_;
    float lowerBound_;
    uint64_t blockIdx;
};
} // namespace RecurrentKda
#endif
