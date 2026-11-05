/**
 * Copyright (c) 2025-2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file recurrent_kda.h
 * \brief Single-kernel fused recurrent KDA implementation.
 */

#ifndef __RECURRENT_KDA_KERNEL_ARCH92_H_
#define __RECURRENT_KDA_KERNEL_ARCH92_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "../recurrent_kda_tiling_data.h"

namespace RecurrentKda {

using namespace matmul;
using namespace AscendC;
using namespace AscendC::MicroAPI;
constexpr uint64_t BUFFER_NUM = 1;
constexpr uint64_t INPUT_BUFFER_NUM = 2;
constexpr uint32_t MAX_OUT_BUFFER_NUM = 2;
constexpr uint64_t MAX_MTP = 8;
constexpr uint64_t BF16_NUM_PER_BLOCK = 16;
constexpr uint64_t FP32_NUM_PER_BLOCK = 8;
constexpr uint32_t REPEAT_LENTH = 64; // 256 bytes for float.
constexpr uint32_t MAX_REPEAT_TIME = 255;
constexpr uint32_t ADD_FOLD_REDUCE_MIN_K = 128;
constexpr uint16_t V_LENGTH = VECTOR_REG_WIDTH / sizeof(float);
constexpr uint16_t TWO_V_LENGTH = 2 * V_LENGTH;
constexpr uint64_t INVALID_STATE_SLOT = static_cast<uint64_t>(-1);
constexpr uint32_t STATE_TRANSPOSE_BLOCK = 16;
constexpr uint32_t DATA_BLOCK_BYTES = 32;

#ifndef RKDA_ENABLE_ADD_FOLD_REDUCE
#define RKDA_ENABLE_ADD_FOLD_REDUCE  1
#endif

// ascend960dt（A6，DAV_9201）向量路径。
// 本文件只按 A6 编译，三发射 / 384KB UB 的优化直接生效，不提供回退 A5 的编译开关：
//   1. token 级融合 VF：一次 __VEC_SCOPE__ 内完成 decay / S@k / delta / outer / S@q，
//      去掉 "decayed state 回写 UB 再读回" 的一整趟状态搬运；
//   2. 行块（V 方向相邻 4 行相互独立）交织成多条短指令流，与 ReduceSum / Sqrt / Div
//      等长指令并行发射；
//   3. 短指令流内部用 MulAddDst 把 "两段乘加 + 累加" 折叠成一条指令（与 A5 通用路径的
//      RKDA_ENABLE_ADD_FOLD_REDUCE 同一写法），缩短归约依赖链；
//   4. 标量链（beta sigmoid、delta、归一化 rstd）收敛到 VL1 mask 上，不再按 64 lane 发射。
// 行块一次处理的行数：K 方向 128 列拆成两段 64 lane，4 行合计 8 条独立的短指令流。
constexpr uint32_t A6_ROW_BLOCK = 4;
// raw gate 变换的行块：每行两段 + 4 行 = 8 条短指令流，用于与 Exp / Ln / Div 长指令同拍。
constexpr uint32_t A6_GATE_ROW_BLOCK = 4;
// L2 归一化 epsilon：与 A2/A3、A5(arch35) 的 1e-6 口径一致，避免 A6 与 A5 出现数值分叉。
constexpr float RKDA_L2_NORM_EPSILON = 1.0e-6F;

struct RKDAInitParams {
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

template <typename inType, typename outType, typename stateType>
class RKDA {
public:
    __aicore__ inline explicit RKDA(const RecurrentKdaTilingData *tilingData)
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
        useAddFoldReduce_ = (RKDA_ENABLE_ADD_FOLD_REDUCE != 0);
        vStep_ = tilingData->vStep;
        stateOutBufferNum_ = (tilingData->stateOutBufferNum == MAX_OUT_BUFFER_NUM) ? MAX_OUT_BUFFER_NUM : BUFFER_NUM;
        attnOutBufferNum_ = (tilingData->attnOutBufferNum == MAX_OUT_BUFFER_NUM) ? MAX_OUT_BUFFER_NUM : BUFFER_NUM;
        restUbSize_ = tilingData->ubRestBytes;
        alignK_ = Ceil(tilingData->dk, BF16_NUM_PER_BLOCK) * BF16_NUM_PER_BLOCK;
        alignV_ = Ceil(tilingData->dv, BF16_NUM_PER_BLOCK) * BF16_NUM_PER_BLOCK;
        eventMte2ToVInitialized_ = false;
        eventVToMte2Initialized_ = false;
        eventVToSInitialized_ = false;
        eventVToMte3Initialized_ = false;
        eventMte3ToVInitialized_ = false;
    }

    __aicore__ inline void Init(const RKDAInitParams &initParams, TPipe *pipe)
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

    __aicore__ inline void SetGlobalTensors(const RKDAInitParams &initParams)
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
        deltaInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(vStep_), buffOffset);
        buffOffset += singleVSize;
        attnInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(vStep_), buffOffset);
        buffOffset += singleVSize;
        vInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(MAX_MTP * alignV_), buffOffset);
        buffOffset += vSize;
        qInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(MAX_MTP * alignK_), buffOffset);
        buffOffset += kSize;
        kInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(MAX_MTP * alignK_), buffOffset);
        buffOffset += kSize;
        stateInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(alignK_ * vStep_), buffOffset);
        buffOffset += cubeSize;
        if (alignK_ == TWO_V_LENGTH) {
            broadTmpInUb = stateInUb;
        } else {
            broadTmpInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(alignK_ * vStep_), buffOffset);
            buffOffset += cubeSize;
        }
        betaInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(betaUbSize / sizeof(float)), buffOffset);
        buffOffset += betaUbSize;
        gateInUb = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(MAX_MTP * alignK_), buffOffset);
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
            CopyInBeta(currentTask.seq0, currentTask.seq1, currentTask.head);
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
            int32_t seqLen = static_cast<int32_t>(seqLen64);
            if (!ValidateStateSlots(batch, seq0, seqLen)) {
                taskValid = false;
                return false;
            }
            uint64_t stateSlot = ResolveInitialStateSlot(batch, seq0, seqLen);
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

    __aicore__ inline bool ValidateCuSeqlens() const
    {
        if (!hasCuSeqlens_) {
            return seqLen_ <= MAX_MTP;
        }
        int64_t seq0 = LoadCuSeqlens(0);
        if (seq0 != 0) {
            return false;
        }
        for (uint64_t i = 0; i < B_; i++) {
            int64_t seq1 = LoadCuSeqlens(i + 1);
            int64_t length = seq1 - seq0;
            if (seq1 < seq0 || seq1 > static_cast<int64_t>(T_) ||
                length > static_cast<int64_t>(MAX_MTP) ||
                (hasSsmStateIndices_ && ssmStateStride_ > 0 && length > ssmStateStride_)) {
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

    __aicore__ inline bool ValidateStateSlots(uint64_t batchIdx, int64_t seq0, int32_t seqLen) const
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
        for (int32_t step = 0; step < seqLen; ++step) {
            if (LoadStateSlot(batchIdx, seq0, seq0 + step) == INVALID_STATE_SLOT) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline uint64_t ResolveInitialStateSlot(uint64_t batchIdx, int64_t seq0, int32_t seqLen) const
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
        Adds(scalar, scalar, 0.0f, 1);
        PipeBarrier<PIPE_V>();
        SyncVToS();
        __ubuf__ float *ptr = (__ubuf__ float *)scalar.GetPhyAddr();
        return ptr[0];
    }

    __aicore__ inline float ExpScalar(float x)
    {
        LocalTensor<float> scalar = scalarBuf_.Get<float>();
        Duplicate(scalar, x, 1);
        PipeBarrier<PIPE_V>();
        Exp(scalar, scalar, 1);
        PipeBarrier<PIPE_V>();
        SyncVToS();
        __ubuf__ float *ptr = (__ubuf__ float *)scalar.GetPhyAddr();
        return ptr[0];
    }

    __aicore__ inline float SigmoidScalar(float x)
    {
        float denom = 1.0f + ExpScalar(-x);
        return 1.0f / denom;
    }

    // A6：整块 L2 归一化收进单个 VF。
    // 每行 128 列拆成前后两段 64 lane，按 4 行一组（V 方向相邻行相互独立）交织：
    // 4 行各自的 "两段平方 + 相加" 短指令填满两条短指令槽，4 条 ReduceSum（长指令）与之并行；
    // rstd 的 Adds / Sqrt / Div 只占 VL1，随后 Duplicate 在寄存器内广播，
    // 不再走 A5 实现里的 "UB 标量回写 + V->S 同步 + 标量读回"。
    // epsilon 取 1e-6，与 A2/A3、A5(arch35) 的归一化口径一致。
    __aicore__ inline void NormalizeRowsVF128(LocalTensor<float> &tensor, int32_t seqLen)
    {
        __ubuf__ float *addr = (__ubuf__ float *)tensor.GetPhyAddr();
        uint16_t rowNum = static_cast<uint16_t>(seqLen);
        uint16_t blockNum = static_cast<uint16_t>(rowNum / A6_ROW_BLOCK);
        __VEC_SCOPE__
        {
            RegTensor<float> r00;
            RegTensor<float> r01;
            RegTensor<float> r10;
            RegTensor<float> r11;
            RegTensor<float> r20;
            RegTensor<float> r21;
            RegTensor<float> r30;
            RegTensor<float> r31;
            RegTensor<float> sq0;
            RegTensor<float> sq1;
            RegTensor<float> sq2;
            RegTensor<float> sq3;
            RegTensor<float> nd0;
            RegTensor<float> nd1;
            RegTensor<float> nd2;
            RegTensor<float> nd3;
            RegTensor<float> rcp;
            RegTensor<float> one;
            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregFirst = CreateMask<float, MaskPattern::VL1>();
            Duplicate(one, 1.0f, pregFirst);
            for (uint16_t i = 0; i < blockNum; ++i) {
                uint32_t rowOffset = static_cast<uint32_t>(i) * A6_ROW_BLOCK * alignK_;
                DataCopy(r00, addr + rowOffset);
                DataCopy(r01, addr + rowOffset + V_LENGTH);
                DataCopy(r10, addr + rowOffset + alignK_);
                DataCopy(r11, addr + rowOffset + alignK_ + V_LENGTH);
                DataCopy(r20, addr + rowOffset + 2 * alignK_);
                DataCopy(r21, addr + rowOffset + 2 * alignK_ + V_LENGTH);
                DataCopy(r30, addr + rowOffset + 3 * alignK_);
                DataCopy(r31, addr + rowOffset + 3 * alignK_ + V_LENGTH);
                // 4 行 × 两段：Mul + MulAddDst 把"平方 + 相加"折叠成两条短指令，
                // 既减少短指令数，也把 "Mul->Add" 的两跳依赖压成一条 FMA
                // （与 token 融合 VF 的归约折叠同一口径，单次舍入差 ≤1 ULP）。
                Mul(sq0, r00, r00, pregFull);
                MulAddDst(sq0, r01, r01, pregFull);
                Mul(sq1, r10, r10, pregFull);
                MulAddDst(sq1, r11, r11, pregFull);
                Mul(sq2, r20, r20, pregFull);
                MulAddDst(sq2, r21, r21, pregFull);
                Mul(sq3, r30, r30, pregFull);
                MulAddDst(sq3, r31, r31, pregFull);
                // 4 条长指令：4 行的整行归约
                ReduceSum(nd0, sq0, pregFull);
                ReduceSum(nd1, sq1, pregFull);
                ReduceSum(nd2, sq2, pregFull);
                ReduceSum(nd3, sq3, pregFull);
                Adds(nd0, nd0, RKDA_L2_NORM_EPSILON, pregFirst);
                Adds(nd1, nd1, RKDA_L2_NORM_EPSILON, pregFirst);
                Adds(nd2, nd2, RKDA_L2_NORM_EPSILON, pregFirst);
                Adds(nd3, nd3, RKDA_L2_NORM_EPSILON, pregFirst);
                Sqrt(nd0, nd0, pregFirst);
                Sqrt(nd1, nd1, pregFirst);
                Sqrt(nd2, nd2, pregFirst);
                Sqrt(nd3, nd3, pregFirst);
                Div(rcp, one, nd0, pregFirst);
                Duplicate(nd0, rcp, pregFull);
                Div(rcp, one, nd1, pregFirst);
                Duplicate(nd1, rcp, pregFull);
                Div(rcp, one, nd2, pregFirst);
                Duplicate(nd2, rcp, pregFull);
                Div(rcp, one, nd3, pregFirst);
                Duplicate(nd3, rcp, pregFull);
                Mul(r00, r00, nd0, pregFull);
                Mul(r01, r01, nd0, pregFull);
                Mul(r10, r10, nd1, pregFull);
                Mul(r11, r11, nd1, pregFull);
                Mul(r20, r20, nd2, pregFull);
                Mul(r21, r21, nd2, pregFull);
                Mul(r30, r30, nd3, pregFull);
                Mul(r31, r31, nd3, pregFull);
                DataCopy(addr + rowOffset, r00, pregFull);
                DataCopy(addr + rowOffset + V_LENGTH, r01, pregFull);
                DataCopy(addr + rowOffset + alignK_, r10, pregFull);
                DataCopy(addr + rowOffset + alignK_ + V_LENGTH, r11, pregFull);
                DataCopy(addr + rowOffset + 2 * alignK_, r20, pregFull);
                DataCopy(addr + rowOffset + 2 * alignK_ + V_LENGTH, r21, pregFull);
                DataCopy(addr + rowOffset + 3 * alignK_, r30, pregFull);
                DataCopy(addr + rowOffset + 3 * alignK_ + V_LENGTH, r31, pregFull);
            }
            // 尾部 0~3 行：单行完成，保持与行块相同的数值路径。
            for (uint16_t row = static_cast<uint16_t>(blockNum * A6_ROW_BLOCK); row < rowNum; ++row) {
                uint32_t rowOffset = static_cast<uint32_t>(row) * alignK_;
                DataCopy(r00, addr + rowOffset);
                DataCopy(r01, addr + rowOffset + V_LENGTH);
                Mul(sq0, r00, r00, pregFull);
                MulAddDst(sq0, r01, r01, pregFull);
                ReduceSum(nd0, sq0, pregFull);
                Adds(nd0, nd0, RKDA_L2_NORM_EPSILON, pregFirst);
                Sqrt(nd0, nd0, pregFirst);
                Div(rcp, one, nd0, pregFirst);
                Duplicate(nd0, rcp, pregFull);
                Mul(r00, r00, nd0, pregFull);
                Mul(r01, r01, nd0, pregFull);
                DataCopy(addr + rowOffset, r00, pregFull);
                DataCopy(addr + rowOffset + V_LENGTH, r01, pregFull);
            }
        }
        // VF 内的 UB 回写与后续普通 V 指令之间保留一次 pipe barrier（与原逐行实现末尾的 barrier 等价）。
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void NormalizeRows(LocalTensor<float> &tensor, int32_t seqLen)
    {
        if (alignK_ == TWO_V_LENGTH) {
            NormalizeRowsVF128(tensor, seqLen);
            return;
        }
        for (int32_t row = 0; row < seqLen; ++row) {
            uint32_t rowOffset = static_cast<uint32_t>(row) * alignK_;
            Mul(broadTmpInUb, tensor[rowOffset], tensor[rowOffset], alignK_);
            PipeBarrier<PIPE_V>();
            ReduceSumDispatch(deltaInUb, broadTmpInUb, 1);
            PipeBarrier<PIPE_V>();
            Sqrt(deltaInUb, deltaInUb, 1);
            PipeBarrier<PIPE_V>();
            SyncVToS();
            float norm = deltaInUb.GetValue(0);
            if (norm > 0.0f) {
                Muls(tensor[rowOffset], tensor[rowOffset], 1.0f / norm, alignK_);
                PipeBarrier<PIPE_V>();
            }
        }
    }

    // A6：整块 gate 变换收进单个 VF。
    // 每行的两段 64 lane 相互独立，再按 4 行一组交织成 8 条短指令流：
    // Exp / Ln / Div 这类长指令只占"长指令槽"，另外 3 行同阶段的 Muls / Adds / DataCopy
    // 短指令填满每拍的 2 个短指令槽，构成 960dt 的"1 长 + 2 短"三发形态。
    // 同时去掉 A5 实现里逐个原子操作之间的 PipeBarrier 以及逐行 dt_bias 加法后的 barrier。
    __aicore__ inline void ApplyGateInKernelVF(float expA, int32_t seqLen)
    {
        __ubuf__ float *gateAddr = (__ubuf__ float *)gateInUb.GetPhyAddr();
        __ubuf__ float *biasAddr = (__ubuf__ float *)broadTmpInUb.GetPhyAddr();
        uint16_t rowNum = static_cast<uint16_t>(seqLen);
        uint16_t quadNum = static_cast<uint16_t>(rowNum / A6_GATE_ROW_BLOCK);
        __VEC_SCOPE__
        {
            RegTensor<float> x00;
            RegTensor<float> x01;
            RegTensor<float> x10;
            RegTensor<float> x11;
            RegTensor<float> x20;
            RegTensor<float> x21;
            RegTensor<float> x30;
            RegTensor<float> x31;
            RegTensor<float> bias0;
            RegTensor<float> bias1;
            RegTensor<float> one;
            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            Duplicate(one, 1.0f, pregFull);
            if (hasDtBias_) {
                DataCopy(bias0, biasAddr);
                DataCopy(bias1, biasAddr + V_LENGTH);
            }
            for (uint16_t i = 0; i < quadNum; ++i) {
                uint32_t rowOffset = static_cast<uint32_t>(i) * A6_GATE_ROW_BLOCK * alignK_;
                DataCopy(x00, gateAddr + rowOffset);
                DataCopy(x01, gateAddr + rowOffset + V_LENGTH);
                DataCopy(x10, gateAddr + rowOffset + alignK_);
                DataCopy(x11, gateAddr + rowOffset + alignK_ + V_LENGTH);
                DataCopy(x20, gateAddr + rowOffset + 2 * alignK_);
                DataCopy(x21, gateAddr + rowOffset + 2 * alignK_ + V_LENGTH);
                DataCopy(x30, gateAddr + rowOffset + 3 * alignK_);
                DataCopy(x31, gateAddr + rowOffset + 3 * alignK_ + V_LENGTH);
                if (hasDtBias_) {
                    Add(x00, x00, bias0, pregFull);
                    Add(x01, x01, bias1, pregFull);
                    Add(x10, x10, bias0, pregFull);
                    Add(x11, x11, bias1, pregFull);
                    Add(x20, x20, bias0, pregFull);
                    Add(x21, x21, bias1, pregFull);
                    Add(x30, x30, bias0, pregFull);
                    Add(x31, x31, bias1, pregFull);
                }
                if (safeGate_) {
                    // 两波调度：wave0（行 0/1）先送 Exp，wave1（行 2/3）的 Muls 落在
                    // Exp 的长指令窗口里；Div 同理。避免 8 条长指令连成一串。
                    Muls(x00, x00, -expA, pregFull);
                    Muls(x01, x01, -expA, pregFull);
                    Muls(x10, x10, -expA, pregFull);
                    Muls(x11, x11, -expA, pregFull);
                    Exp(x00, x00, pregFull);
                    Exp(x01, x01, pregFull);
                    Exp(x10, x10, pregFull);
                    Exp(x11, x11, pregFull);

                    Muls(x20, x20, -expA, pregFull);
                    Muls(x21, x21, -expA, pregFull);
                    Muls(x30, x30, -expA, pregFull);
                    Muls(x31, x31, -expA, pregFull);
                    Exp(x20, x20, pregFull);
                    Exp(x21, x21, pregFull);
                    Exp(x30, x30, pregFull);
                    Exp(x31, x31, pregFull);

                    Adds(x00, x00, 1.0f, pregFull);
                    Adds(x01, x01, 1.0f, pregFull);
                    Adds(x10, x10, 1.0f, pregFull);
                    Adds(x11, x11, 1.0f, pregFull);
                    Div(x00, one, x00, pregFull);
                    Div(x01, one, x01, pregFull);
                    Div(x10, one, x10, pregFull);
                    Div(x11, one, x11, pregFull);

                    Adds(x20, x20, 1.0f, pregFull);
                    Adds(x21, x21, 1.0f, pregFull);
                    Adds(x30, x30, 1.0f, pregFull);
                    Adds(x31, x31, 1.0f, pregFull);
                    Div(x20, one, x20, pregFull);
                    Div(x21, one, x21, pregFull);
                    Div(x30, one, x30, pregFull);
                    Div(x31, one, x31, pregFull);

                    Muls(x00, x00, lowerBound_, pregFull);
                    Muls(x01, x01, lowerBound_, pregFull);
                    Muls(x10, x10, lowerBound_, pregFull);
                    Muls(x11, x11, lowerBound_, pregFull);
                    DataCopy(gateAddr + rowOffset, x00, pregFull);
                    DataCopy(gateAddr + rowOffset + V_LENGTH, x01, pregFull);
                    DataCopy(gateAddr + rowOffset + alignK_, x10, pregFull);
                    DataCopy(gateAddr + rowOffset + alignK_ + V_LENGTH, x11, pregFull);

                    Muls(x20, x20, lowerBound_, pregFull);
                    Muls(x21, x21, lowerBound_, pregFull);
                    Muls(x30, x30, lowerBound_, pregFull);
                    Muls(x31, x31, lowerBound_, pregFull);
                    DataCopy(gateAddr + rowOffset + 2 * alignK_, x20, pregFull);
                    DataCopy(gateAddr + rowOffset + 2 * alignK_ + V_LENGTH, x21, pregFull);
                    DataCopy(gateAddr + rowOffset + 3 * alignK_, x30, pregFull);
                    DataCopy(gateAddr + rowOffset + 3 * alignK_ + V_LENGTH, x31, pregFull);
                } else {
                    // 两波调度：wave0/wave1 的 Exp 先发，wave1 的 Adds/Ln 填 wave0 的长窗口。
                    Exp(x00, x00, pregFull);
                    Exp(x01, x01, pregFull);
                    Exp(x10, x10, pregFull);
                    Exp(x11, x11, pregFull);

                    Exp(x20, x20, pregFull);
                    Exp(x21, x21, pregFull);
                    Exp(x30, x30, pregFull);
                    Exp(x31, x31, pregFull);

                    Adds(x00, x00, 1.0f, pregFull);
                    Adds(x01, x01, 1.0f, pregFull);
                    Adds(x10, x10, 1.0f, pregFull);
                    Adds(x11, x11, 1.0f, pregFull);
                    Ln(x00, x00, pregFull);
                    Ln(x01, x01, pregFull);
                    Ln(x10, x10, pregFull);
                    Ln(x11, x11, pregFull);

                    Adds(x20, x20, 1.0f, pregFull);
                    Adds(x21, x21, 1.0f, pregFull);
                    Adds(x30, x30, 1.0f, pregFull);
                    Adds(x31, x31, 1.0f, pregFull);
                    Ln(x20, x20, pregFull);
                    Ln(x21, x21, pregFull);
                    Ln(x30, x30, pregFull);
                    Ln(x31, x31, pregFull);

                    Muls(x00, x00, -expA, pregFull);
                    Muls(x01, x01, -expA, pregFull);
                    Muls(x10, x10, -expA, pregFull);
                    Muls(x11, x11, -expA, pregFull);
                    DataCopy(gateAddr + rowOffset, x00, pregFull);
                    DataCopy(gateAddr + rowOffset + V_LENGTH, x01, pregFull);
                    DataCopy(gateAddr + rowOffset + alignK_, x10, pregFull);
                    DataCopy(gateAddr + rowOffset + alignK_ + V_LENGTH, x11, pregFull);

                    Muls(x20, x20, -expA, pregFull);
                    Muls(x21, x21, -expA, pregFull);
                    Muls(x30, x30, -expA, pregFull);
                    Muls(x31, x31, -expA, pregFull);
                    DataCopy(gateAddr + rowOffset + 2 * alignK_, x20, pregFull);
                    DataCopy(gateAddr + rowOffset + 2 * alignK_ + V_LENGTH, x21, pregFull);
                    DataCopy(gateAddr + rowOffset + 3 * alignK_, x30, pregFull);
                    DataCopy(gateAddr + rowOffset + 3 * alignK_ + V_LENGTH, x31, pregFull);
                }
            }
            // 尾部 0~3 行：单行完成，数值路径与行块一致。
            for (uint16_t row = static_cast<uint16_t>(quadNum * A6_GATE_ROW_BLOCK); row < rowNum; ++row) {
                uint32_t rowOffset = static_cast<uint32_t>(row) * alignK_;
                DataCopy(x00, gateAddr + rowOffset);
                DataCopy(x01, gateAddr + rowOffset + V_LENGTH);
                if (hasDtBias_) {
                    Add(x00, x00, bias0, pregFull);
                    Add(x01, x01, bias1, pregFull);
                }
                if (safeGate_) {
                    Muls(x00, x00, -expA, pregFull);
                    Muls(x01, x01, -expA, pregFull);
                    Exp(x00, x00, pregFull);
                    Exp(x01, x01, pregFull);
                    Adds(x00, x00, 1.0f, pregFull);
                    Adds(x01, x01, 1.0f, pregFull);
                    Div(x00, one, x00, pregFull);
                    Div(x01, one, x01, pregFull);
                    Muls(x00, x00, lowerBound_, pregFull);
                    Muls(x01, x01, lowerBound_, pregFull);
                } else {
                    Exp(x00, x00, pregFull);
                    Exp(x01, x01, pregFull);
                    Adds(x00, x00, 1.0f, pregFull);
                    Adds(x01, x01, 1.0f, pregFull);
                    Ln(x00, x00, pregFull);
                    Ln(x01, x01, pregFull);
                    Muls(x00, x00, -expA, pregFull);
                    Muls(x01, x01, -expA, pregFull);
                }
                DataCopy(gateAddr + rowOffset, x00, pregFull);
                DataCopy(gateAddr + rowOffset + V_LENGTH, x01, pregFull);
            }
        }
        // 整块 gate 只在 VF 末尾同步一次，替代 A5 实现中每个原子操作后的 PipeBarrier。
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ApplyGateInKernel(uint64_t head, int32_t seqLen)
    {
        uint32_t total = static_cast<uint32_t>(seqLen) * alignK_;
        float expA = hasALog_ ? ExpScalar(ReadFloat(aLogGm_, head)) : 1.0f;

        if (alignK_ == TWO_V_LENGTH && realK_ == alignK_) {
            if (hasDtBias_) {
                CopyVectorIn(broadTmpInUb, dtBiasGm_, head * realK_, realK_);
                SyncMte2ToV();
            }
            ApplyGateInKernelVF(expA, seqLen);
            return;
        }

        if (hasDtBias_) {
            CopyVectorIn(broadTmpInUb, dtBiasGm_, head * realK_, realK_);
            SyncMte2ToV();
            for (int32_t row = 0; row < seqLen; ++row) {
                Add(gateInUb[row * alignK_], gateInUb[row * alignK_], broadTmpInUb, alignK_);
                PipeBarrier<PIPE_V>();
            }
        }

        if (safeGate_) {
            Muls(gateInUb, gateInUb, expA, total);
            PipeBarrier<PIPE_V>();
            Muls(broadTmpInUb, gateInUb, -1.0f, total);
            PipeBarrier<PIPE_V>();
            Exp(broadTmpInUb, broadTmpInUb, total);
            PipeBarrier<PIPE_V>();
            Adds(broadTmpInUb, broadTmpInUb, 1.0f, total);
            PipeBarrier<PIPE_V>();
            Duplicate(gateInUb, 1.0f, total);
            PipeBarrier<PIPE_V>();
            Div(gateInUb, gateInUb, broadTmpInUb, total);
            PipeBarrier<PIPE_V>();
            Muls(gateInUb, gateInUb, lowerBound_, total);
            PipeBarrier<PIPE_V>();
        } else {
            Exp(gateInUb, gateInUb, total);
            PipeBarrier<PIPE_V>();
            Adds(gateInUb, gateInUb, 1.0f, total);
            PipeBarrier<PIPE_V>();
            Ln(gateInUb, gateInUb, total);
            PipeBarrier<PIPE_V>();
            Muls(gateInUb, gateInUb, -expA, total);
            PipeBarrier<PIPE_V>();
        }
    }

    template <typename gateType>
    __aicore__ inline void CopyInGate(uint64_t gateOffset, int32_t seqLen)
    {
        LocalTensor<gateType> gateLocal = gateInQueue_.AllocTensor<gateType>();
        Duplicate<gateType>(gateLocal, static_cast<gateType>(0), alignK_ * static_cast<uint32_t>(seqLen));
        SyncVToMte2();
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
        if constexpr (std::is_same<gateType, float32_t>()) {
            Adds(gateInUb, gateLocal, 0.0f, alignK_ * static_cast<uint32_t>(seqLen));
        } else {
            Cast(gateInUb, gateLocal, AscendC::RoundMode::CAST_NONE,
                 alignK_ * static_cast<uint32_t>(seqLen));
        }
        gateInQueue_.FreeTensor(gateLocal);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyInQKVGate(uint64_t qOffset, uint64_t kOffset, uint64_t vOffset,
                                         uint64_t gateOffset, int32_t seqLen, uint64_t head)
    {
        LocalTensor<inType> qLocal = qInQueue_.AllocTensor<inType>();
        LocalTensor<inType> kLocal = kInQueue_.AllocTensor<inType>();
        LocalTensor<inType> vLocal = vInQueue_.AllocTensor<inType>();

        DataCopyExtParams qInParams{static_cast<uint16_t>(seqLen), static_cast<uint32_t>(realK_ * sizeof(inType)),
                                    static_cast<uint32_t>((queryTokenStride_ - realK_) * sizeof(inType)), 0, 0};
        DataCopyExtParams kInParams{static_cast<uint16_t>(seqLen), static_cast<uint32_t>(realK_ * sizeof(inType)),
                                    static_cast<uint32_t>((keyTokenStride_ - realK_) * sizeof(inType)), 0, 0};
        DataCopyExtParams vInParams{static_cast<uint16_t>(seqLen), static_cast<uint32_t>(realV_ * sizeof(inType)),
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
            CopyInGate<float>(gateOffset, seqLen);
        } else if (gateDtype_ == 1) {
            CopyInGate<bfloat16_t>(gateOffset, seqLen);
        } else {
            CopyInGate<half>(gateOffset, seqLen);
        }

        qLocal = qInQueue_.DeQue<inType>();
        kLocal = kInQueue_.DeQue<inType>();
        vLocal = vInQueue_.DeQue<inType>();
        Cast(qInUb, qLocal, AscendC::RoundMode::CAST_NONE, alignK_ * seqLen);
        Cast(kInUb, kLocal, AscendC::RoundMode::CAST_NONE, alignK_ * seqLen);
        Cast(vInUb, vLocal, AscendC::RoundMode::CAST_NONE, alignV_ * seqLen);
        AscendC::PipeBarrier<PIPE_V>();
        if (useQkL2norm_) {
            NormalizeRows(qInUb, seqLen);
            NormalizeRows(kInUb, seqLen);
        }
        Muls(qInUb, qInUb, scale_, seqLen * alignK_);
        AscendC::PipeBarrier<PIPE_V>();
        if (useGateInKernel_) {
            ApplyGateInKernel(head, seqLen);
        }
        Exp(gateInUb, gateInUb, alignK_ * seqLen);
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
                Cast(stateInUb, stateLocal, AscendC::RoundMode::CAST_NONE, alignK_ * curSingleV);
            }
        } else {
            LocalTensor<stateType> stateTransposeLocal = stateTransposeBuf_.Get<stateType>();
            TransposeKFirstToVFirst(stateTransposeLocal, stateLocal, curSingleV);
            PipeBarrier<PIPE_V>();
            if constexpr (std::is_same<stateType, float32_t>()) {
                Adds(stateInUb, stateTransposeLocal, 0.0f, alignK_ * curSingleV);
            } else {
                Cast(stateInUb, stateTransposeLocal, AscendC::RoundMode::CAST_NONE, alignK_ * curSingleV);
            }
        }
        stateInQueue_.FreeTensor(stateLocal);
    }

    __aicore__ inline void MatVecMul(const LocalTensor<float> &cubeTensor, const LocalTensor<float> &vecTensor,
                                          LocalTensor<float> &dstTensor, uint32_t rows)
    {
        __ubuf__ float* cubeAddr = (__ubuf__ float*)cubeTensor.GetPhyAddr();
        __ubuf__ float* vecAddr = (__ubuf__ float*)vecTensor.GetPhyAddr();
        __ubuf__ float* dstAddr = (__ubuf__ float*)dstTensor.GetPhyAddr();

        uint16_t rowNum = static_cast<uint16_t>(rows);
        uint16_t colLoopTimes = static_cast<uint16_t>(Ceil(alignK_, V_LENGTH));
        uint32_t colLength = alignK_;
        __VEC_SCOPE__
        {
            RegTensor<float> cube;
            RegTensor<float> vec;
            RegTensor<float> dst;
            MaskReg pregLoop;
            for (uint16_t j = 0; j < colLoopTimes; j++) {
                pregLoop = UpdateMask<float>(colLength);
                DataCopy(vec, vecAddr + j * V_LENGTH);
                for (uint16_t i = 0; i < rowNum; i ++) {
                    DataCopy(cube, cubeAddr + i * alignK_ + j * V_LENGTH);
                    Mul(dst, cube, vec, pregLoop);
                    DataCopy(dstAddr + i * alignK_ + j * V_LENGTH, dst, pregLoop);
                }
            }
        }
    }

    // A6 通用回退路径（K != 128）：delta⊗k 外积累加进 state，并计算 state·q 半成品。
    // 与 A5(arch35) 同名实现一致，不依赖任何 A5 特有机制。
    __aicore__ inline void ProcessKQ(const LocalTensor<float> &cubeTensor, const LocalTensor<float> &vec1Tensor,
                                          LocalTensor<float> &dst1Tensor, const LocalTensor<float> &vec2Tensor,
                                          LocalTensor<float> &dst2Tensor, uint32_t rows)
    {
        __ubuf__ float* cubeAddr = (__ubuf__ float*)cubeTensor.GetPhyAddr();
        __ubuf__ float* vec1Addr = (__ubuf__ float*)vec1Tensor.GetPhyAddr();
        __ubuf__ float* vec2Addr = (__ubuf__ float*)vec2Tensor.GetPhyAddr();
        __ubuf__ float* dst1Addr = (__ubuf__ float*)dst1Tensor.GetPhyAddr();
        __ubuf__ float* dst2Addr = (__ubuf__ float*)dst2Tensor.GetPhyAddr();

        uint16_t rowNum = static_cast<uint16_t>(rows);
        uint16_t colLoopTimes = static_cast<uint16_t>(Ceil(alignK_, V_LENGTH));
        uint32_t colLength = alignK_;
        __VEC_SCOPE__
        {
            RegTensor<float> cube;
            RegTensor<float> vec1;
            RegTensor<float> vec2;
            RegTensor<float> dst1;
            RegTensor<float> dst2;
            MaskReg pregLoop;
            for (uint16_t j = 0; j < colLoopTimes; j++) {
                pregLoop = UpdateMask<float>(colLength);
                DataCopy(vec1, vec1Addr + j * V_LENGTH);
                DataCopy(vec2, vec2Addr + j * V_LENGTH);
                for (uint16_t i = 0; i < rowNum; i ++) {
                    DataCopy<float, LoadDist::DIST_BRC_B32>(cube, cubeAddr + i);
                    DataCopy(dst1, dst1Addr + i * alignK_ + j * V_LENGTH);
                    Mul(cube, cube, vec1, pregLoop);
                    Add(dst1, dst1, cube, pregLoop);
                    Mul(dst2, dst1, vec2, pregLoop);
                    DataCopy(dst1Addr + i * alignK_ + j * V_LENGTH, dst1, pregLoop);
                    DataCopy(dst2Addr + i * alignK_ + j * V_LENGTH, dst2, pregLoop);
                }
            }
        }
    }

    // A6 token 级融合 VF（K=V=128）。一次 __VEC_SCOPE__ 内完成
    //   state *= gate ; dot = state·k ; delta = (v - dot) * beta ; state += delta⊗k ; o = state·q
    // 与 A5 两段式实现（DecayMatVecReduce128 + ProcessDeltaKQReduce128）相比：
    //   1. decayed state 不再回写 UB 再读回，状态 UB 读写各少一趟（每 token 省 V×K 的一次往返）；
    //   2. 每 token 少一次 VF 发射与一次 PipeBarrier<PIPE_V>；
    //   3. 行块按 V 方向 4 行一组：每行的 128 列拆成前后两段 64 lane，4 行合计 8 条独立的
    //      短指令流；ReduceSum（长指令）紧跟本行的乘加之后发射，与其余行的短指令并行，
    //      正好构成 960dt 的"两条短指令 + 一条长指令"三发射形态；
    //   4. 段内归约用 MulAddDst 折叠（与 A5 通用路径的 RKDA_ENABLE_ADD_FOLD_REDUCE 同一写法），
    //      每行归约少一条 Add；
    //   5. beta sigmoid 与 delta 标量链收敛到 VL1 mask，只对 1 个 lane 发射。
    template <bool useSigmoid, bool allowNeg>
    __aicore__ inline void ProcessTokenFused128(const LocalTensor<float> &gateTensor,
                                                const LocalTensor<float> &vTensor,
                                                const LocalTensor<float> &betaTensor,
                                                uint64_t betaOffset,
                                                const LocalTensor<float> &kTensor,
                                                LocalTensor<float> &stateTensor,
                                                const LocalTensor<float> &qTensor,
                                                LocalTensor<float> &attnTensor, uint32_t rows)
    {
        __ubuf__ float *gateAddr = (__ubuf__ float *)gateTensor.GetPhyAddr();
        __ubuf__ float *vAddr = (__ubuf__ float *)vTensor.GetPhyAddr();
        uint64_t betaRowStride =
            betaDtype_ == 0 ? FP32_NUM_PER_BLOCK : BF16_NUM_PER_BLOCK;
        __ubuf__ float *betaAddr = (__ubuf__ float *)betaTensor.GetPhyAddr() + betaOffset * betaRowStride;
        __ubuf__ float *kAddr = (__ubuf__ float *)kTensor.GetPhyAddr();
        __ubuf__ float *stateAddr = (__ubuf__ float *)stateTensor.GetPhyAddr();
        __ubuf__ float *qAddr = (__ubuf__ float *)qTensor.GetPhyAddr();
        __ubuf__ float *attnAddr = (__ubuf__ float *)attnTensor.GetPhyAddr();
        uint16_t rowNum = static_cast<uint16_t>(rows);
        uint16_t blockNum = static_cast<uint16_t>(rowNum / A6_ROW_BLOCK);
        __VEC_SCOPE__
        {
            RegTensor<float> gate0;
            RegTensor<float> gate1;
            RegTensor<float> k0;
            RegTensor<float> k1;
            RegTensor<float> q0;
            RegTensor<float> q1;
            RegTensor<float> beta;
            RegTensor<float> s00;
            RegTensor<float> s01;
            RegTensor<float> s10;
            RegTensor<float> s11;
            RegTensor<float> s20;
            RegTensor<float> s21;
            RegTensor<float> s30;
            RegTensor<float> s31;
            RegTensor<float> p0;
            RegTensor<float> p1;
            RegTensor<float> p2;
            RegTensor<float> p3;
            RegTensor<float> d0;
            RegTensor<float> d1;
            RegTensor<float> d2;
            RegTensor<float> d3;
            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregFirst = CreateMask<float, MaskPattern::VL1>();
            // token 常量（gate / k / q 的两段 + beta）每个 token 只装一次
            DataCopy(gate0, gateAddr);
            DataCopy(gate1, gateAddr + V_LENGTH);
            DataCopy(k0, kAddr);
            DataCopy(k1, kAddr + V_LENGTH);
            DataCopy(q0, qAddr);
            DataCopy(q1, qAddr + V_LENGTH);
            DataCopy<float, LoadDist::DIST_BRC_B32>(beta, betaAddr);
            if constexpr (useSigmoid) {
                RegTensor<float> one;
                // 只有 lane0 会被后续 delta 链使用，整条 sigmoid 链都收在 VL1 上。
                Muls(beta, beta, -1.0f, pregFirst);
                Exp(beta, beta, pregFirst);
                Adds(beta, beta, 1.0f, pregFirst);
                Duplicate(one, 1.0f, pregFirst);
                Div(beta, one, beta, pregFirst);
                if constexpr (allowNeg) {
                    Muls(beta, beta, 2.0f, pregFirst);
                }
            }
            for (uint16_t i = 0; i < blockNum; ++i) {
                uint16_t row0 = static_cast<uint16_t>(i * A6_ROW_BLOCK);
                uint32_t rowOffset = static_cast<uint32_t>(row0) * alignK_;
                // 8 条短指令：4 行 state 载入，每行前后两段
                DataCopy(s00, stateAddr + rowOffset);
                DataCopy(s01, stateAddr + rowOffset + V_LENGTH);
                DataCopy(s10, stateAddr + rowOffset + alignK_);
                DataCopy(s11, stateAddr + rowOffset + alignK_ + V_LENGTH);
                DataCopy(s20, stateAddr + rowOffset + 2 * alignK_);
                DataCopy(s21, stateAddr + rowOffset + 2 * alignK_ + V_LENGTH);
                DataCopy(s30, stateAddr + rowOffset + 3 * alignK_);
                DataCopy(s31, stateAddr + rowOffset + 3 * alignK_ + V_LENGTH);
                // 行内交织：每行“2 条衰减 + 2 条乘加 + 1 条长归约”，下一行的短指令
                // 正好落在上一行 ReduceSum 的等待窗口里，避免 4 条长指令扎堆发射。
                Mul(s00, s00, gate0, pregFull);
                Mul(s01, s01, gate1, pregFull);
                Mul(p0, s00, k0, pregFull);
                MulAddDst(p0, s01, k1, pregFull);
                ReduceSum(d0, p0, pregFull);

                Mul(s10, s10, gate0, pregFull);
                Mul(s11, s11, gate1, pregFull);
                Mul(p1, s10, k0, pregFull);
                MulAddDst(p1, s11, k1, pregFull);
                ReduceSum(d1, p1, pregFull);

                Mul(s20, s20, gate0, pregFull);
                Mul(s21, s21, gate1, pregFull);
                Mul(p2, s20, k0, pregFull);
                MulAddDst(p2, s21, k1, pregFull);
                ReduceSum(d2, p2, pregFull);

                Mul(s30, s30, gate0, pregFull);
                Mul(s31, s31, gate1, pregFull);
                Mul(p3, s30, k0, pregFull);
                MulAddDst(p3, s31, k1, pregFull);
                ReduceSum(d3, p3, pregFull);
                // v 广播：只依赖 vAddr，放在最后一个长归约之后，填充其等待窗口
                DataCopy<float, LoadDist::DIST_BRC_B32>(p0, vAddr + row0);
                DataCopy<float, LoadDist::DIST_BRC_B32>(p1, vAddr + row0 + 1);
                DataCopy<float, LoadDist::DIST_BRC_B32>(p2, vAddr + row0 + 2);
                DataCopy<float, LoadDist::DIST_BRC_B32>(p3, vAddr + row0 + 3);
                // 标量链按行收口：每行 3 条 VL1 短指令后立刻广播，行与行之间无依赖
                Sub(d0, p0, d0, pregFirst);
                Mul(d0, d0, beta, pregFirst);
                Duplicate(p0, d0, pregFull);
                Sub(d1, p1, d1, pregFirst);
                Mul(d1, d1, beta, pregFirst);
                Duplicate(p1, d1, pregFull);
                Sub(d2, p2, d2, pregFirst);
                Mul(d2, d2, beta, pregFirst);
                Duplicate(p2, d2, pregFull);
                Sub(d3, p3, d3, pregFirst);
                Mul(d3, d3, beta, pregFirst);
                Duplicate(p3, d3, pregFull);
                // state += delta⊗k，两段各一条 MulAddDst
                MulAddDst(s00, p0, k0, pregFull);
                MulAddDst(s01, p0, k1, pregFull);
                MulAddDst(s10, p1, k0, pregFull);
                MulAddDst(s11, p1, k1, pregFull);
                MulAddDst(s20, p2, k0, pregFull);
                MulAddDst(s21, p2, k1, pregFull);
                MulAddDst(s30, p3, k0, pregFull);
                MulAddDst(s31, p3, k1, pregFull);
                // 读出按行交织：每行“2 条乘加 + 1 条长归约 + 2 条状态写回”，
                // attn 首元素写延迟一行，确保长归约结果已经就绪。
                Mul(p0, s00, q0, pregFull);
                MulAddDst(p0, s01, q1, pregFull);
                ReduceSum(d0, p0, pregFull);
                DataCopy(stateAddr + rowOffset, s00, pregFull);
                DataCopy(stateAddr + rowOffset + V_LENGTH, s01, pregFull);

                Mul(p1, s10, q0, pregFull);
                MulAddDst(p1, s11, q1, pregFull);
                ReduceSum(d1, p1, pregFull);
                DataCopy(stateAddr + rowOffset + alignK_, s10, pregFull);
                DataCopy(stateAddr + rowOffset + alignK_ + V_LENGTH, s11, pregFull);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(attnAddr + row0, d0, pregFirst);

                Mul(p2, s20, q0, pregFull);
                MulAddDst(p2, s21, q1, pregFull);
                ReduceSum(d2, p2, pregFull);
                DataCopy(stateAddr + rowOffset + 2 * alignK_, s20, pregFull);
                DataCopy(stateAddr + rowOffset + 2 * alignK_ + V_LENGTH, s21, pregFull);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(attnAddr + row0 + 1, d1, pregFirst);

                Mul(p3, s30, q0, pregFull);
                MulAddDst(p3, s31, q1, pregFull);
                ReduceSum(d3, p3, pregFull);
                DataCopy(stateAddr + rowOffset + 3 * alignK_, s30, pregFull);
                DataCopy(stateAddr + rowOffset + 3 * alignK_ + V_LENGTH, s31, pregFull);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(attnAddr + row0 + 2, d2, pregFirst);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(attnAddr + row0 + 3, d3, pregFirst);
            }
            // 尾部 0~3 行：单行完成，数值路径与行块一致
            for (uint16_t row = static_cast<uint16_t>(blockNum * A6_ROW_BLOCK); row < rowNum; ++row) {
                uint32_t rowOffset = static_cast<uint32_t>(row) * alignK_;
                DataCopy(s00, stateAddr + rowOffset);
                DataCopy(s01, stateAddr + rowOffset + V_LENGTH);
                Mul(s00, s00, gate0, pregFull);
                Mul(s01, s01, gate1, pregFull);
                Mul(p0, s00, k0, pregFull);
                MulAddDst(p0, s01, k1, pregFull);
                ReduceSum(d0, p0, pregFull);
                DataCopy<float, LoadDist::DIST_BRC_B32>(p0, vAddr + row);
                Sub(d0, p0, d0, pregFirst);
                Mul(d0, d0, beta, pregFirst);
                Duplicate(p0, d0, pregFull);
                MulAddDst(s00, p0, k0, pregFull);
                MulAddDst(s01, p0, k1, pregFull);
                DataCopy(stateAddr + rowOffset, s00, pregFull);
                DataCopy(stateAddr + rowOffset + V_LENGTH, s01, pregFull);
                Mul(p0, s00, q0, pregFull);
                MulAddDst(p0, s01, q1, pregFull);
                ReduceSum(d0, p0, pregFull);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(attnAddr + row, d0, pregFirst);
            }
        }
    }

    __aicore__ inline void ProcessTokenFused128Dispatch(const LocalTensor<float> &gateTensor,
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
                ProcessTokenFused128<true, true>(
                    gateTensor, vTensor, betaTensor, betaOffset,
                    kTensor, stateTensor, qTensor, attnTensor, rows);
            } else {
                ProcessTokenFused128<true, false>(
                    gateTensor, vTensor, betaTensor, betaOffset,
                    kTensor, stateTensor, qTensor, attnTensor, rows);
            }
        } else {
            ProcessTokenFused128<false, false>(
                gateTensor, vTensor, betaTensor, betaOffset,
                kTensor, stateTensor, qTensor, attnTensor, rows);
        }
    }

    __aicore__ inline void ReduceSum64(__ubuf__ float* dstAddr, __ubuf__ float* srcAddr, uint16_t rowNum)
    {
        uint32_t colLength = alignK_;
        __VEC_SCOPE__
        {
            RegTensor<float> src;
            RegTensor<float> sum;
            MaskReg pregLoop = UpdateMask<float>(colLength);
            for (uint16_t i = 0;i < rowNum;i ++) {
                DataCopy(src, srcAddr + i * alignK_);
                ReduceSum(sum, src, pregLoop);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstAddr + i, sum, pregLoop);
            }
        }
    }

    __aicore__ inline void ReduceSum128(__ubuf__ float* dstAddr, __ubuf__ float* srcAddr, uint16_t rowNum)
    {
        uint32_t colLength = alignK_ - V_LENGTH;
        __VEC_SCOPE__
        {
            RegTensor<float> src1;
            RegTensor<float> src2;
            RegTensor<float> sum;
            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregLoop = UpdateMask<float>(colLength);
            for (uint16_t i = 0;i < rowNum;i ++) {
                DataCopy(src1, srcAddr + i * alignK_);
                DataCopy(src2, srcAddr + i * alignK_ + V_LENGTH);
                Add<float, MaskMergeMode::MERGING>(src1, src1, src2, pregLoop);
                ReduceSum(sum, src1, pregFull);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstAddr + i, sum, pregFull);
            }
        }
    }

    __aicore__ inline void ReduceSumVF(__ubuf__ float* dstAddr, __ubuf__ float* srcAddr, uint16_t rowNum)
    {
        uint16_t colLoopTimes = static_cast<uint16_t>(Ceil(alignK_, V_LENGTH));
        __VEC_SCOPE__
        {
            RegTensor<float> src;
            RegTensor<float> tmp;
            RegTensor<float> sum;
            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregLoop;
            for (uint16_t i = 0;i < rowNum;i ++) {
                uint32_t colLength = alignK_;
                Duplicate(tmp, 0.0f);
                for (uint16_t j = 0; j < colLoopTimes; j++) {
                    pregLoop = UpdateMask<float>(colLength);
                    DataCopy(src, srcAddr + i * alignK_ + j * V_LENGTH);
                    Add<float, MaskMergeMode::MERGING>(tmp, tmp, src, pregLoop);
                }
                ReduceSum(sum, tmp, pregFull);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstAddr + i, sum, pregFull);
            }
        }
    }

    __aicore__ inline void ReduceSumDispatch(LocalTensor<float> &dstTensor, LocalTensor<float> &srcTensor,
                                             uint32_t rows)
    {
        __ubuf__ float* srcAddr = (__ubuf__ float*)srcTensor.GetPhyAddr();
        __ubuf__ float* dstAddr = (__ubuf__ float*)dstTensor.GetPhyAddr();
        uint16_t rowNum = static_cast<uint16_t>(rows);
        if (alignK_ <= V_LENGTH) {
            ReduceSum64(dstAddr, srcAddr, rowNum);
        } else if (alignK_ <= TWO_V_LENGTH) {
            ReduceSum128(dstAddr, srcAddr, rowNum);
        } else {
            ReduceSumVF(dstAddr, srcAddr, rowNum);
        }
    }

    __aicore__ inline void Compute(uint32_t curSingleV, uint64_t curQKOffset,
                                   uint64_t curVOffset, uint64_t betaOffset, bool storeState)
    {
        if (alignK_ == TWO_V_LENGTH) {
            // A6：token 级融合 VF，decay / S@k / delta / outer / S@q 在同一个 __VEC_SCOPE__ 内完成。
            ProcessTokenFused128Dispatch(gateInUb[curQKOffset], vInUb[curVOffset], betaInUb,
                                         betaOffset, kInUb[curQKOffset], stateInUb,
                                         qInUb[curQKOffset], attnInUb, curSingleV);
        } else {
            MatVecMul(stateInUb, gateInUb[curQKOffset], stateInUb, curSingleV);
            AscendC::PipeBarrier<PIPE_V>();
            MatVecMul(stateInUb, kInUb[curQKOffset], broadTmpInUb, curSingleV);
            AscendC::PipeBarrier<PIPE_V>();
            ReduceSumDispatch(deltaInUb, broadTmpInUb, curSingleV);
            AscendC::PipeBarrier<PIPE_V>();
            Sub(deltaInUb, vInUb[curVOffset], deltaInUb, curSingleV);
            AscendC::PipeBarrier<PIPE_V>();
            Muls(deltaInUb, deltaInUb, beta_, curSingleV);
            AscendC::PipeBarrier<PIPE_V>();
            ProcessKQ(deltaInUb, kInUb[curQKOffset], stateInUb,
                      qInUb[curQKOffset], broadTmpInUb, curSingleV);
            AscendC::PipeBarrier<PIPE_V>();
            ReduceSumDispatch(attnInUb, broadTmpInUb, curSingleV);
        }
        LocalTensor<outType> attnOutLocal = attnOutQueue_.AllocTensor<outType>();
        if (storeState) {
            LocalTensor<stateType> stateOutLocal = stateOutQueue_.AllocTensor<stateType>();
            if constexpr (std::is_same<stateType, float32_t>()) {
                DataCopy(stateOutLocal, stateInUb, alignK_ * curSingleV);
            } else {
                Cast(stateOutLocal, stateInUb, AscendC::RoundMode::CAST_RINT, alignK_ * curSingleV);
            }
            stateOutQueue_.EnQue<stateType>(stateOutLocal);
        }
        Cast(attnOutLocal, attnInUb, AscendC::RoundMode::CAST_RINT, curSingleV);
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

    template <typename betaType>
    __aicore__ inline void CopyInBetaTyped(int64_t seq0, int64_t seq1, uint64_t head)
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
        if constexpr (std::is_same<betaType, float32_t>()) {
            Adds(betaInUb, betaLocal, 0.0f, static_cast<uint32_t>(betaCount));
        } else {
            Cast(betaInUb, betaLocal, AscendC::RoundMode::CAST_NONE, static_cast<uint32_t>(betaCount));
        }
        betaInQueue_.FreeTensor(betaLocal);
        PipeBarrier<PIPE_V>();
        if (alignK_ != TWO_V_LENGTH) {
            SyncVToS();
        }
    }

    __aicore__ inline void CopyInBeta(int64_t seq0, int64_t seq1, uint64_t head)
    {
        if (betaDtype_ == 0) {
            CopyInBetaTyped<float>(seq0, seq1, head);
        } else if (betaDtype_ == 1) {
            CopyInBetaTyped<bfloat16_t>(seq0, seq1, head);
        } else {
            CopyInBetaTyped<half>(seq0, seq1, head);
        }
    }

    __aicore__ inline uint64_t StateSlotForToken(uint64_t batchIdx, int64_t seq0, int64_t tokenIdx) const
    {
        if (hasSsmStateIndices_) {
            return LoadStateSlot(batchIdx, seq0, tokenIdx);
        }
        return batchIdx;
    }

    __aicore__ inline float LoadBeta(uint64_t tokenOffset)
    {
        uint64_t betaRowStride =
            betaDtype_ == 0 ? FP32_NUM_PER_BLOCK : BF16_NUM_PER_BLOCK;
        float beta = betaInUb.GetValue(tokenOffset * betaRowStride);
        if (useBetaSigmoid_) {
            beta = SigmoidScalar(beta);
            if (allowNegEigval_) {
                beta *= 2.0f;
            }
        }
        return beta;
    }

    __aicore__ inline bool ProcessHead(uint64_t batchIdx, int64_t seq0, int64_t seq1,
                                      uint64_t head_i, uint64_t stateSlot, bool statePrefetched,
                                      bool hasNextTask, uint64_t nextStateSlot, uint64_t nextHead)
    {
        uint64_t qkHead = head_i / (NV_ / NK_);
        uint64_t qOffset = static_cast<uint64_t>(seq0) * queryTokenStride_ + qkHead * queryHeadStride_;
        uint64_t kOffset = static_cast<uint64_t>(seq0) * keyTokenStride_ + qkHead * keyHeadStride_;
        uint64_t vOffset = static_cast<uint64_t>(seq0) * valueTokenStride_ + head_i * valueHeadStride_;
        uint64_t gateOffset = (static_cast<uint64_t>(seq0) * NV_ + head_i) * realK_;
        CopyInQKVGate(
            qOffset, kOffset, vOffset, gateOffset, static_cast<int32_t>(seq1 - seq0), head_i);
        if (realV_ == 0) {
            return false;
        }
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
            for (int64_t seq_i = seq0; seq_i < seq1; seq_i++) {
                uint64_t betaOffset = static_cast<uint64_t>(seq_i - seq0);
                uint64_t curQKOffset = static_cast<uint64_t>(seq_i - seq0) * alignK_;
                uint64_t curVOffset = static_cast<uint64_t>(seq_i - seq0) * alignV_ + v_i;
                uint64_t attnOffset = (static_cast<uint64_t>(seq_i) * NV_ + head_i) * realV_ + v_i;
                uint64_t curStateSlot = StateSlotForToken(batchIdx, seq0, seq_i);
                uint64_t curStateOutSlot = curStateSlot;
                // A6：state 落盘的语义是"任务结束时槽里的值"。没有 ssm_state_indices 时，
                // StateSlotForToken 对同一任务的所有 token 都返回 batchIdx，state 写回地址完全相同，
                // 中间态写回必然被后一个 token 的同地址写回覆盖 —— 只有最后一个 token 的写回可观测。
                // 因此跳过中间态写回：GM 上的最终内容逐位不变，而每 (batch, head) 任务的 state MTE3
                // 从 T 次降到 1 次（T<=8），把 state 写回从"每 token 一次 64KB/32KB"降为常数次。
                // 有 ssm_state_indices 时每个 token 可能命中不同槽，保持逐 token 写回。
                bool storeStateNow = shouldStoreState_;
                if (storeStateNow && !hasSsmStateIndices_ && seq_i + 1 < seq1) {
                    storeStateNow = false;
                }
                if (alignK_ != TWO_V_LENGTH) {
                    beta_ = LoadBeta(betaOffset);
                }
                Compute(curSingleV, curQKOffset, curVOffset, betaOffset, storeStateNow);
                if (attnOutBufferNum_ == BUFFER_NUM) {
                    CopyOutAttn(attnOffset, curSingleV);
                } else {
                    if (hasPendingAttn) {
                        CopyOutAttn(pendingAttnOffset, curSingleV);
                    }
                    pendingAttnOffset = attnOffset;
                    hasPendingAttn = true;
                }
                if (storeStateNow) {
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
    LocalTensor<float> deltaInUb;
    LocalTensor<float> broadTmpInUb;
    LocalTensor<float> attnInUb;
    LocalTensor<float> stateInUb;
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
    bool useAddFoldReduce_;
    float beta_;
    float scale_;
    float lowerBound_;
    uint64_t blockIdx;
};
} // namespace RecurrentKda
#endif
