// Copyright (c) 2025-2026 Tianjin University, Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#pragma once
#include "kernel_operator.h"
#include "../recurrent_kda_tiling_data.h"
#include "recurrent_kda_regbase.h"

namespace RecurrentKda {
using namespace AscendC;
constexpr uint16_t RKDA_DIM_128 = 128;
constexpr uint32_t RKDA_ROW_BF16_BYTES_256 = RKDA_DIM_128 * sizeof(bfloat16_t);
constexpr uint32_t RKDA_ROW_FP32_BYTES_512 = RKDA_DIM_128 * sizeof(float);
constexpr uint32_t RKDA_GATE_DOUBLE_BYTES_1024 = 2 * RKDA_ROW_FP32_BYTES_512;
constexpr uint32_t RKDA_PARAM_SLOT_BYTES_544 = RKDA_ROW_FP32_BYTES_512 + 32;
constexpr uint32_t RKDA_PARAM_DOUBLE_BYTES_1088 = 2 * RKDA_PARAM_SLOT_BYTES_544;
constexpr uint32_t RKDA_GUARD_BYTES_2048 = 4 * RKDA_ROW_FP32_BYTES_512;
constexpr uint64_t RKDA_INVALID_SLOT = static_cast<uint64_t>(-1);
// A6 用 384KB UB 把状态 bank 加到 4 个：递推占 cur/next，剩下两个分别承接上一 token
// 的在途快照和下一任务初态预取，逐 token 落盘（ssm_state_indices）时可再多留一份快照在飞。
constexpr uint32_t RKDA_NO_BANK = RKDA_A6_BANK_COUNT;
// Standalone Mutex allocator; never reserve numeric hardware IDs.
enum RKDAMutexSlot : uint32_t {
    RKDA_MUTEX_STATE_USE = 0,                            // banks: VF reads/writes versus MTE2 replacement
    RKDA_MUTEX_STATE_DMA = RKDA_A6_BANK_COUNT,         // banks: VF writes versus MTE3 snapshot reads
    RKDA_MUTEX_INPUT = 2 * RKDA_A6_BANK_COUNT,         // two tile slots
    RKDA_MUTEX_PARAM = 2 * RKDA_A6_BANK_COUNT + 2,     // two task parameter slots
    RKDA_MUTEX_OUT = 2 * RKDA_A6_BANK_COUNT + 4,       // two tile output slots
    RKDA_MUTEX_COUNT = 2 * RKDA_A6_BANK_COUNT + 6
};
struct RKDATask {
    uint64_t id, request, head;
    int64_t begin, end;
    uint64_t initialSlot;
};
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

// Pure per-core data. Functions and event operations live outside this context.
template <typename inType, typename outType, typename stateType, bool stateVFirst, bool fusedMode = false>
struct RKDAVectorContext {
    const RecurrentKdaTilingData *td_;
    RKDAInitParams args_;
    LocalTensor<uint8_t> ub_;
    GlobalTensor<inType> q_, k_, v_;
    GlobalTensor<stateType> stateIn_, stateOut_;
    GlobalTensor<outType> out_;
    GlobalTensor<float> aLog_, bias_;
    // Fixed A6 V=128 layout: no per-launch window or offset reconstruction.
    static constexpr uint32_t x_ = RKDA_A6_WINDOW_TOKENS;
    static constexpr uint32_t betaSlot_ = 32 * x_;
    static constexpr uint32_t qOffset_ = RKDA_A6_BANK_COUNT * RegBase::StateLayout<stateType, stateVFirst>::BYTES;
    static constexpr uint32_t kOffset_ = qOffset_ + RKDA_ROW_FP32_BYTES_512 * x_;
    static constexpr uint32_t vOffset_ = kOffset_ + RKDA_ROW_FP32_BYTES_512 * x_;
    static constexpr uint32_t gateOffset_ = vOffset_ + RKDA_ROW_FP32_BYTES_512 * x_;
    static constexpr uint32_t betaOffset_ = gateOffset_ + RKDA_GATE_DOUBLE_BYTES_1024 * x_;
    static constexpr uint32_t paramOffset_ = betaOffset_ + 2 * betaSlot_;
    static constexpr uint32_t outOffset_ = paramOffset_ + RKDA_PARAM_DOUBLE_BYTES_1088;
    static constexpr uint32_t eventOffset_ = outOffset_ + RKDA_ROW_FP32_BYTES_512 * x_;
    uint32_t initialBank_, currentBank_, nextBank_;
    uint32_t vfFlags_;
    DataCopyExtParams queryCopy_, keyCopy_, valueCopy_, gateCopy_, betaCopy_, outputCopy_;
    MutexID mutex_[RKDA_MUTEX_COUNT];
    uint32_t prefetchedBank_ = RKDA_NO_BANK;
#ifdef RKDA_EXECUTION_TRACE
    GlobalTensor<uint64_t> trace_;
    uint64_t traceCount_ = 0;
#endif
};
} // namespace RecurrentKda
