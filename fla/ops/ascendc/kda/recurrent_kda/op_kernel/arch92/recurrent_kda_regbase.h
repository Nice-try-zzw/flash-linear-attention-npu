// Copyright (c) 2025-2026 Tianjin University, Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#ifndef RECURRENT_KDA_ARCH92_REGBASE_H
#define RECURRENT_KDA_ARCH92_REGBASE_H

#include "kernel_operator.h"
#include <type_traits>

// A6 legacy 兼容路径共用的寄存器级（RegBase）原语：整条 V=256 路径不再调用
// 任何以 LocalTensor 为操作数的 membase 向量接口，只保留 DMA 与寄存器运算。
namespace RecurrentKda {
namespace RegBase {
using namespace AscendC;
using namespace AscendC::MicroAPI;

// 每个向量寄存器承载 64 个 FP32 lane；低精度输入保持原始位宽留在 UB。
template <class T>
__simd_callee__ inline void Load(RegTensor<float> &dst, __ubuf__ T *src, MaskReg &mask)
{
    if constexpr (std::is_same<T, float>::value) {
        DataCopy(dst, src);
    } else {
        RegTensor<T> raw;
        DataCopy<T, LoadDist::DIST_UNPACK_B16>(raw, src);
        static constexpr CastTrait cast = {RegLayout::ZERO, SatMode::UNKNOWN,
                                           MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        Cast<float, T, cast>(dst, raw, mask);
    }
}

template <class T, bool firstOnly = false>
__simd_callee__ inline void Store(__ubuf__ T *dst, RegTensor<float> &src, MaskReg &mask)
{
    if constexpr (std::is_same<T, float>::value) {
        if constexpr (firstOnly) DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dst, src, mask);
        else DataCopy(dst, src, mask);
    } else {
        RegTensor<T> raw;
        static constexpr CastTrait cast = {RegLayout::ZERO, SatMode::NO_SAT,
                                           MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        if constexpr (firstOnly) {
            uint32_t one = 1;
            MaskReg first = UpdateMask<float>(one);
            Cast<T, float, cast>(raw, src, first);
            DataCopy<T, StoreDist::DIST_FIRST_ELEMENT_B16>(dst, raw, mask);
        } else {
            Cast<T, float, cast>(raw, src, mask);
            DataCopy<T, StoreDist::DIST_PACK_B32>(dst, raw, mask);
        }
    }
}

// 128 维归约：两个 64-lane 分段相加后整体归约并广播到全部 lane。
__simd_callee__ inline void Sum128(RegTensor<float> &sum, RegTensor<float> &a,
                                   RegTensor<float> &b, MaskReg &mask)
{
    Add(sum, a, b, mask);
    ReduceSum(sum, sum, mask);
    Duplicate(sum, sum, mask);
}

// x / sqrt(sum(x*x) + 1e-6)，与 kernel 其它路径的归一化口径一致。
__simd_callee__ inline void Normalize(RegTensor<float> &lo, RegTensor<float> &hi, MaskReg &mask)
{
    RegTensor<float> a, b, norm;
    Mul(a, lo, lo, mask);
    Mul(b, hi, hi, mask);
    Sum128(norm, a, b, mask);
    Adds(norm, norm, 1e-6f, mask);
    Sqrt(norm, norm, mask);
    Div(lo, lo, norm, mask);
    Div(hi, hi, norm, mask);
}

__simd_callee__ inline void Sigmoid(RegTensor<float> &dst, MaskReg &mask)
{
    RegTensor<float> magnitude, numerator, denominator, one, negative;
    MaskReg positive;
    Duplicate(one, 1.0f);
    Abs(magnitude, dst, mask);
    Muls(magnitude, magnitude, -1.0f, mask);
    Exp(magnitude, magnitude, mask);
    Adds(denominator, magnitude, 1.0f, mask);
    Div(numerator, one, denominator, mask);
    Div(negative, magnitude, denominator, mask);
    CompareScalar<float, CMPMODE::GE>(positive, dst, 0.0f, mask);
    Select(dst, numerator, negative, positive);
}

} // namespace RegBase
} // namespace RecurrentKda
#endif // RECURRENT_KDA_ARCH92_REGBASE_H
