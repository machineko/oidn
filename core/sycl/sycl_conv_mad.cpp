// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "sycl_conv_mad.h"

namespace oidn {

  using namespace esimd;

  constexpr int blockOW = 16;
  constexpr int blockIW = blockOW + 3 - 1;

  template<typename T, TensorLayout tensorLayout, TensorLayout weightLayout>
  struct SYCLConvMADKernel
  {
    static constexpr int blockC = TensorAccessor3D<T, tensorLayout>::blockC;

    TensorAccessor3D<T, tensorLayout> src;
    TensorAccessor4D<T, weightLayout> weight;
    TensorAccessor1D<T> bias;
    TensorAccessor3D<T, tensorLayout> dst;

    OIDN_INLINE void operator ()(const WorkItem<3>& it) const SYCL_ESIMD_FUNCTION
    {
      const int oc = it.getId<0>() * blockC;
      const int oh = it.getId<1>();
      const int ow = it.getId<2>() * blockOW;

      // Output row
      simd<T, blockC> dstVec[blockOW];

      // Load biases
      const auto biasVec = block_load<T, blockC, vector_aligned_tag>(&bias(oc));
      #pragma unroll
      for (int i = 0; i < blockOW; ++i)
        dstVec[i] = biasVec;

      // Iterate over input channel blocks
      for (int ic = 0; ic < src.C; ic += blockC)
      {
        // Iterate over kernel height
        #pragma unroll
        for (int kh = 0; kh < 3; ++kh)
        {
          const int ih = oh + kh - 1;
          if (ih < 0 || ih >= src.H)
            continue;

          const int iw = ow - 1;
          const T* srcPtr = &src(ic, ih, iw);
          simd<T, blockIW*blockC> srcVec;

          // Load input row
          if (iw >= 0 && iw + blockIW < src.W)
          {
            srcVec.copy_from(srcPtr, overaligned<32>);
          }
          else
          {
            srcVec = 0;
            #pragma unroll
            for (int i = 0; i < blockIW; ++i)
            {
              if (iw + i >= 0 && iw + i < src.W)
                srcVec.template select<blockC, 1>(i*blockC) = block_load<T, blockC>(srcPtr, vector_aligned);
              srcPtr += blockC;
            }
          }

          // Iterate over kernel width
          const T* weightPtr = &weight(oc, ic, kh, 0);
          
          #pragma unroll
          for (int kw = 0; kw < 3; ++kw)
          {
            // Load weights
            simd<T, blockC*blockC> weightVec;
            weightVec.copy_from(weightPtr, vector_aligned);
            weightPtr += blockC*blockC;

            // Accumulate to output row
            #pragma unroll
            for (int i = 0; i < blockC; ++i)
            {
              #pragma unroll
              for (int j = 0; j < blockOW; ++j)
                dstVec[j] += srcVec.template replicate_w<blockC, 1>((kw+j)*blockC + i) * weightVec.template select<blockC, 1>(i*blockC);
            }
          }
        }
      }

      // Apply ReLU
      #pragma unroll
      for (int i = 0; i < blockOW; ++i)
        dstVec[i] = max(dstVec[i], simd<T, blockC>(0));

      // Store output row
      T* dstPtr = &dst(oc, oh, ow);
      #pragma unroll
      for (int i = 0; i < blockOW; ++i)
      {
        if (ow + i < dst.W)
          block_store(dstPtr, dstVec[i]);
        dstPtr += blockC;
      }
    }
  };

  SYCLConvMAD::SYCLConvMAD(const Ref<SYCLDevice>& device, const ConvDesc& desc)
    : Conv(desc),
      device(device)
  {
    if (srcDesc.layout != TensorLayout::Chw16c || srcDesc.dataType != DataType::Float16)
      throw std::invalid_argument("unsupported convolution source layout/data type");
    if (weightDesc.layout != TensorLayout::OIhw16i16o || weightDesc.dataType != DataType::Float16)
      throw std::invalid_argument("unsupported convolution weight layout/data type");
    if (biasDesc.layout != TensorLayout::x || biasDesc.dataType != DataType::Float16)
      throw std::invalid_argument("unsupported convolution bias layout/data type");
  }

  void SYCLConvMAD::run()
  {
    if (!src || !weight || !bias || !dst)
      throw std::logic_error("convolution argument not set");

    SYCLConvMADKernel<half, TensorLayout::Chw16c, TensorLayout::OIhw16i16o> kernel;
    kernel.src    = *src;
    kernel.weight = *weight;
    kernel.bias   = *bias;
    kernel.dst    = *dst;

    device->runESIMDKernelAsync(WorkDim<3>(dst->getCB(), dst->getH(), ceil_div(dst->getW(), blockOW)), kernel);
  }

} // namespace oidn