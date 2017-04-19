#include "./hadamard_op.h"
#include "./elemwise_binary_op.h"
#include "./elemwise_binary_broadcast_op.h"
#include <mshadow/tensor.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_device_runtime_api.h>
#include <cuda.h>

#ifndef FWT_KERNEL_CUH
#define FWT_KERNEL_CUH
#ifndef fwt_kernel_cuh
#define fwt_kernel_cuh
#define ELEMENTARY_LOG2SIZE 11



namespace mshadow {
namespace cuda {


template <typename DType>
__global__ void hadamard_forward_kernel(const int nthreads, DType *out, DType *indices_p, DType *in, int n_smaples, int in_dim, int out_dim) {

    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= nthreads){
        return;
    }

    //printf("Thiis is id %d \n", index);

    int log2d = in_dim <= 1 ? 0 : log2((double)(in_dim - 1)) + 1;
    DType temp;
    unsigned int k = out_dim;

    in += index*in_dim;
    for (int t = log2d; t; t--) {

        int blockSize = 1 << t;
        int numBlocks = 1 << (log2d - t);

        int halfBlockSize = blockSize >> 1;
        DType *p1 = in;
        DType *p2 = in + halfBlockSize;

        for (int blockIndex = numBlocks; blockIndex; blockIndex--) {
            for (int i = halfBlockSize; i; i--) {
                temp = *p1 + *p2;
                *p2 = *p1 - *p2;
                *p1 = temp;
                p1++;
                p2++;
            }
            p1 += halfBlockSize;
            p2 += halfBlockSize;
        }
    }


    DType *input_p = in;
    out += index*out_dim;
    for (int i = 0; i < k; i++) {
        int ind = (int) *indices_p;
        *out = *(input_p + ind);

        out++;
        indices_p++;
    }

}

template <typename DType>
__global__ void fwtBatch1Kernel(DType *d_Output, DType *d_Input, int log2N){
    const int    N = 1 << log2N;
    const int base = blockIdx.x << log2N;
    int hi = blockIdx.x * blockDim.x + threadIdx.x;
    //std::printf("blockIdx: %d, blockDim: %d", blockIdx.x, blockDim.x);
    //std::printf("threadIdx: %d", threadIdx.x);

    //(2 ** 11) * 4 bytes == 8KB -- maximum s_data[] size for G80
    extern __shared__ float s_data[];

    DType *d_Src = d_Input  + base;
    DType *d_Dst = d_Output + base;

    for(int pos = threadIdx.x; pos < N; pos += blockDim.x)
        s_data[pos] = d_Src[pos];

    //Main radix-4 stages
    const int pos = threadIdx.x;
    for(int stride = N >> 2; stride > 0; stride >>= 2){
        int lo = pos & (stride - 1);
        int i0 = ((pos - lo) << 2) + lo;
        int i1 = i0 + stride;
        int i2 = i1 + stride;
        int i3 = i2 + stride;

        __syncthreads();
        DType D0 = s_data[i0];
        DType D1 = s_data[i1];
        DType D2 = s_data[i2];
        DType D3 = s_data[i3];

        DType T;
        T = D0; D0         = D0 + D2; D2         = T - D2;
        T = D1; D1         = D1 + D3; D3         = T - D3;
        T = D0; s_data[i0] = D0 + D1; s_data[i1] = T - D1;
        T = D2; s_data[i2] = D2 + D3; s_data[i3] = T - D3;
    }

    //Do single radix-2 stage for odd power of two
    if(log2N & 1){
        __syncthreads();
        for(int pos = threadIdx.x; pos < N / 2; pos += blockDim.x){
            int i0 = pos << 1;
            int i1 = i0 + 1;

            DType D0 = s_data[i0];
            DType D1 = s_data[i1];
            s_data[i0] = D0 + D1;
            s_data[i1] = D0 - D1;
        }
    }

    __syncthreads();
    for(int pos = threadIdx.x; pos < N; pos += blockDim.x)
        d_Dst[pos] = s_data[pos];
}


template <typename DType>
__global__ void fwtBatch2Kernel(
    DType *d_Output,
    DType *d_Input,
    int stride
)
{
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int   N = blockDim.x *  gridDim.x * 4;

    DType *d_Src = d_Input  + blockIdx.y * N;
    DType *d_Dst = d_Output + blockIdx.y * N;

    int lo = pos & (stride - 1);
    int i0 = ((pos - lo) << 2) + lo;
    int i1 = i0 + stride;
    int i2 = i1 + stride;
    int i3 = i2 + stride;

    DType D0 = d_Src[i0];
    DType D1 = d_Src[i1];
    DType D2 = d_Src[i2];
    DType D3 = d_Src[i3];

    DType T;
    T = D0;
    D0        = D0 + D2;
    D2        = T - D2;
    T = D1;
    D1        = D1 + D3;
    D3        = T - D3;
    T = D0;
    d_Dst[i0] = D0 + D1;
    d_Dst[i1] = T - D1;
    T = D2;
    d_Dst[i2] = D2 + D3;
    d_Dst[i3] = T - D3;
}


template <typename DType>
__global__ void rsKernel(
    DType *d_Output,
    DType *d_Input,
    DType *indices_p,
    int in_dim,
    int out_dim)
{
    const int real = blockIdx.x * blockDim.x + threadIdx.x;
    const int pos = real%out_dim;
    const int sample_n =  real/out_dim;
    //std::printf("thread: %d \n", real);
    if (pos>=out_dim){
        return;
    }
    int index = (int)*(indices_p + pos);
    DType *d_Dst = d_Output + (sample_n*out_dim+pos);
    DType *d_Src = d_Input + (sample_n*in_dim+index);
    *d_Dst = *d_Src;
}


template <typename DType>
void hadamardTransformG(Tensor<gpu, 2, DType> &out, Tensor<gpu, 2, DType> &value,Tensor<gpu, 1, DType> &indices) {

    int in_dim = (unsigned int) value.shape_[1];
    int n_samples = (unsigned int) value.shape_[0];
    int out_dim = (unsigned int) indices.shape_[1];
    DType *out_p = out.dptr_;
    DType *indices_p = indices.dptr_;
    DType *d_Data = value.dptr_;

    int log2N = in_dim <= 1 ? 0 : log2((double)(in_dim - 1)) + 1;
    int M = n_samples;
    const int THREAD_N = 256;
    int N = 1 << log2N;

    dim3 grid((1 << log2N) / (4 * THREAD_N), M, 1);


    for(; log2N > ELEMENTARY_LOG2SIZE; log2N -= 2, N >>= 2, M <<= 2){
        fwtBatch2Kernel<DType><<<grid, THREAD_N>>>(d_Data, d_Data, N / 4);
    }
    fwtBatch1Kernel<DType><<<M, N / 4, N * sizeof(DType)>>>(d_Data, d_Data, log2N);

    int cal = out_dim%THREAD_N == 0 ? 0:1;

    const int threads_per_block = min(THREAD_N, out_dim);// to make number of threads the same as input
    int nblocks = n_samples*((out_dim + threads_per_block - 1) / threads_per_block) ;
    rsKernel<DType><<<nblocks, threads_per_block>>>(out_p, d_Data, indices_p, in_dim, out_dim);

}


}
}


namespace mxnet {
namespace op {

template<typename xpu>
void hadamardTransformGeneral(const nnvm::NodeAttrs& attrs,
                       const OpContext& ctx,
                       const std::vector<TBlob>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& outputs) {
    using namespace mshadow;
    using namespace mshadow::expr;

    CHECK_EQ(inputs.size(), 2);
    CHECK_EQ(outputs.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();

    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {

            Tensor<xpu, 2, DType> out = outputs[0].FlatTo2D<xpu, DType>(s);
            Tensor<xpu, 2, DType> value = inputs[0].FlatTo2D<xpu, DType>(s);
            Tensor<xpu, 1, DType> indices = inputs[1].FlatTo1D<xpu, DType>(s);

            mshadow::cuda::hadamardTransformG<DType>(out, value, indices);

    });
}


NNVM_REGISTER_OP(dense_inplace)
.set_attr<FCompute>("FCompute<gpu>", hadamardTransformGeneral<gpu>);

NNVM_REGISTER_OP(_backward_dense_inplace)
.set_attr<FCompute>("FCompute<gpu>", hadamardTransformGeneral<gpu>);

}
}

#endif
#endif
