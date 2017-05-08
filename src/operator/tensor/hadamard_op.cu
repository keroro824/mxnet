#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mshadow/tensor.h>
#include <cuda_device_runtime_api.h>
#include <cuda.h>
#include "./hadamard_op.h"
#include "../elemwise_op_common.h"
#include "../mxnet_op.h"
#include "broadcast_reduce_op.h"
#include "./elemwise_binary_op.h"
#include "./elemwise_binary_broadcast_op.h"

#ifndef FWT_KERNEL_CUH
#define FWT_KERNEL_CUH
#ifndef fwt_kernel_cuh
#define fwt_kernel_cuh
#define ELEMENTARY_LOG2SIZE 11
#define THREADS_PER_BLOCK 256


namespace mshadow {
namespace cuda {

//Borrowed from NVDIA Implementation
template <typename DType>
__global__ void fwtBatch1Kernel(DType *d_Output, DType *d_Input, int log2N){
  const int    N = 1 << log2N;
  const int base = blockIdx.x << log2N;

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
  int blockId   = blockIdx.y * gridDim.x + blockIdx.x;        
  int real = blockId * blockDim.x + threadIdx.x; 
  const int pos = real%out_dim;
  const int sample_n =  real/out_dim;
  if (pos>=out_dim){
    return;
  }
  int index = (int)*(indices_p + pos);
  DType *d_Dst = d_Output + (sample_n*out_dim+pos);
  DType *d_Src = d_Input + (sample_n*in_dim+index);
  *d_Dst = *d_Src;
}


template <typename DType>
__global__ void signKernel(
  DType *d_Input,
  DType *sign,
  int in_dim,
  int nsamples)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index>=in_dim*nsamples){
    return;
  }
  *(d_Input + index) *= *(sign + index % in_dim);
}


template <typename DType>
void hadamardTransformG(Tensor<gpu, 2, DType> &out, Tensor<gpu, 2, DType> &value, Tensor<gpu, 1, DType> &indices, Tensor<gpu, 1, DType> &sign) {

  int in_dim = (unsigned int) value.shape_[1];
  int n_samples = (unsigned int) value.shape_[0];
  int out_dim = (unsigned int) indices.shape_[1];
  DType *out_p = out.dptr_;
  DType *indices_p = indices.dptr_;
  DType *sign_p = sign.dptr_;
  DType *d_Data = value.dptr_;

  int log2N = in_dim <= 1 ? 0 : log2((double)(in_dim - 1)) + 1;
  int M = n_samples;
  const int THREAD_N = 256;
  int N = 1 << log2N;

  signKernel<DType><<< (M*in_dim-1)/THREAD_N+1, THREAD_N>>>(d_Data, sign_p, in_dim, M);
  dim3 grid((1 << log2N) / (4 * THREAD_N), M, 1);
  for(; log2N > ELEMENTARY_LOG2SIZE; log2N -= 2, N >>= 2, M <<= 2){
    fwtBatch2Kernel<DType><<<grid, THREAD_N>>>(d_Data, d_Data, N / 4);
  }
  fwtBatch1Kernel<DType><<<M, N / 4, N * sizeof(DType)>>>(d_Data, d_Data, log2N);

  const int threads_per_block = min(THREAD_N, out_dim);// to make number of threads the same as input

  int nblocks = ((out_dim + threads_per_block - 1) / threads_per_block) ;
  dim3 grid_sample(nblocks, n_samples, 1);
  rsKernel<DType><<<grid_sample, threads_per_block>>>(out_p, d_Data, indices_p, in_dim, out_dim);

}


template <typename DType>
__global__ void hadamard_sparse_backward_kernel(DType *out, DType *indices, DType *key, DType *sign, int in_dim, int out_dim) {


  const int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index >= in_dim*out_dim){
    return;
  }
  int ind = (int) *(indices+index/in_dim);
  int keyvalue = index%in_dim;
  int signvalue = (int) *(sign+index%in_dim);
  *(out+index) = ((__popcll(ind & keyvalue) & 1) * -2 + 1) * signvalue;

}


template <typename DType>
inline void hadamardTransformBSparse(Tensor<gpu, 2, DType> &key, Tensor<gpu, 1, DType> &indices, Tensor<gpu, 1, DType> &sign, Tensor<gpu, 2, DType> &in_grad, Tensor<gpu, 2, DType> &workspace) {

  int in_dim = (unsigned int) key.shape_[1];
  int n_samples = (unsigned int) key.shape_[0];
  int out_dim = (unsigned int) indices.shape_[1];

  DType *key_p = key.dptr_;
  DType *workspace_p = workspace.dptr_;
  DType *indices_p = indices.dptr_;
  DType *sign_p = sign.dptr_;

  int batchlen = in_dim*out_dim;
  int threads_per_block = THREADS_PER_BLOCK;
  int nblocks = (batchlen + threads_per_block - 1) / threads_per_block ;

  hadamard_sparse_backward_kernel<DType><<<nblocks, threads_per_block>>>(workspace_p, indices_p, key_p, sign_p, in_dim, out_dim);

}


template <typename DType>
__global__ void hadamard_sparse_backward_kernel(DType *out, DType *indices, DType *key, int in_dim, int out_dim) {


   const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= in_dim*out_dim){
         return;
     }
    int ind = (int) *(key+index/in_dim);
    int keyvalue = index%in_dim;
    *(out+index) = ((__popcll(ind & keyvalue) & 1) * -2 + 1) ;

}


template <typename DType>
inline void hadamardTransformBSparse(Tensor<gpu, 2, DType> &key, Tensor<gpu, 1, DType> &indices, Tensor<gpu, 2, DType> &in_grad, Tensor<gpu, 2, DType> &workspace) {

    int in_dim = (unsigned int) key.shape_[1];
    int n_samples = (unsigned int) key.shape_[0];
    int out_dim = (unsigned int) indices.shape_[1];


    DType *key_p = key.dptr_;
    DType *workspace_p = workspace.dptr_;

    DType *indices_p = indices.dptr_;

    int batchlen = in_dim*out_dim;
    int threads_per_block = THREADS_PER_BLOCK;
    int nblocks = (batchlen + threads_per_block - 1) / threads_per_block ;

    hadamard_sparse_backward_kernel<DType><<<nblocks, threads_per_block>>>(workspace_p, indices_p, key_p, in_dim, out_dim);

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

  CHECK_EQ(inputs.size(), 3);
  CHECK_EQ(outputs.size(), 1);
  Stream<xpu> *s = ctx.get_stream<xpu>();

  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {

    Tensor < xpu, 2, DType > value = inputs[0].FlatTo2D<xpu, DType>(s);
    Tensor < xpu, 1, DType > indices = inputs[1].FlatTo1D<xpu, DType>(s);
    Tensor < xpu, 1, DType > sign = inputs[2].FlatTo1D<xpu, DType>(s);
    Tensor < xpu, 2, DType > out = outputs[0].FlatTo2D<xpu, DType>(s);

    mshadow::cuda::hadamardTransformG<DType>(out, value, indices, sign);

  });
}


template<typename xpu>
void hadamardTransformBack(const nnvm::NodeAttrs& attrs,
                       const OpContext& ctx,
                       const std::vector<TBlob>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;

  CHECK_EQ(inputs.size(), 4);
  CHECK_EQ(outputs.size(), 3);
  Stream<xpu> *s = ctx.get_stream<xpu>();

  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {

    Tensor < xpu, 2, DType > in_grad = inputs[0].FlatTo2D<xpu, DType>(s);
    Tensor < xpu, 2, DType > input_data = inputs[1].FlatTo2D<xpu, DType>(s);
    Tensor < xpu, 1, DType > input_indices = inputs[2].FlatTo1D<xpu, DType>(s);
    Tensor < xpu, 1, DType > sign = inputs[3].FlatTo1D<xpu, DType>(s);
    Tensor < xpu, 2, DType > out_grad = outputs[0].FlatTo2D<xpu, DType>(s);
    Tensor <xpu, 2, DType> workspace = ctx.requested[0].get_space_typed<xpu, 2, DType>(mshadow::Shape2(input_indices.shape_[1], input_data.shape_[1]), s);

    mshadow::cuda::hadamardTransformBSparse<DType>(input_data, input_indices, sign, in_grad, workspace);
    ASSIGN_DISPATCH(out_grad, req[0], dot(in_grad, workspace));
  });
}


NNVM_REGISTER_OP(hadamard_dense)
.set_attr<FCompute>("FCompute<gpu>", hadamardTransformGeneral<gpu>);

NNVM_REGISTER_OP(_backward_hadamard_dense)
.set_attr<FCompute>("FCompute<gpu>", hadamardTransformBack<gpu>);

}
}

#endif
#endif
