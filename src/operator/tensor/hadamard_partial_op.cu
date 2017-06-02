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





// template<typename DType>
// void partialDenseInplace(DType *values, int d, DType *indices, DType *sign_p, int k1, int k2, int offset, int depth, DType *stack) {
//     int j = 0;
//     stack[j++] = k1;
//     stack[j++] = k2;
//     stack[j++] = d;
//     stack[j++] = offset;
//     stack[j++] = depth;
//     stack[j++] = 0;   
//     while (j > 0){
//       int value_add = stack[--j];
//       depth = stack[--j];
//       offset = stack[--j];
//       d =stack[--j];
//       k2 = stack[--j]; 
//       k1 = stack[--j];
//       if (d <= 1 or k1 >= k2)
//           continue;
//       int d2 = d / 2;
//       DType temp;
//       DType *p1 = values+value_add;
//       DType *p2 = values + value_add + d2;
//       for (int i = d2; i; i--) {
//           if (depth == 0){
//             *p1 *= *(sign_p + d2-i);
//             *p2 *= *(sign_p + d2 + d2-i);
//           }
//           temp = *p1;
//           *p1 += *p2;
//           *p2 = temp - *p2;
//           p1++;
//           p2++;
//       }
//       int kd2 = searchSortedIndices<DType>(d2 + offset, indices + k1, k2 - k1) + k1;

//       if (kd2 > k1) {//some indices are in the first half
//           // partialDenseInplace<DType>(values, d2, indices, sign_p, k1, kd2, offset, depth+1);
//         stack[j++] = k1;
//         stack[j++] = kd2;
//         stack[j++] = d2;
//         stack[j++] = offset;
//         stack[j++] = depth+1;
//         stack[j++] = value_add;
//       }
//       if (kd2 < k2) { //some indices are in the second half
//           // partialDenseInplace<DType>(values + d2, d2, indices, sign_p, kd2, k2, offset + d2, depth+1);
//         stack[j++] = kd2;
//         stack[j++] = k2;
//         stack[j++] = d2;
//         stack[j++] = offset+d2;
//         stack[j++] = depth+1;
//         stack[j++] = value_add + d2;
//       }

//   }

// }

template <typename DType>
__global__ void loopkernel(
  int d2,
  DType *sign_p,
  DType *p1,
  DType *p2,
  int depth,
  int n_samples,
  int dim)
{
  
  int blockId   = blockIdx.y * gridDim.x + blockIdx.x;        
  int real = blockId * blockDim.x + threadIdx.x; 
  int index = real%d2;
  int sample = real/d2;
  // printf("blockDim.y is: %d\n", real);
  if (real>=d2*n_samples){
    return;
  }


  p1+=dim*sample+index;
  p2+=dim*sample+index;
  if (depth == 0){
    *p1 *= *(sign_p + index);
    *p2 *= *(sign_p + d2 + index);
  }
  
  DType temp = *p1;
  *p1 += *p2;
  *p2 = temp - *p2;

}

// template<typename DType>
// __global__ void searchSortedIndices(int index, DType *sortedIndices, int k, int* result) {

//     if (k == 0){
//       *result = 0;
//       return;
//     }
    
//     int left = 0;
//     if (index <= (int)(*(sortedIndices+left))){
//       *result = left;
//       return;
//     }
    
//     int right = k - 1;
//     if (index > (int)(*(sortedIndices+right))){
//       *result = right + 1;
//       return;
//     }
//     while (left + 1 < right) {
//         int mid = (left + right) / 2;
//         if (index <= (int)(*(sortedIndices+mid))) {
//             right = mid;
//             if (index > (int)(*(sortedIndices+right))){
//               *result = right + 1;
//               return;
//             } 
//         } else {
//             left = mid;
//             if (index <= (int)(*(sortedIndices+left))){
//               *result = left;
//               return;
//             }
//         }
//     }
//     *result = left + 1;

// }

}
}


namespace mxnet {
namespace op {
template<typename DType>
void searchSortedIndices(int index, DType *sortedIndices, int k, int* result) {

    if (k == 0){
      *result = 0;
      return;
    }
    int left = 0;
    if (index <= (int)(*(sortedIndices+left))){
      *result = left;
      return;
    }
    
    int right = k - 1;
    if (index > (int)(*(sortedIndices+right))){
      *result = right + 1;
      return;
    }
    while (left + 1 < right) {
        int mid = (left + right) / 2;
        if (index <= (int)(*(sortedIndices+mid))) {
            right = mid;
            if (index > (int)(*(sortedIndices+right))){
              *result = right + 1;
              return;
            } 
        } else {
            left = mid;
            if (index <= (int)(*(sortedIndices+left))){
              *result = left;
              return;
            }
        }
    }
    *result = left + 1;

}


template<typename DType>
void partialDenseInplace_recursive(DType *values, int d, DType *indices, DType *sign_p, int k1, int k2, int offset, int depth, int n_samples , Stream<gpu> *s, DType *copy, int dim) {
    if (d <= 1 or k1 >= k2)
        return;
    int d2 = d / 2;
    DType *p1 = values;

    DType *p2 = values + d2;
    const int THREAD_N = 256;
    const int threads_per_block = min(THREAD_N, d2*n_samples);
    int nblocks = ((d2*n_samples + threads_per_block - 1) / threads_per_block) ;

    // dim3 grid_sample(nblocks, n_samples, 1);

    mshadow::cuda::loopkernel<DType><<<nblocks, threads_per_block, 0, s->stream_>>>(d2, sign_p, p1, p2,depth, n_samples, dim);

    int* h_answer;
    h_answer = (int *)malloc(sizeof(int));

    // LOG(INFO)<<"success??"<< depth;
    searchSortedIndices<DType>(d2 + offset, copy + k1, k2 - k1, h_answer);
    // LOG(INFO)<<"success2??"<< depth;
    // int arHost[1];
    // cudaMemcpy(&h_answer, copy, sizeof(int), cudaMemcpyDeviceToHost); 
    
    // LOG(INFO)<<"success??"<< *h_answer;
    int kd2 = *h_answer + k1;
    if (kd2 > k1) {//some indices are in the first half
        partialDenseInplace_recursive<DType>(values, d2, indices, sign_p, k1, kd2, offset, depth+1, n_samples, s, copy, dim);
    }
    if (kd2 < k2) { //some indices are in the second half
        partialDenseInplace_recursive<DType>(values + d2, d2, indices, sign_p, kd2, k2, offset + d2, depth+1, n_samples, s, copy, dim);
    }
    return;
}

template <typename xpu, typename DType>
void hadamardTransformpartialG(Tensor<xpu, 2, DType> &out, Tensor<xpu, 2, DType> &value, Tensor<xpu, 2, DType> &indices, Tensor<xpu, 2, DType> &sign, Stream<xpu> *s, Tensor<xpu, 1, DType> &space) {

  int in_dim = (unsigned int) value.shape_[1];
  int n_samples = (unsigned int) value.shape_[0];
  int out_dim = (unsigned int) indices.shape_[1];
  DType *out_p = out.dptr_;
  DType *indices_p = indices.dptr_;
  DType *sign_p = sign.dptr_;
  DType *d_Data = value.dptr_;
  // DType *space_p = space.dptr_;
  const int THREAD_N = 256;
  DType* copy;
  copy = (DType *)malloc(sizeof(DType)*out_dim); 
  cudaMemcpy(copy, indices_p, sizeof(DType)*out_dim, cudaMemcpyDeviceToHost);
  partialDenseInplace_recursive<DType>(d_Data, in_dim, indices_p,sign_p, 0, out_dim, 0, 0, n_samples, s, copy, in_dim);


  const int threads_per_block = min(THREAD_N, out_dim);// to make number of threads the same as input

  int nblocks = ((out_dim + threads_per_block - 1) / threads_per_block) ;
  dim3 grid_sample(nblocks, n_samples, 1);
  mshadow::cuda::rsKernel<DType><<<grid_sample, threads_per_block, 0, s->stream_ >>>(out_p, d_Data, indices_p, in_dim, out_dim);

}

template<typename xpu>
void hadamardPartialTransformGeneral(const nnvm::NodeAttrs& attrs,
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
    Tensor < xpu, 2, DType > indices = inputs[1].FlatTo2D<xpu, DType>(s);
    Tensor < xpu, 2, DType > sign = inputs[2].FlatTo2D<xpu, DType>(s);
    Tensor < xpu, 2, DType > out = outputs[0].FlatTo2D<xpu, DType>(s);
    Tensor <xpu, 1, DType> workspace = ctx.requested[0].get_space_typed<xpu, 1, DType>(mshadow::Shape1(1), s);

    hadamardTransformpartialG<xpu, DType>(out, value, indices, sign, s, workspace);

  });
}


NNVM_REGISTER_OP(hadamard_partial_dense)
.set_attr<FCompute>("FCompute<gpu>", hadamardPartialTransformGeneral<gpu>);

NNVM_REGISTER_OP(_backward_hadamard_partial_dense)
.set_attr<FCompute>("FCompute<gpu>", hadamardPartialTransformGeneral<gpu>);

}
}

#endif
#endif
