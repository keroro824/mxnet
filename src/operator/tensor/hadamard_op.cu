#include "./hadamard_op.h"
#include "./elemwise_binary_op.h"
#include "./elemwise_binary_broadcast_op.h"
#include <mshadow/tensor.h>

#define WARPS_PER_BLOCK 1
#define THREADS_PER_BLOCK 512

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)


namespace mshadow {
namespace cuda {




__device__ void atomic_add(float* dst, float val) {
	atomicAdd(dst, val);
}

// for double precision
__device__ void atomic_add(double* address, double val) {
  // code example in the official document at:
  // http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
  //      #atomic-functions

  // NOLINT_NEXT_LINE(runtime/int)
  unsigned long long int* address_as_ull = (unsigned long long int*) address;
  // NOLINT_NEXT_LINE(runtime/int)
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(val + __longlong_as_double(assumed)));
    // Note: uses integer comparison to avoid hang in case of NaN
    // (since NaN != NaN)
  } while (assumed != old);
}



template <typename DType>
__global__ void hadamard_forward_kernel(const int nthreads, DType *out, DType *indices_p, DType *in, int n_smaples, int in_dim, int out_dim) {


	// get the target location in the output
     //   const int target = i_sample*out_dim + h[i_indim];
      //  atomic_add(out + target, s[i_indim] * in[index]);


    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= nthreads){
        return;
    }

    int log2d = in_dim <= 1 ? 0 : log2((double)(in_dim - 1)) + 1;
    DType temp;
    unsigned int k = out_dim;


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
    for (int i = 0; i < k; i++) {
        int index = (int) *indices_p;
        *out = *(input_p + index);

        out++;
        indices_p++;
    }

}


template <typename DType>
inline void hadamardTransformG(Tensor<gpu, 2, DType> &out, Tensor<gpu, 2, DType> &value,Tensor<gpu, 1, DType> &indices) {

    int in_dim = (unsigned int) value.shape_[1];
    int n_samples = (unsigned int) value.shape_[0];
    int out_dim = (unsigned int) indices.shape_[1];
    DType *out_p = out.dptr_;
    DType *in_p = value.dptr_;
    DType *indices_p = indices.dptr_;
    int processing_batch_size = 32;

    int upper_bound = n_samples/processing_batch_size;
    if (n_samples%processing_batch_size == 0){
      upper_bound = upper_bound-1;
    }
    upper_bound = upper_bound>0? upper_bound:0;

    int bstart = 0;
    for ( int i = 0; i <= upper_bound; i++ ){
        int batchlen = min(processing_batch_size, n_samples - bstart );
        int nthreads = batchlen * in_dim;
        int threads_per_block = min(THREADS_PER_BLOCK, nthreads);
        int nblocks = (nthreads + threads_per_block - 1) / threads_per_block ;
        printf("n_samples %d  upper_bound %d, nthreads %d, nblocks %d", n_samples, upper_bound, nthreads, nblocks);
        hadamard_forward_kernel<DType><<<nblocks, threads_per_block>>>(nthreads, out_p+bstart*out_dim, indices_p, in_p+bstart*in_dim, batchlen, in_dim, out_dim);
        bstart = (i+1)*batchlen;


    }

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