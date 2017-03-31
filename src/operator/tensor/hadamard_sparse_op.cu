#include "./hadamard_sparse_op.h"
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




template <typename DType>
__global__ void hadamard_sparse_forward_kernel(const int nthreads, DType *out, DType *indices_p, DType *value, DType *key, int n_smaples, int in_dim, int out_dim) {


	// get the target location in the output
     //   const int target = i_sample*out_dim + h[i_indim];
      //  atomic_add(out + target, s[i_indim] * in[index]);


    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= nthreads){
        return;
    }

    int k = out_dim;
    //int nnz = in_dim;

    DType *pKeys = key;
    DType *pValues = value;
    out = 0;

   // for (int j = nnz; j ; j--) {
        DType *pRes = out;
        pKeys += index;
        pValues += index;

        for (int i = k; i; i--) {
            int index = (int) *indices_p;
            int keyvalue = (int) *pKeys;

            *pRes += ((__popcll(index & keyvalue) & 1)*-2 +1) * (*pValues);
            pRes++; indices_p++;
        }
    //    pKeys++; pValues++;
    //}


}


template <typename DType>
inline void hadamardTransformGSparse(Tensor<gpu, 2, DType> &out, Tensor<gpu, 2, DType> &value, Tensor<gpu, 2, DType> &key, Tensor<gpu, 1, DType> &indices) {

    int in_dim = (unsigned int) key.shape_[1];
    int n_samples = (unsigned int) key.shape_[0];
    int out_dim = (unsigned int) indices.shape_[1];
    DType *out_p = out.dptr_;
    DType *value_p = value.dptr_;
    DType *key_p = key.dptr_;

    DType *indices_p = indices.dptr_;
    int processing_batch_size = 128;

    int upper_bound = n_samples/processing_batch_size;
    if (n_samples%processing_batch_size == 0){
      upper_bound = upper_bound-1;
    }
    upper_bound = upper_bound>0? upper_bound:0;

    int bstart = 0;
    for ( int i = 0; i <= upper_bound; i++ ){
        int batchlen = min(processing_batch_size, n_samples - bstart );
        int nthreads = batchlen*in_dim;
        int threads_per_block = min(THREADS_PER_BLOCK, nthreads);
        int nblocks = (nthreads + threads_per_block - 1) / threads_per_block ;
        //printf("n_samples %d  upper_bound %d, nthreads %d, nblocks %d", n_samples, upper_bound, nthreads, nblocks);


        hadamard_sparse_forward_kernel<DType><<<nblocks, threads_per_block>>>(nthreads, out_p+bstart*out_dim, indices_p, value_p+bstart*in_dim, key_p+bstart*in_dim, batchlen, in_dim, out_dim);
        bstart = (i+1)*batchlen;


    }

}
}
}

namespace mxnet {
namespace op {


template<typename xpu>
void hadamardTransformGeneralSparse(const nnvm::NodeAttrs& attrs,
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

            Tensor<xpu, 2, DType> out = outputs[0].FlatTo2D<xpu, DType>(s);
            Tensor<xpu, 2, DType> key = inputs[0].FlatTo2D<xpu, DType>(s);
            Tensor<xpu, 2, DType> value = inputs[1].FlatTo2D<xpu, DType>(s);
            Tensor<xpu, 1, DType> indices = inputs[2].FlatTo1D<xpu, DType>(s);

            mshadow::cuda::hadamardTransformGSparse<DType>(out, value, key,  indices);

    });
}


NNVM_REGISTER_OP(sparse_inplace)
.set_attr<FCompute>("FCompute<gpu>", hadamardTransformGeneralSparse<gpu>);

NNVM_REGISTER_OP(_backward_sparse_inplace)
.set_attr<FCompute>("FCompute<gpu>", hadamardTransformGeneralSparse<gpu>);

}
}