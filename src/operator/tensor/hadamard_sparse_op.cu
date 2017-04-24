#include "./hadamard_sparse_op.h"
#include "./elemwise_binary_op.h"
#include "./elemwise_binary_broadcast_op.h"
#include <mshadow/tensor.h>

#define WARPS_PER_BLOCK 1
#define THREADS_PER_BLOCK 256
#define ELEMENTARY_LOG2SIZE 11

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)


namespace mshadow {
namespace cuda {


template <typename DType>
__global__ void hadamard_sparse_forward_kernel(const int nthreads, DType *out, DType *indices, DType *value, DType *key, int in_dim, int out_dim) {


   const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= nthreads){
         return;
     }

    int k = out_dim;
    int col = index;
    int nnz = in_dim;
    DType *pValues = value;
    DType *pKeys = key;
    DType *pIndices = indices;

    for (int j = nnz; j; j--) {

            int ind = (int) *(pIndices+col);
            int row = (int) *pKeys;
            int keyvalue = (int) *(pKeys+1);
            DType *pRes = out;
            pRes += row*k+col;
            *pRes += ((__popcll(ind & keyvalue) & 1) * -2 + 1) * (*pValues);

            pKeys+=2;
            pValues++;

    }
}


template <typename DType>
inline void hadamardTransformGSparse(Tensor<gpu, 2, DType> &out, Tensor<gpu, 1, DType> &value, Tensor<gpu, 2, DType> &key, Tensor<gpu, 1, DType> &indices) {

    int in_dim = (unsigned int) key.shape_[0];
    int n_samples = (unsigned int) out.shape_[0];
    int out_dim = (unsigned int) indices.shape_[1];
    out = 0;
    DType *out_p = out.dptr_;
    DType *value_p = value.dptr_;
    DType *key_p = key.dptr_;

    DType *indices_p = indices.dptr_;
    int processing_batch_size = 2<<12;
    LOG(INFO)<<processing_batch_size;
    int upper_bound = in_dim/processing_batch_size;
    if (in_dim%processing_batch_size == 0){
      upper_bound = upper_bound-1;
    }
    upper_bound = upper_bound>0? upper_bound:0;

    int bstart = 0;
    for ( int i = 0; i <= upper_bound; i++ ){
        int batchlen = min(processing_batch_size, in_dim - bstart );
        int threads_per_block = min(THREADS_PER_BLOCK, batchlen);
        int nblocks = (batchlen + threads_per_block - 1) / threads_per_block ;

        hadamard_sparse_forward_kernel<DType><<<nblocks, threads_per_block>>>(out_dim, out_p, indices_p, value_p+bstart, key_p+bstart*2, batchlen, out_dim);
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
            Tensor<xpu, 1, DType> value = inputs[1].FlatTo1D<xpu, DType>(s);
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