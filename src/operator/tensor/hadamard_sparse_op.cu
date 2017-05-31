#include "./hadamard_sparse_op.h"
#include "./elemwise_binary_op.h"
#include "./elemwise_binary_broadcast_op.h"
#include "../operator_common.h"
#include <mshadow/tensor.h>
#include "../mshadow_op.h"
#include "../elemwise_op_common.h"
#include "./elemwise_binary_op.h"

#include "../mxnet_op.h"
#include "broadcast_reduce_op.h"

#define WARPS_PER_BLOCK 1
#define THREADS_PER_BLOCK 256


namespace mshadow {
namespace cuda {


template <typename DType>
__global__ void hadamard_sparse_forward_kernel(const int nthreads, DType *out, DType *indices, DType *value, DType *key, int in_dim, int out_dim, DType *sign, DType *save) {


   // const int index = blockIdx.x * blockDim.x + threadIdx.x;
  //     int blockId   = blockIdx.y * gridDim.x + blockIdx.x;        
  // int real = blockId * blockDim.x + threadIdx.x; 
  // const int index = real%out_dim;
  // const int sample_n =  real/out_dim;

    int col1 = blockIdx.x*blockDim.x+threadIdx.x;
  int row = blockIdx.y*blockDim.y+threadIdx.y;
  const int real = col1 + row * gridDim.x *blockDim.x ;
const int index = real%out_dim;
const int sample_n =  real/out_dim;
    if (real >= nthreads){
         return;
     }

    int k = out_dim;
    int col = index;
    //int nnz = in_dim;

    DType *pKeys = key;
    DType *pIndices = indices;
    int sample = sample_n;
    int start;
    if (sample==0){
        start = 0;
        
    }else{
        start = *(save+sample-1); 
    }
    
    int end = *(save+sample);
    DType *pValues = value+start;

   
    for (int j = start; j<end; j++) {

            int ind = (int) *(pIndices+col);
            int row = (int) *(pKeys+j*2);
            int keyvalue = (int) *(pKeys+j*2+1);
            int signvalue = (int) *(sign+ keyvalue);
            DType *pRes = out;
            pRes += row*k+col;
            //printf("hello everyone %d %d %d %d %d\n", index, ind, j, row, keyvalue);
            *pRes += ((__popcll(ind & keyvalue) & 1) * -2 + 1) * (*pValues)* signvalue;

            //pKeys+=2;
            pValues++;

    }
}


template <typename DType>
inline void hadamardTransformGSparse(Tensor<gpu, 2, DType> &out, Tensor<gpu, 1, DType> &value, Tensor<gpu, 2, DType> &key, Tensor<gpu, 1, DType> &indices, Tensor<gpu, 1, DType> &sign, Tensor<gpu, 1, DType> &save) {

    int in_dim = (unsigned int) key.shape_[0];
    int n_samples = (unsigned int) out.shape_[0];
    int out_dim = (unsigned int) indices.shape_[1];
    out = 0;
    DType *out_p = out.dptr_;
    DType *value_p = value.dptr_;
    DType *key_p = key.dptr_;
    DType *sign_p = sign.dptr_;
    DType *save_p = save.dptr_;

    DType *indices_p = indices.dptr_;

    //for ( int i = 0; i <= upper_bound; i++ ){
    //    int batchlen = min(processing_batch_size, n_samples - bstart );
        int nthreads = out_dim*n_samples;
        int threads_per_block = min(THREADS_PER_BLOCK, nthreads);
        // int nblocks = (nthreads + threads_per_block - 1) / threads_per_block ;

        //printf("n_samples %d  upper_bound %d, nthreads %d, nblocks %d", n_samples, upper_bound, nthreads, nblocks);
        //LOG(INFO)<<out_dim<<in_dim<<nthreads<<threads_per_block<<nblocks;
        int nblocks = ((nthreads + threads_per_block - 1) / threads_per_block) ;
        // dim3 grid_sample(nblocks, n_samples, 1);
        dim3 dimBlock(threads_per_block,1);
        dim3 dimGrid(int(ceil(sqrt(nblocks))), int(ceil(sqrt(nblocks))), 1);
        hadamard_sparse_forward_kernel<DType><<<dimGrid, dimBlock>>>(nthreads, out_p, indices_p, value_p, key_p, in_dim, out_dim, sign_p, save_p);
   //     bstart = (i+1)*batchlen;

   // }

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

    CHECK_EQ(inputs.size(), 5);
    CHECK_EQ(outputs.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();

    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {

            Tensor<xpu, 2, DType> out = outputs[0].FlatTo2D<xpu, DType>(s);
            Tensor<xpu, 2, DType> key = inputs[0].FlatTo2D<xpu, DType>(s);
            Tensor<xpu, 1, DType> value = inputs[1].FlatTo1D<xpu, DType>(s);
            Tensor<xpu, 1, DType> indices = inputs[2].FlatTo1D<xpu, DType>(s);
            Tensor<xpu, 1, DType> sign = inputs[3].FlatTo1D<xpu, DType>(s);
            Tensor<xpu, 1, DType> workspace = inputs[4].FlatTo1D<xpu, DType>(s);
            // Tensor<xpu, 1, DType> workspace = ctx.requested[0].get_space_typed<xpu, 1, DType>(mshadow::Shape1(out.shape_[0]), s);
            mshadow::cuda::hadamardTransformGSparse<DType>(out, value, key,  indices, sign, workspace);


    });
}


NNVM_REGISTER_OP(hadamard_sparse)
.set_attr<FCompute>("FCompute<gpu>", hadamardTransformGeneralSparse<gpu>);

NNVM_REGISTER_OP(_backward_hadamard_sparse)
.set_attr<FCompute>("FCompute<gpu>", hadamardTransformGeneralSparse<gpu>);

}
}