#include "./hadamard_op.h"
//#include "./elemwise_binary_op.h"
//#include "./elemwise_binary_broadcast_op.h"

namespace mxnet {
namespace op {


template<typename xpu>
void hadamardTransform(const nnvm::NodeAttrs& attrs,
                       const OpContext& ctx,
                       const std::vector<TBlob>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& outputs) {

  using namespace mshadow;

  CHECK_EQ(inputs.size(), 3);
  CHECK_EQ(outputs.size(), 1);
  Stream<xpu> *s = ctx.get_stream<xpu>();

  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {

    Tensor<xpu, 2, DType> out = outputs[0].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> value = inputs[0].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 1, DType> indices = inputs[1].FlatTo1D<xpu, DType>(s);
    Tensor<xpu, 1, DType> sign = inputs[2].FlatTo1D<xpu, DType>(s);
    unsigned int in_dim = (unsigned int) value.shape_[1];
    unsigned int n_samples = (unsigned int) value.shape_[0];
    DType *out_p = out.dptr_;

    int log2d = in_dim <= 1 ? 0 : log2((double)(in_dim - 1)) + 1;
    DType temp;
    unsigned int k = (unsigned int) indices.shape_[1];

    for (int sample = 0; sample < n_samples; sample++) {
      DType *sign_p = sign.dptr_;

      for (int t = log2d; t; t--) {

        int blockSize = 1 << t;
        int numBlocks = 1 << (log2d - t);

        int halfBlockSize = blockSize >> 1;
        DType *p1 = value.dptr_;
        DType *p2 = value.dptr_ + halfBlockSize;

        for (int blockIndex = 0; blockIndex < numBlocks; blockIndex++) {
          for (int i = 0; i < halfBlockSize; i++) {
            if (t == log2d){
              *p1 *= *(sign_p + halfBlockSize * (blockIndex) + i);
              *p2 *= *(sign_p + halfBlockSize * (blockIndex + 1) + i);
            }
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

      DType *indices_p = indices.dptr_;
      DType *input_p = value.dptr_;
      for (int i = 0; i < k; i++) {
        int index = (int) *indices_p;
        *out_p = *(input_p + index);
        out_p++;
        indices_p++;
      }
      value.dptr_ += in_dim;
    }

  });
}


template<typename xpu>
void hadamardTransform_backwards(const nnvm::NodeAttrs& attrs,
                                 const OpContext& ctx,
                                 const std::vector<TBlob>& inputs,
                                 const std::vector<OpReqType>& req,
                                 const std::vector<TBlob>& outputs) {
  using namespace mshadow;

  CHECK_EQ(inputs.size(), 4);
  CHECK_EQ(outputs.size(), 3);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {

    Tensor < xpu, 2, DType > in_grad = inputs[0].FlatTo2D<xpu, DType>(s);
    Tensor < xpu, 2, DType > input_data = inputs[1].FlatTo2D<xpu, DType>(s);
    Tensor < xpu, 1, DType > input_indices = inputs[2].FlatTo1D<xpu, DType>(s);
    Tensor < xpu, 1, DType > sign = inputs[3].FlatTo1D<xpu, DType>(s);
    Tensor < xpu, 2, DType > out_grad = outputs[0].FlatTo2D<xpu, DType>(s);

    DType *in_grad_p = in_grad.dptr_;
    DType *input_data_p = input_data.dptr_;
    DType *input_indices_p = input_indices.dptr_;
    DType *out_grad_p = out_grad.dptr_;

    Tensor<xpu, 2, DType> workspace = ctx.requested[0].get_space_typed<xpu, 2, DType>(mshadow::Shape2(input_indices.shape_[1], input_data.shape_[1]), s);
    DType *workspace_p = workspace.dptr_;

    int k = input_indices.shape_[1];
    int nnz = workspace.shape_[1];
    DType *pIndices = input_indices.dptr_;

    for (int i = 0; i<k; i++) {
      DType *sign_p = sign.dptr_;
      for (int j = 0; j<nnz; j++) {
        int index = (int) *pIndices;
        int keyvalue = j;
        int signvalue = *(sign_p+j);
        *workspace_p = ((__builtin_popcount(index & keyvalue) & 1) * -2 + 1)*signvalue;
        workspace_p++;
      }
      pIndices++;
    }
    ASSIGN_DISPATCH(out_grad, req[0], dot(in_grad, workspace));

  });
}


MXNET_OPERATOR_REGISTER_HADAMARD(hadamard_dense)
.set_attr<FCompute>("FCompute<cpu>", hadamardTransform<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_hadamard_dense"});

NNVM_REGISTER_OP(_backward_hadamard_dense)
.set_num_inputs(4)
.set_num_outputs(3)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FResourceRequest>("FResourceRequest",
[](const NodeAttrs& attrs) {
return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr<FCompute>("FCompute<cpu>", hadamardTransform_backwards<cpu>);
}
}
