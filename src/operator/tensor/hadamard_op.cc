#include "./hadamard_op.h"

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


NNVM_REGISTER_OP(hadamard_dense)
.describe(R"code(Compute the Subsampled Randomized Hadamard Transform of the input data.

Computes \Omega * X where X is the input data and \Omega is the SRH matrix defined by

\Omega = \frac{n}{l} R H D

The input data is expected to have shape: ``(num_samples, num_features)`` where
`num_features` has been padded with zeros to have its dimension be a power of 2.

The R matrix is a matrix of SIZE samples from R^XXX.

The D matrix is a diagonal matrix of i.i.d samples drawn from [-1, 1].

H is the recursively define noramlized Walsh-Hadamard matrix.

Due to their nature, R and D can be specified by vectors, and H does not need to
be given.

Examples::

data (data.shape = (10, 1000)), 10 input samples which have 1000 features
indices (indices.shape = (1, 100), the vector representation of matrix R
sign (sign.shape = (1. 1024), the vector representation of matrix D

Step 1: pad data with Zeros to shape(10, 1024). 1024 is the nearest power of 2 larger than input feature dimension 1000.
data.shape = (10, 1024)

Step 2: mx.nd.hadamard_dense(data, indices, sign)

Output would have shape (10, 100) which is the resulting samples by applying Sparse Random Projection to the input data.

)code" ADD_FILELINE)
.set_num_inputs(3)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
[](const NodeAttrs& attrs) {
return std::vector<std::string>{"data", "indices", "sign"};
})
.set_attr<nnvm::FInferShape>("FInferShape", HadaShape<3, 1>)
.set_attr<nnvm::FInferType>("FInferType", HadaType<3, 1>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
[](const NodeAttrs& attrs){
return std::vector<std::pair<int, int> >{{0, 0}};
})
.add_argument("data", "NDArray-or-Symbol", "The input array.")
.add_argument("indices", "NDArray-or-Symbol", "The indices representing the columns of the R matrix.")
.add_argument("sign", "NDArray-or-Symbol", "The diagonal elements of the D matrix (+/- 1)")
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
