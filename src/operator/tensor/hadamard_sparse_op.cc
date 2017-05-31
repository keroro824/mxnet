#include "./hadamard_sparse_op.h"

namespace mxnet {
namespace op {

template<typename xpu>
void hadamardTransformSparse(const nnvm::NodeAttrs &attrs,
                             const OpContext &ctx,
                             const std::vector <TBlob> &inputs,
                             const std::vector <OpReqType> &req,
                             const std::vector <TBlob> &outputs) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(inputs.size(), 5);
    CHECK_EQ(outputs.size(), 1);
    Stream <xpu> *s = ctx.get_stream<xpu>();

    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {

        Tensor<xpu, 2, DType> out = outputs[0].FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> keys = inputs[0].FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 1, DType> values = inputs[1].FlatTo1D<xpu, DType>(s);
        Tensor<xpu, 1, DType> indices = inputs[2].FlatTo1D<xpu, DType>(s);
        Tensor<xpu, 1, DType> sign = inputs[3].FlatTo1D<xpu, DType>(s);

        unsigned int k = (unsigned int) indices.shape_[1];
        unsigned int nnz = (unsigned int) keys.shape_[0];
        DType *pKeys = keys.dptr_;
        DType *pValues = values.dptr_;
        DType *sign_p = sign.dptr_;
        out = 0;

        for (int j = nnz; j; j--) {

            DType *rest = out.dptr_;
            DType *pIndices = indices.dptr_;

            for (int i = k; i; i--) {
                int index = (int) *pIndices;
                int row = (int) *pKeys;
                int keyvalue = (int) *(pKeys+1);
                int signvalue = (int) *(sign_p + keyvalue);
                DType *pRes = rest;
                pRes += (row+1) * k - i;

                *pRes += ((__builtin_popcount(index & keyvalue) & 1) * -2 + 1) * (*pValues) * signvalue;
                pIndices++;
            }
            pKeys+=2;
            pValues++;

        }
    });
}


template<typename xpu>
void hadamardTransform_backwards(const nnvm::NodeAttrs &attrs,
                                 const OpContext &ctx,
                                 const std::vector <TBlob> &inputs,
                                 const std::vector <OpReqType> &req,
                                 const std::vector <TBlob> &outputs) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(outputs.size(), 1);
    Stream <xpu> *s = ctx.get_stream<xpu>();

}

DMLC_REGISTER_PARAMETER(InputParam);

NNVM_REGISTER_OP(hadamard_sparse)
.set_num_inputs(5)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
[](const NodeAttrs& attrs) {
  return std::vector<std::string>{"keys", "values", "indices", "sign", "ind"};
})
.set_attr_parser(ParamParser<InputParam>)
.set_attr<nnvm::FInferShape>("FInferShape", HadaShapeSparse<5, 1>)
.set_attr<nnvm::FInferType>("FInferType", HadaTypeSparse<5, 1>)
.set_attr<FCompute>("FCompute<cpu>", hadamardTransformSparse<cpu>)
.set_attr<FResourceRequest>("FResourceRequest",
[](const NodeAttrs& attrs) {
return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_hadamard_sparse"})
.add_arguments(InputParam::__FIELDS__())
.add_argument("keys", "NDArray-or-Symbol", "first input")
.add_argument("values", "NDArray-or-Symbol", "second input")
.add_argument("indices", "NDArray-or-Symbol", "third input")
.add_argument("sign", "NDArray-or-Symbol", "forth input")
.add_argument("ind", "NDArray-or-Symbol", "fifth input");

NNVM_REGISTER_OP(_backward_hadamard_sparse)
.set_num_inputs(1)
.set_num_outputs(5)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FResourceRequest>("FResourceRequest",
[](const NodeAttrs& attrs) {
return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr<FCompute>("FCompute<cpu>", hadamardTransformSparse<cpu>);
}
}
