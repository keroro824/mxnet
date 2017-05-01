#include "./hadamard_sparse_op.h"
//#include "./elemwise_binary_op.h"
//#include "./elemwise_binary_broadcast_op.h"

namespace mxnet {
namespace op {


DMLC_REGISTER_PARAMETER(InputParam);

NNVM_REGISTER_OP(hadamard_sparse)
.set_num_inputs(3)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
[](const NodeAttrs& attrs) {
  return std::vector<std::string>{"keys", "values", "indices"};
})
.set_attr_parser(ParamParser<InputParam>)
.set_attr<nnvm::FInferShape>("FInferShape", HadaShapeSparse<3, 1>)
.set_attr<nnvm::FInferType>("FInferType", HadaTypeSparse<3, 1>)
.set_attr<FCompute>("FCompute<cpu>", hadamardTransformSparse<cpu>)
.set_attr<FResourceRequest>("FResourceRequest",
[](const NodeAttrs& attrs) {
return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_hadamard_sparse"})
.add_arguments(InputParam::__FIELDS__())
.add_argument("keys", "ndarray-or-symbol", "first input")
.add_argument("values", "ndarray-or-symbol", "second input")
.add_argument("indices", "ndarray-or-symbol", "third input");

NNVM_REGISTER_OP(_backward_hadamard_sparse)
.set_num_inputs(1)
.set_num_outputs(3)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FResourceRequest>("FResourceRequest",
[](const NodeAttrs& attrs) {
return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr<FCompute>("FCompute<cpu>", hadamardTransformSparse<cpu>);
}
}
