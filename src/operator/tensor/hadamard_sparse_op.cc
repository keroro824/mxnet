#include "./hadamard_sparse_op.h"
//#include "./elemwise_binary_op.h"
//#include "./elemwise_binary_broadcast_op.h"

namespace mxnet {
namespace op {
MXNET_OPERATOR_REGISTER_HADAMARDSPARSE(sparse_inplace)
.set_attr<FCompute>("FCompute<cpu>", hadamardTransformSparse<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_sparse_inplace"});

NNVM_REGISTER_OP(_backward_sparse_inplace)
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
