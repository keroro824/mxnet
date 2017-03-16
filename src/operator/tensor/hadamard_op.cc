#include "./hadamard_op.h"
#include "./elemwise_binary_op.h"
#include "./elemwise_binary_broadcast_op.h"

namespace mxnet {
namespace op {
MXNET_OPERATOR_REGISTER_HADAMARD(dense_inplace)
.set_attr<FCompute>("FCompute<cpu>", hadamardTransform<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_dense_inplace"});

NNVM_REGISTER_OP(_backward_dense_inplace)
.set_num_inputs(3)
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FResourceRequest>("FResourceRequest",
[](const NodeAttrs& attrs) {
return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr<FCompute>("FCompute<cpu>", hadamardTransform<cpu>);
}
}
