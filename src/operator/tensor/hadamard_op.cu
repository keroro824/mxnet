#include "./hadamard_op.h"
#include "./elemwise_binary_op.h"
#include "./elemwise_binary_broadcast_op.h"

namespace mxnet {
namespace op {
NNVM_REGISTER_OP(dense_inplace)
.set_attr<FCompute>("FCompute<gpu>", hadamardTransform<gpu>);

NNVM_REGISTER_OP(_backward_dense_inplace)
.set_attr<FCompute>("FCompute<gpu>", hadamardTransform<gpu>);
}
}
