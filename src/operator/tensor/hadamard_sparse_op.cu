#include "./hadamard_sparse_op.h"
#include "./elemwise_binary_op.h"
#include "./elemwise_binary_broadcast_op.h"

namespace mxnet {
namespace op {
NNVM_REGISTER_OP(sparse_inplace)
.set_attr<FCompute>("FCompute<gpu>", hadamardsTransform<gpu>);

NNVM_REGISTER_OP(_backward_sparse_inplace)
.set_attr<FCompute>("FCompute<gpu>", hadamardsTransform<gpu>);
}
}
