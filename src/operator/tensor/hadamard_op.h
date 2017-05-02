#ifndef MXNET_OPERATOR_TENSOR_HADAMARD_OP_H_
#define MXNET_OPERATOR_TENSOR_HADAMARD_OP_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <iostream>
#include <mshadow/tensor.h>
#include <mxnet/operator_util.h>
#include "../mshadow_op.h"
#include "../operator_common.h"
#include "../mxnet_op.h"
#include "../elemwise_op_common.h"
#include "./elemwise_binary_op.h"
#include "broadcast_reduce-inl.h"
#include "broadcast_reduce_op.h"


namespace mxnet {
namespace op {
using namespace mshadow;
using namespace std;


template<int n_in, int n_out>
inline bool HadaShape(const nnvm::NodeAttrs& attrs,
                      std::vector<TShape> *in_attrs,
                      std::vector<TShape> *out_attrs) {

  CHECK_EQ(in_attrs->size(), n_in) << " in operator " << attrs.name;
  CHECK_EQ(out_attrs->size(), n_out) << " in operator " << attrs.name;

  const TShape &rshape = (*in_attrs)[0];
  const TShape &cshape = (*in_attrs)[1];
  out_attrs->clear();
  out_attrs->push_back(Shape2(rshape[0], cshape[1]));
  return true;
}


template<int n_in, int n_out>
inline bool HadaType(const nnvm::NodeAttrs& attrs,
                     std::vector<int> *in_attrs,
                     std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), n_in) << " in operator " << attrs.name;
  CHECK_EQ(out_attrs->size(), n_out) << " in operator " << attrs.name;

  int dtype = (*in_attrs)[0];
  out_attrs->clear();
  out_attrs->push_back(dtype);
  return true;
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

Beidi's 1024 example here.

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
.add_argument("sign", "NDArray-or-Symbol", "The diagonal elements of the D matrix (+/- 1)");

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_TENSOR_HADAMARD_OP_H_
