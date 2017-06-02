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

template<typename DType>
int searchSortedIndices(int index, DType *sortedIndices, int k) {
    if (k == 0) return 0;
    int left = 0;
    if (index <= (int)(*(sortedIndices+left))) return left;
    int right = k - 1;
    if (index > (int)(*(sortedIndices+right))) return right + 1;
    while (left + 1 < right) {
        int mid = (left + right) / 2;
        if (index <= (int)(*(sortedIndices+mid))) {
            right = mid;
            if (index > (int)(*(sortedIndices+right))) return right + 1;
        } else {
            left = mid;
            if (index <= (int)(*(sortedIndices+left))) return left;
        }
    }
    return left + 1;
}

template<int n_in, int n_out>
inline bool HadaShape(const nnvm::NodeAttrs& attrs,
                      std::vector<TShape> *in_attrs,
                      std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), n_in) << " in operator " << attrs.name;
  CHECK_EQ(out_attrs->size(), n_out) << " in operator " << attrs.name;

  const TShape &rshape = (*in_attrs)[0];
  const TShape &cshape = (*in_attrs)[1];
  const TShape &dshape = (*in_attrs)[2];

  CHECK(rshape[1] > 0 && (rshape[1] & (rshape[1] - 1)) == 0) << "column dimension must be a power of 2. Consider padding with zeros.";
  CHECK_EQ(rshape[1], dshape[1]) << "Array of diagonals must match second dimension of input data.";

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

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_TENSOR_HADAMARD_OP_H_
