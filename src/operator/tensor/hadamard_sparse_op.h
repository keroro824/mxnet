#ifndef MXNET_OPERATOR_TENSOR_HADAMARDS_OP_H_
#define MXNET_OPERATOR_TENSOR_HADAMARDS_OP_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <string>
#include <utility>
#include "../mshadow_op.h"
#include <mxnet/operator_util.h>
#include "../operator_common.h"
#include <mshadow/tensor.h>

#include <algorithm>
#include "../elemwise_op_common.h"
#include "./elemwise_binary_op.h"
#include "broadcast_reduce-inl.h"
#include <iostream>


namespace mxnet {
namespace op {
using namespace mshadow;
using namespace std;


struct InputParam : public dmlc::Parameter<InputParam> {
    int n_samples;
    DMLC_DECLARE_PARAMETER(InputParam) {
        DMLC_DECLARE_FIELD(n_samples).describe("");
    }
};


template<int n_in, int n_out>
inline bool HadaShapeSparse(const nnvm::NodeAttrs &attrs,
                            std::vector <TShape> *in_attrs,
                            std::vector <TShape> *out_attrs) {

    CHECK_EQ(in_attrs->size(), n_in) << " in operator " << attrs.name;
    CHECK_EQ(out_attrs->size(), n_out) << " in operator " << attrs.name;

    const InputParam &param = nnvm::get<InputParam>(attrs.parsed);
    int dim = param.n_samples;

    const TShape &rshape = (*in_attrs)[0];
    const TShape &cshape = (*in_attrs)[2];
    out_attrs->clear();
    out_attrs->push_back(Shape2(dim, cshape[1]));
    return true;
}


template<int n_in, int n_out>
inline bool HadaTypeSparse(const nnvm::NodeAttrs &attrs,
                           std::vector<int> *in_attrs,
                           std::vector<int> *out_attrs) {
    CHECK_EQ(in_attrs->size(), n_in) << " in operator " << attrs.name;
    CHECK_EQ(out_attrs->size(), n_out) << " in operator " << attrs.name;

    int dtype = (*in_attrs)[1];
    out_attrs->clear();
    out_attrs->push_back(dtype);
    return true;
}
}

}  // namespace mxnet
#endif  // MXNET_OPERATOR_TENSOR_HADAMARD_SPARSE_OP_H_