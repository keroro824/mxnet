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


template<typename xpu>
void hadamardTransformSparse(const nnvm::NodeAttrs& attrs,
                       const OpContext& ctx,
                       const std::vector<TBlob>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& outputs) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(inputs.size(), 3);
    CHECK_EQ(outputs.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();

    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {

            Tensor<xpu, 1, DType> out = outputs[0].FlatTo1D<xpu, DType>(s);
            Tensor<xpu, 1, DType> keys = inputs[0].FlatTo1D<xpu, DType>(s);
            Tensor<xpu, 1, DType> values = inputs[1].FlatTo1D<xpu, DType>(s);
            Tensor<xpu, 1, DType> indices = inputs[2].FlatTo1D<xpu, DType>(s);

            unsigned int k = (unsigned int) indices.shape_[1];
            unsigned int nnz = (unsigned int) keys.shape_[1];

            DType *pKeys = keys.dptr_;
            DType *pValues = values.dptr_;
            out = 0;

            for (int j = nnz; j ; j--) {
                DType *pRes = out.dptr_;

                DType *pIndices = indices.dptr_;
                for (int i = k; i; i--) {
                    int index = (int) *pIndices;
                    int keyvalue = (int) *pKeys;

                    *pRes += ((__builtin_popcount(index & keyvalue) & 1)*-2 +1) * (*pValues);
                    pRes++; pIndices++;
                }
                pKeys++; pValues++;

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
    using namespace mshadow::expr;
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(outputs.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();

}


template<int n_in, int n_out>
inline bool HadaShapeSparse(const nnvm::NodeAttrs& attrs,
                         std::vector<TShape> *in_attrs,
                         std::vector<TShape> *out_attrs) {

    CHECK_EQ(in_attrs->size(), n_in) << " in operator " << attrs.name;
    CHECK_EQ(out_attrs->size(), n_out) << " in operator " << attrs.name;

    const TShape &dshape = (*in_attrs)[2];
    out_attrs->clear();
    out_attrs->push_back(dshape);
    return true;
}


template<int n_in, int n_out>
inline bool HadaTypeSparse(const nnvm::NodeAttrs& attrs,
                         std::vector<int> *in_attrs,
                         std::vector<int> *out_attrs) {
    CHECK_EQ(in_attrs->size(), n_in) << " in operator " << attrs.name;
    CHECK_EQ(out_attrs->size(), n_out) << " in operator " << attrs.name;

    int dtype = (*in_attrs)[1];
    out_attrs->clear();
    out_attrs->push_back(dtype);
    return true;
}


#define MXNET_OPERATOR_REGISTER_HADAMARDSPARSE(name)                        \
  NNVM_REGISTER_OP(name)                                            \
  .set_num_inputs(3)                                                \
  .set_num_outputs(1)                                               \
  .set_attr<nnvm::FListInputNames>("FListInputNames",               \
    [](const NodeAttrs& attrs) {                                    \
      return std::vector<std::string>{"keys", "values", "indices"};                \
    })          \
  .set_attr<nnvm::FInferShape>("FInferShape", HadaShapeSparse<3, 1>)  \
  .set_attr<nnvm::FInferType>("FInferType", HadaTypeSparse<3, 1>)     \
  .add_argument("keys", "ndarray-or-symbol", "first input")                    \
  .add_argument("values", "ndarray-or-symbol", "second input")        \
  .add_argument("indices", "ndarray-or-symbol", "third input")
    }  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_TENSOR_HADAMARD_SPARSE_OP_H_