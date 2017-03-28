#ifndef MXNET_OPERATOR_TENSOR_HADAMARD_OP_H_
#define MXNET_OPERATOR_TENSOR_HADAMARD_OP_H_

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
void hadamardTransform(const nnvm::NodeAttrs& attrs,
                       const OpContext& ctx,
                       const std::vector<TBlob>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& outputs) {
    using namespace mshadow;
    using namespace mshadow::expr;

    CHECK_EQ(inputs.size(), 2);
    CHECK_EQ(outputs.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();

    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {

            Tensor<xpu, 1, DType> out = outputs[0].FlatTo1D<xpu, DType>(s);
            Tensor<xpu, 1, DType> value = inputs[0].FlatTo1D<xpu, DType>(s);
            Tensor<xpu, 1, DType> indices = inputs[1].FlatTo1D<xpu, DType>(s);
            unsigned int in_dim = (unsigned int) value.shape_[1];
            DType *out_p = out.dptr_;

            int log2d = in_dim <= 1 ? 0 : log2((double)(in_dim - 1)) + 1;
            DType temp;
            unsigned int k = (unsigned int) indices.shape_[1];

            for (int t = log2d; t; t--) {

                int blockSize = 1 << t;
                int numBlocks = 1 << (log2d - t);

                int halfBlockSize = blockSize >> 1;
                DType *p1 = value.dptr_;
                DType *p2 = value.dptr_ + halfBlockSize;

                for (int blockIndex = numBlocks; blockIndex; blockIndex--) {
                    for (int i = halfBlockSize; i; i--) {
                        temp = *p1 + *p2;
                        *p2 = *p1 - *p2;
                        *p1 = temp;
                        p1++;
                        p2++;
                    }
                    p1 += halfBlockSize;
                    p2 += halfBlockSize;
                }
            }

            DType *indices_p = indices.dptr_;
            DType *input_p = value.dptr_;
            for (int i = 0; i < k; i++) {
                int index = (int) *indices_p;
                *out_p = *(input_p + index);

                out_p++;
                indices_p++;
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
    CHECK_EQ(outputs.size(), 2);
    Stream<xpu> *s = ctx.get_stream<xpu>();
}


template<int n_in, int n_out>
inline bool HadaShape(const nnvm::NodeAttrs& attrs,
                      std::vector<TShape> *in_attrs,
                      std::vector<TShape> *out_attrs) {

    CHECK_EQ(in_attrs->size(), n_in) << " in operator " << attrs.name;
    CHECK_EQ(out_attrs->size(), n_out) << " in operator " << attrs.name;

    const TShape &dshape = (*in_attrs)[1];
    out_attrs->clear();
    out_attrs->push_back(dshape);
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


#define MXNET_OPERATOR_REGISTER_HADAMARD(name)                        \
  NNVM_REGISTER_OP(name)                                            \
  .set_num_inputs(2)                                                \
  .set_num_outputs(1)                                               \
  .set_attr<nnvm::FListInputNames>("FListInputNames",               \
    [](const NodeAttrs& attrs) {                                    \
      return std::vector<std::string>{"value", "indices"};                \
    })          \
  .set_attr<nnvm::FInplaceOption>("FInplaceOption",     \
   [](const NodeAttrs& attrs){                          \
   return std::vector<std::pair<int, int> >{{0, 0}};    \
   })                                                           \
  .set_attr<nnvm::FInferShape>("FInferShape", HadaShape<2, 1>)  \
  .set_attr<nnvm::FInferType>("FInferType", HadaType<2, 1>)     \
  .add_argument("value", "ndarray-or-symbol", "first input")                    \
  .add_argument("indices", "ndarray-or-symbol", "second input")
    }  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_TENSOR_HADAMARD_OP_H_