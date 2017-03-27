import mxnet as mx
import numpy as np
import random
from mxnet.test_utils import *
from scipy import sparse


#hack some parameters here and will pass in later
dimension = 128
data = sparse.rand(1, dimension, density=0.1, format='dok', dtype=None, random_state=None)
print data


# test hadamard for dense input
def test_dense_inplace_hadamard(data_temp):
	value = mx.symbol.Variable('value')
	in_dim = mx.symbol.Variable('in_dim')
	in_dim_np = np.ones((1,1))
	in_dim_np[:] = dimension

	shape = (1, dimension)
	
	input_mx = mx.nd.array(data_temp.todense())
	in_dim_mx = mx.nd.array(in_dim_np)

	input_shape = mx.nd.ones(input_mx.shape)
	in_dim_mx_shape = mx.nd.ones(in_dim_mx.shape)
	print data_temp.todense()

	test = mx.sym.dense_inplace(value=value, in_dim=in_dim)

	exe_test = test.bind(default_context(), args=[input_mx, in_dim_mx], args_grad=None, grad_req="null")
	exe_test.forward(is_train=False)
	out = exe_test.outputs[0].asnumpy()
	print out

# test hadamard for sparse input
def test_sparse_direct_hadamard(random_mx):
	keys = mx.symbol.Variable('keys')
	values = mx.symbol.Variable('values')
	indices = mx.symbol.Variable('indices')

	keys_np = []
	for key in random_mx.keys():
		keys_np.append(key[1])
	print keys_np
	values_np = random_mx.values()

	indices = []
	for i in range(dimension):
		indices.append(i)

	print random_mx.todense()

	keys_mx = mx.nd.array(keys_np)
	values_mx = mx.nd.array(values_np)
	indices_mx = mx.nd.array(indices)

	test = mx.sym.sparse_inplace(keys=keys)
	exe_test = test.bind(default_context(), args=[keys_mx, values_mx, indices_mx], args_grad=None, grad_req="null")

	exe_test.forward(is_train=False)
	out = exe_test.outputs[0].asnumpy()
	print out

#bad way for checking if they are the same
test_dense_inplace_hadamard(data)
test_sparse_direct_hadamard(data)






