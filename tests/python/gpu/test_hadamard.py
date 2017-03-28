import mxnet as mx
import numpy as np
import random
from mxnet.test_utils import *
from scipy import sparse


# test hadamard for dense input
def test_dense_inplace_hadamard(data_temp):
	value = mx.symbol.Variable('value')
	index = mx.symbol.Variable('indices')
	in_dim_np = np.ones((1,1))
	in_dim_np[:] = in_dimension

	shape = (1, in_dimension)
	
	input_mx = mx.nd.array(data_temp.todense())
	indices_mx = mx.nd.array(indices)

	# print data_temp.todense()

	test = mx.sym.dense_inplace(value=value, indices=index)

	exe_test = test.bind(default_context(), args=[input_mx, indices_mx], args_grad=None, grad_req="null")
	exe_test.forward(is_train=False)
	out = exe_test.outputs[0].asnumpy()
	print out
	return out


# test hadamard for sparse input
def test_sparse_direct_hadamard(random_mx):
	keys = mx.symbol.Variable('keys')
	values = mx.symbol.Variable('values')
	index = mx.symbol.Variable('indices')

	keys_np = []
	for key in random_mx.keys():
		keys_np.append(key[1])
	print keys_np
	values_np = random_mx.values()

	# print random_mx.todense()

	keys_mx = mx.nd.array(keys_np)
	values_mx = mx.nd.array(values_np)
	indices_mx = mx.nd.array(indices)

	test = mx.sym.sparse_inplace(keys=keys, values=values, indices=indices)
	exe_test = test.bind(default_context(), args=[keys_mx, values_mx, indices_mx], args_grad=None, grad_req="null")

	exe_test.forward(is_train=False)
	out = exe_test.outputs[0].asnumpy()
	print out
	return out


def popcount32_table16(v):
	POPCOUNT_TABLE16 = [0] * 2**16
	for index in xrange(len(POPCOUNT_TABLE16)):
		POPCOUNT_TABLE16[index] = (index & 1) + POPCOUNT_TABLE16[index >> 1]
	return (POPCOUNT_TABLE16[ v & 0xffff] + POPCOUNT_TABLE16[(v >> 16) & 0xffff])


def naive_hadamard(random_mx):
    result = [0]*in_dimension
    out = []

    for i in range(in_dimension):
    	for j in range(in_dimension):
    		result[i] += ((popcount32_table16(i & j) & 1) * (-2) + 1) * (random_mx.todense().tolist()[0][j])

    for elem in indices[0]:
    	out.append(result[elem])
    print out 
    return out 


#bad way for checking if they are the same
if __name__ == "__main__":
	in_dimension = 128
	out_dimension = 64
	data = sparse.rand(1, in_dimension, density=0.1, format='dok', dtype=None, random_state=None)
	indices = np.random.randint(in_dimension-1, size=(1,out_dimension))
	# print data
	# print indices
	sparse = test_sparse_direct_hadamard(data)
	dense = test_dense_inplace_hadamard(data)
	print np.allclose(np.array(sparse), np.array(dense), rtol=1.e-5, atol=1.e-8)
	naive = naive_hadamard(data)
	
	print np.allclose(np.array(sparse), np.array(naive))
	