import mxnet as mx
import numpy as np
import random
from mxnet.test_utils import *
from scipy import sparse
from mxnet.test_utils import check_consistency, set_default_context
import sys
import os
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '../unittest'))
from test_operator import *
import mxnet as mx
import numpy as np
from mxnet.test_utils import check_consistency, set_default_context
from numpy.testing import assert_allclose
import time


del test_support_vector_machine_l1_svm
del test_support_vector_machine_l2_svm

# test hadamard for dense input
def test_dense_inplace_hadamard(data_temp, indices):
	value = mx.symbol.Variable('value')
	index = mx.symbol.Variable('indices')
	
	input_mx = mx.nd.array(data_temp.todense())
	indices_mx = mx.nd.array(indices)

	# print data_temp.todense()

	start = time.time()

	test = mx.sym.dense_inplace(value=value, indices=index)

	exe_test = test.bind(default_context(), args=[input_mx, indices_mx], args_grad=None, grad_req="null")
	exe_test.forward(is_train=False)
	out = exe_test.outputs[0].asnumpy()


	end = time.time()
	#print(end - start)
	
	#print out
	return (end - start), out


# test hadamard for sparse input
def test_sparse_direct_hadamard(random_mx):
	keys = mx.symbol.Variable('keys')
	values = mx.symbol.Variable('values')
	index = mx.symbol.Variable('indices')
	# set_default_context(mx.cpu(0))

	keys_np = []
	for key in random_mx.keys():
		keys_np.append(key[1])

	values_np = random_mx.values()

	# print random_mx.todense()

	keys_mx = mx.nd.array([keys_np])
	values_mx = mx.nd.array([values_np])
	indices_mx = mx.nd.array(indices)

	print keys_mx
	start = time.time()

	test = mx.sym.sparse_inplace(keys=keys, values=values, indices=indices)
	exe_test = test.bind(default_context(), args=[keys_mx, values_mx, indices_mx], args_grad=None, grad_req="null")
	exe_test.forward(is_train=False)
	out = exe_test.outputs[0].asnumpy()

	end = time.time()
	#print out
	return (end - start), out


def popcount32_table16(v):
	POPCOUNT_TABLE16 = [0] * 2**16
	for index in xrange(len(POPCOUNT_TABLE16)):
		POPCOUNT_TABLE16[index] = (index & 1) + POPCOUNT_TABLE16[index >> 1]
	return (POPCOUNT_TABLE16[ v & 0xffff] + POPCOUNT_TABLE16[(v >> 16) & 0xffff])


def naive_hadamard(random_mx):
	random_mx = np.matrix(random_mx.todense())
	print random_mx
	final_out  = []
	
	for k in range(random_mx.shape[0]):
		input_vec = random_mx[k]
		result = [0]*in_dimension
		out = []
		for i in range(in_dimension):
			for j in range(in_dimension):
				result[i] += ((popcount32_table16(i & j) & 1) * (-2) + 1) * (input_vec.tolist()[0][j])

		for elem in indices[0]:
			out.append(result[elem])
		final_out.append(out)
	# print final_out
	return final_out	 


#bad way for checking if they are the same
if __name__ == "__main__":
	for i in range(10,20):
		in_dimension =2**i
		out_dimension = 2**(i-2)
		# set_default_context(mx.cpu())
		data = sparse.rand(10000, in_dimension, density=0.1, format='dok', dtype=None, random_state=None)
		indices = np.random.randint(in_dimension-1, size=(1,out_dimension))
		#indices = np.array([range(in_dimension)])

		# print data
		# print indices
		#sparse = test_sparse_direct_hadamard(data)
		set_default_context(mx.cpu())
		timed, densem = test_dense_inplace_hadamard(data, indices)
		set_default_context(mx.gpu())
		times, sparsem = test_dense_inplace_hadamard(data, indices)
		
		print in_dimension, out_dimension, timed, times, np.allclose(np.array(sparsem), np.array(densem), rtol=1.e-5, atol=1.e-8)
	# naive = naive_hadamard(data)
	
	# print np.allclose(np.array(dense), np.array(naive))


