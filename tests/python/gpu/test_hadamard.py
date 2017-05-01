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
def test_dense_inplace_hadamard(data_temp, indices, sign):
	value = mx.symbol.Variable('value')
	index = mx.symbol.Variable('indices')
	signs = mx.symbol.Variable('sign')
	
	input_mx = mx.nd.array(data_temp.todense())
	indices_mx = mx.nd.array(indices)
	sign_mx = mx.nd.array(sign)

	# print data_temp.todense()
	# print indices
	start = time.time()

	test = mx.sym.hadamard_dense(value=value, indices=index, sign=signs)
	arr_grad = [mx.nd.empty(input_mx.shape), mx.nd.empty(indices.shape), mx.nd.empty(sign.shape)]

	exe_test = test.bind(default_context(), args=[input_mx, indices_mx, sign_mx], args_grad=arr_grad)
	exe_test.forward(is_train=True)
	out = exe_test.outputs[0].asnumpy()

	exe_test.backward([mx.nd.array(out)])
	back_out = arr_grad[0].asnumpy()

	end = time.time()
	# print out
	#print(end - start)
	#print back_out
	print back_out
	return (end - start), out, back_out


# test hadamard for sparse input
def test_sparse_direct_hadamard(random_mx, indices, input_dim):
	keys = mx.symbol.Variable('keys')
	values = mx.symbol.Variable('values')
	index = mx.symbol.Variable('indices')
	# set_default_context(mx.cpu(0))
	# keys_np = []
	# for key in random_mx.keys():
	# 	keys_np.append(key[1])

	keys_np = random_mx.keys()
	# ind = np.argsort(keys_np, axis=1)
	values_np = np.array(random_mx.values())
	# keys_np = np.array(keys_np)[ind]
	# values_np = values_np[ind]
	
	
	#print keys_np
	#print random_mx.todense()

	keys_mx = mx.nd.array([keys_np])

	values_mx = mx.nd.array([values_np])
	indices_mx = mx.nd.array(indices)

	start = time.time()

	test = mx.sym.sparse_inplace(keys=keys, values=values, indices=index, n_samples=input_dim )
	exe_test = test.bind(default_context(), args=[keys_mx, values_mx, indices_mx], args_grad=None, grad_req="null")
	exe_test.forward(is_train=False)
	out = exe_test.outputs[0].asnumpy()

	end = time.time()
	print out
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
	for i in range(14,16):
		in_dimension =2**i
		out_dimension = 2**(i-5)
		n_samples = 100
		# set_default_context(mx.cpu())
		data = sparse.rand(n_samples, in_dimension, density=1, format='dok', dtype=None, random_state=None)
		indices = np.random.randint(in_dimension-1, size=(1,out_dimension))
		#indices = np.array([range(in_dimension)])
		sign = np.random.choice(np.array([-1, 1]), in_dimension)
		# print data
		# print indices
		#sparse = test_sparse_direct_hadamard(data)
		set_default_context(mx.cpu())
		times, dense, denseback = test_dense_inplace_hadamard(data, indices, sign)
		set_default_context(mx.gpu())
		timed, densem, densemback = test_dense_inplace_hadamard(data, indices, sign)
		# set_default_context(mx.gpu())
		# times, sparsem = test_dense_inplace_hadamard(data, indices)

		print in_dimension, out_dimension, timed, times, np.allclose(np.array(dense), np.array(densem), rtol=1.e-5, atol=1.e-2), np.allclose(np.array(denseback), np.array(densemback), rtol=1.e-5, atol=1.e-2)


