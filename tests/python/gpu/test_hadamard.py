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
import operator
import math


del test_support_vector_machine_l1_svm
del test_support_vector_machine_l2_svm

# test hadamard for dense input
def test_dense_inplace_hadamard(data_temp, indices, sign):
	data = mx.symbol.Variable('data')
	index = mx.symbol.Variable('indices')
	signs = mx.symbol.Variable('sign')
	
	input_mx = mx.nd.array(data_temp.todense())
	input_mx.wait_to_read()
	indices_mx = mx.nd.array(indices)
	indices_mx.wait_to_read()
	sign_mx = mx.nd.array(sign)
	sign_mx.wait_to_read()

	# print data_temp.todense()
	# print indices
	start = time.time()

	test = mx.sym.hadamard_dense(data=data, indices=index, sign=signs)
	# arr_grad = [mx.nd.empty(input_mx.shape), mx.nd.empty(indices.shape), mx.nd.empty(sign.shape)]

	exe_test = test.bind(default_context(), args=[input_mx, indices_mx, sign_mx], args_grad=None)
	exe_test.forward(is_train=False)
	out = exe_test.outputs[0].asnumpy()
	exe_test.outputs[0].wait_to_read()
	# exe_test.backward([mx.nd.array(out)])
	# back_out = arr_grad[0].asnumpy()

	end = time.time()

	out = exe_test.outputs[0].asnumpy()
	# print out
	#print(end - start)
	#print back_out
	# print out
	return (end - start), out

def test_dense_partial_inplace_hadamard(data_temp, indices, sign):
	data = mx.symbol.Variable('data')
	index = mx.symbol.Variable('indices')
	signs = mx.symbol.Variable('sign')
	
	input_mx = mx.nd.array(data_temp.todense())
	input_mx.wait_to_read()
	indices_mx = mx.nd.array(indices)
	indices_mx.wait_to_read()
	sign_mx = mx.nd.array(sign)
	sign_mx.wait_to_read()

	# print data_temp.todense()
	# print indices
	start = time.time()

	test = mx.sym.hadamard_partial_dense(data=data, indices=index, sign=signs)
	# arr_grad = [mx.nd.empty(input_mx.shape), mx.nd.empty(indices.shape), mx.nd.empty(sign.shape)]

	exe_test = test.bind(default_context(), args=[input_mx, indices_mx, sign_mx], args_grad=None)
	exe_test.forward(is_train=False)
	exe_test.outputs[0].wait_to_read()
	# exe_test.backward([mx.nd.array(out)])
	# back_out = arr_grad[0].asnumpy()

	end = time.time()

	out = exe_test.outputs[0].asnumpy()
	# print out
	#print(end - start)
	#print back_out
	# print out
	return (end - start), out


# test hadamard for sparse input
def test_sparse_direct_hadamard(random_mx, indices, sign, input_dim):
	# keys = mx.symbol.Variable('keys')
	# values = mx.symbol.Variable('values')
	# index = mx.symbol.Variable('indices')
	# signs = mx.symbol.Variable('sign')
	# inds = mx.symbol.Variable('ind')
	# set_default_context(mx.cpu(0))
	# keys_np = []
	# for key in random_mx.keys():
	# 	keys_np.append(key[1])

	keys_np = random_mx.keys()
	values_np = random_mx.values()
	# keys_np = np.array(keys_np)[ind]
	# values_np = values_np[ind]
	# print keys_np
	keys_np, values_np = zip(*sorted(zip(keys_np, values_np),key=operator.itemgetter(0), reverse=False))
	cur_row = 0
	cur_pos = 0
	ind = []
	j=0
	for key in keys_np:
		row = key[0]
		# if row==2 and len(ind)==0:
		# 	ind.append(0)
		if (row>cur_row):
			# if (row != (cur_row+1)):
			for t in range(row-cur_row-1):
				ind.append(cur_pos)
			ind.append(j)
			cur_pos = j
			cur_row = row
		j+=1
	ind.append(len(keys_np))
	#print sign
	#print keys_np
	#print keys_np
	#print random_mx.todense()
	#print ind
	keys_mx = mx.nd.array(keys_np)
	keys_mx.wait_to_read()
	values_mx = mx.nd.array(values_np)
	values_mx.wait_to_read()
	indices_mx = mx.nd.array(indices)
	indices_mx.wait_to_read()
	sign_mx = mx.nd.array(sign)
	sign_mx.wait_to_read()
	ind_mx = mx.nd.array(ind)
	ind_mx.wait_to_read()
	start = time.time()

	# test = mx.sym.hadamard_sparse(keys=keys, values=values, indices=index, sign=signs, ind=inds, n_samples=input_dim )
	# exe_test = test.bind(ctx = mx.gpu(0), args=[keys_mx, values_mx, indices_mx, sign_mx, ind_mx], args_grad=None, grad_req="null")
	# exe_test.forward(is_train=False)
	# print keys_mx, values_mx, indices_mx,sign_mx,  ind_mx
	exe_test=mx.nd.hadamard_sparse(keys_mx, values_mx, indices_mx, sign_mx, ind_mx, n_samples=input_dim)
	out = exe_test.wait_to_read()

	end = time.time()
	out = exe_test.asnumpy()
	#print sign
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


def test_sparsity():
	set_default_context(mx.cpu())
	for i in range(1,15):
		for j in range(7, 21):
			for k in range(int(math.ceil(math.log(j, 2))), j):
				in_dimension =2**j
				out_dimension = 2**k
				n_samples = 1
				# set_default_context(mx.cpu())
				dens = 0.01*i
				data = sparse.rand(n_samples, in_dimension, density=dens, format='dok', dtype=None, random_state=None)
				indices = np.random.randint(in_dimension-1, size=(1,out_dimension))
				indices = np.sort(indices)
				#indices = np.array([range(in_dimension)])
				sign = np.random.choice(np.array([-1, 1]), in_dimension)
				#print sign
				#print indices
				#sign = np.ones((1, in_dimension))
				# print data
				# print indices
				#sparse = test_sparse_direct_hadamard(data)
				# if i==1 and j==7:
				# 	times, dense= test_dense_partial_inplace_hadamard(data, indices, sign)
				timed, densed = test_dense_inplace_hadamard(data, indices, sign)
				timep, densep = test_dense_partial_inplace_hadamard(data, indices, sign)
				times, denses = test_sparse_direct_hadamard(data, indices, sign, n_samples)
				# 
				
				#print in_dimension, out_dimension, timed
				print in_dimension, out_dimension, timed, timep, times, np.allclose(np.array(densed), np.array(denses), rtol=1.e-3, atol=1.e-3), dens


def test_k():
	set_default_context(mx.gpu(0))
	for i in range(7,21):
		for j in range(int(math.ceil(math.log(i, 2))), i):
			in_dimension =2**i
			out_dimension = 2**j
			n_samples = 1
			# set_default_context(mx.cpu())
			dens = 0.01*1
			data = sparse.rand(n_samples, in_dimension, density=dens, format='dok', dtype=None, random_state=None)
			indices = np.random.randint(in_dimension-1, size=(1,out_dimension))
			indices = np.sort(indices)
			#indices = np.array([range(in_dimension)])
			sign = np.random.choice(np.array([-1,1]), in_dimension)
			#print sign
			#print indices
			#sign = np.ones((1, in_dimension))
			# print data
			# print indices
			#sparse = test_sparse_direct_hadamard(data)
			if i==1 and j==7:
				times, dense= test_dense_inplace_hadamard(data, indices, sign)
			timed, densed = test_dense_inplace_hadamard(data, indices, sign)
			timep, densep = test_dense_partial_inplace_hadamard(data, indices, sign)
			times, denses = test_sparse_direct_hadamard(data, indices, sign, n_samples)
			# 
			
			#print in_dimension, out_dimension, timed
			print in_dimension, out_dimension, timed, timep, times, np.allclose(np.array(densed), np.array(denses), rtol=1.e-3, atol=1.e-3), dens


test_sparsity()


# #bad way for checking if they are the same
# if __name__ == "__main__":
# 	test_sparsity()
















# import mxnet as mx
# import numpy as np
# import random
# from mxnet.test_utils import *
# from scipy import sparse
# from mxnet.test_utils import check_consistency, set_default_context
# import sys
# import os
# curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
# sys.path.insert(0, os.path.join(curr_path, '../unittest'))
# from test_operator import *
# import mxnet as mx
# import numpy as np
# from mxnet.test_utils import check_consistency, set_default_context
# from numpy.testing import assert_allclose
# import time
# import operator
# import math


# del test_support_vector_machine_l1_svm
# del test_support_vector_machine_l2_svm

# # test hadamard for dense input
# def test_dense_inplace_hadamard(data_temp, indices, sign):
# 	data = mx.symbol.Variable('data')
# 	index = mx.symbol.Variable('indices')
# 	signs = mx.symbol.Variable('sign')
	
# 	ctx = [mx.gpu(int(x)) for x in range(8)]
	
# 	input_mx = mx.nd.array(data_temp.todense())
# 	input_mx.wait_to_read()
# 	indices_mx = mx.nd.array(indices)
# 	indices_mx.wait_to_read()
# 	sign_mx = mx.nd.array(sign)
# 	sign_mx.wait_to_read()
# 	indices_mx=indices_mx.broadcast_to((8000, indices_mx.shape[1]))
# 	sign_mx=sign_mx.broadcast_to((8000, sign_mx.shape[0]))
# 	# print sign_mx.shape[0], indices_mx.shape[1], indices_mx.shape[0]
# 	# print data_temp.todense()
# 	# print indices
# 	start = time.time()

# 	test = mx.sym.hadamard_dense(data=data, indices=index, sign=signs)
# 	modd = mx.mod.Module(test, context=ctx, data_names=[('data'), ('indices'), ('sign')], label_names=None)
# 	# arr_grad = [mx.nd.empty(input_mx.shape), mx.nd.empty(indices.shape), mx.nd.empty(sign.shape)]

# 	modd.bind(data_shapes=[('data',input_mx.shape), ('indices',indices_mx.shape), ('sign',sign_mx.shape)])
# 	# mod.init_params()
# 	modd.init_params()
# 	modd.forward(mx.io.DataBatch([input_mx, indices_mx, sign_mx], []), is_train=False)
# 	# mod.outputs[0].wait_to_read()
# 	modd.get_outputs()[0]
# 	# exe_test.backward([mx.nd.array(out)])
# 	# back_out = arr_grad[0].asnumpy()
# 	out = modd.get_outputs()[0].asnumpy()
# 	end = time.time()
# 	# print out
# 	#print(end - start)
# 	#print back_out
# 	# print out
# 	return (end - start), out

# def test_dense_partial_inplace_hadamard(data_temp, indices, sign):
# 	data = mx.symbol.Variable('data')
# 	index = mx.symbol.Variable('indices')
# 	signs = mx.symbol.Variable('sign')

# 	ctx = [mx.gpu(int(x)) for x in range(8)]
	
# 	input_mx = mx.nd.array(data_temp.todense())
# 	input_mx.wait_to_read()
# 	indices_mx = mx.nd.array(indices)
# 	indices_mx.wait_to_read()
# 	sign_mx = mx.nd.array(sign)
# 	sign_mx.wait_to_read()
# 	# print sign_mx.asnumpy()
# 	indices_mx=indices_mx.broadcast_to((8000, indices_mx.shape[1]))
# 	sign_mx=sign_mx.broadcast_to((8000, sign_mx.shape[0]))
# 	# print sign_mx.shape[0], indices_mx.shape[1], indices_mx.shape[0]
# 	# print data_temp.todense()
# 	# print sign_mx.asnumpy()
# 	# print indices
# 	start = time.time()

# 	test = mx.sym.hadamard_partial_dense(data=data, indices=index, sign=signs)
# 	modd = mx.mod.Module(test, context=ctx, data_names=[('data'), ('indices'), ('sign')], label_names=None)
# 	# arr_grad = [mx.nd.empty(input_mx.shape), mx.nd.empty(indices.shape), mx.nd.empty(sign.shape)]

# 	modd.bind(data_shapes=[('data',input_mx.shape), ('indices',indices_mx.shape), ('sign',sign_mx.shape)])
# 	# mod.init_params()
# 	modd.init_params()
# 	modd.forward(mx.io.DataBatch([input_mx, indices_mx, sign_mx], []), is_train=False)
# 	# mod.outputs[0].wait_to_read()
# 	# exe_test.backward([mx.nd.array(out)])
# 	# back_out = arr_grad[0].asnumpy()
# 	out = modd.get_outputs()[0].asnumpy()
# 	end = time.time()

# 	# out = exe_test.outputs[0].asnumpy()
	
# 	# print out
# 	#print(end - start)
# 	#print back_out
# 	# print out
# 	return (end - start), out


# # test hadamard for sparse input
# def test_sparse_direct_hadamard(random_mx, indices, sign, input_dim):
# 	# keys = mx.symbol.Variable('keys')
# 	# values = mx.symbol.Variable('values')
# 	# index = mx.symbol.Variable('indices')
# 	# signs = mx.symbol.Variable('sign')
# 	# inds = mx.symbol.Variable('ind')
# 	# set_default_context(mx.cpu(0))
# 	# keys_np = []
# 	# for key in random_mx.keys():
# 	# 	keys_np.append(key[1])

# 	keys_np = random_mx.keys()
# 	values_np = random_mx.values()
# 	# keys_np = np.array(keys_np)[ind]
# 	# values_np = values_np[ind]
# 	# print keys_np
# 	keys_np, values_np = zip(*sorted(zip(keys_np, values_np),key=operator.itemgetter(0), reverse=False))
# 	cur_row = 0
# 	cur_pos = 0
# 	ind = []
# 	j=0
# 	for key in keys_np:
# 		row = key[0]
# 		# if row==2 and len(ind)==0:
# 		# 	ind.append(0)
# 		if (row>cur_row):
# 			# if (row != (cur_row+1)):
# 			for t in range(row-cur_row-1):
# 				ind.append(cur_pos)
# 			ind.append(j)
# 			cur_pos = j
# 			cur_row = row
# 		j+=1
# 	ind.append(len(keys_np))
# 	#print sign
# 	#print keys_np
# 	#print keys_np
# 	#print random_mx.todense()
# 	#print ind
# 	keys_mx = mx.nd.array(keys_np)
# 	keys_mx.wait_to_read()
# 	values_mx = mx.nd.array(values_np)
# 	values_mx.wait_to_read()
# 	indices_mx = mx.nd.array(indices)
# 	indices_mx.wait_to_read()
# 	sign_mx = mx.nd.array(sign)
# 	sign_mx.wait_to_read()
# 	ind_mx = mx.nd.array(ind)
# 	ind_mx.wait_to_read()
# 	start = time.time()

# 	# test = mx.sym.hadamard_sparse(keys=keys, values=values, indices=index, sign=signs, ind=inds, n_samples=input_dim )
# 	# exe_test = test.bind(ctx = mx.gpu(0), args=[keys_mx, values_mx, indices_mx, sign_mx, ind_mx], args_grad=None, grad_req="null")
# 	# exe_test.forward(is_train=False)
# 	# print keys_mx, values_mx, indices_mx,sign_mx,  ind_mx
# 	exe_test=mx.nd.hadamard_sparse(keys_mx, values_mx, indices_mx, sign_mx, ind_mx, n_samples=input_dim)
# 	out = exe_test.wait_to_read()

# 	end = time.time()
# 	out = exe_test.asnumpy()
# 	#print sign
# 	#print out
	
# 	return (end - start), out


# def popcount32_table16(v):
# 	POPCOUNT_TABLE16 = [0] * 2**16
# 	for index in xrange(len(POPCOUNT_TABLE16)):
# 		POPCOUNT_TABLE16[index] = (index & 1) + POPCOUNT_TABLE16[index >> 1]
# 	return (POPCOUNT_TABLE16[ v & 0xffff] + POPCOUNT_TABLE16[(v >> 16) & 0xffff])


# def naive_hadamard(random_mx):
# 	random_mx = np.matrix(random_mx.todense())
# 	print random_mx
# 	final_out  = []
	
# 	for k in range(random_mx.shape[0]):
# 		input_vec = random_mx[k]
# 		result = [0]*in_dimension
# 		out = []
# 		for i in range(in_dimension):
# 			for j in range(in_dimension):
# 				result[i] += ((popcount32_table16(i & j) & 1) * (-2) + 1) * (input_vec.tolist()[0][j])

# 		for elem in indices[0]:
# 			out.append(result[elem])
# 		final_out.append(out)
# 	# print final_out
# 	return final_out	 


# def test_sparsity():
# 	set_default_context(mx.gpu())
# 	for i in range(1,51):
# 		for j in range(7, 21):
# 			in_dimension =2**j
# 			out_dimension = j
# 			n_samples = 1000
# 			# set_default_context(mx.cpu())
# 			dens = 0.01*i
# 			data = sparse.rand(n_samples, in_dimension, density=dens, format='dok', dtype=None, random_state=None)
# 			indices = np.random.randint(in_dimension-1, size=(1,out_dimension))
# 			indices = np.sort(indices)
# 			#indices = np.array([range(in_dimension)])
# 			sign = np.random.choice(np.array([-1, 1]), in_dimension)
# 			#print sign
# 			#print indices
# 			#sign = np.ones((1, in_dimension))
# 			# print data
# 			# print indices
# 			#sparse = test_sparse_direct_hadamard(data)
# 			if i==1 and j==7:
# 				times, dense= test_dense_partial_inplace_hadamard(data, indices, sign)
# 			times, dense = test_dense_partial_inplace_hadamard(data, indices, sign)
# 			#times, dense = test_sparse_direct_hadamard(data, indices, sign, n_samples)
# 			timed, densem = test_sparse_direct_hadamard(data, indices, sign, n_samples)
# 			print in_dimension, out_dimension, timed, times, np.allclose(np.array(dense), np.array(densem), rtol=1.e-1, atol=1.e-1), dens


# def test_k():
# 	# set_default_context(mx.gpu(0))

# 	for i in range(7,21):
# 		for j in range(int(math.ceil(math.log(i, 2))), i):
# 			in_dimension =2**i
# 			out_dimension = 2**j
# 			n_samples = 8000
# 			# set_default_context(mx.cpu())
# 			dens = 0.01*1
# 			data = sparse.rand(n_samples, in_dimension, density=dens, format='dok', dtype=None, random_state=None)
# 			indices = np.random.randint(in_dimension-1, size=(1,out_dimension))
# 			indices = np.sort(indices)
# 			#indices = np.array([range(in_dimension)])
# 			sign = np.random.choice(np.array([-1,1]), in_dimension)
# 			#print sign
# 			#print indices
# 			#sign = np.ones((1, in_dimension))
# 			# print data
# 			# print indices
# 			#sparse = test_sparse_direct_hadamard(data)
# 			# if i==1 and j==7:
# 			# 	times, dense= test_dense_inplace_hadamard(data, indices, sign)
# 			times, dense = test_dense_inplace_hadamard(data, indices, sign)
# 			# timed, densem = test_sparse_direct_hadamard(data, indices, sign, n_samples)
# 			timed, densem = test_dense_partial_inplace_hadamard(data, indices, sign)
# 			#print in_dimension, out_dimension, timed
# 			print in_dimension, out_dimension, timed, times, np.allclose(np.array(dense), np.array(densem), rtol=1.e-1, atol=1.e-1), dens





# #bad way for checking if they are the same
# if __name__ == "__main__":
# 	test_k()

