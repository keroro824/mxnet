import mxnet as mx
import numpy as np
import random
from mxnet.test_utils import *

def test_dense_inplace_hadamard():
	data = mx.symbol.Variable('data')
	dim = mx.symbol.Variable('dim')
	in_dim = mx.symbol.Variable('in_dim')
	dim_np = np.ones((1,1))
	dim_np[:] = 8
	in_dim_np = np.ones((1,1))
	in_dim_np[:] = 4

	shape = (2, 16)
	data_temp = np.random.normal(size=shape)
	
	input_mx = mx.nd.array(data_temp)
	dim_mx = mx.nd.array(dim_np)
	in_dim_mx = mx.nd.array(in_dim_np)

	input_shape = mx.nd.ones(input_mx.shape)
	dim_mx_shape = mx.nd.ones(dim_mx.shape) 
	in_dim_mx_shape = mx.nd.ones(in_dim_mx.shape)
	print data_temp
	print input_mx, input_shape, dim_mx, dim_mx_shape, in_dim_mx, in_dim_mx_shape

	test = mx.sym.dense_inplace(value=data, dim=dim, in_dim=in_dim)

	# exe_test = test.bind(default_context(), args=[input, dim_mx, in_dim_mx], args_grad=[input_shape,dim_mx_shape,in_dim_mx_shape])
	exe_test = test.bind(default_context(), args=[input_mx, dim_mx, in_dim_mx], args_grad=None, grad_req="null")

	exe_test.forward(is_train=False)
	out = exe_test.outputs[0].asnumpy()
	print out


test_dense_inplace_hadamard()





