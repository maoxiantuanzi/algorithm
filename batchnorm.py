"""
@File    :   batchnorm.py    
@Contact :   pengtt0119@gmail.com

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/11/30 3:23 下午   ttpeng      1.0         None
"""
import numpy as np


def batchnorm_train(x, gamma, beta, bn_param):
	"""
	param:x    : 输入数据，设shape(B,L)
	param:gamma : 缩放因子  γ
	param:beta : 平移因子  β
	param:bn_param   : batchnorm所需要的一些参数
		eps      : 接近0的数，防止分母出现0
		momentum : 动量参数，一般为0.9， 0.99， 0.999
		running_mean ：滑动平均的方式计算新的均值，训练时计算，为测试数据做准备
		running_var  : 滑动平均的方式计算新的方差，训练时计算，为测试数据做准备
	"""
	running_mean = bn_param['running_mean']
	running_var = bn_param['running_var']
	momentum = bn_param['momentum']
	eps = bn_param['eps']

	x_mean = x.mean(axis=0)
	x_var = x.var(axis=0)
	x_normalized = (x - x_mean) / np.sqrt(x_var + eps)
	results = gamma * x_normalized + beta

	running_mean = momentum * running_mean + (1 - momentum) * x_mean
	running_var = momentum * running_var + (1 - momentum) * x_var

	bn_param['running_mean'] = running_mean
	bn_param['running_var'] = running_var

	return results, bn_param


def batchnorm_test(x, gamma, beta, bn_param):
	"""
	param:x    : 输入数据，设shape(B,L)
	param:gamma : 缩放因子  γ
	param:beta : 平移因子  β
	param:bn_param   : batchnorm所需要的一些参数
		eps      : 接近0的数，防止分母出现0
		momentum : 动量参数，一般为0.9， 0.99， 0.999
		running_mean ：滑动平均的方式计算新的均值，训练时计算，为测试数据做准备
		running_var  : 滑动平均的方式计算新的方差，训练时计算，为测试数据做准备
	"""
	running_mean = bn_param['running_mean']
	running_var = bn_param['running_var']
	eps = bn_param['eps']

	x_normalized = (x - running_mean) / np.sqrt(running_var + eps)
	results = gamma * x_normalized + beta

	return results, bn_param
