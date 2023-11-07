""" Sigmoid Layer """

import numpy as np

class SigmoidLayer():
	def __init__(self):
		"""
		Sigmoid激活函数: f(x) = 1/(1+exp(-x))
		"""
		self.trainable = False

	def forward(self, Input):

		############################################################################
	    # TODO: 
		# 对输入应用Sigmoid激活函数并返回结果
    print((1/(1+np.exp(-Input))).shape)
    return 1/(1+np.exp(-Input))

	    ############################################################################
       

	def backward(self, delta):

		############################################################################
	    # TODO: 
		# 根据delta计算梯度


	    ############################################################################
