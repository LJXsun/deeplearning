""" ReLU激活层 """
 
import numpy as np

class ReLULayer():
	def __init__(self):
		"""
		ReLU激活函数: relu(x) = max(x, 0)
		"""
		self.trainable = False # 没有可训练的参数

	def forward(self, Input):

		############################################################################
	    # TODO: 
		# 对输入应用ReLU激活函数并返回结果
     # np.maximum用于逐元素比较两个array的大小。
		self.Input = Input
		return np.maximum(0,Input)

	    ############################################################################


	def backward(self, delta):

		############################################################################
	    # TODO: 
		# 根据delta计算梯度

		delta[self.Input<0]=0
		return delta
	    ############################################################################
