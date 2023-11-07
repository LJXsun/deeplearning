""" 欧式距离损失层 """

import numpy as np

class EuclideanLossLayer():
	def __init__(self):
		self.accu = 0.
		self.loss = 0.

	def forward(self, logit, gt):
    """
      输入: (minibatch)
      - logit: 最后一个全连接层的输出结果, 尺寸(batch_size, 10)
      - gt: 真实标签, 尺寸(batch_size, 10)
    """

    ############################################################################
    # TODO: 
    # 在minibatch内计算平均准确率和损失，分别保存在self.accu和self.loss里(将在solver.py里自动使用)
    # 只需要返回self.loss

    ############################################################################
    self.predict_y = logit
    self.lable_y = gt
    L = np.sqrt(np.sum(np.square(self.predict_y - self.lable_y), axis=1))
    self.loss = np.average(L)

    return self.loss

	def backward(self):

		############################################################################
	    # TODO: 
		# 计算并返回梯度(与logit具有同样的尺寸)


	    ############################################################################
