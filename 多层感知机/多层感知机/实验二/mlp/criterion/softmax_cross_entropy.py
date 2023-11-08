""" Softmax交叉熵损失层 """

import numpy as np

# 为了防止分母为零，必要时可在分母加上一个极小项EPS
EPS = 1e-11

class SoftmaxCrossEntropyLossLayer():
	def __init__(self):
		self.acc = 0.
		self.loss = np.zeros(1, dtype='f')

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
        # softmax
		self.real_y = gt
		print("SoftmaxCrossEntropyLossLayer--forward")
		self.predict_y = np.exp(logit)/np.sum(np.exp(logit),axis=1,keepdims=True)
		#print("predic_y.shape =",predic_y.shape)
		row_loss = - np.sum(gt*np.log(predic_y),axis=1,keepdims=True)
		#print("row_loss.shape =",row_loss.shape)
		self.loss = np.average(row_loss)
		self.acc = np.mean(np.argmax(self.predict_y,axis=1)==np.argmax(self.real_y,axis=1))
        
		return self.loss


	def backward(self):

		############################################################################
	    # TODO: 
		# 计算并返回梯度(与logit具有同样的尺寸)


	    ############################################################################
		#局部敏感度
		return self.predic_y - self.real_y
		
		
		
		
		