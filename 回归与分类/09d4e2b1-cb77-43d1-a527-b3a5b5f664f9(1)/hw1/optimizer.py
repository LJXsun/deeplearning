import numpy as np

class SGD(object):
    def __init__(self, model, learning_rate, momentum=0.0):
        self.model = model
        self.learning_rate = learning_rate
        self.momentum = momentum
        # 初始化两个动量
        self.w_v = 0
        self.b_v = 0
        

    def step(self):
        """One updating step, update weights"""

        layer = self.model
        if layer.trainable:
            #pass

            ############################################################################
            # TODO: Put your code here
            # Calculate diff_W and diff_b using layer.grad_W and layer.grad_b.
            # You need to add momentum to this.

            # Weight update with momentum
            self.w_v = self.momentum * self.w_v + self.learning_rate * layer.grad_W
            self.b_v = self.momentum * self.b_v + self.learning_rate * layer.grad_b
#             print("self.w_v.shape =",self.w_v.shape)
#             print("self.b_v.shape =",self.b_v.shape)
#             print("layer.W.shape =",layer.W.shape)
#             print("layer.b.shape =",layer.b.shape)
            layer.W += -self.w_v.T
            layer.b += -self.b_v
            #print("layer.W.shape =",layer.W.shape)
            

            # # Weight update without momentum
            # layer.W += -self.learning_rate * layer.grad_W
            # layer.b += -self.learning_rate * layer.grad_b

            ############################################################################
