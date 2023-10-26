import numpy as np

# a small number to prevent dividing by zero, maybe useful for you
EPS = 1e-11

class SoftmaxCrossEntropyLoss(object):

    def __init__(self, num_input, num_output, trainable=True):
        """
        Apply a linear transformation to the incoming data: y = Wx + b
        Args:
        num_input: size of each input sample
        num_output: size of each output sample
        trainable: whether if this layer is trainable
        """

        self.num_input = num_input
        self.num_output = num_output
        self.trainable = trainable
        self.XavierInit()

    def forward(self, Input, labels):
        """
        Inputs: (minibatch)
        - Input: (batch_size, 784)
        - labels: the ground truth label, shape (batch_size, )
        Input 100 784
        """

        ############################################################################
        # TODO: Put your code here
        # Apply linear transformation (WX+b) to Input, and then
        # calculate the average accuracy and loss over the minibatch
        # Return the loss and acc, which will be used in solver.py
        # Hint: Maybe you need to save some arrays for gradient computing.

        ############################################################################
        loss = 0
        acc = 0
        print("*"*10 + "forward" + "*"*10)
        all_x = Input.T # 784 100
        print(all_x.shape)
        print(self.W.shape)
        print(self.b.shape)
        ans = self.W.T * all_x + self.b
        print(ans.shape)
        np.asarray(tuple1)
        return loss, acc

    def gradient_computing(self):
        pass

        ############################################################################
        # TODO: Put your code here
        # Calculate the gradient of W and b.

        # self.grad_W = 
        # self.grad_b =
        ############################################################################
    # 初始化w和b权重
    def XavierInit(self):
        """
        Initialize the weigths
        """
        raw_std = (2 / (self.num_input + self.num_output))**0.5
        init_std = raw_std * (2**0.5)
        # 784,10
        self.W = np.random.normal(0, init_std, (self.num_input, self.num_output))
        #1 10
        self.b = np.random.normal(0, init_std, (1, self.num_output))
