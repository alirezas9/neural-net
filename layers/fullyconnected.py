import numpy as np

class FC:
    def __init__(self, input_size : int, output_size : int, name : str, initialize_method : str="random"):
        self.input_size = input_size
        self.output_size = output_size
        self.name = name
        self.initialize_method = initialize_method
        self.parameters = [self.initialize_weights(), self.initialize_bias()]
        self.input_shape = None
        self.reshaped_shape = None
    
    def initialize_weights(self):
        if self.initialize_method == "random":
            return np.random.randn(self.output_size, self.input_size) * 0.01
        
        # The xavier initialization method is calculated as a random number 
        # with a uniform probability distribution (U) between the range -(1/sqrt(n)) and 1/sqrt(n),
        # where n is the number of inputs to the node.
        # weight = U [-(1/sqrt(n)), 1/sqrt(n)]

        elif self.initialize_method == "xavier":
            upper, lower = 1 / np.sqrt(self.input_size ), 1 / np.sqrt(self.input_size)
            return np.random.uniform(low=lower, high=upper, size=(self.input_size, self.output_size))
        
        
        # The he initialization method is calculated as a random number with a Gaussian probability distribution (G)
        # with a mean of 0.0 and a standard deviation of sqrt(2/n), where n is the number of inputs to the node.
        # weight = G (0.0, sqrt(2/n))

        elif self.initialize_method == "he":
            return np.random.randn(self.output_size, self.input_size) * np.sqrt(2 / self.input_size)

        else:
            raise ValueError("Invalid initialization method")
    
    def initialize_bias(self):
        return np.zeros((self.output_size, 1))
    
    def forward(self, A_prev):
        """
        Forward pass for fully connected layer.
            args:
                A_prev: activations from previous layer (or input data)
                A_prev.shape = (batch_size, input_size)
            returns:
                Z: output of the fully connected layer
        """
        # NOTICE: BATCH_SIZE is the first dimension of A_prev
        self.input_shape = A_prev.shape
        A_prev_tmp = np.copy(A_prev)

        if len(A_prev.shape) > 2: # check if A_prev is output of convolutional layer
            batch_size = self.input_shape[0]    #A_prev.shape[0]
            A_prev_tmp = A_prev_tmp.reshape(batch_size, -1).T        
        self.reshaped_shape = A_prev_tmp.shape
        
        W, b = self.parameters
        Z = W @ A_prev_tmp + b
        return Z
    
    def backward(self, dZ, A_prev):
        """
        Backward pass for fully connected layer.
            args:
                dZ: derivative of the cost with respect to the output of the current layer
                A_prev: activations from previous layer (or input data)
            returns:
                dA_prev: derivative of the cost with respect to the activation of the previous layer
                grads: list of gradients for the weights and bias
        """
        A_prev_tmp = np.copy(A_prev)
        # check if A_prev is output of convolutional layer
        if len(A_prev.shape) > 2:
            batch_size = A_prev_tmp.shape[0]
            A_prev_tmp = A_prev_tmp.reshape(batch_size, -1).T

        W, b = self.parameters
        dW = (dZ @ A_prev_tmp.T) / A_prev_tmp.shape[1]
        db = np.sum(dZ, axis=1, keepdims=True) / A_prev_tmp.shape[1]
        dA_prev = W.T @ dZ
        grads = [dW, db]        
        # reshape dA_prev to the shape of A_prev
        # check if A_prev is output of convolutional layer 
        if len(A_prev.shape) > 2:
            dA_prev = dA_prev.T.reshape(self.input_shape)
        return dA_prev, grads
    
    def update(self, optimizer, grads):
        """
        Update the parameters of the layer.
            args:
                optimizer: optimizer object
                grads: list of gradients for the weights and bias
        """
        self.parameters = optimizer.update(grads, self.name)
