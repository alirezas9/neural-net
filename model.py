from layers.convolution2d import Conv2D
from layers.maxpooling2d import MaxPool2D
from layers.fullyconnected import FC

from activations import Activation, get_activation

import pickle
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle

class Model:
    def __init__(self, arch, criterion, optimizer, name=None):
        """
        Initialize the model.
        args:
            arch: dictionary containing the architecture of the model
            criterion: loss 
            optimizer: optimizer
            name: name of the model
        """
        if name is None:
            self.model = arch
            self.criterion = criterion
            self.optimizer = optimizer
            self.layers_names = list(arch.keys())
        else:
            self.model, self.criterion, self.optimizer, self.layers_names = self.load_model(name)
    
    def is_layer(self, layer):
        """
        Check if the layer is a layer.
        args:
            layer: layer to be checked
        returns:
            True if the layer is a layer, False otherwise
        """
        return isinstance(layer, (Conv2D, MaxPool2D, FC))

    def is_activation(self, layer):
        """
        Check if the layer is an activation function.
        args:
            layer: layer to be checked
        returns:
            True if the layer is an activation function, False otherwise
        """
        return isinstance(layer, Activation)
    
    def forward(self, x):
        """
        Forward pass through the model.
        args:
            x: input to the model
        returns:
            output of the model
        """
        tmp = []
        A = x
        # NOTICE: we have a pattern of layers and activations
        for l in range(0, len(self.layers_names), 2):
            Z = self.model[self.layers_names[l]].forward(A)
            # print(self.layers_names[l])
            # print(Z)
            tmp.append(np.copy(Z))  # hint add a copy of Z to tmp
            A = self.model[self.layers_names[l + 1]].forward(Z)
            tmp.append(np.copy(A))  # hint add a copy of A to tmp
        return tmp
        
    def backward(self, dAL, tmp, x):
        """
        Backward pass through the model.
        args:
            dAL: derivative of the cost with respect to the output of the model
            tmp: list containing the intermediate values of Z and A
            x: input to the model
        returns:
            gradients of the model
        """
        dA = dAL
        grads = {}
        # NOTICE: we have a pattern of layers and activations
        # for from the end to the beginning of the tmp list
        for l in range((len(tmp) - 1), -1, -2):
            if l > 2:
                Z, A = tmp[l - 1], tmp[l - 2]
            else:
                Z, A = tmp[l - 1], x
            dZ = self.model[self.layers_names[l]].backward(dA, Z)
            dA, grad = self.model[self.layers_names[l - 1]].backward(dZ, A)
            grads[self.layers_names[l - 1]] = grad
        return grads
    
    def update(self, grads):
        """
        Update the model.
        args:
            grads: gradients of the model
        """
        for name in self.layers_names:
            if self.is_layer(self.model[name]) and not (isinstance(self.model[name], MaxPool2D)):    # hint check if the layer is a layer and also is not a maxpooling layer
                self.model[name].update(self.optimizer, grads[name])
    
    def one_epoch(self, x, y):
        """
        One epoch of training.
        args:
            x: input to the model
            y: labels
            batch_size: batch size
        returns:
            loss
        """
        tmp = self.forward(x)
        AL = tmp[-1]
        loss = self.criterion.compute(AL, y)
        dAL = self.criterion.backward(AL, y)
        grads = self.backward(dAL, tmp, x)
        self.update(grads)
        return loss
    
    def save(self, name):
        """
        Save the model.
        args:
            name: name of the model
        """
        with open(name, 'wb') as f:
            pickle.dump((self.model, self.criterion, self.optimizer, self.layers_names), f)
        
    def load_model(self, name):
        """
        Load the model.
        args:
            name: name of the model
        returns:
            model, criterion, optimizer, layers_names
        """
        with open(name, 'rb') as f:
            return pickle.load(f)
        
    def shuffle(self, m, shuffling):
        order = list(range(m))
        if shuffling:
            return np.random.shuffle(order)
        return order

    def batch(self, X, y, batch_size, index, order):
        """
        Get a batch of data.
        args:
            X: input to the model
            y: labels
            batch_size: batch size
            index: index of the batch
                e.g: if batch_size = 3 and index = 1 then the batch will be from index [3, 4, 5]
            order: order of the data
        returns:
            bx, by: batch of data
        """
        last_index = min((index + 1) * batch_size, len(order))   # hint last index of the batch check for the last batch
        batch = order[(index * batch_size):last_index]
        print(f'{batch} , batch')
        # NOTICE: inputs are 4 dimensional or 2 demensional
        if len(X.shape) == 2:
            bx = X[:, batch]
            by = y[:, batch]
            return bx, by
        else:
            bx = X[batch]
            by = y[batch]
            return bx, by

    def compute_loss(self, X, y, batch_size):
        """
        Compute the loss.
        args:
            X: input to the model
            y: labels
            Batch_Size: batch size
        returns:
            loss
        """
        m = X.shape[0] if len(X.shape) == 4 else X.shape[1] 
        order = self.shuffle(m, False)   
        cost = 0
        for b in range(m // batch_size):
            bx, by = self.batch(X, y, batch_size, b, order)
            tmp = self.forward(bx)
            AL = tmp[-1]
            cost += self.criterion.compute(AL, y)(m//batch_size)
        return cost

    def train(self, X, y, epochs, val=None, batch_size=3, shuffling=False, verbose=1, save_after=None):
        """
        Train the model.
        args:
            X: input to the model
            y: labels
            epochs: number of epochs
            val: validation data
            batch_size: batch size
            shuffling: if True shuffle the data
            verbose: if 1 print the loss after each epoch
            save_after: save the model after training
        """
        train_cost = []
        val_cost = []
        # NOTICE: if your inputs are 4 dimensional m = X.shape[0] else m = X.shape[1]
        m = X.shape[0] if len(X.shape) == 4 else X.shape[1]
        for e in tqdm(range(1, epochs + 1)):
            order = self.shuffle(m, shuffling)
            cost = 0
            for b in range(m // batch_size):
                # bx, by = self.batch(X, y, batch_size, b, order)
                # tmp = self.forward(X)
                # AL = tmp[-1]
                # cost += self.criterion.compute(AL, y) * (m//batch_size)
                # dAL = self.criterion.backward(AL, y)
                # grads = self.backward(dAL, tmp, X)
                # self.update(grads)
                cost += (self.one_epoch(X, y)) / (m // batch_size)
            train_cost.append(cost)
            if val is not None:
                val_cost.append(self.compute_loss(val, y, batch_size))
            if verbose != False:
                if e % verbose == 0:
                    print("Epoch {}: train cost = {}".format(e, cost))
                if val is not None:
                    print("Epoch {}: val cost = {}".format(e, val_cost[-1]))
        if save_after is not None:
            self.save(save_after)
        return train_cost, val_cost
    
    def predict(self, X):
        """
        Predict the output of the model.
        args:
            X: input to the model
        returns:
            predictions
        """
        return self.forward(X)[-1]