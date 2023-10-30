from layers.fullyconnected import FC
from activations import *
from model import Model
from optimizers.gradientdescent import GD
from losses.meansquarederror import MeanSquaredError
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

train_data = pd.read_csv("datasets/california_houses_price/california_housing_train.csv").to_numpy()
test_data = pd.read_csv("datasets/california_houses_price/california_housing_test.csv").to_numpy()

input_train = train_data[:, 0:8].T
output_train = train_data[:, 8].T
output_train = np.expand_dims(output_train, axis=-1).T

input_train[1:8, :] /= np.max(input_train[1:8, :], axis=1, keepdims=True)
input_train[0, :] /= np.min(input_train[0, :], keepdims=True)

output_train /= np.max(output_train)

# print((input_train.shape) , (output_train.shape))

input_test = test_data[:, 0:8].T
output_test = test_data[:, 8].T
output_test = np.expand_dims(output_test, axis=-1).T

input_test[1:8, :] /= np.max(input_test[1:8, :], axis=1, keepdims=True)
input_test[0, :] /= np.min(input_test[0, :], keepdims=True)

# print((input_test.shape) , (output_test.shape))

architecture = {
    'FC1': FC(8, 16, 'FC1' ),
    'ACTIVE1': ReLU(),
    'FC2': FC(16, 16, 'FC2'),
    'ACTIVE2': ReLU(),
    'FC3': FC(16, 1, 'FC3'),
    'ACTIVE3': ReLU()
}


criterion = MeanSquaredError()
optimizer = GD(architecture, learning_rate=0.1)
model = Model(architecture, criterion, optimizer)

model.train(input_train, output_train, 30, val=None, batch_size=1000, shuffling=False, verbose=0, save_after='save.txt')
print(model.predict(input_test[:, :8]))
plt.plot(model.predict(input_test[:, :8]))
plt.plot( output_test)
plt.show()