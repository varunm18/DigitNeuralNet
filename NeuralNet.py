import numpy as np
import pickle
import time
import sys
import nnfs
from nnfs.datasets import spiral_data
import warnings

class Layer:
    def __init__(self, input, output, activation):
        # self.weights = np.random.uniform(-1, 1, (input, output))
        # self.biases = np.random.uniform(-1, 1, (1, output))
        self.weights = 0.01 * np.random.randn(input, output)
        self.biases = np.zeros((1, output))
        self.activation_func = activation
    
    def __str__(self):
        return f"Weights:\n{self.weights}\nBiases:\n{self.biases}"

class MLP:
    def __init__(self, layers, activation, loss, optimizer):
        self.layers = []
        for i in range(len(layers)-1):
            self.layers.append(Layer(layers[i], layers[i+1], activation[i]))
        self.classes = layers[-1]
        self.loss_func = loss
        self.optimizer = optimizer

    def forward(self, input, label):
        self.input = np.array(input.copy())
        # Convert Categorical Labels to One-Hot-Encoded
        one_hot = np.array(label.copy())
        if len(one_hot.shape) == 1:
            one_hot = np.eye(self.classes)[one_hot.reshape(-1)]
        self.label = one_hot

        output = self.input.copy()
        for layer in self.layers:
            layer.input = output
            z = np.dot(output, layer.weights) + layer.biases
            layer.z = z

            output = layer.activation_func(z)
            layer.a = output

        self.output = output
        self.loss = self.loss_func(output, self.label)
        self.accuracy = np.mean(np.argmax(output, axis=1)==np.argmax(self.label, axis=1))
        # self.accuracy = np.mean(np.argmax(output, axis=1)==label)

        return output
    
    def backward(self):
        self.optimizer.pre_proccess()

        # check for softmax and cce loss pairing
        if self.layers[-1].activation_func == MLP.softmax and self.loss_func == MLP.categoricalCrossEntropy:
            dz = (self.output - self.label) / len(self.output)
            pair = True
        else:
            da = self.loss_func(self.output, self.label, deriv=True)

        for i in range(len(self.layers)-1, -1, -1):
            layer = self.layers[i]

            if not pair:
                dz = da * layer.activation_func(layer.z, deriv=True)
            else:
                pair = False
            
            layer.dw = np.dot(layer.input.T, dz)
            layer.db = np.sum(dz, axis=0, keepdims=True) # merge all samples together

            da = np.dot(dz, layer.weights.T)

            # print("dw\n", dw)
            # print("db\n", db)
            self.optimizer.update(layer)

        self.optimizer.post_proccess()
    
    def run(self, input):
        output = input.copy()
        for layer in self.layers:
            output = layer.activation_func(np.dot(output, layer.weights) + layer.biases)
        
        return output
    
    # -----ACTIVATION-----

    def relu(input, deriv=False):
        if deriv:
            return (input > 0).astype(int)
        return np.maximum(0, input)
    
    def softmax(input, deriv=False):
        if deriv:
            ai = input[-1].reshape(-1, 1)
            j = np.diagflat(ai) - np.dot(ai, ai.T)
            raise NotImplementedError("Softmax in hidden layers isn't implemented")
        exp = np.exp(input - np.max(input, axis=1, keepdims=True)) # clamp from 0-1
        return exp / np.sum(exp, axis=1, keepdims=True)
        
    # -----LOSS-----

    def categoricalCrossEntropy(input, label, deriv=False):
        input = np.clip(input, 1e-10, 1 - 1e-10) # to avoid log(0), sys.float_info.min
        if deriv:
            return (-label / input) / len(input) # divide by samples for gradient normalization
        return np.sum(label * -np.log(input))/len(label)
        # return np.mean(-np.log(input[range(len(input)), label]))


    # -----OPTIMIZERS-----

    # Vanilla SGD with Decay
    class Optimizer():
        def __init__(self, learning_rate, decay=0):
            self.learning_rate = learning_rate

            self.original_lr = learning_rate
            self.decay = decay
            self.steps = 0

        def pre_proccess(self):
            self.learning_rate = self.original_lr * (1 / (1 + self.decay * self.steps))

        def update(self, layer):
            layer.weights -= self.learning_rate * layer.dw
            layer.biases -= self.learning_rate * layer.db

        def post_proccess(self):
            self.steps+=1
    
    # With Momentum
    class StochasticGradientDescent(Optimizer):
        def __init__(self, learning_rate, decay=0, momentum=0):
            super().__init__(learning_rate, decay)
            
            self.momentum = momentum
        
        def update(self, layer):
            dw = self.learning_rate * layer.dw
            db = self.learning_rate * layer.db
            if self.momentum!=0:
                if self.steps==0:
                    layer.weight_momentum = np.zeros_like(layer.weights)
                    layer.bias_momentum = np.zeros_like(layer.biases)

                dw += self.momentum * layer.weight_momentum
                db += self.momentum * layer.bias_momentum
                layer.weight_momentum = dw
                layer.bias_momentum = db

            layer.weights -= dw
            layer.biases -= db

    # With Cache
    class RMSProp(Optimizer):
        def __init__(self, learning_rate, decay=0, epsilon=1e-7, rho=0.9):
            super().__init__(learning_rate, decay)
            
            self.epsilon = epsilon
            self.rho = rho
        
        def update(self, layer):
            if self.steps==0:
                layer.weight_cache = np.zeros_like(layer.weights)
                layer.bias_cache = np.zeros_like(layer.biases)

            layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dw ** 2
            layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.db ** 2

            layer.weights -= self.learning_rate * layer.dw / (np.sqrt(layer.weight_cache) + self.epsilon)
            layer.biases -= self.learning_rate * layer.db / (np.sqrt(layer.bias_cache) + self.epsilon)

    # Corrected Momentum and Cache
    class Adam(Optimizer):
        def __init__(self, learning_rate, decay=0, epsilon=1e-7, beta1=0.9, beta2=0.999):
            super().__init__(learning_rate, decay)
            
            self.epsilon = epsilon
            self.beta1 = beta1
            self.beta2 = beta2
        
        def update(self, layer):
            # Start with Zeros
            if self.steps==0:
                layer.weight_momentum = np.zeros_like(layer.weights)
                layer.bias_momentum = np.zeros_like(layer.biases)
                layer.weight_cache = np.zeros_like(layer.weights)
                layer.bias_cache = np.zeros_like(layer.biases)
            
            # Momentum and Cache Calcs
            layer.weight_momentum = self.beta1 * layer.weight_momentum + (1 - self.beta1) * layer.dw
            layer.bias_momentum = self.beta1 * layer.bias_momentum + (1 - self.beta1) * layer.db
            layer.weight_cache = self.beta2 * layer.weight_cache + (1 - self.beta2) * layer.dw ** 2
            layer.bias_cache = self.beta2 * layer.bias_cache + (1 - self.beta2) * layer.db ** 2

            # Corrected Momentum and Cache
            weight_momentum = layer.weight_momentum / (1 - self.beta1 ** (self.steps + 1))
            bias_momentum = layer.bias_momentum / (1 - self.beta1 ** (self.steps + 1))
            weight_cache = layer.weight_cache / (1 - self.beta2 ** (self.steps + 1))
            bias_cache = layer.bias_cache / (1 - self.beta2 ** (self.steps + 1))

            # Update
            layer.weights -= self.learning_rate * weight_momentum / (np.sqrt(weight_cache) + self.epsilon)
            layer.biases -= self.learning_rate * bias_momentum / (np.sqrt(bias_cache) + self.epsilon)


    
    def print_iteration(self, epoch, iter):
        print(f"Epoch {epoch}, Iteration {iter}, Loss: {self.loss}, Accuracy: {self.accuracy}")

    def __str__(self):
        str = ""
        for i in range(len(self.layers)):
            str+=f"Layer {i}:\n{self.layers[i]}\n\n"
        return str
    
    def save(self, path):
        with open(path+".pickle", "wb") as outfile:
            pickle.dump(self, outfile)
    
    def load(path):
        with open(path+".pickle", "rb") as infile:
            return pickle.load(infile)

# nnfs.init()
# X, y = spiral_data(samples=100, classes=3)
# # # X = X[:5]
# # # y = y[:5]


# net = MLP(layers=[2, 64, 3], 
#           activation=[MLP.relu, MLP.softmax], 
#           loss=MLP.categoricalCrossEntropy, 
#           optimizer=MLP.Adam(learning_rate=0.05, decay=5e-7))

# for epoch in range(10001):
#     output = net.forward(X, y)
#     if not epoch % 100: 
#         print(f'epoch: {epoch}, ' +
#         f'acc: {net.accuracy:.3f}, ' + f'loss: {net.loss:.3f}')
#     net.backward()

# sgd = MLP.StochasticGradientDescent(learning_rate=1, decay=0.1, momentum=0.9)

# net.layers[0].weights = np.array([[0.2, 0.8, -0.5, 1], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]).T
# net.layers[1].weights = np.array([[0.1, -0.14, 0.5], [-0.5, 0.12, -0.33], [-0.44, 0.73, -0.13]]).T
# net.layers[0].biases = np.array([[2, 3, 0.5]])
# net.layers[1].biases = np.array([[-1, 2, -0.5]])
# print(net.forward([[1, 2, 3, 2.5], [2, 5, -1, 2], [-1.5, 2.7, 3.3, -0.8]]))