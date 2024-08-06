import numpy as np
import pickle
 
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
        
        def __str__(self):
            return f"Base SGD- Learning Rate: {self.original_lr}, Decay: {self.decay}"
    
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
        
        def __str__(self):
            return f"SGD- Learning Rate: {self.original_lr}, Decay: {self.decay}, Momentum: {self.momentum}"

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
        
        def __str__(self):
            return f"RMSProp- Learning Rate: {self.original_lr}, Decay: {self.decay}, Epsilon: {self.epsilon}, Rho: {self.rho}"

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
        
        def __str__(self):
            return f"Adam- Learning Rate: {self.original_lr}, Decay: {self.decay}, Epsilon: {self.epsilon}, Beta 1: {self.beta1}, Beta 2: {self.beta2}"

    
    def print_iteration(self, epoch, iter):
        print(f"Epoch {epoch}, Iteration {iter}, Loss: {self.loss}, Accuracy: {self.accuracy}")
    
    def print_details(self):
        for layer in self.layers:
            print(f"Shape: {layer.weights.shape[0]} to {layer.weights.shape[1]}, Activation: {layer.activation_func}")
        print(f"Loss: {self.loss_func}")
        print(f"{self.optimizer}")

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