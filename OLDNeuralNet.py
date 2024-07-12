import numpy as np
import math
import random
import time
import pickle

class Parameter:
    def __init__(self, val):
        self.val = val
        self.grad = 0
    
    def __str__(self):
        return str(self.val)
    
    def __repr__(self):
        return str(self)

class Neuron:
    def __init__(self, inputs):
        self.bias = Parameter(random.uniform(-1, 1))
        self.weights = [Parameter(random.uniform(-1, 1)) for i in range(inputs)]
    
    def __str__(self):
        return f"weight={self.weights}, bias={self.bias}"

class Layer:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.neurons = [Neuron(inputs) for _ in range(outputs)]

    def __str__(self):
        str = ""
        for i in range(len(self.neurons)):
            str+=f"neuron {i+1}-\n{self.neurons[i]}\n"
        return str

class MLP:
    def __init__(self, inputs, hidden, output, activation, loss):
        if(len(hidden)==0):
            self.layers = [Layer(inputs, output)]
        else:
            self.layers = [Layer(inputs, (hidden[0]))]
            if(len(hidden)==1):
                self.layers += [Layer(hidden[0], output)]
            else:
                self.layers += [Layer(input, output) for input, output in zip(hidden, hidden[1:])] 
                self.layers += [Layer(hidden[-1], output)]

        self.activation_func = activation
        self.loss_func = loss
        
    def forward(self, input, label):
        inputs = input.copy()
        z = []
        a = []
        
        for i in range(len(self.layers)):
            unactivated = []
            activated = []
            for neuron in self.layers[i].neurons:
                w = [item.val for item in neuron.weights]
                val = np.dot(w, inputs) + neuron.bias.val
                unactivated.append(val)

                activated.append(self.activation_func[i](val))
            
            inputs = activated.copy()
            a.append(activated)
            z.append(unactivated)
        
        # a[-1] = MLP.normalize(a[-1])
                
        self.z = z
        self.a = a
        # print(f"z: {self.z}, a: {self.a}")

        self.loss = self.loss_func(a[-1], label)
        # print(f"\nLoss: {self.loss}")
        self.input = input
        self.label = label
        self.output = a[-1]
    
    def backward(self, learning_rate):

        for layer in self.layers:
            for neuron in layer.neurons:
                for weight in neuron.weights:
                    weight.grad = 0
                neuron.bias.grad = 0

        in_da = []
        for i in range(len(self.layers[-1].neurons)):
            in_da.append(self.loss_func(self.a[-1][i], self.label[i], deriv=True))

        for i in range(len(self.layers)-1, -1, -1):
            next_da = [0 for i in range(len(self.layers[i-1].neurons))]
            for n in range(len(self.layers[i].neurons)):
                dz = in_da[n] * self.activation_func[i](self.z[i][n], deriv=True)
                
                if(i==0):
                    dw = np.multiply(dz, self.input)
                else:   
                    dw = np.multiply(dz, self.a[i-1])
                
                db = dz

                for w in range(len(self.layers[i].neurons[n].weights)):
                    self.layers[i].neurons[n].weights[w].grad += dw[w]
                    if i>0:
                        next_da[w] += dz * self.layers[i].neurons[n].weights[w].val

                self.layers[i].neurons[n].bias.grad += db
                
            in_da = next_da.copy()

        # for i in range(len(self.layers)):
        #     print(f"LAYER {i}: \n")
        #     for n in range(len(self.layers[i].neurons)):
        #         print(f"Neuron {n}:")
        #         str = "weights: ["
        #         for w in self.layers[i].neurons[n].weights:
        #             str+=f"{w.grad}, "
        #         print(f"{str[:-2]}], bias: {self.layers[i].neurons[n].bias.grad}") 
        #     print()

        for layer in self.layers:
            for neuron in layer.neurons:
                for weight in neuron.weights:
                    weight.val -= learning_rate * weight.grad
                neuron.bias.val -= learning_rate * neuron.bias.grad
    
    def run(self, input):
        inputs = input.copy()
        for i in range(len(self.layers)):
            activated = []
            for neuron in self.layers[i].neurons:
                w = [item.val for item in neuron.weights]
                val = np.dot(w, inputs) + neuron.bias.val
                
                activated.append(self.activation_func[i](val))
            
            inputs = activated.copy()
        
        return MLP.normalize(activated)
        
    def mean_square(x, label, deriv=False):
        return 2 * (x - label) if deriv else sum(np.subtract(x, label) ** 2)

    def relu(x, deriv=False):
        return float(np.greater(x, 0)) if deriv else max(0, x)

    def tanh(x, deriv=False):
        return 1 - (math.tanh(x) ** 2) if deriv else math.tanh(x)

    def sigmoid(x, deriv=False):
        return MLP.sigmoid(x) * (1 - MLP.sigmoid(x)) if deriv else 1 / (1 + math.exp(-x))
    
    def normalize(x):
        if all(item == 0 for item in x):
            return x
        val = sum(x)
        return [item/val for item in x]
    
    def __str__(self):
        str = ""
        for i in range(len(self.layers)):
            str+=f"LAYER {i}:\n\n{self.layers[i]}\n"
        return str

    def save(self, path):
        with open(path+".pickle", "wb") as outfile:
            pickle.dump(self, outfile)
    
    def load(path):
        with open(path+".pickle", "rb") as infile:
            return pickle.load(infile)
    
# net = MLP(1, [200, 80], 10, activation=[MLP.relu, MLP.sigmoid, MLP.sigmoid], loss=MLP.mean_square)
# net = MLP.load("network")

# net.layers[0].neurons[0].weights = [Parameter(0.28539878179947875)]
# net.layers[0].neurons[1].weights = [Parameter(-0.9459687259330387)]
# net.layers[0].neurons[0].bias = Parameter(0.43144037040913585)
# net.layers[0].neurons[1].bias = Parameter(-0.9356339400196843)

# net.layers[1].neurons[0].weights = [Parameter(-0.33498064056179166), Parameter(0.16930568930048895)]
# net.layers[1].neurons[0].bias = Parameter(-0.1376329332489683)
            
# start = time.time()
# for k in range(600):
#   # forward pass
#   print(k)
#   net.forward([30], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

#   # backward pass
#   net.backward(0.05)
  
#   print(net.output)
            
# end = time.time()
# print(end - start)
# print(f"{net}\n\n")
# net.forward([30], [0.7])
# print(f"\n")
# net.backward(0.05)

