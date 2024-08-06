# Digit Neural Network
A Multi Level Perceptron written from scratch. Used for digit classification(MNIST dataset) and visualization

## Description

#### NeuralNet.py
* NeuralNet.py holds MLP and Layer classes using only numpy
* Allows for following:
  * Input layer, ouput layer and dynamic amount of hidden layers
  * Activation functions (ReLU, softmax, tanh, sigmoid)
  * Loss functions (Categorical Cross Entropy, Mean Square)
  * Optimizers (Stochastic Gradient Descent with decay and momentum, RMSProp, Adam)
* Parallelization with variable batching implemented
* OLDNeuralNet.py and OLDTrainNeuralNet.py are my first attempts without parallelization
* Serialization and loading from memory of MLP through pickle

#### TrainNeuralNet.py
* Converts MNIST CSV data to numpy objects
* Loads data from memory, clamps pixel values from 0-1 and splits based on batch size
* Runs forward and backward propogation through training data for specified amount of epochs
* Runs on testing data and tallys accuracy

### Highest Accuracy of 98.31%
* 784 input, 128 hidden, 128 hidden, and 10 output neurons
* Activation in order: ReLU, ReLU, Softmax
* Categorical Cross Entropy loss
* SGD Optimizer with learning rate of 0.5, decay of 0.0005 and momentum of 0.12

```python
net = MLP(layers=[784, 128, 128, 10], 
              activation=[MLP.relu, MLP.relu, MLP.softmax], 
              loss=MLP.categoricalCrossEntropy,
              optimizer=MLP.StochasticGradientDescent(learning_rate=0.5, decay=0.0005, momentum=0.12))
```
  
## UI

#### Draw.py
* pygame used to create UI for drawing digits
* Automatically rescales and processes drawing pane to be classified by a model loaded from memory
* Displays probabilities distribution of models prediction
* Pixelated toggle button that shows rescaled image
  
![NeuralNetDisplay](https://github.com/user-attachments/assets/d39a3307-bfe1-4c49-8d75-0f33981373e2)
