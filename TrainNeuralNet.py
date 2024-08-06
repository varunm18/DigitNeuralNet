from NeuralNet import MLP
import numpy as np
import math

CONVERT = False
BATCH_SIZE = 32
EPOCHS = 201

def main():
    if CONVERT:
        train_data = np.genfromtxt('MNISTdata/mnist_train.csv', delimiter=',').astype(int)
        np.save('MNISTdata/mnist_train.npy', train_data)

        test_data = np.genfromtxt('MNISTdata/mnist_test.csv', delimiter=',').astype(int)
        np.save('MNISTdata/mnist_test.npy', test_data)

    net = MLP(layers=[784, 128, 128, 10], 
              activation=[MLP.relu, MLP.relu, MLP.softmax], 
              loss=MLP.categoricalCrossEntropy,
              optimizer=MLP.StochasticGradientDescent(learning_rate=0.5, decay=0.0005, momentum=0.12))

    # TRAIN
    train_data = np.load("MNISTdata/mnist_train.npy")
    X, y = train_data[:, 1:], train_data[:, 0]
    X = X / 255.0
    iterations = math.ceil(X.shape[0] / BATCH_SIZE)

    for e in range(EPOCHS):
        for i in range(iterations):
            net.forward(input=X[i * BATCH_SIZE : (i+1) * BATCH_SIZE], label=y[i * BATCH_SIZE : (i+1) * BATCH_SIZE])
            net.backward()
            net.print_iteration(e, i)

    # TEST
    test_data = np.load("MNISTdata/mnist_test.npy")
    X, y = test_data[:, 1:], test_data[:, 0]  
    X = X / 255.0
    iterations = math.ceil(X.shape[0] / BATCH_SIZE)

    correct = 0
    total = 0
    for i in range(iterations):
        predictions = np.argmax(net.run(input=X[i * BATCH_SIZE : (i+1) * BATCH_SIZE]), axis=1)
        correct += np.sum(predictions == y[i * BATCH_SIZE : (i+1) * BATCH_SIZE])
        total += len(predictions)
    
    print(f"Test Accuracy: {correct / total}")

    if(input("Save? ") in ['Y', 'y']):
        net.save(f"Models/{input('Save As: ')}")

if __name__ == "__main__":
    main()