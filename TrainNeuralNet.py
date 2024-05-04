from NeuralNet import MLP
import csv

def main():
    net = MLP(784, [200, 80], 10, activation=[MLP.relu, MLP.relu, MLP.tanh], loss=MLP.mean_square)


if __name__ == "__main__":
    main()