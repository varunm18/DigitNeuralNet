from OLDNeuralNet import MLP
import csv
import time

def main():
    net = MLP(784, [200, 80], 10, activation=[MLP.tanh, MLP.sigmoid, MLP.sigmoid], loss=MLP.mean_square)
    # net = MLP.load("Models/full")

    # train
    start = time.time()
    try:
        with open("MNISTdata/mnist_train.csv") as file:
            train_data = csv.reader(file, delimiter=',')
            v = 0
            for epoch in range(3):
                for row in train_data:
                    num = [0 for i in range(10)]
                    num[int(row[0])] = 1
                    
                    pixels = [int(pixel) for pixel in row[1:]]

                    # forward pass
                    net.forward(input=pixels, label=num)

                    # backward pass
                    net.backward(learning_rate=0.1)

                    v+=1
                    print(f"{v} Loss: {net.loss}")

    except KeyboardInterrupt:
        print('Interrupted')

    # test
    try:
        with open("MNISTdata/mnist_test.csv") as file:
            test_data = csv.reader(file, delimiter=',')
            correct = 0
            total = 0
            for row in test_data:
                num = int(row[0])
                
                pixels = [int(pixel) for pixel in row[1:]]

                distribution = net.run(input=pixels)
                predicted = distribution.index(max(distribution))
                
                if(num==predicted):
                    correct+=1
                total+=1

                print(correct, total)

    except KeyboardInterrupt:
        print('Interrupted')
        
    print(f"Accuracy: {correct/total * 100}%")
    
    if(input("Save? ") in ['Y', 'y']):
        net.save(f"Models/{input('Save As: ')}")

    end = time.time()
    print(end - start)

if __name__ == "__main__":
    main()
        