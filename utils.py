import pickle as pkl # For loading the dataset file

def load_dataset(filename="mnist.pkl"):
    """Load MNIST images & labels from a pickle file.

       Input:
        - filename: A string for the name of the pickle file containing the MNIST samples.

       Output:
        - data_dict: A dictionary that contains the images and the labels.
    """
    infile = open(filename, 'rb')     
    data_dict = pkl.load(infile)
    infile.close()

    return data_dict

def load_network(filename="network_3layer.pkl"):
    """Load a network from a pickle file.

       Input:
        - filename: A string for the name of the pickle file containing the network.

       Output:
        - network: A list of layers, e.g. 
                    [['linear', Weights], 'relu', ['linear', Weights], ...]
    """
    infile = open(filename, 'rb')     
    network = pkl.load(infile)
    infile.close()

    return network


def display_image(X):
    """Display an image using ASCII chars.
       
       Input:
        - X: An image as a vector; i.e. a list with D elements.

       Output: None.
    """
    for i in range(0, 28*28):
        if i % 28 == 0 and i > 0: print("")
        print("." if X[i] < 125 else "@", end="")
    print("")

def display_network(network):
    """Display a network's layers and layer sizes.
       
       Input:
        - network: A list of layers, e.g. 
                    [['linear', Weights], 'relu', ['linear', Weights], ...]

       Output: None.
    """
    print([layer[0]+": " + str(len(layer[1][0]))+"->"+str(len(layer[1])) if type(layer) == list else layer for layer in network])


def calculate_accuracy(dataset, network, predictor):
    """Calculate the accuracy of a network's predictions.
       
       Input:
        - dataset: A dictionary that contains the images and the labels.
        - network: A list of layers, e.g. 
                    [['linear', Weights], 'relu', ['linear', Weights], ...]
        - predictor: The forward-pass function that takes the network and a sample, and returns the outputs
                    of the last layer.
       Output: Accuracy (%).
    """
    X_test = dataset['X_test']
    y_test = dataset['y_test']
    N = len(X_test)

    correct = 0
    for i in range(N):
        X = X_test[i]
        y = y_test[i]
        output = predictor(network, X)
        y_pred = output.index(max(output))
        if y == y_pred: correct += 1

    return (correct / N) * 100