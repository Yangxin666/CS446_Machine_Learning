#!/usr/bin/env python3
import numpy as np
from io import StringIO

NUM_FEATURES = 124 #features are 1 through 123 (123 only in test set), +1 for the bias
DATA_PATH = "/u/cs246/data/adult/" #TODO: if doing development somewhere other than the cycle server, change this to the directory where a7a.train, a7a.dev, and a7a.test are

#returns the label and feature value vector for one datapoint (represented as a line (string) from the data file)
def parse_line(line):
    tokens = line.split()
    x = np.zeros(NUM_FEATURES)
    y = int(tokens[0])
    y = max(y,0) #treat -1 as 0 instead, because sigmoid's range is 0-1
    for t in tokens[1:]:
        parts = t.split(':')
        feature = int(parts[0])
        value = int(parts[1])
        x[feature-1] = value
    x[-1] = 1 #bias
    return y, x

#return labels and feature vectors for all datapoints in the given file
def parse_data(filename):
    with open(filename, 'r') as f:
        vals = [parse_line(line) for line in f]
        (ys, xs) = ([v[0] for v in vals],[v[1] for v in vals])
        return np.asarray([ys],dtype=np.float32).T, np.asarray(xs,dtype=np.float32).reshape(len(xs),NUM_FEATURES,1) #returns a tuple, first is an array of labels, second is an array of feature vectors


class Neural_Nets:
    def __init__(self, layers_size,w1,w2,b1,b2):
        self.layers_size = layers_size
        self.parameters = {}
        self.L = len(self.layers_size)
        self.n = 0
        self.costs = []
        self.w1=w1
        self.w2=w2
        self.b1=b1
        self.b2=b2
 
    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))
 
    def initialize_parameters(self):
        self.parameters["W" + str(1)] = self.w1
        self.parameters["b" + str(1)] = self.b1
        self.parameters["W" + str(2)] = self.w2
        self.parameters["b" + str(2)] = self.b2
 
    def forward(self, x):
        store = {}
 
        A = x.T
        Z = self.parameters["W1"].dot(A) + self.parameters["b1"]
        A = self.sigmoid(Z)
        store["A1"] = A
        
        store["W1"] = self.parameters["W1"]
        store["Z1"] = Z
        
        Z = self.parameters["W2"].dot(A) + self.parameters["b2"]
        A = self.sigmoid(Z)
        store["A2"] = A

        store["W2"] = self.parameters["W2"]
        store["Z2"] = Z
   
 
        return A, store
 
    def sigmoid_derivative(self, Z):
        s = 1 / (1 + np.exp(-Z))
        return s * (1 - s)
 
    def backward(self, X, Y, store):
 
        derivatives = {}
        store["A0"] = X.T
 
        A2 = store["A" + str(self.L)]
        dA2 = -np.divide(Y.T, A2) + np.divide(1 - Y.T, 1 - A2)
        dZ2 = dA2 * self.sigmoid_derivative(store["Z2"])
        dW2 = (dZ2 * store["A1"]).T
        db2 = dZ2
        dA1 = (dZ2 * store["W2"]).T
        dZ1 = dA1 * self.sigmoid_derivative(store["Z1"])
        dW1 = dZ1.dot(store["A0"].T)
        db1 = dZ1
 
        derivatives["dW2"] = dW2
        derivatives["dW1"] = dW1
        derivatives["db1"] = db1
        derivatives["db2"] = db2
 
        return derivatives
 
    def fit(self, X, Y, learning_rate, n_iterations):
        np.random.seed(1)
 
        self.n = X.shape[0]
 
        self.initialize_parameters()
        for j in range(n_iterations):
            for i in range(self.n):
                A, store = self.forward(X[i].reshape(1, 123))
            
                derivatives = self.backward(X[i].reshape(1, 123), Y[i].reshape(1, 1), store)
 
                self.parameters["W1"] = self.parameters["W1"] - learning_rate * derivatives["dW1"]
                self.parameters["b1"] = self.parameters["b1"] - learning_rate * derivatives["db1"]
                self.parameters["W2"] = self.parameters["W2"] - learning_rate * derivatives["dW2"]
                self.parameters["b2"] = self.parameters["b2"] - learning_rate * derivatives["db2"]
 
    def predict(self, X, Y):
        A, cache = self.forward(X)
        err=0
        for i in range(A.shape[1]):
            if A[:,i] > 0.5:
                A[:,i]=1
            else:
                A[:,i]=0
            if A[:,i]!=Y[i]:
                err+=1
        accuracy = 1-err/A.shape[1]
        return accuracy
  
        
def init_model(args):
    w1 = None
    w2 = None

    if args.weights_files:
        with open(args.weights_files[0], 'r') as f1:
            w1 = np.loadtxt(f1)
        with open(args.weights_files[1], 'r') as f2:
            w2 = np.loadtxt(f2)
            w2 = w2.reshape(1,len(w2))
    else:
        #TODO (optional): If you want, you can experiment with a different random initialization. As-is, each weight is uniformly sampled from [-0.5,0.5).
        w1 = np.random.rand(args.hidden_dim, NUM_FEATURES) #bias included in NUM_FEATURES
        w2 = np.random.rand(1, args.hidden_dim + 1) #add bias column

    #At this point, w1 has shape (hidden_dim, NUM_FEATURES) and w2 has shape (1, hidden_dim + 1). In both, the last column is the bias weights.

    b1=w1[:,123]
    b1 = b1.reshape(2, 1) 
    w1=w1[:,0:123]
    b2=w2[:,2]
    w2=w2[:,0:2]
    b2 = b2.reshape(1, 1)
    model = Neural_Nets([2,1],w1,w2,b1,b2)
    return model
			
def train_model(model, train_ys, train_xs, dev_ys, dev_xs, args):
    model.fit(train_xs, train_ys, args.lr, args.iterations)
    return model

def test_accuracy(model, test_ys, test_xs):
    accuracy = model.predict(test_xs, test_ys)
    return accuracy

def extract_weights(model):
    w1 = model.parameters["W1"]
    b1 = model.parameters["b1"]
    w2 = model.parameters["W2"]
    b2 = model.parameters["b2"]
    W1 = np.concatenate((w1, b1), 1)
    W2 = np.concatenate((w2, b2), 1)
    return W1, W2

def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Neural network with one hidden layer, trainable with backpropagation.')
    parser.add_argument('--nodev', action='store_true', default=False, help='If provided, no dev data will be used.')
    parser.add_argument('--iterations', type=int, default=5, help='Number of iterations through the full training data to perform.')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate to use for update in training loop.')

    weights_group = parser.add_mutually_exclusive_group()
    weights_group.add_argument('--weights_files', nargs=2, metavar=('W1','W2'), type=str, help='Files to read weights from (in format produced by numpy.savetxt). First is weights from input to hidden layer, second is from hidden to output.')
    weights_group.add_argument('--hidden_dim', type=int, default=5, help='Dimension of hidden layer.')

    parser.add_argument('--print_weights', action='store_true', default=False, help='If provided, print final lear tned weights to stdout (used in autograding)')

    parser.add_argument('--train_file', type=str, default=os.path.join(DATA_PATH,'a7a.train'), help='Training data file.')
    parser.add_argument('--dev_file', type=str, default=os.path.join(DATA_PATH,'a7a.dev'), help='Dev data file.')
    parser.add_argument('--test_file', type=str, default=os.path.join(DATA_PATH,'a7a.test'), help='Test data file.')


    args = parser.parse_args()

    """
    At this point, args has the following fields:

    args.nodev: boolean; if True, you should not use dev data; if False, you can (and should) use dev data.
    args.iterations: int; number of iterations through the training data.
    args.lr: float; learning rate to use for training update.
    args.weights_files: iterable of str; if present, contains two fields, the first is the file to read the first layer's weights from, second is for the second weight matrix.
    args.hidden_dim: int; number of hidden layer units. If weights_files is provided, this argument should be ignored.
    args.train_file: str; file to load training data from.
    args.dev_file: str; file to load dev data from.
    args.test_file: str; file to load test data from.
    """
    train_ys, train_xs = parse_data(args.train_file)
    dev_ys = None
    dev_xs = None
    if not args.nodev:
        dev_ys, dev_xs= parse_data(args.dev_file)
    test_ys, test_xs = parse_data(args.test_file)
    train_xs=np.squeeze(train_xs)
    test_xs=np.squeeze(test_xs)
    train_xs=train_xs[:,0:123]
    test_xs=test_xs[:,0:123]
    model = init_model(args)
    model = train_model(model, train_ys, train_xs, dev_ys, dev_xs, args)
    accuracy = test_accuracy(model, test_ys, test_xs)
    print('Test accuracy: {}'.format(accuracy))
    if args.print_weights:
        w1, w2 = extract_weights(model)
        with StringIO() as weights_string_1:
            np.savetxt(weights_string_1,w1)
            print('Hidden layer weights: {}'.format(weights_string_1.getvalue()))
        with StringIO() as weights_string_2:
            np.savetxt(weights_string_2,w2)
            print('Output layer weights: {}'.format(weights_string_2.getvalue()))

if __name__ == '__main__':
    main()
