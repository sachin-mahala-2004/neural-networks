import numpy as np
class Network:
    """ sizes here represent sizes of each layer i.e. number of neurons in each layer"""
    def __init__(self,sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.rand(x,1) for x in sizes[1:]]
        self.weights = [np.random.rand(x,y) for x,y in zip(sizes[1:],sizes[:-1])]
        
    def sigmoid(z):
        return 1.0/1+np.exp(-z)
    
    def feedforward(self,a):
        """Return the output of the network if "a" is input."""
        for w,b in zip(self.biases,self.weights):
            a = w@a + b
        return a
        
    def SGD(self,training_data,eta,epoches,mini_batch_size,test_data = None):
        """Train the neural network using mini-batch stochastic gradient descent. 
        The "training_data" is a list of tuples "(x,y)" representing the training inputs 
        and the desired outputs. If the test_data is provided then network will be evaluated 
        against the test data after each epoch, and partial progress printed out.
        This is usefull for tracking progress , but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for epoch in range(epoches):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta)
            if test_data:
                print(f"Epoch {epoch}: {self.evaluate(test_data)}/{n_test}")
            
            else:
                print(f"Epoch {epoch} complete")
    def update_mini_batch(self,mini_batch,eta):
        """Update the network's weights and biases by applying gradient using backpropogation
        to a single mini batch."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b , delta_nabla_w = self.backprop(x,y)
            """It looks at the image, sees what the network guessed, calculates the error,
            and figures out exactly how much to tweak the weights and biases to get closer to the right answer."""
            nabla_b = [nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
            nabla_w = [nw+dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]
        
        self.weights = [w - (eta/len(mini_batch))*nw 
                        for w,nw in zip(self.weights,nabla_w)]       
        self.biases = [b - (eta/len(mini_batch))*nb
                       for b,nb in zip(self.biases,nabla_b)]
        
net = Network([2,3,1])