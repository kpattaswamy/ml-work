import collections
import math
import numpy as np

class Gaussian_Naive_Bayes():
    def fit(self, X_train, y_train):
        """
        fit with training data
        Inputs:
            - X_train: A numpy array of shape (N, D) containing training data; there are N
                training samples each of dimension D.
            - y_train: A numpy array of shape (N,) containing training labels; y[i] = c
                means that X[i] has label 0 <= c < C for C classes.
                
        With the input dataset, function gen_by_class will generate class-wise mean and variance to implement bayes inference.

        Returns:
        None
        
        """
        self.x = X_train
        self.y = y_train  

        self.gen_by_class()
       
    def gen_by_class(self):
        """
        With the given input dataset (self.x, self.y), generate 3 dictionaries to calculate class-wise mean and variance of the data.
        - self.x_by_class : A dictionary of numpy arraies with the keys as each class label and values as data with such label.
        - self.mean_by_class : A dictionary of numpy arraies with the keys as each class label and values as mean of the data with such label.
        - self.std_by_class : A dictionary of numpy arraies with the keys as each class label and values as standard deviation of the data with such label.
        - self.y_prior : A numpy array of shape (C,) containing prior probability of each class
        """
        self.x_by_class = dict()
        self.mean_by_class = dict()
        self.std_by_class = dict()
        self.y_prior = None
        
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # Generate dictionaries.
        # hint : to see all unique y labels, you might use np.unique function, e.g., np.unique(self.y)
        for i in range(len(self.y)):
            if (self.y[i] not in self.x_by_class):
                self.x_by_class[self.y[i]] = []

            self.x_by_class[self.y[i]].append(self.x[i])

        for key, value in self.x_by_class.items():
            self.mean_by_class[key] = []

            x = 0
            for i in range(len(value[0])):
                temp = []
                for i in value:
                    temp.append(i[x])
                avg = self.mean(temp)
                self.mean_by_class[key].append(avg)
                x+=1
                temp = []

        for key, value in self.x_by_class.items():
            self.std_by_class[key] = []

            x = 0
            for i in range(len(value[0])):
                temp = []
                for i in value:
                    temp.append(i[x])
                stdev = self.std(temp)
                self.std_by_class[key].append(stdev)
                x+=1
                temp = []
        
        self.y_prior = []
        
        unique_label = np.unique(self.y)
        for i in unique_label:
            num = len(self.x_by_class[i])
            self.y_prior.append(num/len(self.y))
 

        #print(self.std_by_class[0])
        #exit(1)
        # END_YOUR_CODE
        ############################################################
        ############################################################        

    def mean(self, x):
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # Calculate mean of input x
        total = 0
        for i in x:
            total += i
        mean = total / len(x)
    
        # END_YOUR_CODE
        ############################################################
        ############################################################
        return mean
    
    def std(self, x):
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # Calculate standard deviation of input x, do not use np.std

        avg = self.mean(x)
        var = sum(pow(temp-avg + 1e-6, 2) for temp in x) / (len(x)-1 + 1e-5)
        std = math.sqrt(var)
        # END_YOUR_CODE
        ############################################################
        ############################################################
        return std
    
    def calc_gaussian_dist(self, x, mean, std):
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # calculate gaussian probability of input x given mean and std

        #prob = (1/(std * math.sqrt(2 * math.pi))) * np.exp(-1/2 * math.pow((x - mean) / std, 2 ))

        prob = []
        val = 0

        for i in range(len(x)):
            val = (1/(std[i] * math.sqrt(2 * math.pi) + 1e-5)) * np.exp(-1/2 * math.pow((x[i] - mean[i]) / (std[i]+1e-5), 2))
            prob.append(val)
            val = 0

        #print(prob)
        #print(np.sum(prob))
        #print(self.mean(prob))
        #exit(1)

        # END_YOUR_CODE
        ############################################################
        ############################################################
        return prob
        
    def predict(self, x):
        """
        Use the acquired mean and std for each class to predict class for input x.
        Inputs:

        Returns:
        - prediction: Predicted labels for the data in x. prediction is (N, C) dimensional array, for N samples and C classes.
        """
        n = len(x)
        num_class = len(np.unique(self.y))
        prediction = np.zeros((n, num_class))
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # calculate naive bayes probability of each class of input x
        

        for i in range(len(prediction)):
            prior = 0
            for j in range(len(prediction[0])):
                prior = np.sum(
                np.log(self.calc_gaussian_dist(x[i], self.mean_by_class[j], self.std_by_class[j])
                )) + np.log(self.y_prior[j]+1e-5)
                
                #print(prior)
                #exit(1)

                prediction[i][j] = prior

            
        #print(prediction)

        # END_YOUR_CODE
        ############################################################
        ############################################################
        
        return prediction


class Neural_Network():
    def __init__(self, hidden_size = 64, output_size = 1):
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None    
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        
    def fit(self, x, y, batch_size = 64, iteration = 2000, learning_rate = 1e-3):
        """
        Train this 2 layered neural network classifier using mini-batch stochastic gradient descent.
        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c
          means that X[i] has label 0 <= c < C for C classes.
        - learning_rate: (float) learning rate for optimization.
        - iteration: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        
        Use the given learning_rate, iteration, or batch_size for this homework problem.

        Returns:
        None
        """  

        dim = x.shape[1]
        num_train = x.shape[0]

        '''
        print(f"dim: {dim}")
        print(f"num train: {num_train}")
        print(x[0])
        print(y[0])
        print(len(x[0]))
        '''

        #initialize W
        if self.W1 == None:
            self.W1 = 0.001 * np.random.randn(dim, self.hidden_size)
            self.b1 = 0
            
            self.W2 = 0.001 * np.random.randn(self.hidden_size, self.output_size)
            self.b2 = 0


        for it in range(iteration):
            batch_ind = np.random.choice(num_train, batch_size)

            x_batch = x[batch_ind]
            y_batch = y[batch_ind]

            loss, gradient = self.loss(x_batch, y_batch)
            '''
            print("in fitr")
            print(loss)
            print(gradient)
            '''
            ############################################################
            ############################################################
            # BEGIN_YOUR_CODE
            # Update parameters with mini-batch stochastic gradient descent method
            
            self.W1 -= learning_rate * gradient["dW1"]
            self.W2 -= learning_rate * gradient["dW2"]

            self.b1 -= learning_rate * gradient["db1"]
            self.b2 -= learning_rate * gradient["db2"]

            # END_YOUR_CODE
            ############################################################
            ############################################################
            
            y_pred = self.predict(x_batch)
            acc = np.mean(y_pred == y_batch)
            
            if it % 50 == 0:
                print('iteration %d / %d: accuracy : %f: loss : %f' % (it, iteration, acc, loss))
                
    def loss(self, x_batch, y_batch, reg = 1e-3):
            """
            Implement feed-forward computation to calculate the loss function.
            And then compute corresponding back-propagation to get the derivatives. 

            Inputs:
            - X_batch: A numpy array of shape (N, D) containing a minibatch of N
              data points; each point has dimension D.
            - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
            - reg: hyperparameter which is weight of the regularizer.

            Returns: A tuple containing:
            - loss as a single float
            - gradient dictionary with four keys : 'dW1', 'db1', 'dW2', and 'db2'
            """
            gradient = {'dW1' : None, 'db1' : None, 'dW2' : None, 'db2' : None}


            ############################################################
            ############################################################
            # BEGIN_YOUR_CODE
            # Calculate y_hat which is probability of the instance is y = 0.

            g1 = x_batch.dot(self.W1) + self.b1

            h1 = self.activation(g1)

            h2 = h1.dot(self.W2) + self.b1

            y_hat = self.sigmoid(h2)
            
            # END_YOUR_CODE
            ############################################################
            ############################################################


            ############################################################
            ############################################################
            # BEGIN_YOUR_CODE
            # Calculate loss and gradient

            #dw1 = (-1/np.abs(x_batch[0])).dot(y_hat - y_batch)

            #dw1 = (-1/np.abs(x_batch[0])) * (x_batch.T).dot( (y_batch - y_hat).dot(np.transpose(self.W2)) * np.heaviside(g1, 1))
            

            mult1 = (y_batch - y_hat).dot(np.transpose(self.W2))
            mult2 = np.heaviside(g1, 1)

            initial = (-1/x_batch.shape[0]) * x_batch.T

            dw1 = initial.dot(np.multiply(mult1, mult2)) + (2 * reg * self.b1)



            # + (2*reg*self.b1)


            #dw2 = (-1/np.abs(x_batch[0])).dot(h1.T) * (y_batch - y_hat)
            #dw2 = (-1/np.abs(x_batch[0])).dot(h1.T)


            #dw2 = np.dot(-1/np.abs(x_batch[0]), h1.T)
            #dw2 = dw2.dot(y_batch - y_hat)
            
            
            #dw2 = h1.T.dot(y_batch - y_hat) + (2 * reg * self.b1) + (2 * reg * self.b1)

            right = np.dot(h1.T, (y_batch - y_hat))
            dw2 = (-1/np.abs(x_batch.shape[0])) * right + (2 * reg * self.b2)


            #print(dw1)
            #print(dw2)
            #exit(1)

            #dw2 = np.dot((-1/np.abs(x_batch[0])).dot(h1.T), (y_batch - y_hat))
            #db1 = np.mean(dw1 + (2 * reg * self.b1))
            

            #dw2 = np.dot(x_batch.T, (y_hat - y_batch))

            #dw2 = x_batch.T.dot(y_hat - y_batch)

            #db2 = np.mean(dw2 + (2* reg * self.b1))
            db2 = np.mean((y_batch - y_hat) + (2* reg * self.b1))
            db1 = np.mean((y_batch - y_hat).dot(np.transpose(self.W2)) * np.heaviside(g1, 1) + (2 * reg * self.b1))


            #derW = np.dot(x_batch.T, (y_pred - y_batch))
            #loss = (-1/x_batch[0]) * 

            loss = np.sum(y_batch * np.log(y_hat+1e-5) + (np.log(1-y_hat + 1e-5) * (1-y_batch)))
            loss = (-1 / abs(x_batch.shape[0])) * loss

            gradient["dW1"] = dw1
            gradient["dW2"] = dw2
            gradient["db1"] = db1
            gradient["db2"] = db2


            # END_YOUR_CODE
            ############################################################
            ############################################################
            return loss, gradient

    def activation(self, z):
        """
        Compute the ReLU output of z
        Inputs:
        z : A scalar or numpy array of any size.
        Return:
        s : output of ReLU(z)
        """ 
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # Implement ReLU 
        s = np.maximum(0,z)

        # END_YOUR_CODE
        ############################################################
        ############################################################
        return s
        
    def sigmoid(self, z):
        """
        Compute the sigmoid of z
        Inputs:
        z : A scalar or numpy array of any size.
        Return:
        s : sigmoid of input
        """ 
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE

        s = 1/(1 + (np.clip(np.exp(-z), -500, 500)))

        # END_YOUR_CODE
        ############################################################
        ############################################################
        
        return s
    
    def predict(self, x):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.
        Inputs:

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # Calculate predicted y


        g1 = x.dot(self.W1) + self.b1

        h1 = self.activation(g1)

        h2 = h1.dot(self.W2) + self.b1

        y_hat = self.sigmoid(h2)

        num_trained = y_hat.shape[0]

        for i in range(num_trained):
        #for i in range(num_trained):
            if y_hat[i] >= 0.5:
                y_hat[i] = 1
            else:
                y_hat[i] = 0

        # END_YOUR_CODE
        ############################################################
        ############################################################
        return y_hat

    
