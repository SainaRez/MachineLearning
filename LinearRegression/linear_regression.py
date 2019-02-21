import math
import numpy as np

#-------------------------------------------------------------------------
'''
    Linear Regression
    This file uses python 3
    This file implements the linear regression method based upon gradient descent.
    Xw  = y
    You could test the correctness of the code by typing `nosetests3 -v linear_regression_test.py` in the terminal.
'''

#--------------------------
def compute_Phi(x,p):
    '''
        Computes the feature matrix Phi of x. We will construct p polynoials, the p features of the data samples. 
        The features of each sample, is x^0, x^1, x^2 ... x^(p-1)
        Input:
            x : a vector of samples in one dimensional space, a numpy vector of shape n by 1.
                Here n is the number of samples.
            p : the number of polynomials/features
        Output:
            Phi: the design/feature matrix of x, a numpy matrix of shape (n by p).
    '''
    #########################################

    features = []
    X = np.array(x)
    matrix_shape = np.shape(X)
    for i in range(matrix_shape[0]):
        print (X[i])
        row = []
        for r_index in range(p): 
            row.append(X[i][0]**r_index)
        features.append(row)
    print (features)
    Phi = np.matrix(features)

    #########################################
    return Phi 

#--------------------------
def compute_yhat(Phi, w):
    '''
        Computes the linear logit value of all data instances. z = <w, x>
        Here <w, x> represents the dot product of the two vectors.
        Input:
            Phi: the feature matrix of all data instance, a float numpy matrix of shape n by p. 
            w: the weights parameter of the linear model, a float numpy matrix of shape p by 1. 
        Output:
            yhat: the logit value of all instances, a float numpy matrix of shape n by 1
    '''
    #########################################

    yhat = np.dot(Phi, w)
    
    #########################################
    return yhat

    #--------------------------
def compute_L(yhat,y):
    '''
        Computes the loss function: mean squared error. This function devides the original mean squared error by 2 for making gradient computation simple.  
        Input:
            yhat: the predicted sample labels, a numpy vector of shape n by 1.
            y:  the sample labels, a numpy vector of shape n by 1.
        Output:
            L: the loss value of linear regression, a float scalar.
    '''
    #########################################

    difference = yhat - y
    sum = 0
    for val in difference:
        sum += (val**2)
    L = sum/(2*(np.size(y)))

    #########################################
    return L 

def compute_dL_dw(y, yhat, Phi):
    '''
        Computes the gradients of the loss function L with respect to (w.r.t.) the weights w. 
        Input:
            Phi: the feature matrix of all data instances, a float numpy matrix of shape n by p. 
               Here p is the number of features/dimensions.
            y: the sample labels, a numpy vector of shape n by 1.
            yhat: the predicted sample labels, a numpy vector of shape n by 1.
        Output:
            dL_dw: the gradients of the loss function L with respect to the weights w, a numpy float matrix of shape p by 1. 

    '''
    #########################################

    dL_dw = (np.transpose(Phi) * (yhat - y))/np.size(y)

    #########################################
    return dL_dw

#--------------------------
def update_w(w, dL_dw, alpha = 0.001):
    '''
       Given the instances in the training data, updates the weights w using gradient descent.
        Input:
            w: the current value of the weight vector, a numpy float matrix of shape p by 1.
            dL_dw: the gradient of the loss function w.r.t. the weight vector, a numpy float matrix of shape p by 1. 
            alpha: the step-size parameter of gradient descent, a float scalar.
        Output:
            w: the updated weight vector, a numpy float matrix of shape p by 1.
    '''
    
    #########################################

    w = w - (alpha * dL_dw)

    #########################################
    return w

#--------------------------
def train(X, Y, alpha=0.001, n_epoch=100):
    '''
       Given a training dataset, trains the linear regression model by iteratively updating the weights w using the gradient descent
        repeats n_epoch passes over all the training instances.
        Input:
            X: the feature matrix of training instances, a float numpy matrix of shape (n by p). Here n is the number of data instance in the training set, p is the number of features/dimensions.
            Y: the labels of training instance, a numpy integer matrix of shape n by 1. 
            alpha: the step-size parameter of gradient descent, a float scalar.
            n_epoch: the number of passes to go through the training set, an integer scalar.
        Output:
            w: the weight vector trained on the training set, a numpy float matrix of shape p by 1. 
    '''

    # initialize weights as 0
    w = np.mat(np.zeros(X.shape[1])).T

    for _ in range(n_epoch):

    #########################################
    # Back propagation: compute local gradients 

        yhat = compute_yhat(X, w)
        dL_dw = compute_dL_dw(Y, yhat, X) 

    # update the parameters w
        w = update_w(w, dL_dw, alpha) 

    #########################################
    return w


