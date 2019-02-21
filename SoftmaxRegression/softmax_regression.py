import numpy as np
import math

#-------------------------------------------------------------------------
'''
    Softmax Regression 
    This file uses python 3
    This is the implementaiton of softmax regression for multi-class classification problems.
    The main goal is to extend the logistic regression method to solving multi-class classification problems.
    I use multi-class cross entropy as the loss function and stochastic gradient descent to train the model parameters.

    Notations:
            ---------- input data ----------------------
            p: the number of input features, an integer scalar.
            c: the number of classes in the classification task, an integer scalar.
            x: the feature vector of a data instance, a float numpy matrix of shape p by 1. 
            y: the label of a training instance, an integer scalar value. The values can be 0,1,2, ..., or (c-1).

            ---------- model parameters ----------------------
            W: the weight matrix of softmax regression, a float numpy matrix of shape (c by p). 
            b: the bias values of softmax regression, a float numpy matrix of shape c by 1.
            ---------- values ----------------------
            z: the linear logits, a float numpy matrix of shape c by 1.
            a: the softmax activations, a float numpy matrix of shape c by 1. 
            L: the multi-class cross entropy loss, a float scalar.

            ---------- partial gradients ----------------------
            dL_da: the partial gradients of the loss function L w.r.t. the activations a, a float numpy matrix of shape c by 1. 
                   The i-th element dL_da[i] represents the partial gradient of the loss function L w.r.t. the i-th activation a[i]:  d_L / d_a[i].
            da_dz: the partial gradient of the activations a w.r.t. the logits z, a float numpy matrix of shape (c by c). 
                   The (i,j)-th element of da_dz represents the partial gradient ( d_a[i]  / d_z[j] )
            dz_dW: the partial gradient of logits z w.r.t. the weight matrix W, a numpy float matrix of shape (c by p). 
                   The (i,j)-th element of dz_dW represents the partial gradient of the i-th logit (z[i]) w.r.t. the weight W[i,j]:   d_z[i] / d_W[i,j]
            dz_db: the partial gradient of the logits z w.r.t. the biases b, a float matrix of shape c by 1. 
                   Each element dz_db[i] represents the partial gradient of the i-th logit z[i] w.r.t. the i-th bias b[i]:  d_z[i] / d_b[i]

            ---------- partial gradients of parameters ------------------
            dL_dW: the partial gradients of the loss function L w.r.t. the weight matrix W, a numpy float matrix of shape (c by p). 
                   The i,j-th element dL_dW[i,j] represents the partial gradient of the loss function L w.r.t. the i,j-th weight W[i,j]:  d_L / d_W[i,j]
            dL_db: the partial gradient of the loss function L w.r.t. the biases b, a float numpy matrix of shape c by 1.
                   The i-th element dL_db[i] represents the partial gradient of the loss function w.r.t. the i-th bias:  d_L / d_b[i]

            ---------- training ----------------------
            alpha: the step-size parameter of gradient descent, a float scalar.
            n_epoch: the number of passes to go through the training dataset in order to train the model, an integer scalar.
'''

#-----------------------------------------------------------------
# Forward Pass 
#-----------------------------------------------------------------

#-----------------------------------------------------------------
def compute_z(x,W,b):
    '''
        Computes the linear logit values of a data instance. z =  W x + b
        Input:
            x: the feature vector of a data instance, a float numpy matrix of shape p by 1. Here p is the number of features/dimensions.
            W: the weight matrix of softmax regression, a float numpy matrix of shape (c by p). Here c is the number of classes.
            b: the bias values of softmax regression, a float numpy vector of shape c by 1.
        Output:
            z: the linear logits, a float numpy vector of shape c by 1. 
    '''
    #########################################

    z = np.dot(W, x) + b
    
    #########################################
    return z 

#-----------------------------------------------------------------
def compute_a(z):
    '''
        Computes the softmax activations.
        Input:
            z: the logit values of softmax regression, a float numpy vector of shape c by 1. Here c is the number of classes
        Output:
            a: the softmax activations, a float numpy vector of shape c by 1. 
    '''
    #########################################
    activ = np.zeros(len(z))
    max_val = max(z)
    counter = 0
    for x in z:
        if x == max_val:
            counter += 1
    for index,x in enumerate(z):
        try:
            activ[index] = np.exp(x) / float(sum(np.exp(z)))
        except:
            if x == max_val:
                activ[index] = 1 / counter
            else:
                pass

    a = np.matrix(activ).T
    #########################################
    return a

#-----------------------------------------------------------------
def compute_L(a,y):
    '''
        Computes multi-class cross entropy, which is the loss function of softmax regression. 
        Input:
            a: the activations of a training instance, a float numpy vector of shape c by 1. Here c is the number of classes. 
            y: the label of a training instance, an integer scalar value. The values can be 0,1,2, ..., or (c-1).
        Output:
            L: the loss value of softmax regression, a float scalar.
    '''
    #########################################
    L = 0
    if a[y] == 0:
        L = 10 ** 6
    else:
        L = - (math.log(a[y]))
    
    #########################################
    return L 

#-----------------------------------------------------------------
def forward(x,y,W,b):
    '''
       Forward pass: given an instance in the training data, computes the logits z, activations a and multi-class cross entropy L on the instance.
        Input:
            x: the feature vector of a training instance, a float numpy vector of shape p by 1. Here p is the number of features/dimensions.
            y: the label of a training instance, an integer scalar value. The values can be 0 or 1.
            W: the weight matrix of softmax regression, a float numpy matrix of shape (c by p). Here c is the number of classes.
            b: the bias values of softmax regression, a float numpy vector of shape c by 1.
        Output:
            z: the logit values of softmax regression, a float numpy vector of shape c by 1. Here c is the number of classes
            a: the activations of a training instance, a float numpy vector of shape c by 1. Here c is the number of classes. 
            L: the loss value of softmax regression, a float scalar.
    '''
    #########################################
    
    z = compute_z(x,W,b)
    a = compute_a(z)
    L = compute_L(a,y)

    #########################################
    return z, a, L 

#-----------------------------------------------------------------
# Compute Local Gradients
#-----------------------------------------------------------------

#-----------------------------------------------------------------
def compute_dL_da(a, y):
    '''
        Computes local gradient of the multi-class cross-entropy loss function w.r.t. the activations.
        Input:
            a: the activations of a training instance, a float numpy vector of shape c by 1. Here c is the number of classes. 
            y: the label of a training instance, an integer scalar value. The values can be 0,1,2, ..., or (c-1).
        Output:
            dL_da: the local gradients of the loss function w.r.t. the activations, a float numpy vector of shape c by 1. 
                   The i-th element dL_da[i] represents the partial gradient of the loss function w.r.t. the i-th activation a[i]:  d_L / d_a[i].
    '''
    #########################################
    dL_da = np.zeros(len(a))
    if a[y] == 0.:
        dL_da[y] = -10 ** 6
    else:
        dL_da[y] = (a[y]-1)/(a[y]*(1-a[y]))
    
    dL_da = np.matrix(dL_da).T
    #########################################
    return dL_da 

#-----------------------------------------------------------------
def compute_da_dz(a):
    '''
        Computes local gradient of the softmax activations a w.r.t. the logits z.
        Input:
            a: the activation values of softmax function, a numpy float vector of shape c by 1. Here c is the number of classes.
        Output:
            da_dz: the local gradient of the activations a w.r.t. the logits z, a float numpy matrix of shape (c by c). 
                   The (i,j)-th element of da_dz represents the partial gradient ( d_a[i]  / d_z[j] )
    '''
    #########################################
    c =  len(a)
    da_dz = np.zeros((c, c))
    for i in range(a.shape[0]):
        for j in range(a.shape[0]):
            if i == j:
                da_dz[i][j] = a[j] * (1 - a[j])
            else:
                da_dz[i][j] = -a[i] * a[j]

    da_dz = np.mat(da_dz)
    #########################################
    return da_dz 

#-----------------------------------------------------------------
def compute_dz_dW(x,c):
    '''
        Computes local gradient of the logits function z w.r.t. the weights W.
        Input:
            x: the feature vector of a data instance, a float numpy vector of shape p by 1. Here p is the number of features/dimensions.
            c: the number of classes, an integer. 
        Output:
            dz_dW: the partial gradient of logits z w.r.t. the weight matrix, a numpy float matrix of shape (c by p). 
                   The (i,j)-th element of dz_dW represents the partial gradient of the i-th logit (z[i]) w.r.t. the weight W[i,j]:   d_z[i] / d_W[i,j]
    '''
    #########################################
    
    dz_dW = np.empty((c,len(x)))
    i = 0
    for i in range(c):
        for j in range(x.shape[0]):
            dz_dW[i][j] = x[j]

    dz_dW = np.mat(dz_dW)

    #########################################
    return dz_dW

#-----------------------------------------------------------------
def compute_dz_db(c):
    '''
        Computes local gradient of the logits function z w.r.t. the biases b. 
        Input:
            c: the number of classes, an integer. 
        Output:
            dz_db: the partial gradient of the logits z w.r.t. the biases b, a float vector of shape c by 1. 
                   Each element dz_db[i] represents the partial gradient of the i-th logit z[i] w.r.t. the i-th bias b[i]:  d_z[i] / d_b[i]
    '''
    #########################################

    dz_db = np.ones((c,1))
    dz_db = np.mat(dz_db)
    
    #########################################
    return dz_db

#-----------------------------------------------------------------
# Back Propagation 
#-----------------------------------------------------------------

#-----------------------------------------------------------------
def backward(x,y,a):
    '''
       Back Propagation: given an instance in the training data, computes the local gradients of the logits z, activations a, weights W and biases b on the instance. 
        Input:
            x: the feature vector of a training instance, a float numpy vector of shape p by 1. Here p is the number of features/dimensions.
            y: the label of a training instance, an integer scalar value. The values can be 0,1,2, ..., or (c-1).
            a: the activations of a training instance, a float numpy vector of shape c by 1. Here c is the number of classes. 
        Output:
            dL_da: the local gradients of the loss function w.r.t. the activations, a float numpy vector of shape c by 1. 
                   The i-th element dL_da[i] represents the partial gradient of the loss function L w.r.t. the i-th activation a[i]:  d_L / d_a[i].
            da_dz: the local gradient of the activation w.r.t. the logits z, a float numpy matrix of shape (c by c). 
                   The (i,j)-th element of da_dz represents the partial gradient ( d_a[i]  / d_z[j] )
            dz_dW: the partial gradient of logits z w.r.t. the weight matrix W, a numpy float matrix of shape (c by p). 
                   The i,j -th element of dz_dW represents the partial gradient of the i-th logit (z[i]) w.r.t. the weight W[i,j]:   d_z[i] / d_W[i,j]
            dz_db: the partial gradient of the logits z w.r.t. the biases b, a float vector of shape c by 1. 
                   Each element dz_db[i] represents the partial gradient of the i-th logit z[i] w.r.t. the i-th bias:  d_z[i] / d_b[i]
    '''
    #########################################

    dL_da = compute_dL_da(a, y)
    da_dz = compute_da_dz(a)
    dz_dW = compute_dz_dW(x, a.shape[0])
    dz_db = compute_dz_db(a.shape[0])
    
    #########################################
    return dL_da, da_dz, dz_dW, dz_db

#-----------------------------------------------------------------
def compute_dL_dz(dL_da,da_dz):
    '''
       Given the local gradients, computes the gradient of the loss function L w.r.t. the logits z using chain rule.
        Input:
            dL_da: the local gradients of the loss function w.r.t. the activations, a float numpy vector of shape c by 1. 
                   The i-th element dL_da[i] represents the partial gradient of the loss function L w.r.t. the i-th activation a[i]:  d_L / d_a[i].
            da_dz: the local gradient of the activation w.r.t. the logits z, a float numpy matrix of shape (c by c). 
                   The (i,j)-th element of da_dz represents the partial gradient ( d_a[i]  / d_z[j] )
        Output:
            dL_dz: the gradient of the loss function L w.r.t. the logits z, a numpy float vector of shape c by 1. 
                   The i-th element dL_dz[i] represents the partial gradient of the loss function L w.r.t. the i-th logit z[i]:  d_L / d_z[i].
    '''
    #########################################
    
    dL_dz = da_dz * dL_da
    
    #########################################
    return dL_dz

#-----------------------------------------------------------------
def compute_dL_dW(dL_dz,dz_dW):
    '''
       Given the local gradients, computes the gradient of the loss function L w.r.t. the weights W using chain rule. 
        Input:
            dL_dz: the gradient of the loss function L w.r.t. the logits z, a numpy float vector of shape c by 1. 
                   The i-th element dL_dz[i] represents the partial gradient of the loss function L w.r.t. the i-th logit z[i]:  d_L / d_z[i].
            dz_dW: the partial gradient of logits z w.r.t. the weight matrix W, a numpy float matrix of shape (c by p). 
                   The i,j -th element of dz_dW represents the partial gradient of the i-th logit (z[i]) w.r.t. the weight W[i,j]:   d_z[i] / d_W[i,j]
        Output:
            dL_dW: the global gradient of the loss function w.r.t. the weight matrix, a numpy float matrix of shape (c by p). 
                   Here c is the number of classes.
                   The i,j-th element dL_dW[i,j] represents the partial gradient of the loss function L w.r.t. the i,j-th weight W[i,j]:  d_L / d_W[i,j]
    '''
    #########################################

    dL_dW = np.multiply(dL_dz, dz_dW)
    
    #########################################
    return dL_dW

#-----------------------------------------------------------------
def compute_dL_db(dL_dz,dz_db):
    '''
       Given the local gradients, computes the gradient of the loss function L w.r.t. the biases b using chain rule.
        Input:
            dL_dz: the gradient of the loss function L w.r.t. the logits z, a numpy float vector of shape c by 1. 
                   The i-th element dL_dz[i] represents the partial gradient of the loss function L w.r.t. the i-th logit z[i]:  d_L / d_z[i].
            dz_db: the local gradient of the logits z w.r.t. the biases b, a float numpy vector of shape c by 1. 
                   The i-th element dz_db[i] represents the partial gradient ( d_z[i]  / d_b[i] )
        Output:
            dL_db: the global gradient of the loss function L w.r.t. the biases b, a float numpy vector of shape c by 1.
                   The i-th element dL_db[i] represents the partial gradient of the loss function w.r.t. the i-th bias:  d_L / d_b[i]
    '''
    #########################################

    dL_db = np.multiply(dL_dz, dz_db)    

    #########################################
    return dL_db 

#-----------------------------------------------------------------
# gradient descent 
#-----------------------------------------------------------------

#--------------------------
def update_W(W, dL_dW, alpha=0.001):
    '''
       Updates the weights W using gradient descent.
        Input:
            W: the current weight matrix, a float numpy matrix of shape (c by p). Here c is the number of classes.
            alpha: the step-size parameter of gradient descent, a float scalar.
            dL_dW: the global gradient of the loss function w.r.t. the weight matrix, a numpy float matrix of shape (c by p). 
                   The i,j-th element dL_dW[i,j] represents the partial gradient of the loss function L w.r.t. the i,j-th weight W[i,j]:  d_L / d_W[i,j]
        Output:
            W: the updated weight matrix, a numpy float matrix of shape (c by p).
    '''
    #########################################

    for i in range(W.shape[0]):
        W[i] = W[i] - (alpha * dL_dW[i])

    #########################################
    return W

#--------------------------
def update_b(b, dL_db, alpha=0.001):
    '''
       Updates the biases b using gradient descent.
        Input:
            b: the current bias values, a float numpy vector of shape c by 1.
            dL_db: the global gradient of the loss function L w.r.t. the biases b, a float numpy vector of shape c by 1.
                   The i-th element dL_db[i] represents the partial gradient of the loss function w.r.t. the i-th bias:  d_L / d_b[i]
            alpha: the step-size parameter of gradient descent, a float scalar.
        Output:
            b: the updated of bias vector, a float numpy vector of shape c by 1. 
    '''
    
    #########################################

    for i in range(b.shape[0]):
        b[i] = b[i] - (alpha * dL_db[i])    
    
    #########################################
    return b 

#--------------------------
# train
def train(X, Y, alpha=0.01, n_epoch=100):
    '''
       Given a training dataset, trains the softmax regression model by iteratively updating the weights W and biases b using the gradients computed over each data instance. 
        Input:
            X: the feature matrix of training instances, a float numpy matrix of shape (n by p). Here n is the number of data instance in the training set, p is the number of features/dimensions.
            Y: the labels of training instance, a numpy integer numpy array of length n. The values can be 0 or 1.
            alpha: the step-size parameter of gradient ascent, a float scalar.
            n_epoch: the number of passes to go through the training set, an integer scalar.
        Output:
            W: the weight matrix trained on the training set, a numpy float matrix of shape (c by p).
            b: the bias, a float numpy vector of shape c by 1. 
    '''
    # number of features
    p = X.shape[1]
    # number of classes 
    c = max(Y) + 1

    # randomly initializes W and b
    W = np.asmatrix(np.random.rand(c,p))
    b= np.asmatrix(np.random.rand(c,1))

    for _ in range(n_epoch):
        # goes through each training instance
        for x,y in zip(X,Y):
            x = x.T # convert to column vector
            #########################################
            ## INSERT YOUR CODE HERE
            # Forward pass: computes the logits, softmax and cross_entropy 
            
            z, a, L = forward(x,y,W,b)
            
            # Back Propagation: compute local gradients of cross_entropy, softmax and logits
    
            dL_da, da_dz, dz_dW, dz_db= backward(x,y,a)

            # computes the global gradients using chain rule 
    
            dL_dz = compute_dL_dz(dL_da,da_dz)
            dL_dW = compute_dL_dW(dL_dz, dz_dW)
            dL_db = compute_dL_db(dL_dz, dz_db)

            # updates the paramters using gradient descent

            W = update_W(W, dL_dW, alpha)
            b = update_b(b, dL_db, alpha)

            #########################################
    return W, b

#--------------------------
def predict(Xtest, W, b):
    '''
       Predicts the labels of the instances in a test dataset using softmax regression.
        Input:
            Xtest: the feature matrix of testing instances, a float numpy matrix of shape (n_test by p). Here n_test is the number of data instance in the test set, p is the number of features/dimensions.
            W: the weight vector of the logistic model, a float numpy matrix of shape (c by p).
            b: the bias values of the softmax regression model, a float vector of shape c by 1.
        Output:
            Y: the predicted labels of test data, an integer numpy array of length ntest Each element can be 0, 1, ..., or (c-1) 
            P: the predicted probabilities of test data to be in different classes, a float numpy matrix of shape (ntest,c). Each (i,j) element is between 0 and 1, indicating the probability of the i-th instance having the j-th class label. 
    '''
    n = Xtest.shape[0]
    c = W.shape[0]
    Y = np.zeros(n) # initialize as all zeros
    P = np.asmatrix(np.zeros((n,c)))  
    for i, x in enumerate(Xtest):
        x = x.T # converts to column vector
        #########################################

        z = compute_z(x,W,b)
        a = compute_a(z)
        P[i] = a.T
        Y[i] = np.argmax(a)

        #########################################
    return Y, P 

#-----------------------------------------------------------------
# gradient checking 
#-----------------------------------------------------------------

#-----------------------------------------------------------------
def check_da_dz(z, delta=1e-7):
    '''
        Computes local gradient of the softmax function using gradient checking.
        Input:
            z: the logit values of softmax regression, a float numpy vector of shape c by 1. Here c is the number of classes
            delta: a small number for gradient check, a float scalar.
        Output:
            da_dz: the approximated local gradient of the activations w.r.t. the logits, a float numpy matrix of shape (c by c). 
                   The (i,j)-th element represents the partial gradient ( d a[i]  / d z[j] )
    '''
    c = z.shape[0] # number of classes
    da_dz = np.asmatrix(np.zeros((c,c)))
    for i in range(c):
        for j in range(c):
            d = np.asmatrix(np.zeros((c,1)))
            d[j] = delta
            da_dz[i,j] = (compute_a(z+d)[i,0] - compute_a(z)[i,0]) / delta
    return da_dz 

#-----------------------------------------------------------------
def check_dL_da(a, y, delta=1e-7):
    '''
        Computes local gradient of the multi-class cross-entropy function w.r.t. the activations using gradient checking.
        Input:
            a: the activations of a training instance, a float numpy vector of shape c by 1. Here c is the number of classes. 
            y: the label of a training instance, an integer scalar value. The values can be 0,1,2, ..., or (c-1).
            delta: a small number for gradient check, a float scalar.
        Output:
            dL_da: the approximated local gradients of the loss function w.r.t. the activations, a float numpy vector of shape c by 1.
    '''
    c = a.shape[0] # number of classes
    dL_da = np.asmatrix(np.zeros((c,1))) # initialize the vector as all zeros
    for i in range(c):
        d = np.asmatrix(np.zeros((c,1)))
        d[i] = delta
        dL_da[i] = ( compute_L(a+d,y) 
                        - compute_L(a,y)) / delta
    return dL_da 

#--------------------------
def check_dz_dW(x, W, b, delta=1e-7):
    '''
        computes the local gradient of the logit function using gradient check.
        Input:
            x: the feature vector of a data instance, a float numpy vector of shape p by 1. Here p is the number of features/dimensions.
            W: the weight matrix of softmax regression, a float numpy matrix of shape (c by p). Here c is the number of classes.
            b: the bias values of softmax regression, a float numpy vector of shape c by 1.
            delta: a small number for gradient check, a float scalar.
        Output:
            dz_dW: the approximated local gradient of the logits w.r.t. the weight matrix computed by gradient checking, a numpy float matrix of shape (c by p). 
                   The i,j -th element of dz_dW represents the partial gradient of the i-th logit (z[i]) w.r.t. the weight W[i,j]:   d_z[i] / d_W[i,j]
    '''
    c,p = W.shape # number of classes and features
    dz_dW = np.asmatrix(np.zeros((c,p)))
    for i in range(c):
        for j in range(p):
            d = np.asmatrix(np.zeros((c,p)))
            d[i,j] = delta
            dz_dW[i,j] = (compute_z(x,W+d, b)[i,0] - compute_z(x, W, b))[i,0] / delta
    return dz_dW

#--------------------------
def check_dz_db(x, W, b, delta=1e-7):
    '''
        computes the local gradient of the logit function using gradient check.
        Input:
            x: the feature vector of a data instance, a float numpy vector of shape p by 1. Here p is the number of features/dimensions.
            W: the weight matrix of softmax regression, a float numpy matrix of shape (c by p). Here c is the number of classes.
            b: the bias values of softmax regression, a float numpy vector of shape c by 1.
            delta: a small number for gradient check, a float scalar.
        Output:
            dz_db: the approximated local gradient of the logits w.r.t. the biases using gradient check, a float vector of shape c by 1.
                   Each element dz_db[i] represents the partial gradient of the i-th logit z[i] w.r.t. the i-th bias:  d_z[i] / d_b[i]
    '''
    c,p = W.shape # number of classes and features
    dz_db = np.asmatrix(np.zeros((c,1)))
    for i in range(c):
        d = np.asmatrix(np.zeros((c,1))) 
        d[i] = delta
        dz_db[i] = (compute_z(x,W, b+d)[i,0] - compute_z(x, W, b)[i,0]) / delta
    return dz_db


#-----------------------------------------------------------------
def check_dL_dW(x,y,W,b,delta=1e-7):
    '''
       Computes the gradient of the loss function w.r.t. the weights W using gradient checking.
        Input:
            x: the feature vector of a training instance, a float numpy vector of shape p by 1. Here p is the number of features/dimensions.
            y: the label of a training instance, an integer scalar value. The values can be 0,1,2, ..., or (c-1).
            W: the weight matrix of softmax regression, a float numpy matrix of shape (c by p). Here c is the number of classes.
            b: the bias values of softmax regression, a float numpy vector of shape c by 1.
            delta: a small number for gradient check, a float scalar.
        Output:
            dL_dW: the approximated gradients of the loss function w.r.t. the weight matrix, a numpy float matrix of shape (c by p). 
    '''
    c, p = W.shape    
    dL_dW = np.asmatrix(np.zeros((c,p)))
    for i in range(c):
        for j in range(p):
            d = np.asmatrix(np.zeros((c,p)))
            d[i,j] = delta
            dL_dW[i,j] = ( forward(x,y,W+d,b)[-1] - forward(x,y,W,b)[-1] ) / delta
    return dL_dW

#-----------------------------------------------------------------
def check_dL_db(x,y,W,b,delta=1e-7):
    '''
       Computes the gradient of the loss function w.r.t. the bias b using gradient checking.
        Input:
            x: the feature vector of a training instance, a float numpy vector of shape p by 1. Here p is the number of features/dimensions.
            y: the label of a training instance, an integer scalar value. The values can be 0,1,2, ..., or (c-1).
            W: the weight matrix of softmax regression, a float numpy matrix of shape (c by p). Here c is the number of classes.
            b: the bias values of softmax regression, a float numpy vector of shape c by 1.
            delta: a small number for gradient check, a float scalar.
        Output:
            dL_db: the approxmiated gradients of the loss function w.r.t. the biases, a float vector of shape c by 1.
    '''
    c, p = W.shape    
    dL_db = np.asmatrix(np.zeros((c,1)))
    for i in range(c):
        d = np.asmatrix(np.zeros((c,1)))
        d[i] = delta
        dL_db[i] = ( forward(x,y,W,b+d)[-1] - forward(x,y,W,b)[-1] ) / delta
    return dL_db 

