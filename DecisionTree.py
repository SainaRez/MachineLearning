import math
import numpy as np
from collections import Counter

#-------------------------------------------------------------------------

'''
    Decision Tree (with Discrete Attributes)
    This file uses python 3
'''
        
#-----------------------------------------------
class Node:
    '''
        Decision Tree Node (with discrete attributes)
        Inputs: 
            X: the data instances in the node, a numpy matrix of shape p by n.
               Each element can be int/float/string.
               Here n is the number data instances in the node, p is the number of attributes.
            Y: the class labels, a numpy array of length n.
               Each element can be int/float/string.
            i: the index of the attribute being tested in the node, an integer scalar 
            C: the dictionary of attribute values and children nodes. 
               Each (key, value) pair represents an attribute value and its corresponding child node.
            isleaf: whether or not this node is a leaf node, a boolean scalar
            p: the label to be predicted on the node (i.e., most common label in the node).
    '''
    def __init__(self,X,Y, i=None,C=None, isleaf= False,p=None):
        self.X = X
        self.Y = Y
        self.i = i
        self.C= C
        self.isleaf = isleaf
        self.p = p

#-----------------------------------------------
class Tree(object):
    '''
        Decision Tree (with discrete attributes). 
        We are using ID3(Iterative Dichotomiser 3) algorithm. So this decision tree is also called ID3.
    '''
    #--------------------------
    @staticmethod
    def entropy(Y):
        '''
            Computes the entropy of a list of values.
            Input:
                Y: a list of values, a numpy array of int/float/string values.
            Output:
                e: the entropy of the list of values, a float scalar
        '''
        #########################################

        e = 0
        
        data_count = Counter()
        for val in Y:
            data_count[val] += 1
        d_sum = sum(data_count.values())
        for key in data_count:
            prob = data_count[key]/d_sum
            e += -prob * np.log2(prob)
        #print ("e", e)

        #########################################
        return e    

    #--------------------------
    @staticmethod
    def conditional_entropy(Y,X):
        '''
            Computes the conditional entropy of y given x. The conditional entropy H(Y|X) means average entropy of children nodes, given attribute X. Refer to https://en.wikipedia.org/wiki/Information_gain_in_decision_trees
            Input:
                X: a list of values , a numpy array of int/float/string values. The size of the array means the number of instances/examples. X contains each instance's attribute value. 
                Y: a list of values, a numpy array of int/float/string values. Y contains each instance's corresponding target label. For example X[0]'s target label is Y[0]
            Output:
                ce: the conditional entropy of y given x, a float scalar
        '''
        #########################################

        # H(X|Y) = H(X,Y) - H(X)

        ce = 0
        xc = Counter(X)
        x_sum = sum(xc.values())
        for key in xc:
            prob = xc[key] / x_sum
            a = []
            for i,val in enumerate(X):
                if key == val:
                    a.append(Y[i])
            b = np.array(a)
            ce += prob * Tree.entropy(b)    
        #print ("ce", ce)

        #########################################
        return ce 
    
    
    #--------------------------
    @staticmethod
    def information_gain(Y,X):
        '''
            Computes the information gain of y after spliting over attribute x
            InfoGain(Y,X) = H(Y) - H(Y|X) 
            Input:
                X: a list of values, a numpy array of int/float/string values.
                Y: a list of values, a numpy array of int/float/string values.
            Output:
                g: the information gain of y after spliting over x, a float scalar
        '''
        #########################################

        g = Tree.entropy(Y) - Tree.conditional_entropy(Y,X)
        #print ("g", g)

 
        #########################################
        return g


    #--------------------------
    @staticmethod
    def best_attribute(X,Y):
        '''
            Finds the best attribute to split the node. 
            Here we use information gain to evaluate the attributes. 
            If there is a tie in the best attributes, selects the one with the smallest index.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
            Output:
                i: the index of the attribute to split, an integer scalar
        '''
        #########################################

        X_arr = np.array(X)
        i = -1
        max_g = 0
        for idx,row in enumerate(X_arr):
            gain = Tree.information_gain(Y,row)
            if gain > max_g:
                max_g = gain
                i = idx
   
        #########################################
        return i

        
    #--------------------------
    @staticmethod
    def split(X,Y,i):
        '''
            Splits the node based upon the i-th attribute.
            (1) splits the matrix X based upon the values in i-th attribute
            (2) splits the labels Y based upon the values in i-th attribute
            (3) builds children nodes by assigning a submatrix of X and Y to each node
            (4) builds the dictionary to combine each  value in the i-th attribute with a child node.
    
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
                i: the index of the attribute to split, an integer scalar
            Output:
                C: the dictionary of attribute values and children nodes. 
                   Each (key, value) pair represents an attribute value and its corresponding child node.
        '''
        #########################################

        X_arr = np.array(X)
        counter = Counter(X_arr[i])
        C = dict()
        for key in counter:
            child_list = []
            y_list = []
            for idx, val in enumerate(X_arr[i]):
                if val == key:
                    a = []
                    for row in X_arr:
                        a.append(row[idx])
                    child_list.append(a)
                    y_list.append(Y[idx])
            child_matrix = np.matrix(child_list)
            t_child_matrix = child_matrix.transpose()
            y_arr = np.array(y_list)
            n = Node(t_child_matrix, y_arr)
            C[key] = n

        #########################################
        return C

    #--------------------------
    @staticmethod
    def stop1(Y):
        '''
            Tests condition 1 (stop splitting): whether or not all the instances have the same label. 
    
            Input:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
            Output:
                s: whether or not Conidtion 1 holds, a boolean scalar. 
                True if all labels are the same. Otherwise, false.
        '''
        #########################################

        counter = Counter(Y)
        if len(counter) == 1:
            s = True
        else:
            s = False


        
        #########################################
        return s
    
    #--------------------------
    @staticmethod
    def stop2(X):
        '''
            Tests condition 2 (stop splitting): whether or not all the instances have the same attribute values. 
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
            Output:
                s: whether or not Conidtion 2 holds, a boolean scalar. 
        '''
        #########################################

        X_arr = np.array(X)
        s = True
        for row in X_arr:
            counter = Counter(row)
            if len(counter) == 1:
                s = s and True
            else:
                s = s and False
 
        #########################################
        return s
    
            
    #--------------------------
    @staticmethod
    def most_common(Y):
        '''
            Gets the most-common label from the list Y. 
            Input:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node.
            Output:
                y: the most common label, a scalar, can be int/float/string.
        '''
        #########################################
    
        counter = Counter(Y)
        print (counter)
        max = 0
        for key in counter:
            if counter[key] >= max:
                max = counter[key]
                y = key

        #########################################
        return y
    
    
    #--------------------------
    @staticmethod
    def build_tree(t):
        '''
            Recursively builds tree nodes.
            Input:
                t: a node of the decision tree, without the subtree built.
                t.X: the feature matrix, a numpy float matrix of shape n by p.
                   Each element can be int/float/string.
                    Here n is the number data instances, p is the number of attributes.
                t.Y: the class labels of the instances in the node, a numpy array of length n.
                t.C: the dictionary of attribute values and children nodes. 
                   Each (key, value) pair represents an attribute value and its corresponding child node.
        '''
        #########################################

        #best_att = Tree.best_attribute(t.X, t.Y)    
        #C = Tree.split(t.X, t.Y, best_att)
        t.p = Tree.most_common(t.Y)
        if (Tree.stop2(t.X) or Tree.stop1(t.Y)):
            t.isleaf = True
        else:
            t.i = Tree.best_attribute(t.X, t.Y)
            t.C = Tree.split(t.X, t.Y, t.i)
            for key in t.C:
                #best_attr = Tree.best_attribute(t.C[key].X, t.C[key].Y)
                #Tree.split(t.C[key].X, t.C[key].Y, best_attr)
                #if len(np.array(np.transpose(t.C[key].X))) > 1:
                #if t.C[key].X.size() > 1:
                Tree.build_tree(t.C[key])         

        #########################################
    
    
    #--------------------------
    @staticmethod
    def train(X, Y):
        '''
            Given a training set, trains a decision tree. 
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the training set, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
            Output:
                t: the root of the tree.
        '''
        #########################################
     
        t = Node(X,Y)
        Tree.build_tree(t)

        #########################################
        return t
    
    
    #--------------------------
    @staticmethod
    def inference(t,x):
        '''
            Given a decision tree and one data instance, infers the label of the instance recursively. 
            Input:
                t: the root of the tree.
                x: the attribute vector, a numpy vectr of shape p.
                   Each attribute value can be int/float/string.
            Output:
                y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################

        y_list = []
        count = 0
        i = 0
        n = t
        flag = False
        while True:
            if n.isleaf == True:
                y = n.p
                return y
            else:
                if(i == 0 and flag):
                    y = n.p
                    return y
                if(i == 0):
                    flag = True

                if x[i] in n.C:
                    flag = False
                    count += 1
                    n = n.C[x[i]]
                    if count == len(x):
                        y = n.p
                        return y
            i = (i+1)%x.size
 
        #########################################
        return y
    
    #--------------------------
    @staticmethod
    def predict(t,X):
        '''
            Given a decision tree and a dataset, predicts the labels on the dataset. 
            Input:
                t: the root of the tree.
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the dataset, p is the number of attributes.
            Output:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################

        y_list = []
        X_t = np.transpose(X)
        X_arr = np.array(X_t)
        for row in X_arr:
            y_list.append(Tree.inference(t,np.flip(row)))
        Y = np.array(y_list)

        #########################################
        return Y


    #--------------------------
    @staticmethod
    def load_dataset(filename = 'data1.csv'):
        '''
            Loads dataset 1 from the CSV file: 'data1.csv'. 
            The first row of the file is the header (including the names of the attributes)
            In the remaining rows, each row represents one data instance.
            The first column of the file is the label to be predicted.
            In remaining columns, each column represents an attribute.
            Input:
                filename: the filename of the dataset, a string.
            Output:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the dataset, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################

        y_list = []
        data = np.genfromtxt(filename, delimiter=',', dtype=str)
        #print (data)
        data = data[1:]
        for row in data:
            y_list.append(row[0])
        Y = np.array(y_list)
        t_data = np.transpose(data)
        X = t_data[1:,:]
        print (Y)
        print (X)

        #########################################
        return X,Y

#X, Y = Tree.load_dataset('credit_edited.txt')
#t = Tree.train(X,Y) 
#A = 0
