"""
Pieter van Wyk
Created : 2018-11-25
Updated : 2019-06-24

sigmoid neuron artificial neural network :
Standard neural network of sigmoid neurons that can be 
trained for data classification.
"""
import numpy as np
from functools import reduce

# basic functions :
sum      = lambda ls : reduce( ( lambda x, y: x + y ), ls ) # sum elements in list
ave      = lambda ls : sum(ls)/len(ls) # average of list
sqls     = lambda ls : list( map( lambda l : l*l, ls) ) # square elements in list
var      = lambda ls : ave( sqls (ls) ) - ave(ls)*ave(ls) # variance of list
vec_diff = lambda ls1, ls2 : list( map( lambda p : p[0]-p[1] , list( zip(ls1,ls2) ) ) ) # vector subtraction
rms      = lambda ls : np.sqrt(sum(sqls(ls))/len(ls)) # root mean square of vector
rmse     = lambda ls1,ls2 : rms( vec_diff(ls1,ls2) ) # rms error of two vectors

# sigmoid function and its derivative (activation function)
sigmoid       = lambda x : 1 / ( 1 + np.exp(-x) )
deriv_sigmoid = lambda x : sigmoid(x) * ( 1 - sigmoid(x) )

# forward propagation (x_in and syn_in must be column vectors)
forward_prop = lambda x_in,syn_in : sigmoid(np.dot(x_in,syn_in))

# backward propagation
def back_prop(x_in,syn_in,y_error,alp) :
    delta   = y_error * deriv_sigmoid(np.dot(x_in,syn_in))
    syn_out = syn_in + alp*np.dot(x_in.T,delta)
    return syn_out

# Artificial Neural Network Function for Training :
"""
- input  : x_in    - input data
           y_given - given output data
           syn_ar  - array of synapse weights (before after back propagation)
           alp     - learning rate
- output : syn_ar_new  - array of synapse weights (after back propagation)
           y_error     - network error
           y_out_trail - network output data
"""
def ann_train(x_in,y_given,syn_ar,alp) :

    # forward propagation :
    l_ar = ann(x_in,syn_ar)

    # network output and error :
    y_out_trail = l_ar[-1]
    y_error     = y_given - y_out_trail

    # back propagation (gradient descent) :
    #--------------------------------------
    syn_ar_new = []          # synapse array after back propagation (initialization)
    num_syn    = len(syn_ar) # number of synapses

    # output layer :
    l_bp_error = y_error
    l_bp_delta = l_bp_error * deriv_sigmoid(l_ar[num_syn - 0])
    syn_bp_new = back_prop(l_ar[num_syn - 1],syn_ar[num_syn - 1],l_bp_error,alp)
    syn_ar_new = [syn_bp_new] + syn_ar_new

    # hiden layers :
    for i_bp in range( 1 , num_syn ) :
        l_bp_error = np.dot(l_bp_delta,syn_ar[num_syn - i_bp].T)
        l_bp_delta = l_bp_error * deriv_sigmoid(l_ar[num_syn - i_bp])
        syn_bp_new = back_prop(l_ar[num_syn - i_bp - 1],syn_ar[num_syn - i_bp - 1],l_bp_error,alp)
        syn_ar_new = [syn_bp_new] + syn_ar_new

    return [syn_ar_new,y_error,y_out_trail]

# Artificial Neural Network Function for Output :
"""
- input  : x_in  - input data
           syn_0 - hidden layer 1 input weights
           syn_1 - hidden layer 1 output weights
- output : y_out - network output data
"""
def ann(x_in,syn_ar) :

    # input layer
    l0 = x_in # input layer

    # forward propagation :
    l_ar = [l0] # array of node values after sigmoid function (initialization)
    l_i  = l0
    for i_l in range( 1 , len(syn_ar) + 1) :
        l_i = forward_prop(l_i,syn_ar[i_l-1])
        l_ar.append(l_i)

    return l_ar

# generate input array of synapse weights
def syn_weights(layers,dim_in,dim_out,hidden_nodes) :

    syn_ar = [] # array of initial synapses (initialization)

    # input layer :
    syn_in = 2*np.random.random( ( dim_in,hidden_nodes  ) ) - 1
    syn_ar.append(syn_in)

    # hidden layers :
    for i_lay in range( 1 , layers - 1 ) :
        syn_i = 2*np.random.random( ( hidden_nodes,hidden_nodes ) ) - 1
        syn_ar.append(syn_i)

    # output layer :
    syn_out = 2*np.random.random( ( hidden_nodes,dim_out ) ) - 1
    syn_ar.append(syn_out)

    return syn_ar

# Training Function for Artificial Neural Network :
"""
-input   : data_set_x_in_train  - input data set
           data_set_y_out_train - (expected) output data set
           layers               - number of layers in ann (including input and output layers)
           hidden_nodes         - number of nodes in hidden layer
           num_epoch            - number of epochs
- output : syn_ar - array of synapse weights adjusted after training
"""
def training_ann(data_set_x_in_train,data_set_y_out_train,layers,hidden_nodes,num_epoch) :

    # initializing synapse weights :
    dim_in  = len(data_set_x_in_train[0])  # number of input nodes
    dim_out = len(data_set_y_out_train[0]) # number of output nodes
    syn_ar  = syn_weights(layers,dim_in,dim_out,hidden_nodes) # synapse weight input array

    # setup for early stop (initialization) :
    early_stop = [] # vector for early stopping
    R = 1.0         # radius of convergence

    # training sample size
    sample_size = len(data_set_x_in_train)

    # coefficient for back-propagation :
    alp = 0.005

    # iterating over neural network (epoch)
    for i_ep in range( 1 , num_epoch + 1 ) :

        # iterating neural network over training dataset :
        for i_in in range(0,sample_size) :
            y_out = np.array([data_set_y_out_train[i_in]])
            x_in  = np.array([data_set_x_in_train[i_in]])

            # ANN for current iteration :
            ann_res = ann_train(x_in,y_out,syn_ar,alp)

            # iteration results :
            syn_ar      = ann_res[0]
            y_error     = ann_res[1]
            y_out_trail = ann_res[2]

        if ( i_ep % 100 ) == 0 :
            print("Error : " + str(np.mean(abs(y_error))))

        early_stop.append(np.mean(abs(y_error)))
        if (i_ep >= 100) :
            vec_stop = []
            for i_ear in range(0,100) :
                vec_stop.append(early_stop[-i_ear - 1])
            R = var(vec_stop)

        if (R < 1.0e-14) :
            print("Exit after ",i_ep," iterations.")
            break

    # tuned synaptic weights after training :
    return syn_ar

# Splitting Dataset for Training and Testing :
"""
- input  : data_set   - input data set
           train_size - percentage of dataset to use for training
- output : dataset_y_test_train - array containing training set (component 0),
                                  and test set (component 1)
"""
def split_data(data_set,train_size) :

    # training dataset :
    data_set_train  = []
    for i_trn in range( 0 , int( len(data_set) * train_size ) ) :
        data_set_train.append(data_set[i_trn])

    # testing dataset :
    data_set_test  = []
    for i_tst in range( int( len(data_set) * train_size ), len(data_set) ) :
        data_set_test.append(data_set[i_tst])

    dataset_y_test_train = [np.array(data_set_train),np.array(data_set_test)]

    return dataset_y_test_train

# END
