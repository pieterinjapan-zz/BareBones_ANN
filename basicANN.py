"""
Pieter van Wyk
Created : 2018-11-25
Updated : 2019-06-24

Testing sigmoidANN :
Testing the neural network defined in sigmoidANN.py for
a generated input/output dataset.
"""
import sigmoidANN as s_ann
import numpy as np


# Generate Dateset for Testing ANN :
#---------------------------------------------------------

# generate network input :
def gen_x_in(size,dim) :

    x_in = []
    for j in range(0,size) :

        x_in_i = []
        for i in range(0,dim) :
            n_ran = 3.0*np.random.random() - 1.5
            x_in_i.append(n_ran)

        x_in.append(x_in_i)

    return np.array(x_in)

# generate network output :
def gen_y_out(x_in) :

    y_out = []
    for i_y in range( 0 , len(x_in) ) :
        y_out_i_1 = sum(s_ann.sqls(x_in[i_y]))/(2*len(x_in[i_y]))
        y_out_i_2 = s_ann.var(x_in[i_y])/len(x_in[i_y])
        y_out.append([y_out_i_1,y_out_i_2])

    return np.array(y_out)

# generating network input / output :
sample_size = 100   # size of dataset
dim_in      = 4     # dimension of input data (number of input nodes)
data_x_in  = gen_x_in(sample_size,dim_in) # input dataset
data_y_out = gen_y_out(data_x_in)         # output dataset


# splitting the dataset into a training set and test set :
#---------------------------------------------------------

# neural network input data :
data_set_x_in = data_x_in
print( "Size of Input Dataset  :" , len(data_set_x_in ) )

# neural network output data :
data_set_y_out = data_y_out
print( "Size of Output Dataset :" , len(data_set_y_out) )
print("")

train_size = 0.8 # percentage of dataset to use for training
dataset_x_test_train = s_ann.split_data(data_set_x_in ,train_size)
dataset_y_test_train = s_ann.split_data(data_set_y_out,train_size)

# network training input / output :
data_set_x_in_train  = dataset_x_test_train[0]  # network training input
data_set_y_out_train = dataset_y_test_train[0]  # network training output

# network testing input / output :
data_set_x_in_test  = dataset_x_test_train[1]  # network testing input
data_set_y_out_test = dataset_y_test_train[1]  # network testing output

print( "Taining dataset size :" , len(data_set_x_in_train) )
print( "Testing dataset size :" , len(data_set_y_out_test) )
print("")

sample_size = len(data_set_x_in_train)
dim_in      = len(data_set_x_in_train[0])
dim_out     = len(data_set_y_out_train[0])
print("Size of Training Dataset :" , sample_size )
print("Number of Input Nodes    :" , dim_in )
print("Number of Output Nodes   :" , dim_out )
print("")

# Training Network :
#---------------------------------------------------------

# Dimensions of ANN :
layers       = 6     # layers in network
hidden_nodes = 10    # dimension of hidden nodes
num_epoch    = 1000  # number of training epochs

# training ANN :
syn_ar_new = s_ann.training_ann(data_set_x_in_train,data_set_y_out_train,layers,hidden_nodes, num_epoch)

# Validating Network :
#---------------------------------------------------------
test_size = len(data_set_y_out_test) # test dataset size
rmse_ls   = [] # list for rms_error data (initialization)

for i_tst in range( 0 , test_size ) :

    # test input :
    x_in_test  = np.array([data_set_x_in_test[i_tst]])

    # actual system output :
    y_out_test = np.array([data_set_y_out_test[i_tst]])
    print( i_tst,"Actual System Output : ", y_out_test       )

    # trained network output :
    y_out_trail = s_ann.ann(x_in_test,syn_ar_new)[-1]
    print( i_tst,"Network Trail Output : ", y_out_trail      )

    # root mean square error :
    rms_diff = s_ann.rmse(y_out_test[0],y_out_trail[0] )
    print( i_tst,"RMS Mean Difference  : ",rms_diff          )

    rmse_ls.append(rms_diff)

print("")
print("Mean RMSE    : ", s_ann.ave(rmse_ls) )
print("Maximum RMSE : ", max(rmse_ls) )

print("")
print("all is well!")

# END
