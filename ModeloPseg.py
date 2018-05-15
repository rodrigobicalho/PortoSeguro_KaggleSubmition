# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 13:16:18 2017

@author: Rodrigo
"""

import pandas as pd
import numpy as np
import csv as csv

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier

#Shuffle the datasets
from sklearn.utils import shuffle

#Learning curve
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from imblearn.over_sampling import SMOTE

from sklearn.decomposition import PCA

import tensorflow as tf
import math
from tensorflow.python.framework import ops
import time

path1 = 'C:\\Users\\Rodrigo\\Documents\\Curso Data Science Big Data\\Projects\\Porto Seguro\\Files formato csv\\train.csv'
path2 = 'C:\\Users\\Rodrigo\\Documents\\Curso Data Science Big Data\\Projects\\Porto Seguro\\Files formato csv\\test.csv'

data_train = pd.read_csv(path1, sep=",")
data_test_init = pd.read_csv(path2, sep=",")
data_test = data_test_init

#Getting column names
col_labels =  list(data_train.columns.values)

# Checking NAs in each column
NAtrain = data_train[data_train == -1].count()
NATest = data_test[data_test == -1].count()

# Checking how NAs behave in the target variable
col_labels_na = list(['ps_ind_02_cat','ps_ind_04_cat','ps_ind_05_cat','ps_reg_03','ps_car_01_cat','ps_car_02_cat','ps_car_03_cat','ps_car_05_cat','ps_car_07_cat',
'ps_car_09_cat','ps_car_11','ps_car_12','ps_car_14'])
#for i in range(14):
 #   print(data_train[["target",str(col_labels_na[i])]].groupby([str(col_labels_na[i])], as_index=False).mean().sort_values(by=str(col_labels_na[i]), ascending=False))

# Solving NA for categorical values and creating dummies as appropriate for all categoricals

#For all categorical variables, one extra dummy will be created for NA data
col_labels_cat = list(['ps_ind_02_cat',  'ps_ind_04_cat', 'ps_ind_05_cat', 'ps_car_01_cat', 'ps_car_02_cat', 'ps_car_03_cat', 'ps_car_04_cat', 'ps_car_05_cat', 'ps_car_06_cat', 'ps_car_07_cat', 'ps_car_08_cat', 'ps_car_09_cat', 'ps_car_10_cat', 'ps_car_11_cat'])
    
#Adjusting data_train for all categoricals, including NA variables
for i in range(len(col_labels_cat)):
    column_names = []
    dummy_i = pd.get_dummies(data_train[str(col_labels_cat[i])])
    dummy_i = dummy_i.drop(dummy_i.columns[[1]],axis=1)
    data_train = data_train.drop(str(col_labels_cat[i]),axis=1)
    
    for j in range(dummy_i.shape[1]):
        column_names_i = str(col_labels_cat[i]+'_X'+str(j))
        column_names.append(column_names_i)
    
    dummy_i.columns = [column_names]
    data_train = pd.concat([data_train,dummy_i],axis=1)
    
#Adjusting data_test for NA variables categorical
for i in range(len(col_labels_cat)):
    column_names = []
    dummy_i = pd.get_dummies(data_test[str(col_labels_cat[i])])
    dummy_i = dummy_i.drop(dummy_i.columns[[1]],axis=1)
    data_test = data_test.drop(str(col_labels_cat[i]),axis=1)
    
    for j in range(dummy_i.shape[1]):
        column_names_i = str(col_labels_cat[i]+'_X'+str(j))
        column_names.append(column_names_i)
    
    dummy_i.columns = [column_names]
    data_test = pd.concat([data_test,dummy_i],axis=1)
    

col_labels_na_cont = list(['ps_reg_03','ps_car_14'])
print(data_train[["target",'ps_reg_03']].groupby(['ps_reg_03'], as_index=False).mean().sort_values(by=str('ps_reg_03'), ascending=True))
print(data_train[["target",'ps_car_14']].groupby(['ps_car_14'], as_index=False).mean().sort_values(by=str('ps_car_14'), ascending=True))
data_train["ps_reg_03_NA"] = data_train["ps_reg_03"].apply(lambda x:1 if x==-1 else 0)
data_train["ps_reg_03"] = data_train["ps_reg_03"].apply(lambda x:0 if x==-1 else x)
data_test["ps_reg_03_NA"] = data_test["ps_reg_03"].apply(lambda x:1 if x==-1 else 0)
data_test["ps_reg_03"] = data_test["ps_reg_03"].apply(lambda x:0 if x==-1 else x)

data_train["ps_car_14_NA"] = data_train["ps_car_14"].apply(lambda x:1 if x==-1 else 0)
data_train["ps_car_14"] = data_train["ps_car_14"].apply(lambda x:0 if x==-1 else x)
data_test["ps_car_14_NA"] = data_test["ps_car_14"].apply(lambda x:1 if x==-1 else 0)
data_test["ps_car_14"] = data_test["ps_car_14"].apply(lambda x:0 if x==-1 else x)

# Checking if there is any other NA
NAtrain = data_train[data_train == -1].count()
NATest = data_test[data_test == -1].count()

#Shuffling Training Set
#data_train = shuffle(data_train)
Train_X = data_train.iloc[0:575212,2:217]
cv_X = data_train.iloc[575213:585212,2:217]
Train_Y = data_train.iloc[0:575212,1]
cv_Y = data_train.iloc[575213:585212,1:2]
Test_X = data_train.iloc[585213:595212,2:217]
Test_Y = data_train.iloc[585213:595212,1:2]

id_test = data_test.iloc[:,0:1]
data_test = data_test.iloc[:,1:216]

# Smote undersampling majority class
#from collections import Counter 
#from sklearn.datasets import fetch_mldata 
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(ratio = 'majority') 
Train_X_res, Train_Y_res = rus.fit_sample(Train_X,Train_Y)

print("Y=1(%): ",str(np.sum(Train_Y_res)/Train_X_res.shape[0]))
## Without Undersampling
# Train_X_res = Train_X
# Train_Y_res = Train_Y

# Normalizing the data
maxi = np.max(Train_X_res,axis = 0)
avgi = np.average(Train_X_res,axis = 0)
Train_X_res = (Train_X_res - avgi)/(maxi+0.0000000000000000001)
cv_X = (cv_X - avgi)/(maxi+0.0000000000000000001)
Test_X = (Test_X - avgi)/(maxi+0.0000000000000000001)
data_test = (data_test - avgi)/(maxi+0.0000000000000000001)

# Reshuffling the data
Train_Y_res = Train_Y_res.reshape((Train_Y_res.shape[0],1))     
a = np.concatenate((Train_Y_res,Train_X_res),axis=1)
a = shuffle(a)
Train_X_res = a[:,1:217]
Train_Y_res = a[:,0:1]

# Checking the importance of the variables
pca = PCA(n_components=0.99, svd_solver='full',copy=False)
pca.fit(Train_X_res)
Variance_ratio = pca.explained_variance_ratio_
Acc_Var_ratio = np.cumsum(Variance_ratio)
#Train_X_res = pca.fit_transform(Train_X_res)
#cv_X = pca.fit_transform(cv_X)
#Test_X = pca.fit_transform(Test_X)
#data_test = pca.fit_transform(data_test)

# Generating predictions for another test set

def prob_predict(data_test,parameters):
    
    #ops.reset_default_graph()
    import itertools
    AL = 0
    AL =L_model_forward(tf.cast(data_test,tf.float32), parameters,1)
    AL = tf.sigmoid(AL)
    
    with tf.Session() as sess:
        p_testset = sess.run([AL])
        
    p_testset = list(itertools.chain(*p_testset))
    p_testset = pd.DataFrame(p_testset)
    p_testset = np.reshape(p_testset,(p_testset.shape[0],1))
    #p_testset = pd.DataFrame(p_testset)
    return p_testset

def gini(actual, pred, cmpcol = 0, sortcol = 1):
     assert( len(actual) == len(pred) )
     all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
     all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
     totalLosses = all[:,0].sum()
     giniSum = all[:,0].cumsum().sum() / totalLosses

     giniSum -= (len(actual) + 1) / 2.
     return giniSum / len(actual)
 

def gini_normalized(a, p):
     return gini(a, p) / gini(a, a)

# Neural Nets

def create_placeholders(n,yj):
    X = tf.placeholder(tf.float32, name = 'X', shape = (None,n))
    Y = tf.placeholder(tf.float32, name = 'Y', shape = (None,yj))
    
    
    return X,Y

def initialize_parameters(layer_dims):
    """
    Arguments:
        layer_dims -- a Python array(list) containing the dimensions of each layer in the NN
        
        Returns:
        parameters -- a dictionary of tensors containing W1, b1, ..... Wl,bl
                      Wl = matrix of shape (layer_dims[l],layer_dims[l-1])
                      b1 = bias of shape (layer_dims[l],1)
"""


    parameters = {}
    L = len(layer_dims)
    
    for l in range(1,L):
        parameters['W'+str(l)] = tf.get_variable("W"+ str(l), [layer_dims[l-1],layer_dims[l]], initializer = tf.contrib.layers.xavier_initializer())
        parameters['b'+str(l)] = tf.get_variable("b"+str(l),[1,layer_dims[l]],initializer = tf.zeros_initializer())
        
    return parameters

   

def random_mini_batches(X,Y,mini_batch_size=64):
    
    """Creates a list of random minibatches from (X, Y)
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (10, number of examples)
    mini_batch_size -- size of the mini-batches, integer
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    m = X.shape[0]
    mini_batches = []
    
    #Shuffle X and Y
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:]
    shuffled_Y = Y[permutation,:]
    
    # partition 
    
    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : (k + 1)* mini_batch_size,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : (k + 1)* mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
        
    #handling the end case (last batch that may not have same number os examples)
        if m % mini_batch_size != 0:
            mini_batch_X = shuffled_X[num_complete_minibatches : m,:]
            mini_batch_Y = shuffled_Y[num_complete_minibatches : m,:]
        ### END CODE HERE ###
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

    return mini_batches

def linear_forward(A,W,b):
    Z = tf.add(tf.matmul(A,W),b)
    return Z

def relu_activation(Z):
     A  = tf.nn.relu(Z)
     return A
 
def L_model_forward(X,parameters,keep_probs):
    
    A = X
    L = len(parameters)//2
    
    # Relu activation function
    for l in range(1,L):
        A_prev = A
        A_prev = tf.nn.dropout(A_prev,keep_probs)
        A = relu_activation(linear_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)]))
        #A = tf.tanh(linear_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)]))

    #last layer
    AL = linear_forward(A,parameters['W'+str(L)],parameters['b'+str(L)])
    
    return AL

def regularization(parameters,lamb=0.1):
    
    regularizer = 0
    L = len(parameters)//2
    
    for l in range(1,L+1):
        regularizer = regularizer + lamb * tf.add(tf.nn.l2_loss(parameters['W'+str(l)]),tf.nn.l2_loss(parameters['b'+str(l)]))
    
    return regularizer

def compute_cost(Z3,Y,pos_weight = 0):
    """
Computes the cost
Arguments:
Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6,
number of examples)
Y -- "true" labels vector placeholder, same shape as Z3
Returns:
cost - Tensor of the cost function
"""
    # The function tf.nn.softmax_cross_entropy_with_logits require transpose
    logits = Z3
    labels = Y

    
    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))
    #cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=labels))
    cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=logits,targets=labels,pos_weight = pos_weight))
    return cost

def model(Train_X,Train_Y,cv_X,cv_Y,Test_X,Test_Y,layer_dims,minibatch_size=32,alfa = 0.0001,lamb=0,keep_probs=1,pos_weight=0,num_epochs=1500,print_cost = True):
    """
    Train_X,Train_Y,cv_X,cv_Y,Test_X,Test_Y = Input your datasets
    layer_dims: a Python array(list) containing the dimensions of each layer in the NN
    minibatch_size: size of the minibatch. Select the number 2**X, with X=[5,10] for better performance
    alfa = Learning rate. Select a small enough for your model not to diverge, and big enough for fast optimization
    lamb = Regularization Lambda. Increase to avoid overfitting. Reduce to avoid bias
    num_epochs = Number of iterations of the model
    Print_cost = True if you want the costs to be printed as the model is learning
    """
    tic = time.time()
    ops.reset_default_graph()
    (m,n) = Train_X.shape # m = training examples n= number of features
    costs = []
    yj = Train_Y.shape[1]
    global_step = tf.Variable(0,trainable=False) #decaying alpha
    #keep_probs = tf.placeholder(tf.float32)
    
    X, Y = create_placeholders(n, yj)
    parameters = initialize_parameters(layer_dims)
    AL =L_model_forward(X, parameters,keep_probs)
    cost = compute_cost(AL, Y,pos_weight)
    regularizer = regularization(parameters,lamb)
    cost = tf.reduce_mean(regularizer)+cost
    #Decaying learning rate
    #decay_alfa = alfa
    decay_alfa = tf.train.exponential_decay(alfa,global_step,250,0.25,staircase=False)
    #current_alfa = decay_alfa
    #optimization
    optimizer = tf.train.AdamOptimizer(learning_rate = decay_alfa).minimize(cost,global_step=global_step)
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        # Run the initialization
        sess.run(init)
        
        for epoch in range(num_epochs):
            epoch_cost = 0
            num_minibatches = int(m/minibatch_size)
            minibatches = random_mini_batches(Train_X,Train_Y,minibatch_size)
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                epoch_cost = epoch_cost + minibatch_cost/num_minibatches
                
            if epoch % 10 == 0:
               # current_alfa2 = sess.run([current_alfa])
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
                #print("The current alpha is",str(current_alfa2))
            costs.append(epoch_cost)
            
        
        #Plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations')
        plt.title("Learning rate =" + str(alfa))
        plt.show()
        
        # Save parameters in the variable parameters
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")
        
        #Calculate predictions on Training Set
        #correct_prediction = tf.equal(tf.argmax(Z3,1), tf.argmax(Y,1))
        AL = tf.sigmoid(AL)
        prediction = tf.round(AL)
        
        correct_prediction = tf.equal(tf.cast(prediction, "int32"),tf.cast(Y, "int32"))
        accuracy = 100* tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print ("Train Accuracy Training set: ", accuracy.eval({X: Train_X, Y: Train_Y}))
        print ("Test Accuracy CV set: ", accuracy.eval({X: cv_X, Y: cv_Y}))
        print ("Test Accuracy Test set: ", accuracy.eval({X: Test_X, Y: Test_Y}))
        
        pred_train = sess.run([prediction], feed_dict={X:Train_X})
        pred_cv = sess.run([prediction], feed_dict={X:cv_X})
        pred_test = sess.run([prediction], feed_dict={X:Test_X})
        
        #AL_train = sess.run([AL], feed_dict={X:Train_X})
        #AL_cv = sess.run([AL], feed_dict={X:cv_X})
        #AL_test = sess.run([AL], feed_dict={X:Test_X})
        
        toc = time.time()
        print("Time: " + str(math.floor(toc-tic)))
        
        p_train = prob_predict(Train_X_res,parameters)
        p_cv = prob_predict(cv_X,parameters)
        p_test = prob_predict(Test_X,parameters)
        print("Gini Train: ",str(gini_normalized(Train_Y_res,p_train)))
        print("Gini CV: ",str(gini_normalized(cv_Y,p_cv)))
        print("Gini Test: ",str(gini_normalized(Test_Y,p_test)))
        
               
        return parameters, pred_train, pred_cv, pred_test #, AL_train, AL_cv, AL_test
    
    
    # Testing the model
#layer_dims = [Train_X.shape[1],100,90,80,70,60,50,40,30,20,10,1]
layer_dims = [Train_X.shape[1],200,150,125,100,75,50,25,1]
parameters, pred_train, pred_cv, pred_test = model(Train_X_res,Train_Y_res,cv_X,cv_Y,Test_X,Test_Y,layer_dims,minibatch_size=2**19,alfa = 0.02,lamb=0.015,keep_probs=1,pos_weight=200,num_epochs=250,print_cost = True)


#Precision and recall curve

def test_results(actual,prediction):
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

precision, recall, thresholds = precision_recall_curve(cv_Y, p_cv)
average_precision = average_precision_score(cv_Y, p_cv)
F1_list = (2*precision*recall)/(precision+recall)

thresholds = np.append(thresholds,[1])
Confusion_matrix = np.vstack((thresholds,precision,recall,F1_list))
Confusion_matrix = Confusion_matrix.T

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))

plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))



print('F1 Threshold Tradeoff: {0:0.2f}'.format(
      average_precision))
plt.step(F1_list,thresholds, color='b', alpha=0.2,
         where='post')
plt.fill_between(F1_list,thresholds, step='post', alpha=0.2,
                 color='b')



plt.xlabel('Treshold')
plt.ylabel('F1 Score')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('F1 Threshold Tradeoff')
: AP={0:0.2f}'.format(
          average_precision))

#Diagnosis
p_cv = prob_predict(cv_X,parameters)
p_test = prob_predict(Test_X,parameters)
p_train = prob_predict(Train_X_res,parameters)
gini_train = gini_normalized(Train_Y_res,p_train)
#Predicting in two parts
ops.reset_default_graph()
a1 = data_test.iloc[0:500000,:]
a2 = data_test.iloc[500000:(data_test.shape[0]+1)]
p_sub1 = prob_predict(a1,parameters)
p_sub2 = prob_predict(a2,parameters)
p_submission = np.vstack((p_sub1,p_sub2))
p_submission = pd.DataFrame(p_submission)
p_submission = np.reshape(p_submission,(p_submission.shape[0],1))
del a1
del a2
del p_sub1
del p_sub2
#Predicting all at once
p_submission = prob_predict(data_test,parameters)

# If you want Gini
gini_cv = gini_normalized(cv_Y,p_cv)
gini_test = gini_normalized(Test_Y, p_test)

#Creating final file for submission

p_submission["id"] = id_test
#p_testset = p_testset[['id',int('0')]]
p_submission = p_submission.rename(columns={int('0'):'target'})
p_submission = p_submission[['id','target']]


path3 = "C:\\Users\\Rodrigo\\Documents\\Curso Data Science Big Data\\Projects\\Porto Seguro\\Scripts\\submission.csv"

p_submission.to_csv(path_or_buf=path3, sep=',', na_rep='', float_format=None, columns=None, header=True, index=False, index_label=None, mode='w', encoding=None, compression=None, quoting=None, quotechar='"', line_terminator='\n', chunksize=None, tupleize_cols=False, date_format=None, doublequote=True, escapechar=None, decimal='.')