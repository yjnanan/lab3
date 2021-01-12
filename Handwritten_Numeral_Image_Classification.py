import numpy as np
import random
import matplotlib.pyplot as plt
from load_data import loadLabel,loadImage

def der_activation_function(x,type):
    if type==1:
        return 1 - np.power(np.tanh(x), 2)
    elif type==2:
        return (1/(1+np.exp(-x)))*(1-1/(1+np.exp(-x)))
    else:
        x[x<=0]=0.25
        x[x>0]=1
        return x

def activation_function(x,type):
    if type==1:
        return np.tanh(x)
    elif type==2:
        return 1/(1+np.exp(-x))
    else:
        return np.where(x<=0,0.25*x,x)

def MLP_train(data,labels,hidden_nodes,epoch,test_data,test_labels):
    alpha=0.002
    size=data.shape
    w1=np.zeros((hidden_nodes,size[1]))
    for i in range(hidden_nodes):
        for j in range(size[1]):
            w1[i,j]=random.uniform(-0.4,0.4)
    w2=np.zeros((10,hidden_nodes))
    for i in range(10):
        for j in range(hidden_nodes):
            w2[i,j]=random.uniform(-0.4,0.4)
    b1=np.zeros(hidden_nodes)
    b2=np.zeros(10)
    for i in range(epoch):
        for x,y in zip(data,labels):
            u=np.dot(w1,x.T)+b1
            h=activation_function(u,3)
            v=np.dot(w2,h)+b2
            output=activation_function(v,3)

            delta2=(output-y.T)*der_activation_function(v,3)
            delta1=der_activation_function(u,3)*np.dot(w2.T,delta2)
            d_w1=np.dot(np.expand_dims(delta1,axis=1),np.expand_dims(x,axis=0))
            d_w2=np.dot(np.expand_dims(delta2,axis=1),np.expand_dims(h,axis=0))

            w1=w1-alpha*d_w1
            w2=w2-alpha*d_w2
            b1=b1-alpha*delta1
            b2=b2-alpha*delta2
    u_test=np.dot(w1,test_data.T)+np.expand_dims(b1,axis=1)
    h_test=activation_function(u_test,3)
    v_test=np.dot(w2,h_test)+np.expand_dims(b2,axis=1)
    output_test=activation_function(v_test.T,3)
    right_times=0
    for i in range(len(output_test)):
        if np.argmax(output_test[i])==np.argmax(test_labels[i]):
            right_times+=1
    accuracy=right_times/len(output_test)
    print(accuracy)


if __name__=='__main__':
    train_imgs=loadImage("train-images-idx3-ubyte")
    train_labels=loadLabel("train-labels-idx1-ubyte")
    test_imgs=loadImage("t10k-images-idx3-ubyte")
    random.seed(2)
    test_labels=loadLabel("t10k-labels-idx1-ubyte")
    # MLP_train(train_imgs,train_labels,25,15,test_imgs,test_labels)
    for nodes in range(30,60,10):
        print('activation function: PReLU')
        print(nodes,"hidden nodes:")
        MLP_train(train_imgs, train_labels, nodes, 30, test_imgs, test_labels)