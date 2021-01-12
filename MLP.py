import numpy as np
import random
import struct
import matplotlib.pyplot as plt

#load Images
def loadImage(filename):
    #open binary file
    binfile=open(filename,'rb')
    buffers=binfile.read()
    #analyse header information
    head=struct.unpack_from('>IIII',buffers,0)
    offset=struct.calcsize('>IIII')
    imgNum=head[1]
    width=head[2]
    height=head[3]
    #analyse the data set and get the data of each image
    bits=imgNum*width*height
    bitsString='>'+str(bits)+'B'
    imgs=struct.unpack_from(bitsString,buffers,offset)
    binfile.close()
    #store image into an array
    imgs=np.reshape(imgs,[imgNum,width*height])
    #data normalization
    imgs=imgs.astype(np.float)
    imgs=imgs/255.0
    return imgs

#load labels
def loadLabel(filename):
    #open binary file
    binfile=open(filename,'rb')
    buffers=binfile.read()
    #analyse header information
    head=struct.unpack_from('>II',buffers,0)
    imgNum=head[1]
    #get the data of labels
    offset=struct.calcsize('>II')
    numString='>'+str(imgNum)+"B"
    labels=struct.unpack_from(numString,buffers,offset)
    binfile.close()
    #store labels in an array
    labels=np.reshape(labels,[imgNum])
    #transform labels eg: 0 to [1,0,0,0,0,0,0,0,0,0]
    binlabels = np.zeros((imgNum,10))
    for i,s in enumerate(labels): binlabels[i,s]=1
    return binlabels

#derivative of activation function
def der_activation_function(x,type):
    x_ = x.copy()
    if type==1:
        #tanh
        return 1 - np.power(np.tanh(x_), 2)
    elif type==2:
        #sigmoid
        return (1/(1+np.exp(-x_)))*(1-1/(1+np.exp(-x_)))

#activation function
def activation_function(x,type):
    x_=x.copy()
    if type==1:
        #tanh
        return np.tanh(x_)
    elif type==2:
        #sigmoid
        return 1/(1+np.exp(-x_))

#MLP train and test
def MLP_train(data,labels,hidden_nodes,epoch,test_data,test_labels,act_f):
    #set learning rate
    alpha=0.01
    size=data.shape
    #initialize weight
    w1=np.zeros((hidden_nodes,size[1]))#m*784
    for i in range(hidden_nodes):
        for j in range(size[1]):
            w1[i,j]=random.uniform(-0.4,0.4)
    w2=np.zeros((10,hidden_nodes))#10*m
    for i in range(10):
        for j in range(hidden_nodes):
            w2[i,j]=random.uniform(-0.4,0.4)
    #initialize bias
    b1=np.zeros(hidden_nodes)#m*1
    b2=np.zeros(10)#10*1
    # print('wight:w_ji:',w1,'bias:w_j0:',b1)
    # x_label=[]
    # y_label=[]
    # train(stochastic gradient descent)
    for i in range(epoch):
        # x_label.append(i+1)
        loss=0
        for x,y in zip(data,labels):
            #forward propagration
            u=np.dot(w1,x.T)+b1
            h=activation_function(u,act_f)
            v=np.dot(w2,h)+b2
            output=activation_function(v,act_f)

            #back propagration
            delta2=(output-y.T)*der_activation_function(v,act_f)#gradient of b2
            delta1=der_activation_function(u,act_f)*np.dot(w2.T,delta2)#gradient of b1
            d_w1=np.dot(np.expand_dims(delta1,axis=1),np.expand_dims(x,axis=0))#gradient of w1
            d_w2=np.dot(np.expand_dims(delta2,axis=1),np.expand_dims(h,axis=0))#gradient of w2

            #update weight and bias
            w1=w1-alpha*d_w1
            w2=w2-alpha*d_w2
            b1=b1-alpha*delta1
            b2=b2-alpha*delta2

            #calculate loss
            loss+=0.5*np.sum(np.multiply(output-y,output-y))
        average_loss=loss/60000
        # print training process
        print(i+1,'epoch average error:',average_loss)
        # y_label.append(average_loss)

    # plt.plot(x_label,y_label)
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.show()

    #test
    u_test=np.dot(w1,test_data.T)+np.expand_dims(b1,axis=1)
    h_test=activation_function(u_test,act_f)
    v_test=np.dot(w2,h_test)+np.expand_dims(b2,axis=1)
    output_test=activation_function(v_test.T,act_f)

    #calculate accuracy
    right_times=0
    for i in range(len(output_test)):
        if np.argmax(output_test[i])==np.argmax(test_labels[i]):
            right_times+=1
    accuracy=100*right_times/len(output_test)
    print('accuracy:',accuracy)
    return accuracy


if __name__=='__main__':
    np.seterr(divide='ignore', invalid='ignore')
    train_imgs=loadImage("train-images-idx3-ubyte")
    train_labels=loadLabel("train-labels-idx1-ubyte")
    test_imgs=loadImage("t10k-images-idx3-ubyte")
    test_labels=loadLabel("t10k-labels-idx1-ubyte")
    random.seed(17)
    result=MLP_train(train_imgs,train_labels,270,100,test_imgs,test_labels,2)
    # for act in range(1,3):
    #     print('no data normalization')
    #     if act==1:
    #         print('activation function: tanh')
    #     if act==2:
    #         print('activation function: sigmoid')
    #     # elif act==3:
    #     #     print('activation function: PReLU')
    #     for nodes in range(30,330,30):
    #         print(nodes,'hidden nodes:')
    #         result=MLP_train(train_imgs, train_labels, nodes, 30, test_imgs, test_labels,act)