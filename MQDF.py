import numpy as np
import matplotlib.pyplot as plt
from load_data import loadImage,loadLabel

#classify training data
def split_data(data,labels):
    data_list=[[],[],[],[],[],[],[],[],[],[]]
    size=labels.shape
    for i in range(size[0]):
        for j in range(size[1]):
            if labels[i,j]==1:
                data_list[j].append(data[i,:])
    for i in range(10):
        data_list[i]=np.array(data_list[i])
    return data_list

#calculate mean of training data of each class
def calculate_mean(splited_data):
    mean_data=splited_data.copy()
    for i in range(10):
        mean_data[i]=np.mean(mean_data[i],axis=0)
    return mean_data

#calculate covariance matrix of training data of each class
def calculate_cov(splited_data):
    cov_data=splited_data.copy()
    for i in range(10):
        cov_data[i]=np.cov(cov_data[i].T)
    return cov_data

#decompose covariance matrix into eigenvalues and eigenvectors
def calculate_eig(cov_matrix,k):
    c_m=cov_matrix.copy()
    eigen_result=[[],[],[],[],[],[],[],[],[],[]]
    for i in range(10):
        eigen_value,eigen_vector=np.linalg.eig(c_m[i])
        #select k eigenvalues and eigenvectors
        eigen_vector=eigen_vector[:,0:k].real
        #get best value(maximum likelihood estimation)
        delta=np.sum(eigen_value[k:784].real)/(784-k)
        eigen_value = eigen_value[0:k].real
        eigen_result[i].append(eigen_value)
        eigen_result[i].append(eigen_vector)
        eigen_result[i].append(delta)
    return eigen_result

def test(train_imgs,train_labels,test_imgs,test_labels,k):
    data_list = split_data(train_imgs, train_labels)
    mean = calculate_mean(data_list)
    cov = calculate_cov(data_list)
    eig_result = calculate_eig(cov,k)
    right_times=0#record the times of correct prediction
    for i in range(len(test_imgs)):
        output=np.zeros(10)
        for j in range(10):
           #calculate discriminant function
           output[j]=-np.sum((np.dot((test_imgs[i]-mean[j]),(eig_result[j])[1])**2)/(eig_result[j])[0])\
                     -(np.sum((test_imgs[i]-mean[j])**2)-np.sum(np.dot((test_imgs[i]-mean[j]),(eig_result[j])[1])**2))/(eig_result[j])[2]\
                     -np.sum(np.log((eig_result[j])[0]))-(784-k)*np.log((eig_result[j])[2])
        #get classification result
        print('discriminant function result(probability):\n',output)
        if np.argmax(output)==np.argmax(test_labels[i]):
            right_times+=1
            print("prediction result is correct")
        else:
            print('prediction result is wrong')
    #calculate accuracy
    accuracy=100*right_times/len(test_labels)
    print('accuracy:',accuracy)
    return accuracy



if __name__=='__main__':
    train_imgs=loadImage("train-images-idx3-ubyte")
    train_labels=loadLabel("train-labels-idx1-ubyte")
    test_imgs=loadImage("t10k-images-idx3-ubyte")
    test_labels=loadLabel("t10k-labels-idx1-ubyte")
    k_list=[]
    accuracy_list=[]
    for k in range(1,201,1):
        k_list.append(k)
        print('K:',k)
        accuracy_list.append(test(train_imgs,train_labels,test_imgs,test_labels,k))
    print(k_list)
    print(accuracy_list)
    plt.plot(k_list,accuracy_list,marker = "o")
    plt.xlabel('K')
    plt.ylabel('accuracy')
    plt.title('MQDF')
    plt.show()