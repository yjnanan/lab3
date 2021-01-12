from input_data_SVM import get_labels,get_images
from sklearn import svm
import pickle
import numpy as np

def train_test(c,k):
    # train model
    train_data = get_images('train-images-idx3-ubyte', length=60000)
    train_labels = get_labels('train-labels-idx1-ubyte')

    clf = svm.SVC(C=c, kernel=k)
    train_data = np.asmatrix(train_data[:(60000 * 784)]).reshape(60000, 784)

    clf.fit(train_data, train_labels[:60000])

    # # save the model to disk
    # filename = 'finalized_model_60000_f_poly.sav'
    # pickle.dump(clf, open(filename, 'wb'))
    # print("Succeed!")

    # test model
    test_data = get_images('t10k-images-idx3-ubyte', True)  # True: for full length
    test_labels = get_labels('t10k-labels-idx1-ubyte')

    test_data = np.asmatrix(test_data).reshape(10000, 784)
    #get accuracy
    result = clf.score(test_data, test_labels)
    print("Accuracy: ", result)

if __name__=='__main__':
    kernel_list=['rbf','poly']
    for k in kernel_list:
        print('kernel function:',k)
        for c in range(1,11,1):
            print('C(penalty coefficient)=',c)
            train_test(c,k)