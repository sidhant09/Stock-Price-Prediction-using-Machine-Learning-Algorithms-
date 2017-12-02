##PACKAGES USED
import csv
import numpy as np
from sklearn import svm 
from sklearn.metrics import *
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import *
from pybrain.datasets import *
from pybrain.structure.modules import *

##READING THE DATA FROM THE CSV
def read_data(fp):
    all_features = []
    timestamp_list =[]
    close_list = []
    high_list = []
    low_list = []
    open_price_list =[]
    volume_list = []
    count=0
    datasetname = fp
    for line in datasetname:
        l=line.split(',')
        count+=1
        x = list(l[len(l)-1])
        x = x[0:len(x)-1]
        x = ''.join(x)
        l[len(l)-1]=x
        all_features.append(l)
        timestamp, close, high, low, open_price , volume = l
        timestamp_list.append(int(timestamp))
        close_list.append(float(close))
        high_list.append(float(high))
        low_list.append(float(low))
        open_price_list.append(float(open_price))
        volume_list.append(float(volume))
    return timestamp_list, close_list, high_list, low_list, open_price_list, volume_list

##TO FIND PRICE INCREASING OR DECREASING
def creating_binary_labels(close_list, open_price_list):
    label_list = close_list - open_price_list
    label_list = label_list[1:-1]
    for i in range(len(label_list)):
        if(label_list[i]>0):
            label_list[i]=1
        else:
            label_list[i]=0
    return label_list
    
##EXTRACTING FEATURES (6 TO 12)
def fearure_creation(timestamp_list, close_list, high_list, low_list, open_price_list, volume_list, x):
    open_change_percentage_list=[]
    close_change_percentage_list=[]
    low_change_percentage_list=[]
    high_change_percentage_list=[]
    volume_change_percentage_list=[]    
    volume_diff_percentage_list=[]
    open_diff_percentage_list=[]
    Open_price_moving_average_list=[]
    Close_price_moving_average_list=[]
    High_price_moving_average_list=[]
    Low_price_moving_average_list=[]

    highest_open_price = open_price_list[0]
    lowest_open_price = open_price_list[0]
    highest_volume = volume_list[0]
    lowest_volume = volume_list[0]
    if(x>len(open_price_list)):
        x = len(open_price_list)
    for i in range(len(close_list)-x,len(close_list)):
        if(highest_open_price<open_price_list[i]):
            highest_open_price=open_price_list[i]
        if(lowest_open_price>open_price_list[i]):
            lowest_open_price=open_price_list[i]
        if(highest_volume<volume_list[i]):
            highest_volume=volume_list[i]
        if(lowest_volume>volume_list[i]):
            lowest_volume=volume_list[i]

    opensum=open_price_list[0]
    closesum=close_list[0]
    highsum=high_list[0]
    lowsum=low_list[0]
    for i in range(1, len(close_list)-1):
        close_change_percentage = (close_list[i] - close_list[i-1])/close_list[i-1]
        close_change_percentage_list.append(close_change_percentage)
        
        open_change_percentage = (open_price_list[i+1] - open_price_list[i])/open_price_list[i]
        open_change_percentage_list.append(open_change_percentage)

        high_change_percentage = (high_list[i] - high_list[i-1])/high_list[i-1]
        high_change_percentage_list.append(high_change_percentage)
        if volume_list[i-1]==0:
            volume_list[i-1] = volume_list[i-2]

        volume_change_percentage = (volume_list[i] - volume_list[i-1])/volume_list[i-1]
        volume_change_percentage_list.append(volume_change_percentage)

        low_change_percentage = (low_list[i] - low_list[i-1])/low_list[i-1]
        low_change_percentage_list.append(low_change_percentage)

        volume_diff = (volume_list[i] - volume_list[i-1])/(highest_volume-lowest_volume)
        volume_diff_percentage_list.append( volume_diff)

        open_diff = (open_price_list[i+1] - open_price_list[i])/(highest_open_price - lowest_open_price)
        open_diff_percentage_list.append(open_diff)

        opensum+=open_price_list[i]
        closesum+=close_list[i]
        highsum+=high_list[i]
        lowsum+=low_list[i]

        Open_price_moving_average = float(opensum/i+1) / open_price_list[i+1]
        Open_price_moving_average_list.append(Open_price_moving_average)

        High_price_moving_average = float(highsum/i+1) / high_list[i+1]
        High_price_moving_average_list.append(High_price_moving_average)

        Close_price_moving_average = float(closesum/i+1) / close_list[i+1]
        Close_price_moving_average_list.append(Close_price_moving_average)

        Low_price_moving_average = float(lowsum/i+1) / low_list[i+1]
        Low_price_moving_average_list.append(Low_price_moving_average)
   
    
    close_change_percentage_list = np.array(close_change_percentage_list)
    high_change_percentage_list = np.array(high_change_percentage_list)
    low_change_percentage_list = np.array(low_change_percentage_list)
    volume_change_percentage_list = np.array(volume_change_percentage_list)
    open_price_list = np.array(open_price_list)
    close_list = np.array(close_list)
    open_diff_percentage_list=np.array(open_diff_percentage_list)
    volume_change_percentage_list=np.array(volume_change_percentage_list)
    
    ##WITH 5 FEATURES
    feature1 = np.column_stack((open_change_percentage_list, close_change_percentage_list, high_change_percentage_list, low_change_percentage_list, volume_change_percentage_list))  
    
    ##WITH 7 FEATURES
    feature2 = np.column_stack((open_change_percentage_list, close_change_percentage_list, high_change_percentage_list, low_change_percentage_list, volume_change_percentage_list, open_diff_percentage_list, volume_diff_percentage_list))  
    
    ##WITH 9 FEATURES
    feature3 = np.column_stack((open_change_percentage_list, close_change_percentage_list, high_change_percentage_list, low_change_percentage_list, volume_change_percentage_list, Open_price_moving_average_list, Close_price_moving_average_list, High_price_moving_average_list, Low_price_moving_average_list))  
    
    ##WITH 11 FEATURES
    feature4 = np.column_stack((open_change_percentage_list, close_change_percentage_list, high_change_percentage_list, low_change_percentage_list, volume_change_percentage_list, open_diff_percentage_list, volume_diff_percentage_list,Open_price_moving_average_list, Close_price_moving_average_list, High_price_moving_average_list, Low_price_moving_average_list))
    
    label_list = creating_binary_labels(close_list, open_price_list)
    return feature1, feature2, feature3, feature4, label_list

##SUPPORT VECTOR MACHINE
def svm_rbf(feature, label_list):
    length_feature = len(feature)
    
    ##TRAINIG DATA = 75% OF THE TOTAL DATA
    len_train = int(0.75*length_feature)
    train_feature = feature[0: len_train]
    test_feature = feature[len_train: ]

    train_label = label_list[0:len_train]
    test_label = label_list[len_train:]

    clf = svm.SVC(C=100000,kernel='rbf')
    clf.fit(train_feature, train_label)
    predicted = clf.predict(test_feature)
    print "Accuracy: ",accuracy_score(predicted, test_label)*100, "%"
    return predicted, test_label, train_feature, train_label, test_feature


##NEURAL NETWORK
def neural_networks(train_feature, train_label, test_features, test_labels):
    net = buildNetwork(len(train_feature[0]), 30, 1, hiddenclass = TanhLayer, outclass = TanhLayer,recurrent = True)
    ds = ClassificationDataSet(len(train_feature[0]), 1)
    for i, j in zip(train_feature, train_label):
        ds.addSample(i, j)
    trainer = BackpropTrainer(net, ds)
    epochs = 9
    for i in range(epochs):
        trainer.train()
    predicted = list()
    for i in test_features:
        predicted.append(int(net.activate(i)>0.5))
    predicted = np.array(predicted)

    print "Accuracy:", accuracy_score(test_labels, predicted)*100, "%"
    return predicted

##MAIN FUNCTION
if __name__ == '__main__':
    ##DATA SOURCE
    fp2= open("MSFT_37_2.csv",'r+')
    timestamp_list, close_list, high_list, low_list, open_price_list, volume_list = read_data(fp2)
    fp2.close()

    x = 5
    feature1, feature2, feature3, feature4, label_list = fearure_creation(timestamp_list, close_list, high_list, low_list, open_price_list, volume_list, x )
    print
    print "-----------------------------------------------------------------------"
    print "SVM with Features : "
    print "Open Change%, Close Change%, High Change%, Low Change%, Volume Change%"
    predicted1, test_label1, train_feature1, train_label1, test_feature1 = svm_rbf(feature1, label_list)
    print "-----------------------------------------------------------------------"
    
 
    print
    print "-----------------------------------------------------------------------"
    print "SVM with Features : "
    print "Open Change%, Close Change%, High Change%, Low Change%, Volume Change%,"
    print "Open Difference% , Volume Difference%, "
    predicted2, test_label2, train_feature2, train_label2, test_feature2= svm_rbf(feature2, label_list)
    print "-----------------------------------------------------------------------"

    print
    print "-----------------------------------------------------------------------"
    print "SVM with Features : "
    print "Open Change%, Close Change%, High Change%, Low Change%, Volume Change%,"
    print "Open Price Moving Avg, Close Price Moving Avg, High Price Moving Avg, Low Price Moving Avg"
    predicted3, test_label3, train_feature3, train_label3, test_feature3= svm_rbf(feature3, label_list)
    print "-----------------------------------------------------------------------"
        
    print
    print "-----------------------------------------------------------------------"
    print "SVM with Features : "
    print "Open Change%, Close Change%, High Change%, Low Change%, Volume Change%"
    print "Open Difference% , Volume Difference%, Open Price Moving Avg"
    print "Close Price Moving Avg, High Price Moving Avg, Low Price Moving Avg"
    predicted4, test_label4, train_feature4, train_label4, test_feature4 =  svm_rbf(feature4, label_list)
    print "-----------------------------------------------------------------------"
    
       
    print
    print
    print "*******************************RNN*************************************"
    print
    print "-----------------------------------------------------------------------"
    print "RNN Kernel with Features : "
    print "Open Change%, Close Change%, High Change%, Low Change%, Volume Change%"
    predicted_NN_1=neural_networks(train_feature1, train_label1, test_feature1, test_label1)
    print "-----------------------------------------------------------------------"

    print
    print "-----------------------------------------------------------------------"
    print "RNN Kernel with Features : "
    print "Open Change%, Close Change%, High Change%, Low Change%, Volume Change%,"
    print "Open Difference% , Volume Difference%, "
    predicted_NN_2=neural_networks(train_feature2, train_label2, test_feature2, test_label2)
    print "-----------------------------------------------------------------------"
   
    print
    print "-----------------------------------------------------------------------"
    print "RNN Kernel with Features : "
    print "Open Change%, Close Change%, High Change%, Low Change%, Volume Change%,"
    print "Open Price Moving Avg, Close Price Moving Avg, High Price Moving Avg, Low Price Moving Avg"
    predicted_NN_3=neural_networks(train_feature3, train_label3, test_feature3, test_label3)
    print "-----------------------------------------------------------------------"

    print
    print "-----------------------------------------------------------------------"
    print "RNN Kernel with Features : "
    print "Open Change%, Close Change%, High Change%, Low Change%, Volume Change%"
    print "Open Difference% , Volume Difference%, Open Price Moving Avg"
    print "Close Price Moving Avg, High Price Moving Avg, Low Price Moving Avg"
    predicted_NN_4=neural_networks(train_feature4, train_label4, test_feature4, test_label4)
    print "-----------------------------------------------------------------------"     