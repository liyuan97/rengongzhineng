import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
#from sklearn.model_selection import train_test_split #废弃！！
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#from sklearn.neural_network import BernoulliRBM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import ClusterCentroids,RandomUnderSampler
from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter
import math


h=0.2

# 读取训练数据
def readdata():
    fake = pd.read_csv('fake.csv')  
    real = pd.read_csv('real.csv')
    ALL=fake.append(real)
    all=ALL.drop(['Unnamed: 0'],axis=1)
    X=np.array(ALL)
    #X = StandardScaler().fit_transform(X)
    y=X[:,24]
    #y=np.append(np.zeros(fake.shape[0]),np.ones(real.shape[0]))
    X=X[:,1:-1]
    return X,y


def allmodel():
    classifiers = [
    linear_model.LogisticRegression(C=1e5),#1
    KNeighborsClassifier(5),#2
    SVC(kernel="linear", C=0.025),#3
    SVC(gamma='auto', C=1),#4
    DecisionTreeClassifier(max_depth=5),#5
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),#6
    AdaBoostClassifier(),#7
    GaussianNB(),#8
    MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1),#9
    ]
    names = ['LogisticRegression',#1
         "Nearest Neighbors", #2
         "Linear SVM", #3
         "RBF SVM",#4
         "Decision Tree",#5 
         "Random Forest", #6
         "AdaBoost",#7
         "Naive Bayes",#8
         'MLPClassifier',#9
         ]
    index=['origional',
           "overSampler",
           'underSampler',
           'smotesampler',
           'adasynsampler'
            ]
    return classifiers,names,index

########################
#print('%-15s  %-15s  %-15s'%('name','score','roc_auc','wrong'))
def runmodel(input_x,input_y,index_name):
    X_train, X_test, y_train, y_test = train_test_split(input_x,input_y, test_size=.5, random_state=1)
    print("Method                 ACC       AUC      RECALL")    
    for name, clf in zip(names, classifiers):
    #    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        #confusion=confusion_matrix(y_test, clf.predict(X_test))
        #
        re=metrics.recall_score(y_test, clf.predict(X_test), average=None)[0]
        #metrics.f1_score(y_test, clf.predict(X_test))
        fpr, tpr, thresholds = metrics.roc_curve(y_test, clf.predict(X_test))
        roc_auc = metrics.auc(fpr, tpr)
        print('%-20s, %f, %f, %f '%(name,score,roc_auc,re))
        pre_score[name][index_name]=np.array([score,roc_auc,re])
#        print('%-15s  %-15s  %-15s  %-15s '%(name,score,roc_auc,sum(abs(clf.predict(X_test)-y_test))))
        #,confusion)
        #pred = 
        #print(metrics.classification_report(y_test, clf.predict(X_test)))
#######################
def runmodel_tra(input_x,input_y,index_name):
    X_train, X_test, y_train, y_test = train_test_split(input_x,input_y, test_size=.5, random_state=1)
    X_test,y_test=X_train,y_train
    print("Method                 ACC       AUC      RECALL")    
    for name, clf in zip(names, classifiers):
    #    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        #confusion=confusion_matrix(y_test, clf.predict(X_test))
        #
        re=metrics.recall_score(y_test, clf.predict(X_test), average=None)[0]
        #metrics.f1_score(y_test, clf.predict(X_test))
        fpr, tpr, thresholds = metrics.roc_curve(y_test, clf.predict(X_test))
        roc_auc = metrics.auc(fpr, tpr)
        print('%-20s, %f, %f, %f '%(name,score,roc_auc,re))
        tra_score[name][index_name]=np.array([score,roc_auc,re])
#####################
def plot2(X,y,title):
    pca = PCA(n_components=2)
    newData=pca.fit_transform(X)
    cValue=[]
    for i in range(y.size):
        if y[i]==0:
            cValue.append('r')
        else:
            cValue.append('b')
    plt.scatter(newData[:,0], newData[:,1] , c=cValue, marker='o')  #cmap=plt.cm.Paired    
    title=title+str(sorted(Counter(y.astype(int)).items()))
    plt.title(title)    
    plt.show()
##################
def overSampler(X,y): 
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_sample(X, y)
    return X_resampled,y_resampled
##################
def underSampler(X,y):
    cc = RandomUnderSampler(random_state=0)
    X_resampled, y_resampled = cc.fit_sample(X, y)
    return X_resampled,y_resampled
######################
def smotesampler(X,y):
    X_resampled_smote, y_resampled_smote = SMOTE().fit_sample(X, y)
    return X_resampled_smote, y_resampled_smote
def adasynsampler(X,y):    
    X_resampled_adasyn, y_resampled_adasyn = ADASYN().fit_sample(X, y)
    return X_resampled_adasyn, y_resampled_adasyn
#######################
#sorted(Counter(y).items())

if __name__ == '__main__':
    classifiers,names,index=allmodel()
    X,y=readdata()


    
    X_resampled_over,y_resampled_over=overSampler(X,y)
    X_resampled_under,y_resampled_under=underSampler(X,y)
    X_resampled_smote, y_resampled_smote=smotesampler(X,y)
    X_resampled_adasyn, y_resampled_adasyn=adasynsampler(X,y)
    
    plot2(X,y,"All DATA(red=fake,blue=real)")  
    plot2(X_resampled_over,y_resampled_over,"overSampler(red=fake,blue=real)")
    plot2(X_resampled_under,y_resampled_under,"underSampler(red=fake,blue=real)")  
    plot2(X_resampled_smote, y_resampled_smote,"smote(red=fake,blue=real)")    
    plot2(X_resampled_adasyn, y_resampled_adasyn,"adasyn(red=fake,blue=real)")

    pre_score=pd.DataFrame(index=index,columns=names)
    tra_score=pd.DataFrame(index=index,columns=names)
    
    runmodel(X,y,index[0])
    runmodel(X_resampled_over,y_resampled_over,index[1])
    runmodel(X_resampled_under,y_resampled_under,index[2])
    runmodel(X_resampled_smote, y_resampled_smote,index[3])
    runmodel(X_resampled_adasyn, y_resampled_adasyn,index[4]) 

    
    runmodel_tra(X,y,index[0])
    runmodel_tra(X_resampled_over,y_resampled_over,index[1])
    runmodel_tra(X_resampled_under,y_resampled_under,index[2])
    runmodel_tra(X_resampled_smote, y_resampled_smote,index[3])
    runmodel_tra(X_resampled_adasyn, y_resampled_adasyn,index[4]) 
    
    for name, clf in zip(names, classifiers):
        
        print('\n%-20s, %-15s, %s, %s '%(name,'pre_auc','tra_auc','alpha'))
        for index_name in index:
            pre_auc=pre_score[name][index_name][1]
            tra_auc=tra_score[name][index_name][1]
            myscore=-math.log(pre_auc/tra_auc)
            print('%-20s, %-15s, %f, %f, %f '%(name,index_name,pre_auc,tra_auc,myscore))
    
    
    
    