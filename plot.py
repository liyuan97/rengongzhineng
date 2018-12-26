from collections import Counter
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
from imblearn.under_sampling import ClusterCentroids
from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, SMOTENC
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import ClusterCentroids
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
def plot_decision_function(X, y, clf, ax):
    plot_step = 0.2
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.4)
    ax.scatter(X[:, 0], X[:, 1], alpha=0.8, c=y, edgecolor='k')


#########################
X,y=readdata()
pca = PCA(n_components=2)
X=pca.fit_transform(X)

def plotone(X,y,sampler_name,clf,ax,name):
#    clf = make_pipeline(sampler, clf)
    clf.fit(X, y)
    plot_decision_function(X, y, clf, ax)
    ax.set_title('{}+{}'.format(name,sampler_name))
    print('{}+{}'.format(name,sampler_name))



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


    i=0
    X_resampled_over,y_resampled_over=overSampler(X,y)   
    X_resampled_under,y_resampled_under=underSampler(X,y)    
    X_resampled_smote, y_resampled_smote=smotesampler(X,y)
    X_resampled_adasyn, y_resampled_adasyn=adasynsampler(X,y)

    X_resampled_over=pca.fit_transform(X_resampled_over)
    X_resampled_under=pca.fit_transform(X_resampled_under)
    X_resampled_smote=pca.fit_transform(X_resampled_smote)
    X_resampled_adasyn=pca.fit_transform(X_resampled_adasyn)
    
    
for i in range(9):
    clf=classifiers[i] 
    clf.fit(X, y)
    fig,(ax0,ax1,ax2,ax3,ax4) = plt.subplots(1,5, figsize=(30,6))
    plot_decision_function(X, y, clf, ax0)
    ax0.set_title('{}+unbalanced'.format(clf.__class__.__name__))
    plotone(X_resampled_over,y_resampled_over,RandomOverSampler(random_state=0).__class__.__name__,clf,ax1,clf.__class__.__name__)
    plotone(X_resampled_under,y_resampled_under,ClusterCentroids(random_state=0).__class__.__name__,clf,ax2,clf.__class__.__name__)
    plotone(X_resampled_smote, y_resampled_smote,SMOTE().__class__.__name__,clf,ax3,clf.__class__.__name__)
    plotone(X_resampled_adasyn, y_resampled_adasyn,ADASYN().__class__.__name__,clf,ax4,clf.__class__.__name__)
    fig.tight_layout()
    fig.show()
    fig.savefig('D:\\大三\\郭小波 复杂数据分析\\homework\\HM2\\image\\{}'.format(clf.__class__.__name__))



def same_model_five_data():
    clf=classifiers[i] 
    clf.fit(X, y)
    

