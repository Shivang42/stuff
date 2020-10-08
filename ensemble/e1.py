from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from ensemble1 import MajorityVoteClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve
from itertools import product
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np

colors = ['red','blue','green','orange']
ls = ['--','-.',':','-']
flow = datasets.load_iris()
X,y = flow.data[50:,[1,2]],flow.target[50:]
xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=0.5,random_state=1,stratify=y)
print(xtrain.shape,xtest.shape)
c1 = KNeighborsClassifier(n_neighbors=1,p=2,metric='minkowski')
c2 = DecisionTreeClassifier(max_depth=1,criterion='entropy',random_state=0)
c3 = LogisticRegression(penalty='l2',C=100.0,random_state=1)
p1 = Pipeline([['sc',StandardScaler()],['clf',c1]],verbose=0)
p3 = Pipeline([['sc',StandardScaler()],['clf',c3]],verbose=0)
mvc = MajorityVoteClassifier(classifiers=[p1,c2,p3])
clf_labels = ['Logistic Regression','Decision Tree','KNN','MVC']
sc = StandardScaler()
xtrainstd  = sc.fit_transform(xtrain)
xmin = xtrainstd[:,0].min()-1
xmax = xtrainstd[:,0].max()+1
ymin = xtrainstd[:,1].min()-1
ymax = xtrainstd[:,1].max()+1
xx,yy = np.meshgrid(np.arange(xmin,xmax,0.1),np.arange(ymin,ymax,0.1))
f,aa = plt.subplots(nrows=2,ncols=2,sharex='col',sharey='row',figsize=(7,5))
for idx,clf,lab in zip(product([0,1],[0,1]),[p1,c2,p3,mvc],clf_labels):
    clf.fit(xtrainstd,ytrain)
    z = clf.predict(np.c_[xx.ravel(),yy.ravel()])
    z = z.reshape(xx.shape)
    aa[idx[0],idx[1]].contourf(xx,yy,z,alpha=0.3)
    aa[idx[0],idx[1]].scatter(xtrainstd[ytrain==0,0],xtrainstd[ytrain==0,1],c='red',marker='o',s=50)
    aa[idx[0],idx[1]].scatter(xtrainstd[ytrain==1,0],xtrainstd[ytrain==1,1],c='blue',marker='*',s=50)
    aa[idx[0],idx[1]].set_title(lab)
plt.show()
print(mvc.get_params())
param = {
    'decisiontreeclassifier__max_depth':[1,2],
    'pipeline-2__clf__C':[0.001,0.1,100.0]
    }
grid = GridSearchCV(estimator=mvc,param_grid=param,cv=10,scoring='roc_auc')
grid.fit(xtrainstd,ytrain)
'''
for clf,label,c,l in zip([p1,c2,p3,mvc],clf_labels,colors,ls):
    ypred = clf.fit(xtrain,ytrain).predict_proba(xtest)[:,1]
    ytest1 = np.array([1 if k==2 else 0 for k in ytest])
    print(ypred)
    fpr,tpr,thresholds = roc_curve(y_true=ytest1,y_score=ypred)
    ra = auc(x=fpr,y=tpr)
    plt.plot(fpr,tpr,color=c,linestyle=l,label='%s (auc = %0.2f)' %(label,ra))
plt.legend(loc='best')
plt.plot([0,1],[0,1],linestyle='--',color='pink',linewidth=2)
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.grid(alpha=0.5)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()
'''
