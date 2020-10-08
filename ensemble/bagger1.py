import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

wine = pd.read_csv('C:/Users/Cyberdex42/Downloads/wine.data',header=None)
wine.columns = ['Class label', 'Alcohol','Malic acid', 'Ash','Alcalinity of ash','Magnesium', 'Total phenols','Flavanoids', 'Nonflavanoid phenols','Proanthocyanins','Color intensity', 'Hue','OD280/OD315 of diluted wines','Proline']
wine = wine[wine['Class label']!=1]
y = wine['Class label'].values
x = wine[['Alcohol','OD280/OD315 of diluted wines']].values
le = LabelEncoder()
y = le.fit_transform(y)
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,stratify=y,random_state=1)
tree = DecisionTreeClassifier(criterion='entropy',max_depth=1,random_state=1)
bag1 = BaggingClassifier(base_estimator=tree,n_estimators=500,max_samples=1.0,max_features=1.0,bootstrap=True,bootstrap_features=False,n_jobs=2,random_state=1)
tree = tree.fit(xtrain,ytrain)
ytrpred = tree.predict(xtrain)
ytepred = tree.predict(xtest)
treetrain = accuracy_score(ytrain,ytrpred)
treetest = accuracy_score(ytest,ytepred)
ada = AdaBoostClassifier(base_estimator=tree,n_estimators=500,learning_rate=0.1,random_state=1)
ada = ada.fit(xtrain,ytrain)
ytrpred1 = ada.predict(xtrain)
ytepred1 = ada.predict(xtest)
adatrain = accuracy_score(ytrain,ytrpred1)
adatest = accuracy_score(ytest,ytepred1)
print('Ratio of train test: %2.2f / %2.2f' % (treetrain,treetest))
print('Ratio of train test: %2.2f / %2.2f' % (adatrain,adatest))
'''
print('Ratio of train test: %2.2f / %2.2f' % (treetrain,treetest))
bag1 = bag1.fit(xtrain,ytrain)
ytrpred1 = bag1.predict(xtrain)
ytepred1 = bag1.predict(xtest)
bagtrain = accuracy_score(ytrain,ytrpred1)
bagtest = accuracy_score(ytest,ytepred1)
print('Ratio of train test: %2.2f / %2.2f' % (bagtrain,bagtest))
'''
xmin = xtrain[:,0].min()-1
xmax = xtrain[:,0].max()+1
ymin = ytrain.min()-1
ymax = ytrain.max()+1
xx,yy = np.meshgrid(np.arange(xmin,xmax,0.1),np.arange(ymin,ymax,0.1))
f,aa = plt.subplots(nrows=1,ncols=2,sharex='col',sharey='row',figsize=(8,3))
for idx,clf,lab in zip([0,1],[tree,ada],['Decision Tree','Ada Boost Classifier']):
    clf.fit(xtrain,ytrain)
    z = clf.predict(np.c_[xx.ravel(),yy.ravel()])
    z = z.reshape(xx.shape)
    aa[idx].contourf(xx,yy,z,alpha=0.3)
    aa[idx].scatter(xtrain[ytrain==0,0],xtrain[ytrain==0,1],color='red',marker='o')
    aa[idx].scatter(xtrain[ytrain==1,0],xtrain[ytrain==1,1],color='green',marker='^')
    aa[idx].set_title(lab)
plt.show()

