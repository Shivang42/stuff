'''
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
def ee(n,error):
    start = int(math.ceil(n/2))
    probs = [comb(n,k)*error**k*(1-error)**(n-k) for k in range(start,n+1)]
    p = sum(probs)
    return p
range1 = np.arange(0,1.01,0.01)
erc = [ee(11,e) for e in range1]
plt.plot(range1,erc,linewidth=1.5,color='yellow')
plt.plot(range1,range1,linewidth=1.5,color='green')
plt.grid(alpha=0.3)
plt.show()
k = np.array([[0.9,0.1],
              [0.8,0.2],
              [0.4,0.6]])
h = np.average(k,axis=0,weights=[0.2,0.2,0.6])
print(np.argmax(h))
'''
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import six
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import numpy as np
import operator

class MajorityVoteClassifier(BaseEstimator,ClassifierMixin):
    def __init__(self,classifiers,vote='classlabel',weights=None):
        self.classifiers = classifiers
        self.named_classifiers = {key:value for key,value in _name_estimators(classifiers)}
        self.vote = vote
        self.weights = weights
    def fit(self,x,y):
        self.le = LabelEncoder()
        self.le.fit(y)
        self.classifiers_ = []
        for clif in self.classifiers:
            clf1 = clone(clif).fit(x,self.le.transform(y))
            self.classifiers_.append(clf1)
        return self
    def predict(self,x):
        if self.vote=='probability':
            mv = np.argmax(self.predict_proba(x),axis=1)
        else:
            predictions = np.asarray([cl.predict(x) for cl in  self.classifiers_]).T
            mv = np.apply_along_axis(lambda x:np.argmax(np.bincount(x,weights=self.weights)),axis=1,arr=predictions)
        mv = self.le.inverse_transform(mv)
        return mv
    def predict_proba(self,x):
        probas = np.asarray([clf.predict_proba(x) for clf in self.classifiers_])
        av = np.average(probas,axis=0,weights=self.weights)
        return av
    def get_params(self,deep=True):
        if not deep:
            return super(MajorityVoteClassifier,self).get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
            for name,step in six.iteritems(self.named_classifiers):
                for key,value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name,key)] = value
            return out


























        
