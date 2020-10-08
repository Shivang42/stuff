from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd

colors = ['red','blue','green','yellow','brown']
markers = ['o','*','d','D','s']
x,y = make_blobs(n_samples=150,n_features=2,centers=3,cluster_std=0.75,shuffle=True,random_state=0)
km = KMeans(n_clusters=5,init='random',n_init=10,max_iter=300,tol=1e-04,random_state=0)
ac = AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='complete')
labels = ac.fit_predict(x)
ykm = km.fit_predict(x)
for i,c,m in zip(range(5),colors,markers):
    plt.scatter(x[labels==i,0],x[labels==i,1],c=c,marker=m,s=50,edgecolor='black')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],c='gold',marker='D',s=75)
plt.grid()
plt.show()
#Elbow technique
'''
distortions = []
for i in range(1,11):
    km1 = KMeans(n_clusters=i,init='k-means++',n_init=10,max_iter=300,random_state=0)
    km1.fit(x)
    distortions.append(km1.inertia_)
plt.plot(range(1,11),distortions,marker='o')
plt.xlabel('No. of clusters')
plt.ylabel('Distortion')
plt.show()

#Clustering analysis bars
cluster_labels = np.unique(ykm)
n_clusters = cluster_labels.shape[0]
xup,xlow = 0,0
yts = []
sil_vals = silhouette_samples(x,ykm,metric='euclidean')
for i,c in enumerate(cluster_labels):
    csilvals = sil_vals[ykm==c]
    csilvals.sort()
    xup+=len(csilvals)
    plt.barh(range(xlow,xup),csilvals,color=cm.jet(float(i)/n_clusters),edgecolor=None)
    yts.append((xlow+xup)/2.0)
    xlow+=len(csilvals)
plt.yticks(yts,cluster_labels+1)
silavg = np.mean(sil_vals)
plt.axvline(silavg,linestyle='--',color='red')
plt.show()
'''


