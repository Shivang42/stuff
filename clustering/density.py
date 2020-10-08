import matplotlib.pyplot as plt
from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN
from sklearn.datasets import make_moons

x,y = make_moons(n_samples=500,noise=0.05,random_state=0)
plt.scatter(x[:,0],x[:,1],c='red')
plt.show()

f,(ax1,ax2) = plt.subplots(1,2,figsize=(8,3))
km = KMeans(n_clusters=2,random_state=0)
ac = AgglomerativeClustering(n_clusters=2,affinity='euclidean',linkage='complete')
db = DBSCAN(eps=0.2,min_samples=5,metric='euclidean')
ykm1 = km.fit_predict(x)
ykm2 = ac.fit_predict(x)
ykm3 = db.fit_predict(x)

#Kmeans Clustering Plot
ax1.scatter(x[ykm1==0,0],x[ykm1==0,1],c='red',edgecolor='black',marker='o',label='Cluster 1')
ax1.scatter(x[ykm1==1,0],x[ykm1==1,1],c='blue',edgecolor='black',marker='*',label='Cluster 2')
#Agglomerative Clustering Plot
ax2.scatter(x[ykm2==0,0],x[ykm2==0,1],c='red',edgecolor='black',marker='o',label='Cluster 1')
ax2.scatter(x[ykm2==1,0],x[ykm2==1,1],c='blue',edgecolor='black',marker='*',label='Cluster 2')
ax1.set_title("Kmeans Clustering Plot")
ax2.set_title("Agglomerative Clustering Plot")
plt.legend()
plt.show()
#DBSCAN Plot            
plt.scatter(x[ykm3==0,0],x[ykm3==0,1],c='red',edgecolor='black',marker='o',label='Cluster 1')
plt.scatter(x[ykm3==1,0],x[ykm3==1,1],c='blue',edgecolor='black',marker='*',label='Cluster 2')
plt.legend()
plt.show()
