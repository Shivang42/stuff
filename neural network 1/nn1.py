import os
import struct
import numpy as np
import matplotlib.pyplot as plt

def load(path,kind='train'):
    labpath = os.path.join(path,'%s-labels.idx1-ubyte' % kind)
    imgpath = os.path.join(path,'%s-images.idx3-ubyte' % kind)
    with open(labpath,'rb') as lp:
        magic,n = struct.unpack('>II',lp.read(8))
        labels = np.fromfile(lp,dtype=np.uint8)
    with open(imgpath,'rb') as ip:
        magic,num,rows,cols = struct.unpack('>IIII',ip.read(16))
        images = np.fromfile(ip,dtype=np.uint8)
        images= ((images/255.)-0.5)*2
    return images,labels

#xtest,ytest = load('',kind='t10k')
#xtrain,ytrain = load('')
npz = np.load('mnist.npz')
xtrain,xtest = npz[npz.files[0]],npz[npz.files[1]]
ytrain,ytest = npz[npz.files[2]],npz[npz.files[3]]
xtrain = xtrain.reshape(60000,28*28)
xtest = xtest.reshape(10000,28*28)
fig,ax = plt.subplots(nrows=2,ncols=5,sharex=True,sharey=True)
ax = ax.flatten()

for k in range(10):
    ax[k].imshow(xtrain[ytrain==0][k].reshape(28,28),cmap='Greys')
plt.show()
