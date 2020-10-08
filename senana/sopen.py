import re
import os
import pickle
import numpy as np
import mysql.connector as tor
from sklearn.feature_extraction.text import HashingVectorizer
from nltk.stem import PorterStemmer

ps = PorterStemmer()
def tokenizer(text):
    text = re.sub('<[^>]*>','',str(text))
    emots = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text)
    text = re.sub('[\W]+',' ',text.lower())+' '.join(emots).replace('-','')
    king = [ps.stem(w) for w in text.split() if w not in stop]
    return king
hv = HashingVectorizer(decode_error='ignore',n_features=2**21,preprocessor=None,tokenizer=tokenizer)
a = tor.connect(host='localhost',user='root',passwd='12345',database='world')
b = a.cursor()

b.execute("insert into reviews values({},'{}',{},{}),({},'{}',{},{});".format(1,"i liked this movie",1,"curdate()",2,"I hated this movie.Hated,hated,hated this movie",0,"curdate()"))
a.commit()
a.close()
print("SQL Part done")

clf = pickle.load(open('movieclassifier\PKL\classifier.pkl','rb+'))
stop = pickle.load(open('movieclassifier\PKL\stopword.pkl','rb+'))
label = {0:'negative',1:'positive'}
ex = ["I hated this movie"]
x = hv.transform(ex)
print(label[clf.predict(x)[0]])
