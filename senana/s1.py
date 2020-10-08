import os
import re
import nltk
import pickle
import pyprind
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from nltk.stem.porter import PorterStemmer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,TfidfTransformer,HashingVectorizer

stop = stopwords.words('english')
df = pd.read_csv('reviews.csv',encoding='utf-8')
count = CountVectorizer(stop_words=stop,max_df=.1,max_features=50)
x = count.fit_transform(df['review'].values)
lda = LatentDirichletAllocation(n_components=10,random_state=123,learning_method='batch')
xtops = lda.fit_transform(x)
print(xtops.shape)
fnames = count.get_feature_names()
for idx,topic in enumerate(lda.components_):
    print("Topic "+str(idx)+" : ")
    print(" ".join([fnames[i] for i in topic.argsort()[:-6:-1]]))

df = pd.read_csv('reviews.csv',encoding='utf-8')
ps = PorterStemmer()

tfidf = TfidfTransformer(norm='l2',use_idf=True,smooth_idf=True)
np.set_printoptions(precision=2)
def tokenizer(text):
    text = re.sub('<[^>]*>','',text)
    emots = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text)
    text = re.sub('[\W]+',' ',text.lower())+' '.join(emots).replace('-','')
    king = [ps.stem(w) for w in text.split() if w not in stop]
    return king
def tokenizer1(text):
    return [ps.stem(word) for word in text.split()]
def streamdocs(path):
    with open(path,'r',encoding='utf-8') as csv:
        next(csv)
        for line in csv:
            review1,label1 = line[:-3],int(line[-2])
            yield review1,label1
def gmb(docstr,size):
    docs,y =[],[]
    try:
        for _ in range(size):
            text2,label2 = next(docstr)
            docs.append(text2)
            y.append(label2)
    except StopIteration:
        return None,None
    return docs,y
hv = HashingVectorizer(decode_error='ignore',n_features=2**21,preprocessor=None,tokenizer=tokenizer)
sgdc = SGDClassifier(loss='log',random_state=1,max_iter=1)
stream1 = streamdocs(path='reviews.csv')
k = pyprind.ProgBar(3)
classes = np.array([0,1])
for _ in range(3):
    xtra,ytra = gmb(stream1,size=10)
    if not xtra:
        break
    xtra = hv.transform(xtra)
    sgdc.partial_fit(xtra,ytra,classes=classes)
    k.update()
xtest,ytest = gmb(stream1,size=10)
xtest = hv.transform(xtest)
print(sgdc.score(xtest,ytest))

xtrain,ytrain,xtest,ytest = df.loc[1:10,'review'].values,df.loc[1:10,'sentiment'].values,df.loc[10:20,'review'].values,df.loc[10:20,'sentiment'].values
tfidf = TfidfVectorizer(strip_accents=None,lowercase=False,preprocessor=None)
param_grid = [{'tf__ngram_range':[(1,1)],
               'tf__stop_words':[stop,None],
               'tf__tokenizer':[tokenizer],
               'clf__penalty':['l1','l2'],
               'clf__C':[1.0,10.0,100.0]},
              {'tf__ngram_range':[(1,1)],
               'tf__stop_words':[stop,None],
               'tf__tokenizer':[tokenizer],
               'tf__use_idf':[False],
               'tf__norm':[None],
               'clf__penalty':['l1','l2'],
               'clf__C':[1.0,10.0,100.0]}]
mn = LogisticRegression(random_state=0)
pp = Pipeline([('tf',tfidf),('clf',mn)])
print(xtrain.shape)
gcsv = GridSearchCV(pp,param_grid,scoring='accuracy',cv=5,n_jobs=2)
gcsv.fit(xtest,ytest)
direc = os.path.join('movieclassifier','PKL')
if not os.path.exists(direc):
    os.makedirs(direc)
pickle.dump(stop,open(os.path.join(direc,'stopword.pkl'),'wb+'),protocol=4)
pickle.dump(gcsv,open(os.path.join(direc,'classifier.pkl'),'wb+'),protocol=4)










