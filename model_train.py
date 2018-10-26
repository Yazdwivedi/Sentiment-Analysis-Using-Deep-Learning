import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def preparedata(data):
    l=[]
    for i in range(len(data)):
        l.append(data.iloc[i,-1].split("\t"))  

    a=[y[1] for y in l]    
    a=pd.DataFrame(a)    

    b=[y[0] for y in l]
    b=pd.DataFrame(b)

    data=pd.concat([b,a],axis=1)    
    return data

data1=pd.read_csv("amazon_cells_labelled.txt",delimiter="\n",header=None)
data1=preparedata(data1)

data2=pd.read_csv("imdb_labelled.txt",delimiter="\n",header=None)
data2=preparedata(data2)

data3=pd.read_csv("yelp_labelled.txt",delimiter="\n",header=None)
data3=preparedata(data3)

data=pd.concat([data1,data2,data3])
        

data=data.reset_index()
del data["index"]

values=[]
for i in range(len(data.values[:,1])):
    if len(data.values[i,1])>1:
        values.append(i)

data.drop(values,inplace=True)        


from nltk.corpus import stopwords
import re
import nltk
corpus=[]

for x in data.values:
    temp=re.sub("[^a-zA-Z]"," ",x[0])
    temp=nltk.word_tokenize(temp)
    temp=[l for l in temp if l not in set(stopwords.words("english"))]
    corpus.append(temp)

y=data.iloc[:,1].values

y=y.astype("float64")

import gensim

wordvec=gensim.models.Word2Vec.load("word2vec.bin")

for i in range(len(corpus)):
    if len(corpus[i])>8:
        corpus[i][8:]=[]


vec_x=[]
for sent in corpus:
    sent_x=[wordvec[l] for l in sent if l in wordvec.wv.vocab]
    vec_x.append(sent_x)
            
sentend=np.ones((300,),dtype=np.float32)

for i in range(len(vec_x)):
    if len(vec_x[i])<8:
        for j in range(8-len(vec_x[i])):
            vec_x[i].append(sentend)

vec_x=np.array(vec_x)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(vec_x,y,test_size=0.1)

            
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout

model=Sequential()

model.add(LSTM(150,input_shape=X_train.shape[1:],return_sequences=True,activation="relu"))
model.add(Dropout(0.3))
model.add(LSTM(150,return_sequences=True,activation="relu"))
model.add(Dropout(0.3))
model.add(LSTM(150,return_sequences=True,activation="relu"))
model.add(Dropout(0.3))
model.add(LSTM(150,activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(64,activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(1,activation="sigmoid"))

model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])

model.fit(X_train,y_train,epochs=4,batch_size=64,validation_data=(X_test,y_test)) 

model.save("sentiment.h5");
    




