import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
import re
import nltk

from keras.models import load_model 

model=load_model("sentiment.h5")

inp=re.sub("[^a-zA-Z]"," ",inp)
value=nltk.word_tokenize(inp)
value=[wordvec[x] for x in value if x not in set(stopwords.words("english")) and x in wordvec.wv.vocab]
final=[]
final.append(value)

for i in range(len(final)):
    if len(final[i])<8:
        for j in range(8-len(final[i])):
            final[i].append(sentend)

final=np.array(final)

pred=model.predict(final)

if(pred>0.5):
    print("Positive")
else:
    print("Negative")    
