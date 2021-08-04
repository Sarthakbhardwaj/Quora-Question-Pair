import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import numpy as np
from gensim.models import KeyedVectors
from sklearn.cross_decomposition import CCA
import seaborn as sns

df=pd.read_csv("../input/question-pairs-dataset/questions.csv")
print("number of data points",df.shape[0])

df.info()

df.isnull().sum(


df=df.dropna()
df.isnull().sum()

nan_rows=df[df.isnull().any(1)]
print(nan_rows)

df.groupby('is_duplicate')['id'].count().plot.bar()


print("question pairs that are not similar {}".format(100-round(df['is_duplicate']).mean()*100,2))
print("question pairs that are similar {}".format(round(df['is_duplicate']).mean()*100))

vecfile = "../input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin"
vecs = KeyedVectors.load_word2vec_format(vecfile, binary =True)
vecs.init_sims(replace=True)


import nltk
nltk.download('punkt')
nltk.download('stopwords')

pip install inflect

import inflect
import re
from nltk.corpus import stopwords
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from scipy.stats.mstats import spearmanr as spr

def remove_punctuation(words):
    #new_words = []
    #for word in wordings:
    new_word = re.sub(r'[^\w\s]', "", words)

    return new_word

def replace_numbers(words):
    p = inflect.engine()
    new_word = words.split()
    new_words=[]
    for word in new_word:
        if word.isdigit():
            new = p.number_to_words(word)
            new_words.append(new)
        else:
            new_words.append(word)
    return new_words

def remove_stopwords(words):
    words= [w for w in words if not w in stopwords.words('english')]
    return words

def normalize(words):
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words= remove_stopwords(words)
    return words

print(df['question1'][0])
print(df['question2'][0])

normalize(df['question1'][0])

from wordcloud import WordCloud
import matplotlib.pyplot as plt

qw = " ".join(review for review in df.question1)
print ("There are {} words in the combination of all review.".format(len(qw)))
qe = " ".join(review for review in df.question2)
print ("There are {} words in the combination of all review.".format(len(qe)))

wordcloud = WordCloud(background_color="white").generate(qw)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

wordcloud = WordCloud(background_color="white").generate(qe)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

def compute_cosine_similarity(vector1, vector2):
    cos=vector1.dot(vector2)/(np.linalg.norm(vector1, ord=2) * np.linalg.norm(vector2, ord=2))
    if np.isnan(cos):
        return 0.500
    else:
        return cos

def cca_sem(df, vecs):
    #write_str  = []
    sims = []
    sim = 0
    original_value=[]
    for i in range(500):
        sent1 = df['question1'][i]
        sent2 = df['question2'][i]
        
        original_value.append(df['is_duplicate'][i])
        words_in_sent1 = normalize(sent1)
        words_in_sent2 = normalize(sent2)
        stems1 = []
        for word in words_in_sent1:
            if word in vecs:
                stems1.append(word)
        stems2 = []
        for word in words_in_sent2:
            if word in vecs:
                stems2.append(word)

    
        len_1 = len(stems1)
        len_2 = len(stems2)
        len_min = min(len_1, len_2,8)

        
        sim=[]
        if len_min == 1:
            sims.append(0.500)
    
        v1 = np.asarray(vecs["hi"])
        for word in stems1:
            x = np.asarray(vecs[word])
            v1 = np.vstack((v1, x))

        v2 = np.asarray(vecs["hi"])
        for word in stems2:
            x = np.asarray(vecs[word])
            v2 = np.vstack((v2, x))
    
        v1 = np.delete(v1, 0, 0)
        v2 = np.delete(v2, 0, 0)
        b = len_min
        cca = CCA(n_components=b,max_iter=15000)
        cca.fit(v1.T,v2.T)
        X_c, Y_c = cca.transform(v1.T, v2.T)        
        sim =0

        X_T = X_c.T
        Y_T = Y_c.T
        for i in range(b):
            v1 = []
            v1 = X_T[i]
            w11 = np.asarray(v1)
            v2 = []
            v2 = Y_T[i]
            w21 = np.asarray(v2)
            sim_1 = compute_cosine_similarity(v1, v2)
            sim =sim + sim_1
            
        sim = sim/b
        sims.append(sim)
       

    print(sims)
    count=0
    arrr=[]
    for i in range(len(sims)):
        if sims[i]>0.7 or sims[i]<0.001:
            arrr.append(1)
        else:
            arrr.append(0)
    for i in range(len(sims)):
        if df['is_duplicate'][i]==arrr[i]:
            count+=1
    print(arrr)
    #print(count)
    ann=count/500
    print(ann*100,"% results matched")

df=pd.read_csv("../input/question-pairs-dataset/questions.csv")
cca_sem(df,vecs)


