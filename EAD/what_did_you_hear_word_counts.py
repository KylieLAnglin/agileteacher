# %%
import os

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
import gensim
import nltk
import spacy

import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
import pystout
from sklearn.cluster import KMeans

from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

from spacy.lang.en import English

nlp = English()

from agileteacher.library import start
from agileteacher.library import qualtrics
from agileteacher.library import clean_text
from agileteacher.library import process_text


SURVEY1_PATH = (
    start.RAW_DATA_PATH
    + "EAD During Survey #1 (Reflection after Student Responses)_April 23, 2021_12.34.csv"
)

# %%
survey1 = pd.read_csv(SURVEY1_PATH)
survey1_labels = qualtrics.extract_column_labels(SURVEY1_PATH)
survey1 = qualtrics.select_valid_rows(survey1)

cols = qualtrics.search_column_labels(
    survey1_labels, "What did you hear students say in the discussion?"
)

survey1["text"] = survey1.Q65

texts = list(survey1.text)
texts_cleaned = [
    process_text.process_text_nltk(
        text=text,
        lower_case=True,
        remove_punct=True,
        remove_stopwords=True,
        lemma=False,
    )
    for text in texts
]

corpus = ""
for text in texts_cleaned:
    corpus = corpus + text

allWords = nltk.tokenize.word_tokenize(corpus)
allWordDist = nltk.FreqDist(w.lower() for w in allWords)
mostCommon = allWordDist.most_common(50)
mostCommon


# %% Vectorize
df = survey1[["Q82", "text"]].rename(columns={"Q82": "email"}).set_index("email")
df["clean_text"] = [
    process_text.process_text_nltk(
        text, lower_case=True, remove_punct=True, remove_stopwords=True, lemma=False
    )
    for text in df.text
]

matrix = process_text.vectorize_text(
    df=df,
    text_col="clean_text",
    remove_stopwords=True,
    tfidf=False,
    lemma=False,
    lsa=False,
)

print(len(list(matrix.columns)))


# %% Cluster
clusters = linkage(matrix, method="ward")
fig = plt.figure(figsize=(25, 10))
dn = dendrogram(clusters)

# Two clusters

cluster = AgglomerativeClustering(n_clusters=2, affinity="euclidean", linkage="ward")
cluster.fit_predict(matrix)

df["cluster2"] = cluster.fit_predict(matrix)

# %% Word counts in each cluster
# Corpus 1


corpus1 = ""
for text in df[df.cluster2 == 0].clean_text:
    corpus1 = corpus1 + text


words1 = nltk.tokenize.word_tokenize(corpus1)
word_distribution1 = nltk.FreqDist(w.lower() for w in words1)
most_common_words1 = word_distribution1.most_common(10)
most_common_words1

corpus2 = ""
for text in df[df.cluster2 == 1].text:
    corpus2 = corpus + text
words2 = nltk.tokenize.word_tokenize(corpus2)
word_distribution2 = nltk.FreqDist(w.lower() for w in words2)
most_common_words2 = word_distribution2.most_common(10)
most_common_words2
