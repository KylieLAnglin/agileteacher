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
df = survey1[["Q82", "text"]].rename(columns={"Q82": "email"}).set_index("email")

# %%
df["clean_text"] = [
    process_text.process_text(
        text=text,
        lower_case=True,
        remove_punct=True,
        remove_stopwords=True,
        lemma=False,
    )
    for text in df.text
]

df["clean_text"] = df.clean_text.str.replace("heard", "")
df["clean_text"] = df.clean_text.str.replace("one", "")
df["clean_text"] = df.clean_text.str.replace("students", "")
df["clean_text"] = df.clean_text.str.replace("student", "")
df["clean_text"] = df.clean_text.str.replace("dev ", " ")
df["clean_text"] = df.clean_text.str.replace("jasmine ", " ")
df["clean_text"] = df.clean_text.str.replace("harrison ", " ")


matrix = process_text.vectorize_text(
    df=df,
    text_col="clean_text",
    remove_stopwords=True,
    tfidf=False,
    lemma=False,
    lsa=False,
)

df["word_count"] = df["text"].str.split().str.len()
df.word_count.mean()

# %%


# %%

# Create Dictionary
id2word = gensim.corpora.Dictionary(df["clean_text"].str.split())
corpus = [id2word.doc2bow(text) for text in df["clean_text"].str.split()]

# %%
# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(
    corpus=corpus,
    id2word=id2word,
    num_topics=2,
    random_state=100,
    update_every=1,
    chunksize=100,
    passes=10,
    alpha="auto",
    per_word_topics=True,
)
# Print the Keyword in the 10 topics
print(lda_model.print_topics())
doc_lda = lda_model[corpus]

# Visualize the topics
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
vis

## With two topics, we have topic 1 = process, topic 2 = content

# %% Three topics
# Build LDA model
id2word_3 = gensim.corpora.Dictionary(df["clean_text"].str.split())
corpus_3 = [id2word.doc2bow(text) for text in df["clean_text"].str.split()]
lda_model3 = gensim.models.ldamodel.LdaModel(
    corpus=corpus,
    id2word=id2word,
    num_topics=3,
    random_state=100,
    update_every=1,
    chunksize=100,
    passes=10,
    alpha="auto",
    per_word_topics=True,
)
# Print the Keyword in the 10 topics
print(lda_model3.print_topics())
doc_lda3 = lda_model[corpus]


pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model3, corpus_3, id2word_3)
vis
# %%
# %% CLUSTER

matrix = process_text.vectorize_text(
    df, text_col="clean_text", remove_stopwords=True, tfidf=True
)

num_clusters = 2
km = KMeans(n_clusters=num_clusters)
km.fit(matrix)
clusters = km.labels_.tolist()
df["cluster"] = clusters

grouped = df["text"].groupby(df["cluster"])


# %%
print("Top terms per cluster:")
print()
# sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
for i in range(num_clusters):
    print("Cluster %d words:" % i, end="")
    print("")
    print(
        matrix.loc[list(df[df.cluster == i].index)]
        .agg("mean")
        .sort_values(ascending=False)
        .head(10)
    )
