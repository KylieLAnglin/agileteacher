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

texts = survey1.text
list(survey1.sample().text)


# %% Average length
token_lists = [nltk.word_tokenize(text) for text in texts]
lengths = [len(tokens) for tokens in token_lists]
sum(lengths) / len(lengths)


# %% Collect Bigrams and Trigrams
# Build the bigram and trigram models
bigram = gensim.models.Phrases(token_lists, min_count=3, threshold=1)
bigram_mod = gensim.models.phrases.Phraser(bigram)

text_bigrams = [bigram_mod[doc] for doc in token_lists]

# %%
text_processed = [
    process_text.process_tokens_nltk(
        tokens=text,
        lower_case=True,
        remove_punct=True,
        remove_stopwords=True,
        lemma=True,
    )
    for text in text_bigrams
]

# %%

# Create Dictionary
id2word = gensim.corpora.Dictionary(text_processed)


# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in text_processed]

# %%
# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(
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
print(lda_model.print_topics())
doc_lda = lda_model[corpus]

# %%
# Visualize the topics
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
vis

# %% CLUSTER
df = survey1[["text"]]


matrix = process_text.vectorize_text(
    df, text_col="text", remove_stopwords=True, tfidf=True
)

num_clusters = 3
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
