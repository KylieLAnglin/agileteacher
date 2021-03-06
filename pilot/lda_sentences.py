# %%
import os

import pandas as pd
import numpy as np
import scipy
import gensim
import nltk
import spacy
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import spacy

nlp = spacy.load("en_core_web_sm")

from agileteacher.library import start
from agileteacher.library import clean

# %%
text_df = pd.read_csv(os.path.join(start.clean_data_path, "text.csv"))
text = clean.create_corpus_from_series(text_df.text_clean)


# %% Create list of sentences
def sent_to_words(sentences):
    for sentence in sentences:
        yield (
            gensim.utils.simple_preprocess(str(sentence), deacc=True)
        )  # deacc=True removes punctuations


text_sentences = nlp(text)
for sentence in text_sentences.sents:
    print(sentence.text)

# %%


data_words = list(sent_to_words(text_sentences.sents))

# %%

# Build the bigram and trigram models
bigram = gensim.models.Phrases(
    data_words, min_count=3, threshold=20
)  # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=20)

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
print(trigram_mod[bigram_mod[data_words[6]]])

# %%
# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    stop_words = nltk.corpus.stopwords.words("english")
    return [
        [
            word
            for word in gensim.utils.simple_preprocess(str(doc))
            if word not in stop_words
        ]
        for doc in texts
    ]


def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


def lemmatization(texts, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append(
            [token.lemma_ for token in doc if token.pos_ in allowed_postags]
        )
    return texts_out


# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

nlp = spacy.load("en", disable=["parser", "ner"])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(
    data_words_bigrams, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]
)

print(data_lemmatized[:1])
# %%

# Create Dictionary
id2word = gensim.corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
print(corpus[:1])

# [[(id2word[id], freq) for id, freq in cp] for cp in corpus]
# %%
# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(
    corpus=corpus,
    id2word=id2word,
    num_topics=20,
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

# %%
lda_

# %%
def lda_to_csv(model, corpus):
    """This function takes a gensim lda model as input, and outputs a csv with topics probs by document"""
    topic_probs = model.get_document_topics(
        corpus
    )  # get the list of topic probabilities by doc
    topic_dict = [
        dict(x) for x in topic_probs
    ]  # convert to dictionary to convert to data frame
    df = pd.DataFrame(topic_dict).fillna(
        0
    )  # convert to data frame, fill topics < 0.01 as 0
    return df


docs = lda_to_csv(lda_model, corpus)

# %%
docs.sort_values(by=0, ascending=False)
data_words[349]

# %%
data_words[82]

# %%
data_words[631]

# %%
docs.sort_values(by=1, ascending=False)

# %%
data_words[405]


# %%
data_words[609]

# %%
data_words[381]

# %%
docs.sort_values(by=2, ascending=False)
data_words[458]
data_words[501]


# %%
