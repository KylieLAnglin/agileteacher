# %%
import spacy
import nltk
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from agileteacher.library import start

nlp = spacy.load("en", disable=["parser", "ner"])

# %% Replace spacy stop word list with nltks
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
nltk_stopwords = set(nltk.corpus.stopwords.words("english"))

for word in spacy_stopwords:
    if word not in nltk_stopwords:
        if not any(substring in word for substring in ["‘", "’", "'"]):
            lexeme = nlp.vocab[word]
            lexeme.is_stop = False

# %%
def vectorize_text(
    df: pd.DataFrame,
    text_col: str,
    remove_stopwords: bool = False,
    tfidf: bool = False,
    lemma: bool = False,
    lsa: bool = False,
):
    docs = list(df[text_col])
    if (remove_stopwords == False) & (lemma == False):
        docs = [" ".join([token.text for token in nlp(text)]) for text in df[text_col]]

    elif (remove_stopwords == True) & (lemma == False):
        docs = [
            " ".join([token.text for token in nlp(text) if not token.is_stop])
            for text in df[text_col]
        ]

    elif (remove_stopwords == False) & (lemma == True):
        docs = [
            " ".join([token.lemma_ for token in nlp(text)]) for text in df[text_col]
        ]

    elif (remove_stopwords == True) & (lemma == True):
        docs = [
            " ".join([token.lemma_ for token in nlp(text) if not token.is_stop])
            for text in df[text_col]
        ]

    if tfidf == False:
        vec = CountVectorizer()

    elif tfidf:
        vec = TfidfVectorizer()

    X = vec.fit_transform(docs)
    matrix = pd.DataFrame(X.toarray(), columns=vec.get_feature_names(), index=df.index)

    print("Number of words: ", len(matrix.columns))

    if lsa:
        lsa_dfs = create_lsa_dfs(matrix=matrix)
        matrix = lsa_dfs.matrix
        print("Number of dimensions: ", len(matrix.columns))

    return matrix