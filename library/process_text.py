# %%
import spacy
import nltk
import pandas as pd
import collections

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
def process_text(
    df: pd.DataFrame,
    text_col: str,
    lower_case: bool = True,
    remove_punct: bool = True,
    remove_stopwords: bool = False,
    lemma: bool = False,
):

    raw = list(df[text_col])
    docs = [" ".join([token.text for token in nlp(text)]) for text in raw]

    if lower_case:
        docs = [" ".join([token.lower_ for token in nlp(text)]) for text in raw]

    if remove_punct:
        docs = [
            " ".join([token.text for token in nlp(text) if not token.is_punct])
            for text in docs
        ]

    if remove_stopwords:
        docs = [
            " ".join([token.text for token in nlp(text) if not token.is_stop])
            for text in docs
        ]

    if lemma:
        docs = [" ".join([token.lemma_ for token in nlp(text)]) for text in docs]

    return docs


def vectorize_text(
    df: pd.DataFrame,
    text_col: str,
    remove_stopwords: bool = False,
    tfidf: bool = False,
    lemma: bool = False,
    lsa: bool = False,
):
    docs = process_text(
        df=df,
        text_col=text_col,
        lower_case=False,
        remove_punct=False,
        remove_stopwords=remove_stopwords,
        lemma=lemma,
    )

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


def create_lsa_dfs(
    matrix: pd.DataFrame, n_components: int = 100, random_state: int = 100
):

    lsa = TruncatedSVD(n_components=n_components, random_state=random_state)
    lsa_fit = lsa.fit_transform(matrix)
    lsa_fit = Normalizer(copy=False).fit_transform(lsa_fit)
    print(lsa_fit.shape)

    #  Each LSA component is a linear combo of words
    word_weights = pd.DataFrame(lsa.components_, columns=matrix.columns)
    word_weights.head()
    word_weights_trans = word_weights.T

    # Each document is a linear combination of components
    matrix_lsa = pd.DataFrame(lsa_fit, index=matrix.index, columns=word_weights.index)
    matrix_lsa.sample(5)

    word_weights = word_weights_trans.sort_values(by=[0], ascending=False)

    LSA_tuple = collections.namedtuple("LSA_tuple", ["matrix", "word_weights"])
    new = LSA_tuple(matrix_lsa, word_weights)

    return new


def create_corpus_from_series(series: pd.Series):
    text = ""
    for row in series:
        text = text + row
    return text


def remove_tags(text: str, regex_str: str):
    text = re.sub(regex_str, " ", text)
    return text