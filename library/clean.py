import sklearn.feature_extraction
import nltk
import string
import pandas as pd


def doc_len_list(doc_list: list, ignore_less_than: int = 0):
    """list of number of tokens in spacy docs

    Args:
        doc_list (list): list of spacy Docs
        ignore_empty (bool, optional): Include empty documents? Defaults to True.

    Returns:
        float: average number of tokens in spacy docs
    """
    lens = []
    [
        lens.append(len(doc)) if len(doc) >= ignore_less_than else lens.append(None)
        for doc in doc_list
    ]

    return lens


# Preprocessing function
def preprocessing(
    data: pd.DataFrame,
    text_col: int = -1,
    remove_stopwords: bool = False,
    filler_words: list = [],
    stem: bool = False,
    tfidf: bool = False,
    lsa: bool = False,
):
    df = data.copy()

    text = data.iloc[:, text_col]

    if remove_stopwords:
        # define stopwords
        filler_words = set(nltk.corpus.stopwords.words("english")).union(
            set(string.punctuation), set(filler_words)
        )
        if stem:  # stopwords must be removed before stemming
            # tokenize text
            text = text.apply(lambda x: nltk.tokenize.casual.casual_tokenize(x))
            # remove stopwords and stem
            text = text.apply(
                lambda x: [
                    nltk.stem.SnowballStemmer("english").stem(item)
                    for item in x
                    if item not in filler_words
                ]
            )
            # combine
            text = text.apply(lambda x: " ".join([item for item in x]))
            # set filler_words to None so stop words don't get removied twice
            filler_words = None

    if not remove_stopwords:
        filler_words = None

    if stem and not remove_stopwords:
        # tokenize text
        text = text.apply(lambda x: nltk.tokenize.casual.casual_tokenize(x))
        # stem
        text = text.apply(
            lambda x: [nltk.stem.SnowballStemmer("english").stem(item) for item in x]
        )
        # combine
        text = text.apply(lambda x: " ".join([item for item in x]))

    if tfidf and not lsa:
        vectors = sklearn.feature_extraction.text.TfidfVectorizer(
            lowercase=True, stop_words=filler_words
        ).fit_transform(text.tolist())

        dense = vectors.todense()
        denselist = dense.tolist()

        df["matrix"] = denselist
        return df

    if lsa and tfidf:
        # Add tfidf vectorizer and remove stopwords
        vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(
            lowercase=True, stop_words=filler_words
        )
        text = vectorizer.fit_transform(text)

        # Define the LSA function
        vectors = sklearn.decomposition.TruncatedSVD(n_components=100, random_state=100)

        # Convert text to vectors
        vectors.fit(text)
        svd_matrix = vectors.fit_transform(text)
        svd_matrix = sklearn.preprocessing.Normalizer(copy=False).fit_transform(text)

        dense = svd_matrix.todense()
        denselist = dense.tolist()

        df["matrix"] = denselist
        return df

    if not tfidf and not lsa:
        # Vectorization
        vectors = sklearn.feature_extraction.text.CountVectorizer(
            lowercase=True, stop_words=filler_words
        ).fit_transform(text.tolist())
        dense = vectors.todense()
        denselist = dense.tolist()
        df["matrix"] = denselist
        return df
