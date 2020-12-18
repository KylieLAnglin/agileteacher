from agileteacher.library import process_text

import pandas as pd


text = "so now we're going to take ten seconds for each partner"
text_df = pd.DataFrame({"text": [text]})


result = process_text.vectorize_text(
    text_df, "text", remove_stopwords=False, tfidf=True, lemma=False, lsa=False
)
result


def test_vectorize_text():
    # no processing
    result = process_text.vectorize_text(
        text_df, "text", remove_stopwords=False, tfidf=False, lemma=False, lsa=False
    )
    assert len(result.columns) == 12

    # no stop words
    result = process_text.vectorize_text(
        text_df, "text", remove_stopwords=True, tfidf=False, lemma=False, lsa=False
    )
    assert len(result.columns) == 5

    # no stop words
    result = process_text.vectorize_text(
        text_df, "text", remove_stopwords=False, tfidf=False, lemma=True, lsa=False
    )
    assert len(result.columns) == 12
    assert "be" in list(result.columns)
