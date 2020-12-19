from agileteacher.library import process_text

import pandas as pd


text = "So, now we're going to take ten seconds for each partner."
text_df = pd.DataFrame({"text": [text]})


def test_process_text():
    text_result = process_text.process_text(
        text,
        lower_case=True,
        remove_punct=True,
        remove_stopwords=False,
        lemma=False,
    )
    assert text_result == "so now we 're going to take ten seconds for each partner"

    text_result2 = process_text.process_text(
        text,
        lower_case=True,
        remove_punct=True,
        remove_stopwords=True,
        lemma=False,
    )
    assert "so" not in text_result2[0]

    text_result4 = process_text.process_text(
        text,
        lower_case=True,
        remove_punct=True,
        remove_stopwords=True,
        lemma=True,
    )
    print(text_result4)
    assert "-pron-" not in text_result4


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
    assert "so" not in list(result.columns)

    result = process_text.vectorize_text(
        text_df, "text", remove_stopwords=False, tfidf=False, lemma=True, lsa=False
    )
    assert len(result.columns) == 12
    assert "be" in list(result.columns)
