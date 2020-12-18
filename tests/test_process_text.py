from agileteacher.library import process_text

import pandas as pd


text = "So, now we're going to take ten seconds for each partner."
text_df = pd.DataFrame({"text": [text]})


result = process_text.vectorize_text(
    text_df, "text", remove_stopwords=False, tfidf=True, lemma=False, lsa=False
)
result


def test_process_text():
    text_result = process_text.process_text(
        text_df,
        "text",
        lower_case=True,
        remove_punct=True,
        remove_stopwords=False,
        lemma=False,
    )
    assert text_result == ["so now we 're going to take ten seconds for each partner"]

    text_result2 = process_text.process_text(
        text_df,
        "text",
        lower_case=True,
        remove_punct=True,
        remove_stopwords=True,
        lemma=False,
    )
    assert "so" not in text_result2[0]

    text_result3 = process_text.process_text(
        text_df,
        "text",
        lower_case=True,
        remove_punct=True,
        remove_stopwords=False,
        lemma=True,
    )
    assert "be" in text_result3[0]

    text_result4 = process_text.process_text(
        text_df,
        "text",
        lower_case=True,
        remove_punct=True,
        remove_stopwords=True,
        lemma=True,
    )
    assert "seconds" not in text_result4[0] and "so" not in text_result4[0]


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
