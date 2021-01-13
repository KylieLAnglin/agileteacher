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
    assert "so" not in text_result2

    text_result3 = process_text.process_text(
        text,
        lower_case=True,
        remove_punct=True,
        remove_stopwords=False,
        lemma=True,
    )
    assert "be" in text_result3

    text_result4 = process_text.process_text(
        "text",
        lower_case=True,
        remove_punct=True,
        remove_stopwords=True,
        lemma=True,
    )
    assert "seconds" not in text_result4 and "so" not in text_result4


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


def test_ave_word_embedding_for_doc():
    test1 = "Hi, class. So next, we're going to work on a small activity."
    assert len(process_text.ave_word_embedding_for_doc(test1)) == 300


def test_doc_matrix_with_embeddings():
    result = process_text.doc_matrix_with_embeddings(text_df, "text")
    assert len(result) == 1
    assert len(result.columns == 300)


tfidf_test = pd.DataFrame({"the": [1], "cat": [2]})


def test_weighted_ave_word_embedding_for_doc():
    result = process_text.weighted_ave_word_embedding_for_doc(tfidf_test, 0)
    return result


result = test_weighted_ave_word_embedding_for_doc()
