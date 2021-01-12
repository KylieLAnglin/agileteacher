# %%
import collections
import os
import re

import pandas as pd
import spacy

nlp = spacy.load("en", disable=["parser", "ner"])


def create_corpus_from_series(series: pd.Series):
    text = ""
    for row in series:
        text = text + row
    return text


def remove_tags(text: str, regex_str: str):
    text = re.sub(regex_str, " ", text)
    return text