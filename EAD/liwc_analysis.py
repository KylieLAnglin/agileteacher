# %%
import os
import re

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
import liwc
from collections import Counter

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

survey1 = pd.read_csv(SURVEY1_PATH)
survey1_labels = qualtrics.extract_column_labels(SURVEY1_PATH)
survey1 = qualtrics.select_valid_rows(survey1)

cols = qualtrics.search_column_labels(
    survey1_labels, "What did you hear students say in the discussion?"
)

survey1["text"] = survey1.Q65

texts = survey1.text
texts = [text.lower() for text in survey1.texts]
# %%
def tokenize(text):
    tokens = nltk.word_tokenize(text)
    return tokens


# def tokenize(text):
#     # you may want to use a smarter tokenizer
#     for match in re.finditer(r"\w+", text, re.UNICODE):
#         yield match.group(0)


parse, category_names = liwc.load_token_parser("/Users/kylie/LIWC.dic")

# %%
tokens = tokenize(test)

from collections import Counter

counts = Counter(category for token in tokens for category in parse(token))
print(counts["function"])
# => Counter({'funct': 58, 'pronoun': 18, 'cogmech': 17, ...})

# %%
# Compare what did you hear to what did you hope to hear
