# %%
# %%
import os

import pandas as pd
import scipy
import spacy
from spacy import displacy

from agileteacher.library import start
from agileteacher.library import clean_text
from agileteacher.library import process_text


nlp = spacy.load("en_core_web_lg", disable=["parser", "ner"])

# %%
df = pd.read_csv(os.path.join(start.clean_data_path, "text.csv"))
df["new_index"] = df["id"].map(str) + df["attempt"].map(str)
df = df.set_index("new_index").sort_index()

text = df.loc["kylie3"].text_clean


doc = nlp(text)

# %%


# %%
doc1 = nlp("talk")
doc2 = nlp("speak")
doc3 = nlp("angry")
print(doc1.similarity(doc3))
# %%
doc1 = nlp(df.loc["kylie1"].text_clean)
doc3 = nlp(df.loc["kylie3"].text_clean)

print(doc1.similarity(doc3))

# %%
doc1 = nlp(df.loc["kylie1"].text_processed)
doc3 = nlp(df.loc["jonathan1"].text_processed)

print(doc1.similarity(doc3))

# %%
from sent2vec.vectorizer import Vectorizer

sentences = list(df.text_clean)
vectorizer = Vectorizer()
vectorizer.bert(sentences)
vectors = vectorizer.vectors
# %%
dist_1 = spatial.distance.cosine(vectors[0], vectors[1])
dist_2 = spatial.distance.cosine(vectors[0], vectors[2])
print("dist_1: {0}, dist_2: {1}".format(dist_1, dist_2))
assert dist_1 < dist_2
# dist_1: 0.043, dist_2: 0.192

# %%
