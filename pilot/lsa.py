# %%
import os

import pandas as pd

from agileteacher.library import start
from agileteacher.library import clean


# %%
df = pd.read_csv(os.path.join(start.clean_data_path, "text.csv"))
df = df[["id", "attempt", "text"]]
df.sample(5)

# %%

matrix = clean.vectorize_text(
    df, text_col="text", remove_stopwords=True, tfidf=True, lemma=False, lsa=False
)
lsas = clean.create_lsa_dfs(matrix=matrix, n_components=21)
lsas.word_weights.head(30)
# %%
