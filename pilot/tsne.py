# %%
# %%
import os

import pandas as pd
import numpy as np
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import seaborn as sns

from agileteacher.library import start
from agileteacher.library import clean

# %%
text_df = pd.read_csv(os.path.join(start.clean_data_path, "text.csv"))
matrix = clean.vectorize_text(
    text_df, text_col="text", remove_stopwords=True, tfidf=False, lemma=True, lsa=True
)
# %%
tsne = TSNE(n_components=2, verbose=1, perplexity=10, n_iter=5000)
tsne_results = tsne.fit_transform(matrix)

# %%
text_df["marker_size"] = text_df.attempt ^ 3
text_df["tsne-2d-one"] = tsne_results[:, 0]
text_df["tsne-2d-two"] = tsne_results[:, 1]
plt.figure(figsize=(16, 10))
sns.scatterplot(
    x="tsne-2d-one",
    y="tsne-2d-two",
    hue="id",
    palette=sns.color_palette("hls", 7),
    data=text_df,
    legend="full",
    alpha=1,
    size="marker_size",
)

# %%
