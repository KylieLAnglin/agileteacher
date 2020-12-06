# %%
import os

import pandas as pd
import numpy as np
import sklearn
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

from agileteacher.library import start
from agileteacher.library import clean


# %%
text_df = pd.read_csv(os.path.join(start.clean_data_path, "text.csv"))
matrix = clean.vectorize_text(
    text_df,
    text_col="text_clean",
    remove_stopwords=True,
    tfidf=True,
    lemma=True,
    lsa=False,
)

# %%
pca = PCA(n_components=7)
pca_result = pca.fit_transform(matrix)
text_df["pca-one"] = pca_result[:, 0]
text_df["pca-two"] = pca_result[:, 1]
text_df["pca-three"] = pca_result[:, 2]
text_df["pca-four"] = pca_result[:, 3]
text_df["pca-five"] = pca_result[:, 4]
print(
    "Explained variation per principal component: {}".format(
        pca.explained_variance_ratio_
    )
)


# %%
plt.figure(figsize=(16, 10))
sns.scatterplot(
    x="pca-one",
    y="pca-two",
    hue="id",
    palette=sns.color_palette("hls", 7),
    data=text_df,
    legend="full",
    alpha=1,
)

plt.savefig(start.results_path + "Pilot Study/pca_fig")
# %%
components = (
    pd.DataFrame(data=pca.components_, columns=matrix.columns)
    .transpose()
    .sort_values(0)
)
components.to_csv(start.results_path + "Pilot Study/pca.csv")
components.head()
# %%
components.sort_values(by=0, ascending=False).head(10)

# %%
components.sort_values(by=0, ascending=False).tail(10)

# %%
components.sort_values(by=1, ascending=False).head(10)

# %%
components.sort_values(by=1, ascending=False).tail(10)

# %%
