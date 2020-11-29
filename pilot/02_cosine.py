# %%
import os

import pandas as pd
from openpyxl import load_workbook
import scipy

from agileteacher.library import start
from agileteacher.library import clean

# %%
df = pd.read_csv(os.path.join(start.clean_data_path, "text.csv"))
df = df[["id", "attempt", "text_clean"]]

df["new_index"] = df["id"].map(str) + df["attempt"].map(str)
df = df.set_index("new_index").sort_index()

# %%
matrix = clean.vectorize_text(
    df, text_col="text_clean", remove_stopwords=True, tfidf=True, lemma=True, lsa=False
)

# %%
file = start.results_path + "Pilot Study/Cosine Replicability.xlsx"
wb = load_workbook(file)
ws = wb.active
# %%
col = 2
for main in list(df.index):
    row = 2
    for comp in list(df.index):
        dist = 1 - scipy.spatial.distance.cosine(matrix.loc[main], matrix.loc[comp])
        ws.cell(row=row, column=col).value = round(dist, 2)
        row = row + 1
    col = col + 1

wb.save(file)

# %%
