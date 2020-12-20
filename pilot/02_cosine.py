# %%
import os

import pandas as pd
from openpyxl import load_workbook
import scipy

from agileteacher.library import start
from agileteacher.library import process_text

# %%
df = pd.read_csv(os.path.join(start.clean_data_path, "text.csv")).set_index(
    "id_attempt"
)
df = df[["id", "attempt", "text_clean"]]


# %%
matrix = process_text.vectorize_text(
    df,
    text_col="text_clean",
    remove_stopwords=False,
    tfidf=True,
    lemma=False,
    lsa=False,
)


file = start.results_path + "Pilot Study/Cosine Replicability All Words.xlsx"
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
