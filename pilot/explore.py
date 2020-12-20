import text2vec

import pandas as pd
from openpyxl import load_workbook
import scipy

from agileteacher.library import start
from agileteacher.library import process_text

# %%
df = pd.read_csv(start.clean_data_path + "text.csv").set_index("id_attempt")

# %%
t2v = text2vec(list(df.text_clean))
# %%
