# %%
# %%
import os

import pandas as pd
import scipy
import spacy

from agileteacher.library import start
from agileteacher.library import clean_text
from agileteacher.library import process_text

nlp = spacy.load("en", disable=["parser", "ner"])
# %%
df = pd.read_csv(os.path.join(start.clean_data_path, "text.csv"))

text = text_df.loc["jonathan3"].text_clean

doc = nlp(text)

# %%