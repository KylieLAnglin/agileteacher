# %%
import os
import re
from collections import namedtuple

import pandas as pd
import numpy as np


from agileteacher.library import start
from agileteacher.library import process_text
from agileteacher.library import clean_text

# %%
RAW_EXCEL_FILE_PATH = start.RAW_DATA_PATH + "pilot_study_data.xlsx"

COLUMN_NAMES = {"Quote from transcript": "quote", "Coding Period": "period"}

PARTICIPANTS = ["esteban", "jonathan", "kaycie", "kylie", "miles", "naomi", "samantha"]

PARTICIPANT_SHEETS = {
    "esteban": "Esteban - lower codes and time",
    "jonathan": "Jonathan - lower codes and time",
    "kaycie": "Kaycie -lower codes and time (H",
    "kylie": "Kylie - lower codes + time  (Ha",
    "miles": "Miles -lower codes + time (Happ",
    "naomi": " Naomi - lower codes + time (Ha",
    "samantha": "Samantha - lower codes + time (",
}


ATTEMPTS = [
    "First Directions Elements",
    "Second Directions Elements",
    "Third Directions Elements",
]

OUTPUT_FILE = start.CLEAN_DATA_PATH + "text.csv"

text = {}
# %%
def create_participant_df(df: pd.DataFrame, name: str):
    df = df.rename(columns=COLUMN_NAMES)
    df = df[["period", "quote"]]
    df = df.loc[~df.period.isnull()]

    i = 1
    attempts = []
    texts = []
    for attempt in ATTEMPTS:
        text = ""
        for row in df[df.period == attempt].quote:
            text = text + row
        attempts.append(i)
        i = i + 1
        texts.append(text)

    new_df = pd.DataFrame()
    new_df["attempt"] = attempts
    new_df["text"] = texts
    new_df["id"] = name

    return new_df[["id", "attempt", "text"]]


# %%


text_df = pd.DataFrame(columns=["id", "attempt", "text"])

for participant in PARTICIPANTS:
    df = pd.read_excel(
        RAW_EXCEL_FILE_PATH,
        sheet_name=PARTICIPANT_SHEETS[participant],
        skiprows=5,
        engine="openpyxl",
    )
    text_df = text_df.append(create_participant_df(df=df, name=participant))

text_df["id_attempt"] = text_df["id"].map(str) + text_df["attempt"].map(str)
text_df = text_df.set_index("id_attempt").sort_index()


# %%
text_df["text_clean"] = [
    clean_text.remove_tags(txt, r"\[(.*?)\]") for txt in text_df.text
]

text_df["text_clean"] = [re.sub(r"30", "thirty", txt) for txt in text_df.text_clean]
text_df["text_clean"] = [re.sub(r"20", "twenty", txt) for txt in text_df.text_clean]
text_df["text_clean"] = [re.sub(r"10", "ten", txt) for txt in text_df.text_clean]
text_df["text_clean"] = [
    re.sub(r"35", "thirty-five", txt) for txt in text_df.text_clean
]


text_df["text_clean"] = [
    clean_text.add_whitespace_after_punct(txt) for txt in text_df.text_clean
]

text_df["text_clean"] = [
    clean_text.remove_trailing_hyphen(txt) for txt in text_df.text_clean
]

text_df["text_clean"] = [re.sub(r"/\n/", "", txt) for txt in text_df.text_clean]

# %%

text_df["text_processed"] = [
    process_text.process_text(
        text=text,
        lower_case=True,
        remove_punct=True,
        remove_stopwords=True,
        lemma=True,
    )
    for text in text_df.text_clean
]

# %%
text_df.to_csv(OUTPUT_FILE)
