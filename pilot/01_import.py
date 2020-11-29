# %%
import os
import re

import pandas as pd
import numpy as np


from agileteacher.library import start
from agileteacher.library import clean

# %%
column_names = {"Quote from transcript": "quote", "Coding Period": "period"}
text = {}
# %%
def create_participant_df(df: pd.DataFrame, name: str):
    df = df.rename(columns=column_names)
    df = df[["period", "quote"]]
    df = df.loc[~df.period.isnull()]

    i = 1
    attempts = []
    texts = []
    for attempt in [
        "First Directions Elements",
        "Second Directions Elements",
        "Third Directions Elements",
    ]:
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

df = pd.read_excel(
    os.path.join(start.raw_data_path, "pilot_study_data.xlsx"),
    sheet_name="Esteban - lower codes and time",
    skiprows=5,
)
text_df = create_participant_df(df=df, name="esteban")

participant_list = [
    ("Jonathan - lower codes and time", "jonathan"),
    ("Kaycie -lower codes and time (H", "kaycie"),
    ("Kylie - lower codes + time  (Ha", "kylie"),
    ("Miles -lower codes + time (Happ", "miles"),
    (" Naomi - lower codes + time (Ha", "naomi"),
    ("Samantha - lower codes + time (", "samantha"),
]
for participant in participant_list:
    df = df = pd.read_excel(
        os.path.join(start.raw_data_path, "pilot_study_data.xlsx"),
        sheet_name=participant[0],
        skiprows=5,
    )
    text_df = text_df.append(create_participant_df(df=df, name=participant[1]))

text_df["new_index"] = text_df["id"].map(str) + text_df["attempt"].map(str)
text_df = text_df.set_index("new_index").sort_index()
# %%
text_df["text_clean"] = [clean.remove_tags(txt, r"\[(.*?)\]") for txt in text_df.text]

text_df["text_clean"] = [re.sub(r"30", "thirty", txt) for txt in text_df.text_clean]
text_df["text_clean"] = [re.sub(r"20", "twenty", txt) for txt in text_df.text_clean]
text_df["text_clean"] = [re.sub(r"10", "ten", txt) for txt in text_df.text_clean]


# %%
text_df.to_csv(os.path.join(start.clean_data_path, "text.csv"))
