# %%
import os

import pandas as pd
import numpy as np

from agileteacher.library import start
from agileteacher.library import qualtrics
from agileteacher.library import clean

# %% Survey Filenames
survey1_name = "*EAD Preparation Activity_November 6, 2020_18.16.csv"
survey2_name = "*EAD During Survey #1 (Reflection after Student Responses)_November 6, 2020_18.18.csv"
survey3_name = "*EAD During Survey #2 Self-Reflection_November 6, 2020_18.19.csv"
# %%
survey1_path = os.path.join(start.raw_data_path, survey1_name)
survey1_labels = qualtrics.extract_column_labels(survey1_path)
survey1 = pd.read_csv(survey1_path)
survey1 = qualtrics.select_valid_rows(
    survey=survey1, keep_previews=True, min_duration=45
)

survey2_path = os.path.join(start.raw_data_path, survey2_name)
survey2_labels = qualtrics.extract_column_labels(survey2_path)
survey2 = pd.read_csv(survey2_path)
survey2 = qualtrics.select_valid_rows(survey2, keep_previews=False, min_duration=45)
survey2 = qualtrics.drop_meta_data(survey2)

survey2 = survey2.rename(
    columns={"Q65": "hear_text", "Q66": "hope_hear_text", "Q82": "email"}
)
survey2 = survey2.loc[
    (
        (survey2.email.str.startswith("test") != True)
        & (survey2.email.str.startswith("rhonda.bondie") != True)
    )
]
survey2["survey"] = "during1"


survey3_path = os.path.join(start.raw_data_path, survey3_name)
survey3_labels = qualtrics.extract_column_labels(survey3_path)
survey3 = pd.read_csv(survey3_path)
survey3 = qualtrics.select_valid_rows(survey3, keep_previews=False, min_duration=45)
survey3 = qualtrics.drop_meta_data(survey3)
survey3 = survey3.rename(
    columns={"Q83": "hear_text", "Q84": "hope_hear_text", "Q1": "email"}
)
survey3 = survey3.loc[
    (
        (survey3.email.str.startswith("test") != True)
        & (survey3.email.str.startswith("rhonda.bondie") != True)
    )
]
survey3["survey"] = "during2"
survey3["participant"] = np.where(
    ((survey3.Q32 == "It was just me!") | (survey3.Q21 == "Giving Directions")), 1, 0
)

survey4_path = os.path.join(
    start.raw_data_path, "*EAD Post Simulation Reflection_November 6, 2020_18.17.csv"
)
survey4_labels = qualtrics.extract_column_labels(survey4_path)
survey4 = pd.read_csv(survey4_path)
survey4 = qualtrics.select_valid_rows(survey4, keep_previews=False, min_duration=45)
survey4 = qualtrics.drop_meta_data(survey4)
survey4 = survey4.rename(
    columns={"Q36": "email", "Q87": "hear_text", "Q88": "hope_hear_text"}
)
survey4 = survey4.loc[
    (survey4.email.str.startswith("test") != True)
    & (survey2.email.str.startswith("rhonda.bondie") != True)
]
survey4["survey"] = "post"
survey4["participant"] = np.where(
    (survey4.Q28 == "It was just me!") | (survey4.Q27 == "Giving Directions"), 1, 0
)
