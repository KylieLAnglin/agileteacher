# %%
import os

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm

import pystout


from spacy.lang.en import English
nlp = English()

from agileteacher.library import start
from agileteacher.library import qualtrics
from agileteacher.library import clean

# %%
# What did you hear not present.
survey1_path = os.path.join(start.raw_data_path, 
"*EAD Preparation Activity_November 6, 2020_18.16.csv")
survey1_labels = qualtrics.extract_column_labels(survey1_path)
survey1 = qualtrics.select_valid_rows(survey1_path)

survey2_path = os.path.join(start.raw_data_path, 
"*EAD During Survey #1 (Reflection after Student Responses)_November 6, 2020_18.18.csv")
survey2_labels = qualtrics.extract_column_labels(survey2_path)
survey2 = qualtrics.select_valid_rows(survey2_path)
survey2 = survey2.loc[((~survey2.Q65.isnull()) | (~survey2.Q66.isnull())) &
                    (~survey2.Q82.str.startswith('test'))]
survey2 = qualtrics.drop_meta_data(survey2)

survey3_path = os.path.join(start.raw_data_path, 
"*EAD During Survey #2 Self-Reflection_November 6, 2020_18.19.csv")
survey3_labels = qualtrics.extract_column_labels(survey3_path)
survey3 = qualtrics.select_valid_rows(survey3_path)
survey3 = qualtrics.drop_meta_data(survey3)
