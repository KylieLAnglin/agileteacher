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
survey2 = qualtrics.drop_meta_data(survey2)
survey2 = survey2.rename(columns = {'Q65':'hear_text', 'Q66':'hope_hear_text',
                                    'Q82':'email'})
survey2 = survey2.loc[((~survey2.hear_text.isnull()) | (~survey2.hope_hear_text.isnull())) &
                    (survey2.email.str.startswith('test') != True) &
                    (survey2.email.str.startswith('rhonda.bondie') != True)]
survey2['survey'] = 'during1'


survey3_path = os.path.join(start.raw_data_path, 
"*EAD During Survey #2 Self-Reflection_November 6, 2020_18.19.csv")
survey3_labels = qualtrics.extract_column_labels(survey3_path)
survey3 = qualtrics.select_valid_rows(survey3_path)
survey3 = qualtrics.drop_meta_data(survey3)
survey3 = survey3.rename(columns = {'Q83':'hear_text', 'Q84':'hope_hear_text',
                                    'Q1': 'email'})
survey3 = survey3.loc[((~survey3.hear_text.isnull()) | (~survey3.hope_hear_text.isnull())) &
                    (survey3.email.str.startswith('test') != True) &
                    (survey2.email.str.startswith('rhonda.bondie') != True)]
survey3['survey'] = 'during2'


survey4_path = os.path.join(start.raw_data_path, 
"*EAD Post Simulation Reflection_November 6, 2020_18.17.csv")
survey4_labels = qualtrics.extract_column_labels(survey4_path)
survey4 = qualtrics.select_valid_rows(survey4_path)
survey4 = qualtrics.drop_meta_data(survey4)
survey4 = survey4.rename(columns = {'Q1': 'email'})
survey4 = survey4.loc[
                    (survey4.email.str.startswith('test') != True) &
                    (survey2.email.str.startswith('rhonda.bondie') != True)]
survey4['survey'] = 'post'

