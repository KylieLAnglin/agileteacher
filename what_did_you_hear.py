# %%
import os

import pandas as pd
import numpy as np
from scipy import stats

from spacy.lang.en import English
nlp = English()

from agileteacher.library import start
from agileteacher.library import qualtrics
from agileteacher.library import clean

# %%
survey1_path = os.path.join(start.raw_data_path, 
"*EAD Preparation Activity_November 6, 2020_18.16.csv")

survey1_labels = qualtrics.extract_column_labels(survey1_path)
survey1 = qualtrics.select_valid_rows(survey1_path)

survey2_path = os.path.join(start.raw_data_path, 
"*EAD During Survey #1 (Reflection after Student Responses)_November 6, 2020_18.18.csv")
survey2_labels = qualtrics.extract_column_labels(survey2_path)
survey2 = qualtrics.select_valid_rows(survey2_path)
survey2 = qualtrics.drop_meta_data(survey2)

survey3_path = os.path.join(start.raw_data_path, 
"*EAD During Survey #2 Self-Reflection_November 6, 2020_18.19.csv")
survey3_labels = qualtrics.extract_column_labels(survey3_path)
survey3 = qualtrics.select_valid_rows(survey3_path)
survey3 = qualtrics.drop_meta_data(survey3)

# %%
cols = qualtrics.search_column_labels(survey3_labels, 
'What did you hear students say in the discussion?')
cols = cols + qualtrics.search_column_labels(survey3_labels,
'Did you have a partner or did you participate today by yourself?')
cols = cols + qualtrics.search_column_labels(survey3_labels, 
'Were you observing or giving directions in this simulation?')

survey3 = survey3[cols]

survey3 = survey3.replace(np.nan, '', regex=True)

survey3['q20doc'] = [nlp(text) for text in survey3.Q20]
q20lens = clean.doc_len_list(survey3.q20doc)
#print(clean.ave_num_words(list(survey3.Q20doc)))

survey3['q83doc'] = [nlp(text) for text in survey3.Q83]
q83lens = clean.doc_len_list(survey3.q83doc)

t, p = stats.ttest_ind(q20lens, q83lens)
# %%
qualtrics.search_column_labels(survey1_labels, 
'What did you hear students say in the discussion?')

# %%
