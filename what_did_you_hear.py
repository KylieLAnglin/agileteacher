import os
import pandas as pd

from agileteacher.library import start
from agileteacher.library import qualtrics

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

cols = qualtrics.search_column_labels(survey3_labels, 
'What did you hear students say in the discussion?')

# create average word count for Q83 vs Q20

# create top words for Q83 vs Q20