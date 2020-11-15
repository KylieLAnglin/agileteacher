import os
import pandas as pd

from agileteacher.library import start
from agileteacher.library import qualtrics

survey1_path = os.path.join(start.raw_data_path, 
"*EAD Preparation Activity_November 6, 2020_18.16.csv")

survey1_labels = qualtrics.extract_column_labels(survey1_path)
survey1 = qualtrics.select_valid_rows(survey1_path)


