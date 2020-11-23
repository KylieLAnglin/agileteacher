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

from agileteacher import import_surveys

# %% Build
hear_df = pd.DataFrame()
hear_df["survey"] = import_surveys.survey2.survey
hear_df["email"] = import_surveys.survey2.email
hear_df["hear_text"] = import_surveys.survey2.hear_text
hear_df["hope_hear_text"] = import_surveys.survey2.hope_hear_text
hear_df = hear_df.append(
    import_surveys.survey3[
        ["survey", "email", "participant", "hear_text", "hope_hear_text"]
    ]
)
hear_df = hear_df.append(
    import_surveys.survey4[
        ["survey", "email", "participant", "hear_text", "hope_hear_text"]
    ]
)

# %% Clean and create variables

hear_df = qualtrics.replace_missing_text(
    hear_df, columns=["hear_text", "hope_hear_text"]
)
hear_df["hear_doc"] = [nlp(text) for text in hear_df.hear_text]
hear_df["hope_hear_doc"] = [nlp(text) for text in hear_df.hope_hear_text]

hear_df["survey1"] = np.where(hear_df.survey == "pre", 1, 0)
hear_df["survey2"] = np.where(hear_df.survey == "during1", 1, 0)
hear_df["survey3"] = np.where(hear_df.survey == "during2", 1, 0)
hear_df["survey4"] = np.where(hear_df.survey == "post", 1, 0)

hear_df["survey1_trt"] = hear_df.survey1 * hear_df.participant
hear_df["survey2_trt"] = hear_df.survey2 * hear_df.participant
hear_df["survey3_trt"] = hear_df.survey3 * hear_df.participant
hear_df["survey4_trt"] = hear_df.survey4 * hear_df.participant

# %%
hear_df["hear_len"] = clean.doc_len_list(hear_df.hear_doc, ignore_less_than=2)
hear_df["hope_hear_len"] = clean.doc_len_list(hear_df.hope_hear_doc, ignore_less_than=2)

# %% Means
print('overall:', hear_df.hear_len.mean())
print(hear_df[hear_df.survey1 == 1].hear_len.min())
print(hear_df[hear_df.survey2 == 1].hear_len.mean())
print(hear_df[hear_df.survey3 == 1].hear_len.mean())
print(hear_df[hear_df.survey4 == 1].hear_len.mean())

survey3 = hear_df[hear_df.survey3 == 1]
survey3.hear_len.mean()
survey3[survey3.participant==1].hear_len.mean()
survey3[survey3.participant==0].hear_len.mean()

survey4 = hear_df[hear_df.survey3 == 1]
survey4.hear_len.mean()
survey4[survey4.participant==1].hear_len.mean()
survey4[survey4.participant==0].hear_len.mean()


# %%
model = "hear_len ~ 1 + participant"
hear_df.loc[:, "const"] = 1
results = smf.ols(model, data=hear_df[(hear_df.survey3 == 1) & 
(~hear_df.participant.isnull())]).fit()
print(results.summary())

# %%
model = "hear_len ~ 1 + participant"
hear_df.loc[:, "const"] = 1
results = smf.ols(model, data=hear_df[~hear_df.participant.isnull()]).fit()
print(results.summary())

model = "hear_len ~ 1 + participant + survey3"
hear_df.loc[:, "const"] = 1
results = smf.ols(model, data=hear_df[~hear_df.participant.isnull()]).fit()
print(results.summary())
# %% Ttest survey 2 (q66) to survey 3 (q20doc)
survey2 = survey2.replace(np.nan, "", regex=True)
survey2["q66doc"] = [nlp(text) for text in survey2.Q66]
q66lens = clean.doc_len_list(survey2.q66doc, ignore_less_than=2)

print(sum(q66lens) / len(q66lens))
print(sum(q20lens) / len(q20lens))
t, p = stats.ttest_ind(q20lens, q66lens)
print(p)


# %%
