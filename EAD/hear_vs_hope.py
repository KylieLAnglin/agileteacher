# %%
import pandas as pd
import statsmodels.formula.api as smf


HEAR = start.RAW_DATA_PATH + "LIWC_Q65.csv"

HOPE = start.RAW_DATA_PATH + "LIWC_Q66.csv"
hear = pd.read_csv(HEAR)
hope = pd.read_csv(HOPE)


hear = qualtrics.select_valid_rows(hear)
hope = qualtrics.select_valid_rows(hope)

cols = qualtrics.search_column_labels(
    survey1_labels, "What did you hear students say in the discussion?"
)

hear = hear[["Q65", "RecipientEmail", "WC", "Analytic", "Clout", "Authentic", "Tone"]]
hear = hear.rename({"Q65": "text", "RecipientEmail": "email"})
hear["hear"] = 1
hear["hope"] = 0
hope = hope[
    [
        "Q66",
        "RecipientEmail",
        "WC",
        "Analytic",
        "Clout",
        "Authentic",
        "Tone",
    ]
]
hope = hope.rename({"Q66": "text", "RecipientEmail": "email"})
hope["hear"] = 0
hope["hope"] = 1
df = hear.append(hope)

# %% Compare word counts
# On average, teachers use 15 more words in describing what they heard than describing what they hope to hear
model = "WC ~ 1 + hope"
results = smf.ols(model, data=df).fit()
print(results.summary())

# %% More analytic when they talk about what they hope to hear
# a	high	number	reflects	formal,	logical,	and	hierarchical
# thinking;	lower	numbers	reflect	more	informal,	personal,	here-and-now,	and
# narrative	thinking.
model = "Analytic ~ 1 + hope"
results = smf.ols(model, data=df).fit()
print(results.summary())

# %%

# %%
###
# Clout	-- a	high	number	suggests	that	the	author	is	speaking	from	the	perspective
# of	high	expertise	and	is	confident;	low	Clout	numbers	suggest	a	more	tentative,
# humble,	even	anxious	style
##
model = "Clout ~ 1 + hope"
results = smf.ols(model, data=df).fit()
print(results.summary())

# %%
####
# Authentic	-- - higher	numbers	are	associated	with	a	more	honest,	personal,	and
# disclosing	text;	lower	numbers	suggest	a	more	guarded,	distanced	form	of
# discourse
##
model = "Authentic ~ 1 + hope"
results = smf.ols(model, data=df).fit()
print(results.summary())
# %%
# Emotional - a	high	number	is	associated	with	a	more	positive,	upbeat	style;
# a	low	number	reveals	greater	anxiety,	sadness,	or	hostility.		A	number	around	50
# suggests	either	a lack	of	emotionality	or	different	levels	of	ambivalence.

model = "Tone ~ 1 + hope"
results = smf.ols(model, data=df).fit()
print(results.summary())
# %%
