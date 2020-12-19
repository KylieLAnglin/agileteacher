# %%
import spacy

nlp = spacy.load("en_core_web_sm")

import nltk


# %%
df = pd.read_csv(os.path.join(start.clean_data_path, "text.csv")).set_index("new_index")

# %%
doc = nlp(df.loc["naomi2"]["text_clean"])
for token in doc:
    print(token.text)

" ".join([token.text for token in doc if not token.is_stop])
# %%
for token in doc:
    print(
        token.text,
        token.lemma_,
        token.pos_,
        token.tag_,
        token.dep_,
        token.shape_,
        token.is_alpha,
        token.is_stop,
    )
# %%
tokens = nltk.word_tokenize(df.loc["naomi2"]["text_clean"])
# %%
