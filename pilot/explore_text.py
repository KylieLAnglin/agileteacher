# %%
import pandas as pd

from agileteacher.library import process_text
from agileteacher.library import start

# %%
df = pd.read_csv(os.path.join(start.clean_data_path, "text.csv")).set_index(
    "id_attempt"
)

matrix = process_text.vectorize_text(
    df,
    text_col="text_clean",
    remove_stopwords=True,
    tfidf=True,
    lemma=True,
    lsa=False,
)

words = process_text.what_words_matter(matrix, "naomi1", "naomi2", show_num=5)
