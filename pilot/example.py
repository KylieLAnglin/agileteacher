# %%
# %%
import os

import pandas as pd
import scipy

from agileteacher.library import start
from agileteacher.library import process_text

# %%


example1 = "All right, everybody. Hello. We are going to be introducing our space race unit now."
example2 = (
    "All right. Hello, everybody. We're going to start our unit on the space race."
)
example3 = (
    "Alrighty. Hello, everyone. So now we're going to start our unit on the space race"
)

df = pd.DataFrame({"text": [example1, example2, example3]})
# %%
