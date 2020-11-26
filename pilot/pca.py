# %%
import os

import pandas as pd
import numpy as np


from agileteacher.library import start
from agileteacher.library import clean


import time
import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
%matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
# %%
text_df = pd.read_csv(os.path.join(start.clean_data_path, "text.csv"))
matrix = clean.vectorize_text(text_df, remove_stopwords=True)

# %%
pca = PCA(n_components=5)
pca_result = pca.fit_transform(matrix)
text_df['pca-one'] = pca_result[:,0]
text_df['pca-two'] = pca_result[:,1] 
text_df['pca-three'] = pca_result[:,2]
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
