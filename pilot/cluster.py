# %%
import os

import pandas as pd
from sklearn.cluster import KMeans

from agileteacher.library import start
from agileteacher.library import clean


# %%
df = pd.read_csv(os.path.join(start.clean_data_path, "text.csv"))
df = df[["id", "attempt", "text"]]
df["new_index"] = df["id"].map(str) + df["attempt"].map(str)
df.sample(5)

# %%
matrix = clean.vectorize_text(
    df, text_col="text", remove_stopwords=True, tfidf=True, lemma=False, lsa=False
)
# %%
num_clusters = 3
km = KMeans(n_clusters=num_clusters)
km.fit(matrix)
clusters = km.labels_.tolist()
df["cluster"] = clusters

grouped = df["text"].groupby(df["cluster"])


# %%
print("Top terms per cluster:")
print()
# sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
for i in range(num_clusters):
    print("Cluster %d words:" % i, end="")
    print("")
    print(
        matrix.loc[list(df[df.cluster == i].index)]
        .agg("mean")
        .sort_values(ascending=False)
        .head(10)
    )


# %%
# set up colors per clusters using a dict
cluster_colors = {0: "#1b9e77", 1: "#d95f02", 2: "#7570b3", 3: "#e7298a", 4: "#66a61e"}

# set up cluster names using a dict
cluster_names = {
    0: "0",
    1: "1",
    2: "Relationships",
    3: "3",
    4: "4",
}

# %%
from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(matrix)


# %%
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.manifold import MDS

MDS()

# convert two components as we're plotting points in a two-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify `random_state` so the plot is reproducible.
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

xs, ys = pos[:, 0], pos[:, 1]

# %%
new_df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, name=df.new_index)) 
groups = new_df.groupby('label')

#some ipython magic to show the matplotlib plots inline
%matplotlib inline 

# set up plot
fig, ax = plt.subplots(figsize=(17, 9)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
            label=cluster_names[name], color=cluster_colors[name], 
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    ax.tick_params(\
        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelleft='off')
    
ax.legend(numpoints=1)  #show legend with only 1 point

#add label in x,y position with the label as the film title
for i in range(len(new_df)):
    ax.text(new_df.loc[i]['x'], new_df.loc[i]['y'], new_df.loc[i]['name'], size=8)  

    
    
plt.show() #show the plot


# %%
