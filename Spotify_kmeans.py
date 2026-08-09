import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import os

os.makedirs('images', exist_ok=True)


if not os.path.exists("dataset_sample.csv"):
    print("creating sample __")
    df_full = pd.read_csv("dataset.csv")
    df_sample = df_full.sample(n=10000, random_state=42)
    df_sample.to_csv("dataset_sample.csv", index=False)
    print("sample created!")
else:
    print("sample already exists - skipping")


# load the dataset
df = pd.read_csv('dataset_sample.csv')
df.columns = df.columns.str.strip()

# just checking whats in here
print(df.shape)
print(df.columns.tolist())
df.info()
print(df.describe())

# checking for nulls and dupes
print("\nmissing values:")
print(df.isnull().sum())
print("duplicate rows:", df.duplicated().sum())


# drop rows where track name is missing
df = df.dropna(subset=['track_name'])

# these are the features i want to cluster on
# tried including duration_ms at first but it messed up the clusters
features = [
    'danceability', 'energy', 'key', 'loudness',
    'mode', 'speechiness', 'acousticness',
    'instrumentalness', 'liveness', 'valence',
    'tempo', 'time_signature'
]

show_cols = ['track_id', 'track_name', 'artists', 'album_name', 'popularity']

print("\nnulls in audio features:")
print(df[features].isnull().sum())

df = df.dropna(subset=features)
print("rows left:", len(df))

# sampling because running elbow on 100k rows takes a long time
sample = df.sample(n=min(10000, len(df)), random_state=42)
X_sample = sample[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sample)


# elbow method to find good k
inertias = []
k_vals = range(2, 21)

for k in k_vals:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    print(f"k={k} inertia={km.inertia_:.0f}")

plt.figure(figsize=(10, 5))
plt.plot(k_vals, inertias, 'bo-')
plt.xlabel('k')
plt.ylabel('inertia')
plt.title('elbow plot')
plt.grid(True)
plt.savefig('images/elbow.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close()


# checking the drops manually
drops = np.diff(inertias)
for i, d in enumerate(drops, start=2):
    print(f"k={i} to k={i+1}: {abs(d):.0f}")


# silhouette - higher is better, checking k=4 to 8
sil_scores = []
for k in range(4, 9):
    km2 = KMeans(n_clusters=k, random_state=42, n_init=10)
    labs = km2.fit_predict(X_scaled)
    sc = silhouette_score(X_scaled, labs, sample_size=5000, random_state=42)
    sil_scores.append(sc)
    print(f"k={k} silhouette={sc:.3f}")

plt.figure(figsize=(7, 4))
plt.plot(range(4, 9), sil_scores, 'go-')
plt.xlabel('k')
plt.ylabel('silhouette score')
plt.title('silhouette vs k')
plt.grid(True)
plt.savefig('images/silhouette.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close()


# k=6 looked like the best balance from both plots
best_k = 6

# now fitting on the full dataset not just the sample
X_full = df[features]
X_full_scaled = scaler.fit_transform(X_full)

final_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['cluster'] = final_model.fit_predict(X_full_scaled)

print("\ncluster sizes:")
print(df['cluster'].value_counts().sort_index())


# looking at what each cluster actually sounds like
print("\naverage features per cluster:")
print(df.groupby('cluster')[features].mean().round(3))

# printing some example songs per cluster
for c in range(best_k):
    print(f"\ncluster {c} — {(df['cluster'] == c).sum()} songs")
    top = df[df['cluster'] == c].sort_values('popularity', ascending=False)
    print(top[show_cols].head(5).to_string(index=False))


# recommendation function
# finds the song then returns other songs in the same cluster
def get_recommendations(song, n=10):
    # exact match
    result = df[df['track_name'].str.lower() == song.lower()]

    # if nothing found try partial match
    if result.empty:
        result = df[df['track_name'].str.lower().str.contains(song.lower(), na=False)]

    if result.empty:
        print(f"couldnt find '{song}'")
        return

    c = result.iloc[0]['cluster']
    name = result.iloc[0]['track_name']

    recs = df[(df['cluster'] == c) & (df['track_name'] != name)]
    recs = recs.sort_values('popularity', ascending=False)

    print(f"\nif you like '{name}' (cluster {c}) you might also like:")
    print(recs[show_cols].head(n).to_string(index=False))


get_recommendations("Blinding Lights")


# pca just to visualize the clusters in 2d
# it wont be perfect since we have 12 dimensions
pca = PCA(n_components=2, random_state=42)
pca_coords = pca.fit_transform(X_full_scaled)
df['pc1'] = pca_coords[:, 0]
df['pc2'] = pca_coords[:, 1]

print(f"\npca variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")

# plotting a random 5000 so it doesnt get too cluttered
viz = df.sample(n=min(5000, len(df)), random_state=42)

plt.figure(figsize=(11, 7))
sc = plt.scatter(viz['pc1'], viz['pc2'], c=viz['cluster'],
                 cmap='tab10', alpha=0.4, s=15)
plt.colorbar(sc, label='cluster')
plt.xlabel('pc1')
plt.ylabel('pc2')
plt.title(f'song clusters (k={best_k})')
plt.grid(True, alpha=0.3)
plt.savefig('images/clusters.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close()