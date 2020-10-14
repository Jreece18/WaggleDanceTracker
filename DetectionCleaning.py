import pandas as pd
import numpy as np

prefix = 'Bees10'
waggle_df = pd.read_pickle('WaggleDetections-Bees10.pkl')
# Sort by cluster and then frame so the dataset is ordered in blocks of clusters
waggle_df = waggle_df.sort_values(by=['Cluster', 'frame']).reset_index().drop(['index'], axis=1)

# All rows duplicated on frame and cluster
dup = waggle_df[waggle_df.duplicated(subset=['frame', 'Cluster'], keep=False)] # Returns all rows that match on frame and cluster
# All non duplicated rows
non_dup = waggle_df[~waggle_df.duplicated(subset=['frame', 'Cluster'], keep=False)] # Returns all rows that match on frame and cluster

a = dup.index.values
b = dup.index.values - 1 # Rows before duplicate
c = dup.index.values + 1 # Rows after duplicate

# Concatenate removing duplicate indices
idx = np.unique(np.concatenate((a, b, c)))
df = waggle_df[waggle_df.index.isin(idx)].reset_index().reset_index()

# level_0 used for the below indexing, 'index' used for returning these values in df
pts = df[['level_0', 'x','y', 'index', 'frame']].values

#pts = pts[np.argsort(pts[:, -1], axis = 0)]    # to sort about last column if not sorted
# Returns indices of duplicates
same_pts = [np.argwhere(i[0] == pts[:, -1]) for i in np.array(np.unique(pts[:, -1], return_counts=True)).T if i[1]>=2]
save_row = []
for i in same_pts:
    dist = []
    pre = min(i)-1
    for j in i:
        # Euclidean distance between the previous point and each duplicate point
        dist_pre = np.sqrt((pts[pre, 1]-pts[j, 1])**2 + (pts[pre, 2]-pts[j, 2])**2)
        dist.append(dist_pre)
    # Save level_0 of point with the shortest distance
    save_row.append(i[np.argmin(dist)][0])
# Return points of shortest distance based on index
final_pts = pts[[save_row]]

# Concatenate final_pts index with non_duplicate index, removing unique values
final_idx = np.unique(np.concatenate((non_dup.index.values, final_pts[:, 3])))
waggle_df = waggle_df.reset_index()
# Return all rows where index is in final_idx
df = waggle_df[waggle_df['index'].isin(final_idx)]

# Check all duplicates removed
dup = df[df.duplicated(subset=['frame', 'Cluster'], keep=False)]
dup.head()

# Calculate Euclidean Distance between each point in cluster and the next point
for i in list(df['Cluster'].unique()):
    clust = df[df['Cluster']==i]
    clust.loc[:, 'euclid'] = np.sqrt(np.square(df['x'] - df['x'].shift(1)) + np.square(df['y'] - df['y'].shift(1)))
    df.loc[clust.index, 'euclid'] = clust['euclid']

# Fill NaNs (first of each cluster) with 0 
df.fillna(0, inplace=True)
# Remove any points where euclidean distance is above the 90th percentile
quant = df.euclid.quantile(0.9)
df = df[df['euclid'] < quant]

# Save cleaned dataset
df.to_pickle('WaggleDetections-{}-Cleaned.pkl'.format(prefix))