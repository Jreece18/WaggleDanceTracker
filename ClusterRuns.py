import cv2 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from scipy import signal
from scipy import stats
from sklearn.mixture import GaussianMixture
import itertools 
import math

warnings.filterwarnings('ignore')

# Returns the slope (gradient) from linear regression of x, y coordinates over time
# Gives information on the direction of movement of the bee during the waggle run
def getSlope(x, y, frames):
    f = frames.astype('float32')
    x = x.astype('float32')
    y = y.astype('float32')
    
    slopex, _, _, _, _ = stats.linregress(f, x)
    slopey, _, _, _, _ = stats.linregress(f, y)
    
    return (slopex, slopey)


# Perform GMM on group of x, y slopes, return any points that do not belong to the population
def removeOutliers(slopes):
    results = GaussianMixture(n_components=1, covariance_type='full', verbose=1).fit_predict(X=np.array(slopes))
    results = results.tolist()
    mode = stats.mode(results).mode.tolist()[0]
    outliers = [i for i, x in enumerate(results) if x != mode]
    
    return outliers

# Divides distance information into 4 quadrants of general direction
def directionQuadrant(distance):
    x, y = distance
    
    # if x, -y
    if x >= 0 and y < 0:
        quadrant = 1
    # if x, y
    if x >= 0 and y >= 0:
        quadrant = 2
    # if -x, y
    if x < 0 and y >= 0:
        quadrant = 3
    # if -x, -y
    if x < 0 and y < 0:
        quadrant = 4
    return quadrant

# Finds outliers in long dances
def findOutliers(x_slopes, y_slopes, x_median, y_median):
    # Convert lists to Boolean depending on the median x, y
    if x_median >= 0:
        print('a')
        x = [i >= 0 for i in x_slopes]
    else:
        x = [i < 0 for i in x_slopes]
    if y_median >= 0:
        y = [i >= 0 for i in y_slopes]
    else:
        y = [i < 0 for i in y_slopes]
    
    # Multiply Boolean so (True, False) returns False
    outliers = [a*b  for a, b in zip(x, y)]
    return outliers

df = pd.read_pickle('WaggleRuns.pkl')

# get mean for x, y, frame of each waggle run
waggles = pd.DataFrame(columns=['xmean', 'ymean', 'x0', 'y0', 'x1', 'y1', 'frame0', 'frame1', 'time_taken', 'frequency',\
                                'cluster', 'angle', 'distance'])

# For each cluster, get simplified data
for i in df['cluster'].unique():
    clust = df[df['cluster'] == i]
    distance = getSlope(clust.x, clust.y, clust.frame)
    waggles.loc[len(waggles)] = clust.x.mean(), clust.y.mean(),  clust.x.min(), clust.y.min(), clust.x.max(), \
                                clust.y.max(), clust.frame.min(), clust.frame.max(), clust.frame.max() - clust.frame.min(), \
                                len(signal.find_peaks(clust.angle.values)[0]) / ((clust.frame.max() - clust.frame.min() + 1) / 60), \
                                clust.cluster.max(), clust.angle.mean(), distance

waggles = waggles[waggles['time_taken'] != 0]
# len(clust[clust.angle > clust.angle.mean()]) / ((clust.frame.max() - clust.frame.min() + 1) / 60)

# Find nearest neighbour for each waggle run
for i in waggles['cluster'].unique():    
    target = waggles[waggles['cluster'] == i]
    frame = float(target.frame1)
    angle = float(target.angle)
    # Only look for neighbours in a short window of frames
    search = waggles[waggles['frame0'].between(frame+5, frame+125)]
#     search = search[search['angle'].between(angle-20, angle+20)]
    search.loc[:, 'xmean'] = search['xmean'] - float(target['xmean'])
    search.loc[:, 'ymean'] = search['ymean'] - float(target['ymean'])
    search.loc[:, 'euclid'] = np.sqrt(np.square(search['xmean']) + np.square(search['ymean']))
    
    # Save next waggle as the closest point
    min_euclid = search.euclid.min()
    next_waggle = search[search['euclid'] == min_euclid]

    if len(next_waggle) > 0:
        waggles.loc[target.index, 'next_cluster'] = int(next_waggle.iloc[0, :]['cluster'])
        waggles.loc[target.index, 'next_euclid'] = next_waggle.iloc[0, :]['euclid']
        waggles.loc[target.index, 'next_angle_diff'] = abs(angle) - abs(next_waggle.iloc[0, :]['angle'])
        
        
### Remove Duplicates
        
# Divide all waggle runs by next_cluster into duplicates, non duplicates, and NaN values
dup = waggles[waggles.duplicated(subset=['next_cluster'], keep=False)]
non_dup = waggles[~waggles.duplicated(subset=['next_cluster'], keep=False)]
na = waggles[pd.isnull(waggles['next_cluster'])]
pts = dup[['cluster', 'next_cluster', 'next_angle_diff', 'next_euclid']].dropna().values
dup.head()

# Sort pts by next_cluster 
pts = pts[np.argsort(pts[:, 1], axis=0)]
# Separate pts into lists of duplicates
same_pts = [np.argwhere(i[0] == pts[:, 1]) for i in np.array(np.unique(pts[:, 1], return_counts=True)).T if i[1] >= 2]
save_row = []
# For each set of duplicates, save shortest distance
for i in same_pts:
    dist = []
    for j in i:
        pt = pts[j, -1]
        dist.append(pt)
    save_row.append(i[np.argmin(dist)][0])
final_pts = pts[[save_row]]

# Save duplicates that are in final_pts
waggles = dup[dup['cluster'].isin(list(final_pts[:, 0].astype(int)))]
# If pt not in final_pts, replace next_cluster, next_euclid with NaN
other_dup = dup[~dup['cluster'].isin(list(final_pts[:, 0].astype(int)))]
other_dup.loc[:, ['next_cluster', 'next_euclid']] = np.nan
# Concatenate all back into a single df
waggles = pd.concat([waggles, non_dup, na, other_dup])
waggles = waggles.sort_index().drop_duplicates()
# Remove next waggles if euclidean distance is in 0.85 quantile
# waggles.loc[waggles['next_euclid'] > waggles.next_euclid.quantile(0.85), ['next_cluster', 'next_euclid']] = np.nan
waggles.head()

quantile = waggles.next_euclid.quantile(0.85)

waggles.loc[:, 'next_cluster'] = np.where((waggles['next_euclid'] >= quantile), np.nan, waggles['next_cluster'])
waggles.loc[:, 'next_euclid'] = np.where((waggles['next_euclid'] >= quantile), np.nan, waggles['next_euclid'])

## Create lists of all clusters belonging to a dance
index = waggles.cluster.tolist()
next_index = waggles.next_cluster.tolist()

final = []

for i in index:
    # If i already in final list, continue
    if i in itertools.chain(*final):
        continue
    
    current_no = i
    current_idx = index.index(current_no)
    current_list = []

    current_list.append(current_no)

    # Fill current list with sequence
    while not (math.isnan(current_no)):
#         print(current_no)
        
        # Break if current no out of bounds or if next_index is NaN
        if current_no > (len(index) - 1):
            break
        if math.isnan(next_index[current_idx]):
            break
        # Else, append next number in sequence to list
        else:
            current_no = int(next_index[current_idx])
            current_idx = int(index.index(current_no))
            current_list.append(current_no)
    
    final.append(current_list)

# Get list of all long dances and short dances
long_dances = [x for x in final if len(x) > 2]
short_dances = [x for x in final if len(x) == 2]

final_short_dances = []
for dance in short_dances:
    distance = []
    angle = []
    next_euclid = []
    for run in dance:
        waggle = waggles[waggles['cluster']==run].iloc[0,:]
        distance.append(waggle.distance)
        angle.append(waggle.angle)
        next_euclid.append(waggle.next_euclid)
    
    # Catch value errors where NaN values occur
    try:
        if next_euclid[0] >= 100:
            continue
        # if x slopes are not in range of each other, skip
        if distance[0][0] not in np.arange(distance[1][0]-0.5, distance[1][0]+0.5):
            continue
        # if y slopes are not in range of each other
        if distance[0][1] not in np.arange(distance[1][1]-0.5, distance[1][1]+0.5):
            continue
        if angle[0] not in np.arange(angle[1]+10, angle[0]-10):
            continue
    except ValueError:
        continue

    final_short_dances.append(dance)

# Label all clusters belonging to a long dance (>2)
for i, dance in enumerate(long_dances):
    for run in dance:
        waggles.loc[waggles[waggles['cluster']==run].index, 'dance'] = i
        
waggles[waggles['dance'].notna()]['dance'].unique()
waggles.head()


for i in waggles[waggles['dance'].notna()]['dance'].unique():
    df = waggles[waggles['dance']==i]
    clusters = list(df.loc[:, 'cluster'])
    slopes = df.distance.tolist()
    # Divide slopes into x,y and find median 
    slopex, slopey = [i[0] for i in slopes], [i[1] for i in slopes]
    slopex_median, slopey_median = np.median(slopex), np.median(slopey)
    slopex_sign, slopey_sign = 'negative', 'negative'
        
    # Find slopes with a different sign to the median 
    slopes_med = findOutliers(slopex, slopey, slopex_median, slopey_median)
    # Gaussian Mixture Model on the slopes to find outliers
    gm = GaussianMixture(n_components=2, covariance_type='tied', verbose=1).fit_predict(X=np.array(slopes))
    gm_count = stats.mode(gm)[1][0]
    gm_mode = stats.mode(gm)[0][0]
    gm_bool = [i == gm_mode for i in list(gm)]
    
    combined_avg = [a + b for a, b in zip(gm_bool, slopes_med)]
    print(gm_bool, slopes_med)
    print(combined_avg)
    # Return index of values where combined_avg is False
    idx = []
    if combined_avg.count(False) >= 1 and gm_bool.count(False) <= len(gm_bool)/2:
        # remove this value from list
        idx = [i for i, x in enumerate(combined_avg) if x == False]
        
    
    outliers = list(df.iloc[idx, :]['cluster'])
    print(outliers)
    # If values in outliers, replace the dance with NaN 
    if len(outliers) >= 1:
        print('yes')
        waggles.loc[waggles[waggles['cluster'].isin(outliers)].index, 'Dance'] = np.nan
    
    # If 3 or more values in outliers, make new dance
    if combined_avg.count(False) >= 3:
        # make false values into own bee cluster
        print('b')
        waggles.loc[waggles[waggles['cluster'].isin(idx)].index, 'Dance'] = len(waggles['Dance'].unique())


long_dances_new = []
for i in waggles['Dance'].unique():
    df = waggles[waggles['Dance']==i]
    print(df.cluster.values)
    long_dances_new.append(df.cluster.values)
    
    