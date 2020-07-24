import cv2 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Function to find large child contours in the frame and return the x,y coordinates and the frame in which the contour was found
def findChildContours(frame, frame_count):
    contours, hierarchy = cv2.findContours(frame, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    child_contours = []
    # Code taken from: https://stackoverflow.com/questions/22240746/recognize-open-and-closed-shapes-opencv
    hierarchy = hierarchy[0] 
    for i, c in enumerate(contours):
        # Return only innermost contours with no child contours 
        if hierarchy[i][2] < 0 and hierarchy[i][3] < 0:
            child_contours.append(c)
    x_coords = []
    y_coords = []
    size = []
    cont = []
    for c in child_contours:
        if cv2.contourArea(c) > 240: # Only save large contours # Originally 200
            m = cv2.moments(c)
            # Find Contour centre 
            x = m['m10'] / m['m00']
            y = m['m01'] / m['m00']
            x_coords.append(x)
            y_coords.append(y)
            size.append(cv2.contourArea(c))
            cont.append(c)
    frame_counts = [frame_count] * len(x_coords) # Which frame these contours found in
    return list(zip(x_coords, y_coords, frame_counts, size, cont)) # Zip lists to list of tuples # Size removed

### Motion Detection on smaller video to find early waggle-run clusters

cap = cv2.VideoCapture('TestWaggle.mp4')

delay_time = 40
prev_frame = None
counter = 0
potential_waggles = []
rois = list(np.arange(1,40))
saved_frames = []

while True:
    counter += 1
    ret, frame = cap.read()
    if ret == False:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (21,21), 0)
    
    thresh = cv2.threshold(gray_blur, 120, 230, cv2.THRESH_BINARY)[1]
    
    # If first frame, set current frame as prev_frame
    if prev_frame is None:
        prev_frame = thresh
    current_frame = thresh
    frame_diff = cv2.absdiff(current_frame, prev_frame) # Background Subtraction    
        
    _, hierarchy = cv2.findContours(frame_diff, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    # Catch blank images 
    if hierarchy is None:
        continue
    else:
        find_waggles = findChildContours(frame_diff, counter)
        potential_waggles = potential_waggles + find_waggles
    
    cv2.imshow('Gray', gray_blur)
    cv2.imshow('Frame Diff', frame_diff)
    cv2.imshow('Frame', frame)
    #cv2.waitKey(100)
    
    if counter > 40:
        pass
        #break
    # q to quit
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    prev_frame = current_frame
    
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1);

# Create df from motion detection
waggle_df = pd.DataFrame(potential_waggles, columns=['x', 'y', 'frame', 'size', 'contour'])
X = waggle_df.drop(['size', 'contour'], axis=1)

# Find 'waggle runs' via clustering
from sklearn.cluster import DBSCAN
clust = DBSCAN(eps=25, min_samples=8).fit(X)
waggle_df.loc[:, 'Cluster'] = clust.labels_

df = waggle_df[waggle_df['Cluster'] == 0].reset_index()

