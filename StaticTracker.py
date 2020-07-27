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


### Tracking Functions ###

# Find contour based on prior coordinates
def findFullContourPoints(img, x, y):
    # Threshold Image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    thresh = cv2.threshold(gray, 180, 220, cv2.THRESH_BINARY)[1]
    
    contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for c in contours[0]:
        dist = cv2.pointPolygonTest(c, (x,y), False)
        if dist == 1.0:
            final_contour = c
            print('Contour Found')
            break
        else:
            final_contour = None
    #print(final_contour)
    return final_contour, thresh

# Find largest contour in ROI
def findFullContour(img, bbox):
    bbox = map(int, bbox)
    x, y, w, h = bbox
    # Threshold Image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    thresh = cv2.threshold(gray, 180, 220, cv2.THRESH_BINARY)[1]
    thresh_roi = thresh[y:y+h, x:x+w]
    # Mask of black pixels so only ROI is shown
    mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    mask[y:y+h, x:x+w] = thresh_roi
    
    # Taken from: https://stackoverflow.com/questions/54615166/find-the-biggest-contour-opencv-python-getting-errors
    contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contour_areas = [(cv2.contourArea(contour), contour) for contour in contours[0]]
    
    final_c = max(contour_areas, key=lambda x: x[0]) # Find max contour area
    return final_c[1], thresh
    
# Find Centre coordinates of contour
def getContourMoment(contour):
    m = cv2.moments(contour)
    # Find Contour centre 
    x = m['m10'] / m['m00']
    y = m['m01'] / m['m00']
    return int(x), int(y) 


# Fits a bounding box tightly around the contour
def getFittedBox(contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return rect, box

# Converts a fitted bounding box to a straight one with no rotation 
def rotatedBoxConverter(box):
    box_t = np.array(box).T
    x, y = min(box_t[0]), min(box_t[1])
    w, h = max(box_t[0]) - x, max(box_t[1]) - y
    return x, y, w, h

# Create mask from image
def createMask(img):
    mask = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    mask.fill(255)
    plt.imshow(mask)
    return mask

# Track centre coordinates and trace to mask
def traceMask(mask, prev_pt, current_pt):
    cv2.line(mask, prev_pt, current_pt, (0,0,0))

# Find centre of a box
def boxCentre(box):
    x, y, w, h = box
    cntr_x = x + w/2
    cntr_y = y - h/2
    return int(cntr_x), int(cntr_y)

# Calculate avg area of box
def avgArea(box, total, count):
    x, y, w, h = box
    total += w * h
    avg = total / count
    return total, avg

# Prevent incorrect updates
def anchorBox(box, prev_box, avg):
    global counter
    x0, y0, w0, h0 = prev_box
    x, y, w, h = box
    
    # If bounding box is too far from previous box
    if abs(x - x0) > w0/2 and abs(y - y0) > h0/2:
        print('Box Lost')
        success = False
    # If bounding box is too large
    elif (w * h) > (avg * 1.5):
        print('Box Lost (Expanded)')
        print(str(counter))
        success = False
    # prev_box only updated if box found
    else:
        print('Box Found')
        prev_box = box 
        success = True

    return success, prev_box

# If contour is none, box probably too small, expand incrementally until contour found
def expandBox(img, bbox):
    contour = None
    x, y, w, h = bbox
    while contour is None:
        x -= 5
        w += 10
        y -= 5
        h += 10
        bbox = (x, y, w, h)
        print(bbox)
        contour, thresh = findFullContour(img, prev_bbox)
    return bbox, contour, thresh
