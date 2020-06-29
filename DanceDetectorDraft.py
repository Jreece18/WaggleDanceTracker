import cv2
import numpy as np
import matplotlib.pyplot as plt

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
    for c in child_contours:
        if cv2.contourArea(c) > 240: # Only save large contours # Originally 200
            m = cv2.moments(c)
            # Find Contour centre 
            x = m['m10'] / m['m00']
            y = m['m01'] / m['m00']
            x_coords.append(x)
            y_coords.append(y)
            size.append(cv2.contourArea(c))
    frame_counts = [frame_count] * len(x_coords) # Which frame these contours found in
    return list(zip(x_coords, y_coords, frame_counts)) # Zip lists to list of tuples # Size removed

cap = cv2.VideoCapture('Bees10.mov')

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Type font for adding frame counter to video
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

delay_time = 1 # Delay next loop for easier viewing
prev_frame = None 
# Save frames for isolated study
saved_frames = [] 
rois = list(np.arange(240,250))

counter = 0 # Frame counter
potential_waggles = [] # List to save potential waggles

while True:
    counter += 1
    ret, frame = cap.read()
#    frame = frame[300:600, 700:1050]

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (21,21), 0)
    
    thresh = cv2.threshold(gray_blur, 108, 230, cv2.THRESH_BINARY)[1]
    
    # If first frame, set current frame as prev_frame
    if prev_frame is None:
        prev_frame = thresh
    current_frame = thresh

    frame_diff = cv2.absdiff(current_frame, prev_frame) # Background Subtraction    
    
    _, hierarchy = cv2.findContours(frame_diff, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    # Catch blank images and skip frame
    if hierarchy is None:
        continue
    else:
        find_waggles = findChildContours(frame_diff, counter)
        potential_waggles = potential_waggles + find_waggles
    
    # Frame Counter
    cv2.putText(thresh, str(counter), (40,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (150,0,0), 2)
    
    # Display the resulting frame
    cv2.imshow('Thresholded', thresh)
    cv2.imshow('Frame Diff', frame_diff)
    cv2.imshow('Grayscale', gray)
    cv2.imshow('GrayBlur', gray_blur)
    cv2.waitKey(delay_time)
    
    
    # Make current frame the previous frame for the next loop
    prev_frame = current_frame
    
    # q to quit
    if cv2.waitKey(delay_time) & 0xFF == ord('q'):
        break
    
    if counter in rois:
        saved_frames.append(frame_diff)
    
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1);