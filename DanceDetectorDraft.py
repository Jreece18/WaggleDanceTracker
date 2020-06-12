import cv2
import numpy as np
import matplotlib.pyplot as plt

cam = cv2.VideoCapture('Bees10.mov')

width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

delay_time = 40
prev_frame = None

# Frames to save for individual study
saved_frames = []
rois = list(np.arange(110,120,1)) + list(np.arange(260,270,1))
# Frame counter
counter = 0
while True:
    counter += 1
    ret, frame = cam.read()
    frame = frame[300:600, 700:1050]
    
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (21,21), 0)
#    gray_blur = gray
    
    thresh = cv2.threshold(gray_blur, 108, 230, cv2.THRESH_BINARY)[1]
    
    # If first frame, set current frame as prev_frame
    if prev_frame is None:
        prev_frame = thresh
    current_frame = thresh

    frame_diff = cv2.absdiff(current_frame, prev_frame)    
    
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
    
cam.release()
cv2.destroyAllWindows()
cv2.waitKey(1)