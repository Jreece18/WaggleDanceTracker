import pandas as pd
import numpy as np
import cv2

df = pd.read_pickle('WaggleRunsFinal.pkl')

prefix = 'Bees10'
cap = cv2.VideoCapture('../Bees10.mov')
# Dimensions of video
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

for dance in df.Dance.unique():
    dance = df[df['Dance']==dance]
    # Set video bounds
    x0 = dance.x0.min() - 100
    x1 = dance.x1.max() + 100
    y0 = dance.y0.min() - 100
    y1 = dance.y1.max() + 100
    frame0 = dance.frame0.min()
    frame1 = dance.frame1.max()
    
    # Adjust if bounds beyond video bounds
    if frame0 < 0:
        frame0 = 0
    if x0 < 0:
        x0 = 0
    if y0 < 0:
        y0 = 0 
    if x1 > width:
        x1 = width
    if y1 > height:
        y1 = height
    
    # Create output video
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('{}-{}.mp4'.format(prefix, str(int(dance.Dance.max()))), -1, fps, (x1-x0,y1-y0))
    print('{}-{}.mp4'.format(prefix, str(int(dance.Dance.max()))))
    
    # Set video start
    cap.set(1, frame0-5)
    counter = frame0-5
    run = False
    while True:
        ret, frame = cap.read()
        
        # If video counter within range of a run or not
        if counter in list(dance.frame0):
            run = True
        elif counter in list(dance.frame1):
            run = False
        
        # If in range of run, draw rectangle around bee
        if run == True:
            loc = np.where(((dance['frame0'] <= counter).values * (dance['frame1'] >= counter).values) == True)[0][0]
            loc = dance.iloc[loc, :]
            cv2.rectangle(frame, (loc.x0 - 20, loc.y0 - 20), (loc.x1 + 20, loc.y1 + 20), (200, 200, 200), 2)
        
        # Write to output video
        out.write(frame[y0:y1, x0:x1])
        cv2.imshow('Frame', frame)
        cv2.waitKey(1)       
        
        
        counter += 1
        if counter-10 >= dance.iloc[-1, :]['frame1']:
            out.release()
            break

    if counter > 200:
        break

cap.release()
cv2.destroyAllWindows()