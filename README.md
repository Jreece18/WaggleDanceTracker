# WaggleDanceTracker
Data Science MSc - City, University of London
Final Project

## Description
The code should be run in the following order:
1. DanceDetector.py
2. DetectionCleaning.py
3. WaggleRunTracker.py
4. ClusterRuns.py
5. SaveDances.py

### DanceDetector.py
- Requires a static video of an observation hive with any number of bees performing the waggle dance.
- Takes the video and detects 'waggle-like' activity based on contour detection.
- Clusters the activity based on density to isolate waggle runs.
### DetectionCleaning.py
- Removes duplicate points (two coordinates at the same frame in the same cluster).
### WaggleRunTracker.py
- Using the detection dataset as a skeleton, tracks the bee throughout the waggle run.
- Saves data on orientation, waggle frequency, spatial coordinates for each cluster.
### ClusterRuns.py
- Clusters each waggle run into a waggle dance based on a forward search (within 5s of the previous run ending).
- Clustered based on proximity.
### SaveDances.py
- Saves each waggle dance as a separate cropped video.
- The bee is highlighted during the waggle run as a visual aid.


Paper: https://www.biorxiv.org/content/10.1101/2020.11.21.354019v1
