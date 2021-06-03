import cv2
import json
import os, sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import path
from tracking import Tracker
from feed_input import DummyFeed


# load the json data containing list of frames with detections
json_path = "./detection.json"
f = open(json_path)
data = json.load(f)

# initialize the trackers
tracker = Tracker(thresh1=500, thresh2=10)

# cam
cam = DummyFeed()

# define the field of view of model

a = (227,1447)
b = (674,884)
c = (994,502)
d = (1024,204)
e = (1485, 164)
f = (1808,512)
g = (2100, 1061)
h = (2205,1685)


contour = np.array([a,b,c,d,e,f,g,h])
polygon = path.Path([a,b,c,d,e,f,g,h])

submask = None



for frames in tqdm(data):

    frame = cam.get_frame()

    if submask is None:

        frame_width,frame_height,_ = frame.shape 

        # out = cv2.VideoWriter('./output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))

        submask = np.ones_like(frame)
        cv2.fillPoly(submask, pts =[contour], color=(255,255,255))

    
    frame_id = frames["frame_id"]
    bboxes   = frames["bboxs"]

    tracker.update(bboxes)

    for i in range(len(tracker.trackerList)):

        k = tracker.trackerList[i].prediction
        c = tracker.trackerList[i].centroid

        # if c is not None:
        #     if polygon.contains_points(c):
                # print(tracker.trackerList[i].centroid)
        
                # cv2.rectangle(frame, (int(k[0]),int(k[1])), 
                #     (int(k[2]),int(k[3])),tracker.trackerList[i].color, 8)


    # frame[submask !=  255] = frame[submask !=  255]/4
    frame = frame.astype(np.uint8)

    # out.write(frame)
    fr = cv2.resize(frame,(0,0),fx=0.50,fy=0.50)
    cv2.imwrite(f"./result/{frame_id}.jpg",fr)

# out.release()
cam.__del__()
