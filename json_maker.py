import cv2
import numpy 
from feed_input import DummyFeed
from detector import Detector
from tqdm import tqdm
import json

cam = DummyFeed()
model = Detector()

# cv2.namedWindow("win",cv2.WINDOW_NORMAL)


frame_id = 0

data_ls = []


while True:

    frame = cam.get_frame()

    if frame is not None:
        frame_data = {}
        frame_id+=1
        bboxs,image = model.infer(frame)
        
        frame_data["frame_id"] = frame_id
        frame_data["bboxs"] = bboxs
    
    else:

        print("Processed")

    data_ls.append(frame_data)

    if frame_id == 10:
        break

with open('detection.json', 'w') as file:
    json.dump(data_ls, file)