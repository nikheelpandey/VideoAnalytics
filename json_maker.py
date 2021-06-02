import cv2
# import numpy as np 
from feed_input import DummyFeed
from detector import Detector
from tqdm import tqdm
import json

cam = DummyFeed()
model = Detector()



frame_id = 0

data_ls = []


# while True:

for i in tqdm(range(5000)):

    frame = cam.get_frame()

    if frame is not None:
        frame_data = {}
        frame_id+=1
        bboxs,image = model.infer(frame)
        
        frame_data["frame_id"] = frame_id
        frame_data["bboxs"] = bboxs
    
    # else:

    #     print("Processed")
    #     break


    data_ls.append(frame_data)

    # if frame_id == 5000:
    #     break

    with open('detection.json', 'w') as file:
        json.dump(data_ls, file)