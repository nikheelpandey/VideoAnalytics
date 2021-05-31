import cv2
import numpy 
from feed_input import DummyFeed
from detector import Detector
"""
camera test
"""
cam = DummyFeed()

model = Detector()
cv2.namedWindow("win",cv2.WINDOW_NORMAL)
frame_id = 0



while True:

    frame = cam.get_frame()

    if frame is not None:
        bboxs,image = model.infer(frame)

    cv2.imshow("win",image)
    cv2.waitKey(0)