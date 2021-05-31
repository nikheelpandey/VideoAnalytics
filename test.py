import cv2
import numpy 
from feed_input import DummyFeed
from detector import Detector

"""
camera test
"""

cam = DummyFeed()

# while True:
#     frame = cam.get_frame()
#     print(frame.shape)
#     # cv2.imwrite("./sample.jpg",frame)


# """
# detector test
# """

model = Detector()
cv2.namedWindow("win",cv2.WINDOW_NORMAL)

while True:

    frame = cam.get_frame()
    bboxs,image = model.infer(frame)

    cv2.imshow("win",image)
    cv2.waitKey(0)