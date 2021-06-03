
import os
import imageio
from tqdm import tqdm
import cv2

out = None

for i in range(1,212):

    im = cv2.imread("./result/"+str(i)+".jpg")
    # print(im.shape)
    frame_height,frame_width,_ = im.shape

    if out is None:
        out = cv2.VideoWriter('./sample_video.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))
    
    out.write(im)



out.release()