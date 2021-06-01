import cv2
import json
import os, sys
import numpy as np
import pandas as pd
from tracking import Tracker


json_path = "./detection.json"

# load the json data containing list of frames with detections
f = open(json_path)
data = json.load(f)



