import numpy as np 
import cv2 
import os

class Detector(object):

    """
    A class used to load the detection model and infer from it 

    ...

    Attributes
    ----------
    model_path : str
        root folder of the model

    CLASSES : list    
        total classes supported by the model

    COLORS : list  
        unique color for each class

    model: caffeemodel
        detection model 

    meta_class:list
        concerned class

    Methods
    -------
    infer(image)
        returns bounding boxes for moterway

    """


    def __init__(self, model_path=None):

        if model_path is None:
            self.model_path = "./model"
        else:
            self.model_path = model_path
        
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]

        self.meta_class = ["bus","moterbike","car"]
        self.COLORS = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))

        self.model = cv2.dnn.readNetFromCaffe(os.path.join(self.model_path,"model.txt"),os.path.join(self.model_path,"model.caffemodel"))

    
    def infer(self,image):
        bboxs = []

        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image.copy(), (1920, 1080)), 0.007843, (1920, 1080), 127.5)

        self.model.setInput(blob)
        detections = self.model.forward()

        for i in np.arange(0,detections.shape[2]):
            
            idx = int(detections[0, 0, i, 1])
            confidence = detections[0,0,i,2]
            
            if (confidence > 0.75) and (self.CLASSES[idx] in self.meta_class):

                idx = int(detections[0,0,i,1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                box = list(box)
                bboxs.append(box)

        return bboxs, image