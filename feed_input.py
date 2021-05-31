import cv2

class DummyFeed(object):
    def __init__(self, file_path = None):
        self.file_path = "./data/sample.webm"
        self.load_file()
    
    def load_file(self):
        self.video = cv2.VideoCapture(self.file_path)
               
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, image = self.video.read()
        
        if success:
            return image
        
        else:
            print("Feed completed.")
            self.__del__()
            return None
            # self.load_file()
