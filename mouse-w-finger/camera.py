import cv2
from threading import Thread
from sys import exit

class WebcamStream():
    def __init__(self):        
        self.vcap = cv2.VideoCapture(0, cv2.CAP_ANY)
        
        if not self.vcap.isOpened():
            exit(0)
            
        self.vcap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.vcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.frame = self.vcap.read()[1]
        
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True
        
        self.tFlag = False
        
    def start(self):
        self.t.start() 

    def update(self):
        while True:
            self.frame = self.vcap.read()[1]
            cv2.waitKey(1)
            if self.tFlag:
                break
            
        self.vcap.release()
        exit(0)
    
    def read(self):
        return self.frame
    
    def stop(self, tFlag=False):
        self.tFlag = tFlag
        self.tFlag = True
        self.t.join()