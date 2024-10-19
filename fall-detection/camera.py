import cv2 
from threading import Thread
from sys import exit

class WebcamStream():
    def __init__(self):
        try:  
            self.vcap = cv2.VideoCapture(0,cv2.CAP_ANY).release()
            self.vcap = cv2.VideoCapture(0,cv2.CAP_ANY)
        
            if self.vcap.isOpened() == False:
                exit(0)
            
            self.vcap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.vcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
            self.frame = self.vcap.read()[1]
        
            self.t = Thread(target=self.update, args=())
            self.t.daemon = False
            self.tFlag = False
            
        except Exception as initCameraError:
            print(f"Kamera başlatma hatası: {initCameraError}")
            exit(0)
        
    def start(self):
        self.t.start() 

    def update(self):
        while True :
            try:
                self.frame = self.vcap.read()[1]
                if self.tFlag == True:
                    break
            except Exception as cameraError:
                print(f"Görsel hatası: {cameraError}")
                exit(0)
            
        self.vcap.release()
        exit(0)
    
    def read(self):
        return self.frame
    
    def stop(self, tFlag = False):
        self.tFlag = tFlag
        self.tFlag = True
        self.t.join()