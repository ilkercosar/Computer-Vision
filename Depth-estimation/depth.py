import cv2
import torch
import numpy as np
from threading import Thread, Lock
from sys import exit
from time import time

class WebcamStream:
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
        start = time()
        count = 0
        self.fps = 15
        while True:
            ret, self.frame = self.vcap.read()
            
            if self.tFlag or not ret:
                break
            
            count += 1
            
            if count == 10:
                end = time()
                self.fps = int(10 / (end - start))
                start = end
                count = 0
            
            self.frame = cv2.putText(self.frame, str(self.fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            
        self.vcap.release()
        exit(0)
    
    def read(self):
        return self.frame
    
    def stop(self):
        self.tFlag = True
        self.t.join()


class DepthEstimation:
    def __init__(self):
        self.midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
        self.midas.to('cpu')
        self.midas.eval()

        self.transform = torch.hub.load('intel-isl/MiDaS', 'transforms')
        self.transform = self.transform.small_transform

        self.frame = None
        self.depth_frame = None

        self.frame_lock = Lock()

        self.t = Thread(target=self.update, args=())
        self.t.daemon = True
        self.tFlag = False
    
    def start(self):
        self.t.start()

    def update(self):
        while True:
            if self.tFlag:
                break
            
            self.frame_lock.acquire()
            frame = self.frame
            self.frame_lock.release()
            
            if frame is None:
                continue

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imgbatch = self.transform(img).to('cpu')

            with torch.no_grad():
                prediction = self.midas(imgbatch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode='bicubic',
                    align_corners=False
                ).squeeze()

                output = prediction.cpu().numpy()

                # Normalize the output
                depth_frame = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX)
                depth_frame = np.uint8(depth_frame)

                self.frame_lock.acquire()
                self.depth_frame = depth_frame
                self.frame_lock.release()
    
    def update_frame(self, frame):
        self.frame_lock.acquire()
        self.frame = frame
        self.frame_lock.release()

    def read(self):
        self.frame_lock.acquire()
        depth_frame = self.depth_frame
        self.frame_lock.release()
        return depth_frame

    def stop(self):
        self.tFlag = True
        self.t.join()

if __name__ == "__main__":
    webcam_stream = WebcamStream()
    depth_estimation = DepthEstimation()
    
    webcam_stream.start()
    depth_estimation.start()
    
    while True:
        frame = webcam_stream.read()
        depth_estimation.update_frame(frame)

        depth_frame = depth_estimation.read()
        
        if frame is not None:
            cv2.imshow('normal', frame)
        if depth_frame is not None:
            cv2.imshow('depth', depth_frame)
                
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    webcam_stream.stop()
    depth_estimation.stop()
    cv2.destroyAllWindows()

