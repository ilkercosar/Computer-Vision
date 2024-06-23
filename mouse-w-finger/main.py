import cv2
import mediapipe as mp
from sys import exit
import ctypes
import camera
from multiprocessing import Process, Queue

def camera_process(queue):
    mpHands = mp.solutions.holistic
    hands = mpHands.Holistic()
    mpDraw = mp.solutions.drawing_utils
    
    webcam = camera.WebcamStream()
    webcam.start()
    winW, winH = [ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1)]
    
    while True:
        frame = webcam.read()
        copy = frame.copy()
        copy = cv2.resize(copy, (winW, winH), interpolation=cv2.INTER_LINEAR)
        rgbFrame = cv2.cvtColor(copy, cv2.COLOR_BGR2RGB)
        
        result = hands.process(rgbFrame)
        mpDraw.draw_landmarks(copy, result.left_hand_landmarks, mpHands.HAND_CONNECTIONS)
        
        if result.left_hand_landmarks:
            index_finger_tip = result.left_hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP.value]
            x = int(index_finger_tip.x * winW * 1.25)
            y = int(index_finger_tip.y * winH * 1.25)
            
            queue.put((x, y))
        
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(copy, (100, 100), interpolation=cv2.INTER_LINEAR)
        cv2.imshow('frame', frame)
    
        key = cv2.waitKey(1)
        if key == ord('q') or key == ord('Q'):
            break

    cv2.destroyAllWindows()
    webcam.stop(tFlag=True)
    exit(0)

def cursor_process(queue):
    while True:
        if not queue.empty():
            x, y = queue.get()
            ctypes.windll.user32.SetCursorPos(x, y)

if __name__ == "__main__":
    queue = Queue()
    
    camera_proc = Process(target=camera_process, args=(queue,))
    cursor_proc = Process(target=cursor_process, args=(queue,))
    
    camera_proc.start()
    cursor_proc.start()
    
    camera_proc.join()
    cursor_proc.terminate()
