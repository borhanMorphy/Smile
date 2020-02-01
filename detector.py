import cv2
import numpy as np


class Detector:
    def __init__(self,cascade_path:str=None):
        if isinstance(cascade_path,type(None)):
            cascade_path = cv2.data.haarcascades+'haarcascade_frontalface_default.xml'
        self._cascade_detector = cv2.CascadeClassifier(cascade_path)
        

    def detect(self,frame:np.ndarray) -> list:
        """Takes frame as BGR and return faces
        
        Args:
            frame (np.ndarray): bgr frame
        
        Returns:
            list: for each face returns (x,y,w,h)
        """
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY,2,7)
        return self._cascade_detector.detectMultiScale(gray)


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    det = Detector()
    RED = (0,0,255) # as BGR

    while 1:
        ret,frame = cap.read()
        if ret:
            faces = det.detect(frame)
            for x,y,w,h in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),RED,2)
            cv2.imshow("face detector",frame)
        if cv2.waitKey(1) == 27: # if esc pressed then quit
            break
