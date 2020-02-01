from detector import Detector
from classifier import SmileClassifier
import cv2

if __name__ == '__main__':
    sc_model = SmileClassifier(nnetwork="lenet")
    det_model = Detector()

    cap = cv2.VideoCapture(0)
    
    RED = (0,0,255)   # as BGR
    GREEN = (0,255,0) # as BGR
    while 1:
        ret,frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces = det_model.detect(frame)
            
            for x,y,w,h in faces: # TODO make it batch operation
                result = sc_model(gray[y:y+h,x:x+w])
                pred = result[0]["label"]
                score = result[0]["score"]
                if pred == "neutral":
                    cv2.rectangle(frame,(x,y),(x+w,y+h),RED,2)
                else:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),GREEN,2)
            cv2.imshow("face detector",frame)
        if cv2.waitKey(1) == 27: # if esc pressed then quit
            break
