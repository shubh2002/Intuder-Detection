import coordinates
import cv2
import numpy as np 
import detection

cap = cv2.VideoCapture('test_video.mp4')
font = cv2.FONT_HERSHEY_PLAIN
point = None
title = 'Frame'
cv2.namedWindow(title, cv2.WINDOW_NORMAL)
cv2.resizeWindow(title, (853,480))

_, frame = cap.read()
point = coordinates.get_points(frame)
while True:

    _, frame = cap.read()
    if frame is not None:
        overlay = frame.copy()   #copying frame for overlay
        height, width = frame.shape[0:2]   #capturing height and width of the frame
        mask = np.zeros((height, width), np.int8)  #creating mask 
        cv2.polylines(mask, np.array([point], dtype = np.int32), True, 255, 3)   #drawing polygon on mask
        cv2.fillPoly(mask, [np.array(point)], 255)   #filling polygon with white color
        masked_image = cv2.bitwise_and(frame, frame, mask = mask)
        rect = detection.people_detection(masked_image, 0.3, 0.3)
        cv2.polylines(frame, [np.array(point)], True, (0, 0, 255), 2)
        cv2.fillPoly(frame, [np.array(point)], (0, 0, 255))
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        text = f'{len(rect)}:intuder detected'
        cv2.putText(frame, text, (50,50), font, 3, (0,255,0), 2)
        for i in rect:
            (x,y,w,h) = i
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
        cv2.imshow('Image', frame)
       
    else: 
        break
        
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()