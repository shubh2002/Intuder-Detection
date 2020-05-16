import cv2
import numpy as np 
#import person_detection

points = []

current_frame = None
title = 'Frame'
exitFlag = False

def onmouse(events, x, y, flags, param):
    global current_frame, points
    if events == cv2.EVENT_LBUTTONDOWN:
        print(f'{x},{y}')       
        points.append([x, y])

def get_points(frame):
    global current_frame, points, exitFlag
    # points = []
    current_frame = frame.copy()
    
    cv2.setMouseCallback(title, onmouse)
    cv2.imshow(title, current_frame)
    cv2.waitKey(0)
    cv2.destroyWindow(title)
        
    return points