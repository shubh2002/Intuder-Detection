#importing neccessary pacakges

import numpy as np 
import cv2
import argparse



#loading our model
print('[INFO] loading YOLO from disk....')
net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
#getting all layers name
layer_names = net.getLayerNames()
#determining only output layers that we want from YOLO
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def people_detection(frames, minConfidence, minthreshold):
    (H,W) = frames.shape[0:2]
    

    #constructing blob from frames
    #and passing those blobs into detector to get bounding boxes and associated probabilities
    blob = cv2.dnn.blobFromImage(frames, 1/255.0, (416, 416), swapRB = True, crop = False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    #intialising list of confidence, boxes, classIds
    boxes =[]
    confidences = []
    classIDs = []
    rect =[]

    #loop over each output layer
    for output in outs:
        #loop over each detection
        for detection in output:
            #extracting class id and confidence
            scores = detection[5:]
            classid = np.argmax(scores)
            confidence = scores[classid]

            #filter out weak detection 
            if confidence > minConfidence:
                #scale the coordinates of bounding box relative to frame
                #yolo gives coordinates of center and width and height of box
                center_x = int(detection[0] * W)
                center_y = int(detection[1] * H)
                w = int(detection[2] * W)
                h = int(detection[3] * H)
                #using center coordinates to drive top left coordinates
                x = int(center_x - (w/2))
                y = int(center_y - (h/2))

                #updating the list of bounding box, confidence and classid
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                classIDs.append(classid)

    #filtering for person detection as the model gives other detection too
    personBoxes = []
    personBoxesForNMS = []
    personConfidences = [] 

    #enumerate classIDs for all detected object
    for i, detectedClass in enumerate(classIDs):
        #check if detected class is person
        if detectedClass == 0:
            (x, y, x1, y1) = boxes[i]
            personBoxes.append([int(x), int(y), int(x1), int(y1)])
            personBoxesForNMS.append(boxes[i])
            personConfidences.append(confidences[i])

    #applying NMS
    idxs = cv2.dnn.NMSBoxes(personBoxes, personConfidences, minConfidence, minthreshold)
    
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            rect.append([x,y,w,h])
            # draw a bounding box rectangle and label on the frame
            # color = [int(c) for c in COLORS[classIDs[i]]]
            # cv2.rectangle(frames, (x, y), (x + w, y + h), color, 2)
            # text = "{}: {:.4f}".format(LABELS[classIDs[i]],
            #     confidences[i])
            # cv2.putText(frames, text, (x, y - 5),
            #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                

    return rect




