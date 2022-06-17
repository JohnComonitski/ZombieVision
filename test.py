import torch
import numpy as np
import cv2

model = torch.hub.load('ultralytics/yolov5', 'custom', path='model/weights/last.pt', force_reload=True)

#Capture OBS Virtual Camera
cap = cv2.VideoCapture(1)

#Detectiuon Loop
while cap.isOpened():
    ret, frame = cap.read()
    
    # Make detections 
    results = model(frame)
    
    cv2.imshow('YOLO', np.squeeze(results.render()))
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()