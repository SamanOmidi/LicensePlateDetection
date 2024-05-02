from ultralytics import YOLO
import numpy as np
import cv2

# load model
model = YOLO("models/plate-detector.pt")

original_image = "cars/L2P_118_jpg.rf.25bf7d5c75577f2a44f4e22c5cb5dc94.jpg"

# Run batched inference on a list of images
results = model(original_image)  # return a list of Results objects

box = None

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    # xywh
    box = boxes.xywh[0]
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    result.show()  # display to screen
    result.save(filename='result.jpg')  # save to disk


img = cv2.imread(original_image)
x, y, w, h = map(int, box)
# crop_img = img[y:y+h, x:x+w]
crop_img = img[y-h:y+h, x-w:x+w]
cv2.imshow("cropped", crop_img)
cv2.waitKey(0)

char_detector = YOLO("models/character-detector.pt")

results = char_detector(crop_img)

for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    result.show()  # display to screen

