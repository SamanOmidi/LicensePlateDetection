from ultralytics import YOLO
from plate import make_plate
import os
import cv2

def detect_plate(model: YOLO, image_src: str):
    result = model(image_src, device="cpu")[0]
    return result
    

def plate(result, image_src: str):
    original_img = cv2.imread(image_src)
    # plate x, y, w, h
    x, y, w, h = map(int, result.boxes.xywh[0])
    # cropped_img = original_img[y - h//2:y + h//2, x - w//2:x + w//2]
    cropped_img = original_img[(y - h//2) - 5:(y + h//2) + 5,
                               (x - w//2) - 5:(x + w//2) + 5]
    cv2.imwrite("output/result.jpg", cropped_img)


def main():
    model = YOLO("../models/plate-detector.pt")
    image_src = "cars/car-4.jpg"
    result = detect_plate(model, image_src)
    if result is None:
        print("ERROR: Could not detect the plate.")
        os.exit(1)
    plate(result, image_src)
    make_plate("output/result.jpg")


if __name__ == "__main__":
    main()