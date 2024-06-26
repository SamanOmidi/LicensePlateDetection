from ultralytics import YOLO
from plate import make_plate
from extractor import extract
import sys
import cv2

def detect_plate(model: YOLO, image_src: str):
    result = model(image_src, device="cpu")[0]
    return result
    

def plate(result, image_src: str):
    original_img = cv2.imread(image_src)
    # plate x, y, w, h
    x, y, w, h = map(int, result.boxes.xywh[0])
    cropped_img = original_img[(y - h//2) - 5:(y + h//2) + 5,
                               (x - w//2) - 5:(x + w//2) + 5]
    cv2.imwrite("output/detected_plate.jpg", cropped_img)
    return cropped_img
    


def main():
    model = YOLO("../models/plate-detector.pt")
    image_src = "cars/pic (1).jpg"
    result = detect_plate(model, image_src)
    if len(result) == 0:
        print("ERROR: Could not detect any plate.")
        sys.exit(1)
    else:
        cropped_image = plate(result, image_src)
        make_plate(cropped_image)
        plate_number = extract("output/plate.png")
        print(plate_number)


if __name__ == "__main__":
    main()