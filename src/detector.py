from ultralytics import YOLO
import cv2

def detect_plate(model: YOLO, image_src: str):
    # TODO Check results and take the closest object
    # results = model(image_src)
    # For now we only take the first element
    result = model(image_src)[0]
    return result
    

def plate(result, image_src: str):
    original_img = cv2.imread(image_src)
    # plate x, y, w, h
    x, y, w, h = map(int, result.boxes.xywh[0])
    cropped_img = original_img[y - h:y + h, x - w:x + w]
    cv2.imwrite("result.jpg", cropped_img)


def main():
    model = YOLO("../models/plate-detector.pt")
    image_src = "cars/car-0.jpg"
    result = detect_plate(model, image_src)
    plate(result, image_src)


if __name__ == "__main__":
    main()