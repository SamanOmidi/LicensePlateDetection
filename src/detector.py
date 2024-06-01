from ultralytics import YOLO
from plate import rotate, shear, make_plate
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
    # cropped_img = original_img[y - h//2:y + h//2, x - w//2:x + w//2]
    cropped_img = original_img[(y - h//2) - 5:(y + h//2) + 5,
                               (x - w//2) - 5:(x + w//2) + 5]
    # cropped_img = original_img[y - h:y + h, x - w:x + w]
    # cropped_img = cv2.resize(cropped_img, (100,32))
    cv2.imwrite("output/result.jpg", cropped_img)


def main():
    model = YOLO("../models/plate-detector.pt")
    # image_src = "cars/L2P_153_jpg.rf.fe183bacc99247562dffe5091d5d5f45.jpg"
    image_src = "cars/11.jpg"
    result = detect_plate(model, image_src)
    plate(result, image_src)
    flag = rotate("output/result.jpg")
    if flag:
        shear("output/blurred.jpg")
        # make_plate("output/sheared.jpg")
        make_plate("output/blurred.jpg")
    else:
        make_plate("output/blurred.jpg")


if __name__ == "__main__":
    main()