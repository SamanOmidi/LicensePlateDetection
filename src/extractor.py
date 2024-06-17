from ultralytics import YOLO
import numpy as np


def find_order(predicted_classes, original_xyxy, sorted_xyxy):
    l = []
    sorted_xyxy = np.array(sorted_xyxy)
    original_xyxy = np.array(original_xyxy)
    for t in sorted_xyxy:
        for i in range(len(original_xyxy)):
            check = np.equal(t, original_xyxy[i])
            if False in check:
                continue
            else:
                l.append(predicted_classes[i])
                break
    return l


def main():
    model = YOLO("../models/char_detector.pt")
    names = model.names
    result = model("output/plate.png", device="cpu", agnostic_nms=True)[0]
    # result = model("output/temp.jpg", device="cpu", agnostic_nms=True)[0]
    result.show()
    # Sort the predictions from left to right based on the x-coordinate of the bounding box
    predicted_classes = result.boxes.cls.to('cpu').tolist()
    original_xyxy = list(result.boxes.xyxy)
    sorted_xyxy = list(sorted(result.boxes.xyxy, key=lambda x: x[0]))
    order = find_order(predicted_classes, original_xyxy, sorted_xyxy)
    plate = ""
    for i in range(6):
        plate += names[order[i]] + ' '
    print(plate)


if __name__ == "__main__":
    main()