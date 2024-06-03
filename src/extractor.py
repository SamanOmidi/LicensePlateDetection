# from hezar.models import Model
from ultralytics import YOLO

def main():
    # reader = easyocr.Reader(['fa'])
    # result = reader.readtext('result.jpg')
    # for (bbox, text, prob) in result:
    #     print(f'Text: {text}, Probability: {prob}')
    # model = Model.load("hezarai/crnn-fa-64x256-license-plate-recognition")
    model = YOLO("../models/best.pt")
    # plate_text = model.predict("output/blurred.jpg")
    results = model("output/plate.png", device="cpu")
    print(len(results))
    # print(plate_text)
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        result.show()  # display to screen
        # result.save(filename="result.jpg")  # save to disk


if __name__ == "__main__":
    main()