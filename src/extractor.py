import easyocr
from hezar.models import Model

def main():
    # reader = easyocr.Reader(['fa'])
    # result = reader.readtext('result.jpg')
    # for (bbox, text, prob) in result:
    #     print(f'Text: {text}, Probability: {prob}')
    model = Model.load("hezarai/crnn-fa-64x256-license-plate-recognition")
    plate_text = model.predict("output/blured.jpg")
    print(plate_text)


if __name__ == "__main__":
    main()