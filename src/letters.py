import cv2
import numpy as np
import keras
import imutils

image_src = "output/detected_plate.jpg"

image = cv2.imread(image_src)
image = imutils.resize(image, 1300, 1300)


height, width = image.shape[:2]

h1 = 50
h2 = height - 50

# First number
first_number = image[h1:h2, 170:290]
# Second number
second_number = image[h1:h2, 290:430]
# Letter
letter = image[h1:h2, 430:615]
# Third number
third_number = image[h1:h2, 615:760]
# Fourth number
fourth_number = image[h1:h2, 760:870]
# Fifth number
fifth_number = image[h1:h2, 870:1000]

# Load model
model = keras.models.load_model("models/full_model.h5")

classes = ["0", "1", "2", "3", "4", "5", "6",
           "7", "8", "9", "alef", "be", "dal", "ghaf",
           "he", "je", "lam", "mim", "non", "pe", "ain",
           "pwd", "sad", "sin", "ta", "taxi", "vav", "ye"]

l = [first_number, second_number, letter, third_number, fourth_number, fifth_number]
plate = ""
for img in l:
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.imwrite("letters/letter.jpg", img)
    img = keras.preprocessing.image.load_img("letters/letter.jpg", target_size=(64, 64), color_mode="grayscale")
    img_array = keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    print(predicted_class)
    plate += classes[predicted_class[0]] + ' '

print(plate)
