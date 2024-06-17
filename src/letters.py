import cv2
import numpy as np
import functools

image_src = "output/plate.png"

image = cv2.imread(image_src)

# # Convert image so keras can read it
# image = cv2.resize(image, (64,64))
# image = image[...,::-1].astype(np.float32) / 255.0
# # Convert to grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

height, width = image.shape[:2]

h1 = 30
h2 = height - 30

# First number
first_number = image[h1:h2, 85:145]
# Second number
second_number = image[h1:h2, 145:215]
# Letter
letter = image[h1:h2, 215:300]
# Third number
third_number = image[h1:h2, 300:370]
# Fourth number
fourth_number = image[h1:h2, 370:430]
# Fifth number
fifth_number = image[h1:h2, 430:490]

import keras
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
    cv2.destroyAllWindows()
    cv2.imwrite("letters/letter.jpg", img)
    img = keras.preprocessing.image.load_img("letters/letter.jpg", target_size=(64, 64), color_mode="grayscale")
    img_array = keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    print(predicted_class)
    plate += classes[predicted_class[0]] + ' '

print(plate)
