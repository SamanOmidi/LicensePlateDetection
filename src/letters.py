import cv2

image_src = "output/plate.png"
image = cv2.imread(image_src)

height, width = image.shape[:2]

# First number -> 53 - 93
first_number = image[0:height, 53:93]
# Second number -> 94 - 140
second_number = image[0:height, 92:138]
# Letter -> 140 - 203
letter = image[0:height, 140:203]
# Third number -> 203 - 245
third_number = image[0:height, 203:245]
# Fourth number -> 245 - 280
fourth_number = image[0:height, 245:280]
# Fifth number -> 280 - 324
fifth_number = image[0:height, 280:324]

l = [first_number, second_number, letter, third_number, fourth_number, fifth_number]
for img in l:
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
