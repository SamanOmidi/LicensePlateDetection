import cv2
import math
import numpy as np
import imutils

def rotate(image_src: str):
    image = cv2.imread(image_src)

    # Resize the image based on resolution
    # We don't don anything to the images till we get the resolution of the
    # original image which is not determined at the moment

    # Convert to gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply blur to make edges more apparent
    blured = cv2.GaussianBlur(gray, (5,5), 0)

    # Perform edge detection to find the edges
    edged = cv2.Canny(blured, 50, 100)

    # Find lines
    lines = cv2.HoughLinesP(edged, 1, math.pi / 180, 2, None, 5, 1)

    angles = {}

    if lines is not None:
        for line in lines:
            line = line[0]
            pt1 = (line[0], line[1])
            pt2 = (line[2], line[3])
            if line[2] - line[0] != 0:
                # M represents the m of the line
                m = (line[3] - line[1]) / (line[2] - line[0])
                # Convert m to degree
                m = np.degrees(np.arctan(m))
                if angles == {}:
                    angles[m] = 1
                else:
                    flag = True
                    for angle in angles:
                        if m < angle + 5 and m > angle - 5:
                            angles[angle] += 1
                            flag = False
                            break
                    if flag:
                        if m not in angles:
                            angles[m] = 1
                        else:
                            angles[m] += 1
            cv2.line(blured, pt1, pt2, (255,0,255),1)

        rotation_angle = 0.0
        cnt = 0
        for angle in angles:
            if cnt == 0:
                cnt = angles[angle]
                rotation_angle = angle
            else:
                if angles[angle] > cnt:
                    cnt = angles[angle]
                    rotation_angle = angle
        
        height, width = image.shape[:2]
        center_of_image = (width / 2, height / 2)
        rotate_matrix = cv2.getRotationMatrix2D(center=center_of_image,
                                                angle=rotation_angle,
                                                scale=1)
        rotated_image = cv2.warpAffine(src=image,
                                    M=rotate_matrix,
                                    dsize=(width, height))

        cv2.imwrite("output/blurred.jpg", rotated_image)
        return True
    else:
        print("ERROR: No lines found.")
        return False


# TODO Check if the plate image need shearing

def shear(image_src: str):
    pass

# Change plate to a specific size for later classification
def make_plate(image_src: str):
    image = cv2.imread(image_src)
    resized_image = imutils.resize(image, width=420, height=110)
    cv2.imwrite("output/plate.png", resized_image)