import cv2
import math
import numpy as np
import imutils


def rotate(image):
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

        cv2.imwrite("output/xrotated.jpg", rotated_image)
        print("Rotation applied to the image.\n")
        return rotated_image
    else:
        print("No rotation applied to the image.\n")
        return image


def rotate_yaxis(image):
    proj2dto3d = np.array([[1, 0, -image.shape[1]/2],
                           [0, 1, -image.shape[0]/2],
                           [0, 0, 0],
                           [0, 0, 1]], np.float32)
    
    ry = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]], np.float32)
    
    trans = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 300],
                      [0, 0, 0, 1]], np.float32)

    proj3dto2d = np.array([[200, 0, image.shape[1]/2, 0],
                           [0, 200, image.shape[0]/2, 0],
                           [0, 0, 1, 0]], np.float32)    
    
    y = 8.0

    ay = float(y * (math.pi / 180.0))

    ry[0, 0] = math.cos(ay)
    ry[0, 2] = -math.sin(ay)
    ry[2, 0] = math.sin(ay)
    ry[2, 2] = math.cos(ay)

    r = ry

    final = proj3dto2d.dot(trans.dot(r.dot(proj2dto3d)))
    dst = cv2.warpPerspective(image, final, (image.shape[1], image.shape[0]), None,
                               cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, (255,255,255))
    
    cv2.imwrite("output/yrotated.jpg", dst)
    return dst

# Change plate to a specific size for later classification
def make_plate(cropped_image):
    xrotated_image = rotate(cropped_image)
    yrotated_image = rotate_yaxis(xrotated_image)
    resized_image = imutils.resize(yrotated_image, width=640, height=640)
    cv2.imwrite("output/plate.png", resized_image)
