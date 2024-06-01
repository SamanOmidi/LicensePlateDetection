import cv2
import os
import math
import numpy as np


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def computer_skew(image):
    if len(image.shape) == 3:
        h, w, _ = image.shape
    elif len(image.shape) == 2:
        h, w = image.shape
    else:
        print("ERROR: Unsupported image type.")
        os.exit(1)
    
    blured = cv2.medianBlur(image, 3)
    edges = cv2.Canny(blured, threshold1 = 30, threshold2 = 100,
                       apertureSize = 3, L2gradient = True)
    lines = cv2.HoughLinesP(edges, 1, math.pi / 180, 30,
                             minLineLength = w / 4.0, maxLineGap = h / 4.0)
    
    angle = 0.0
    nlines = lines.size

    cnt = 0

    for x1, y1, x2, y2 in lines[0]:
        ang = np.arctan2(y2 - y1, x2 - x1)
        if math.fabs(ang) <= 30:
            angle += ang
            cnt += 1
    
    if cnt == 0:
        return 0.0
    else:
        val = (angle / cnt) * 180 / math.pi
        return val


def rotate(result_image: str):
    image = cv2.imread(result_image)
    result = rotate_image(image, computer_skew(image))
    cv2.imwrite("output/rotated.jpg", result)
    cv2.imshow("result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    