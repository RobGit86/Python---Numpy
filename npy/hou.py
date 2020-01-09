import cv2
import math
import numpy as np
from numpy import unravel_index
import houghFunc as hf

src_str = 'img/test-out-gray.jpg'
img = cv2.imread(src_str)
img = cv2.Canny(img, 400, 600)

number_of_lines = 50

x_max = img.shape[1]
y_max = img.shape[0]

ro_min = 0.0
ro_max = math.hypot(x_max, y_max)

acc = np.zeros((int(ro_max), 180))

for y in range(y_max):
    for x in range(x_max):
        if img[y, x] == 255:
            for m in range(180):
                ro = int((x * math.cos(math.radians(m))) + (y * math.sin(math.radians(m))))
                acc[ro][m] = acc[ro][m] + 1

maximums = np.empty((0, 2))

for x in range(number_of_lines):
    maxidx = unravel_index(acc.argmax(), acc.shape)
    maximums = np.append(maximums, maxidx)
    acc[maxidx[0]][maxidx[1]] = 0

lines = hf.HoughFunc(src_str)


def draw_line(rr, angg, img):

    lin = False

    if angg > 90:
        rr1 = rr - ro_max
        lin = True

    for x in range(lines.width):
        if angg == 0:
            y = int((rr - x * math.cos(math.radians(angg))) / 0.001)
        else:
            y = int((rr - x * math.cos(math.radians(angg))) / (math.sin(math.radians(angg))))
        if lin:
            if angg == 0:
                y1 = int((rr1 - x * math.cos(math.radians(angg))) / 0.001)
            else:
                y1 = int((rr1 - x * math.cos(math.radians(angg))) / (math.sin(math.radians(angg))))

        if 0 < y < lines.height:
            img.image[y, x, 0] = 0
            img.image[y, x, 1] = 0
            img.image[y, x, 2] = 255

        if lin and 0 < y1 < lines.height:
            img.image[y1, x, 0] = 0
            img.image[y1, x, 1] = 0
            img.image[y1, x, 2] = 255


for i in range(0, 2*number_of_lines, 2):
    draw_line(maximums[i], maximums[i+1], lines)

cv2.imshow('Canny Algorithm', img)
cv2.imshow('Edges', lines.image)
cv2.waitKey(0)
cv2.destroyAllWindows()
