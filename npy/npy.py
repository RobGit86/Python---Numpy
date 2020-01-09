import func as im
import cv2

# Source image string
str_src = 'img/test.jpg'
image = im.Img(str_src)
image.img_info()

# Grayscale
grayscale = image.gray_scale()

# Binary
binary = grayscale.binary()

cv2.imshow('Source', image.image)
cv2.imshow('Grayscale', grayscale.image)
cv2.imshow('Binary', binary.image)
cv2.waitKey(0)
cv2.destroyAllWindows()
