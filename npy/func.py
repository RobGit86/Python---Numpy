import cv2
import os
import numpy as np


class Img:
    def __init__(self, image):
        self.image_string = image
        self.image = cv2.imread(image)
        self.height = len(self.image)
        self.width = len(self.image[0])
        self.pixels = self.width * self.height

    def gray_scale(self):
        image = self.image.copy()
        xs = np.sum(image, axis=2)
        xs = xs / 3
        xs = xs.astype(int)
        image[:, :, 0] = xs
        image[:, :, 1] = xs
        image[:, :, 2] = xs

        filename = os.path.splitext(self.image_string)
        fullname = filename[0] + '-out-gray.jpg'
        cv2.imwrite(fullname, image)

        grayscale = Img(fullname)
        return grayscale

    def binary(self):
        img = self.image.copy()
        pi = np.zeros(256)

        for x in range(self.height):
            for y in range(self.width):
                idx = img[x][y][0]
                pi[idx] = pi[idx] + 1

        pi = pi / self.pixels
        threshold = 0
        tmp = 0.0

        for k in range(255):
            w0 = np.sum(pi[:k])
            w1 = np.sum(pi[k:])
            u0 = 0.0
            u1 = 0.0
            for g in range(k + 1):
                if w0 != 0:
                    u0 += ((pi[g] * g) / w0)
            for h in range(k + 1, 254):
                if w1 != 0:
                    u1 += ((pi[h] * h) / w1)

            if (w0 * w1) * (u1 - u0) * (u1 - u0) > tmp:
                tmp = (w0 * w1) * (u1 - u0) * (u1 - u0)
                threshold = k

        for x in range(self.height):
            for y in range(self.width):
                if img[x][y][0] <= threshold:
                    img[x][y][:] = 0
                else:
                    img[x][y][:] = 255

        filename = os.path.splitext(self.image_string)
        fullname = filename[0][:-9] + '-out-bin.jpg'
        cv2.imwrite(fullname, img)

        binary = Img(fullname)
        return binary

    def img_info(self):
        print('Width: ' + str(self.width))
        print('Height: ' + str(self.height))
        print('Pixels: ' + str(self.pixels))
