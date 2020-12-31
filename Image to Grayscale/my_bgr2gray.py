import cv2
import numpy as np

def my_bgr2gray(src):
    dst1 = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    dst2 = (src[:,:,0] * 0.333) + (src[:,:,1] * 0.333) + (src[:,:,2] * 0.333)
    dst3 = src[:,:,0]
    dst2 = (dst2+0.5).astype(np.uint8)
    return dst1, dst2, dst3

src = cv2.imread('Penguins.png')

dst1, dst2, dst3 = my_bgr2gray(src)

cv2.imshow('original', src)
cv2.imshow('gray(cvtColor)', dst1)
cv2.imshow('gray(1/3)', dst2)
cv2.imshow('gray(one channel)', dst3)

cv2.waitKey()
cv2.destroyAllWindows()