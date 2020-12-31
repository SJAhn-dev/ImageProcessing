import cv2
import numpy as np

def my_filtering(src, ftype, fsize):
    (h, w) = src.shape
    dst = np.zeros((h, w))

    if ftype == 'average':
        print('average filtering')
        mask = np.ones(fsize, np.float32) / (fsize[0] * fsize[1])
        #mask 확인
        print(mask)

    elif ftype == 'sharpening':
        print('sharpening filtering')
        mask = np.ones(fsize, np.float32) / (fsize[0] * fsize[1]) * -1
        middle_h = int(fsize[0] / 2)
        middle_w = int(fsize[1] / 2)

        mask[middle_h][middle_w] = mask[middle_h][middle_w] + 2
        #mask 확인
        print(mask)

    left = int(-fsize[1] / 2)
    right = int(fsize[1] / 2)
    top = int(-fsize[0] / 2)
    bottom = int(fsize[0] / 2)
    for row in range(h):
        if(row + top <= 0 or row + bottom >= h):
            continue
        for col in range(w):
            if(col + left <= 0 or col + right >= w):
                continue
            data = src[ row+top:row+bottom+1 , col+left:col+right+1 ]
            dst[row,col] = np.sum(mask*data)

            if(dst[row,col] < 0):
                dst[row,col] = 0
            elif(dst[row,col] > 255):
                dst[row,col] = 255

    dst = (dst+0.5).astype(np.uint8)

    return dst

if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    # 3x3 filter
    # dst_average = my_filtering(src, 'average', (3,3))
    # dst_sharpening = my_filtering(src, 'sharpening', (3,3))

    # 11x13 filter
    dst_average = my_filtering(src, 'average', (11,13))
    dst_sharpening = my_filtering(src, 'sharpening', (11,13))

    cv2.imshow('original', src)
    cv2.imshow('average filter', dst_average)
    cv2.imshow('sharpening filter', dst_sharpening)
    cv2.waitKey()
    cv2.destroyAllWindows()
