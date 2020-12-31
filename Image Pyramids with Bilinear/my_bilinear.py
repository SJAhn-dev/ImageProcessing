import cv2
import numpy as np

def my_bilinear(src, dst_shape):
    (h, w) = src.shape
    (h_dst, w_dst) = dst_shape
    scale_h = h_dst / h
    scale_w = w_dst / w

    s = 1-(1/scale_w)
    t = 1/scale_h

    dst = np.zeros(dst_shape, np.uint8)
    for row in range(h_dst):
        for col in range(w_dst):
            bot = min( int(np.ceil( row / scale_h )), h-1 )
            top = int(np.floor( row / scale_h))
            right = min ( int(np.ceil ( col / scale_w )) , w-1 )
            left = int(np.floor( col / scale_w ))

            top_left = src[top, left] * s * (1-t)
            top_right = src[top, right] * (1-s) * (1-t)
            bot_left = src[bot, left] * s * t
            bot_right = src[bot, right] * (1-s) * t

            dst[row, col] = top_left + top_right + bot_left + bot_right

    return dst

if __name__ == '__main__':
    src = cv2.imread('../Images/Lena.png', cv2.IMREAD_GRAYSCALE)

    #이미지 크기 ??x??로 변경
    my_dst_mini = my_bilinear(src, (128, 128))

    #이미지 크기 512x512로 변경(Lena.png 이미지의 shape는 (512, 512))
    my_dst = my_bilinear(my_dst_mini, (512, 512))

    cv2.imshow('original', src)
    cv2.imshow('my bilinear mini', my_dst_mini)
    cv2.imshow('my bilinear', my_dst)

    cv2.waitKey()
    cv2.destroyAllWindows()
