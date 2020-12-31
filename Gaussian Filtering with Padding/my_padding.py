import numpy as np
import cv2

def my_padding(src, pad_shape, pad_type = 'zero'):
    (h, w) = src.shape
    (p_h, p_w) = pad_shape
    pad_img = np.zeros((h + 2 * p_h, w + 2 * p_w))
    pad_img[p_h:h + p_h, p_w:w + p_w] = src

    if pad_type == 'repetition':
        #print('repetition padding')
        #up
        pad_img[ :p_h, p_w:p_w + w] = src[0, :]
        #down
        pad_img[p_h + h: , p_w:p_w + w] = src[h-1,:]

        #left
        pad_img[:,:p_w] = pad_img[:,p_w:p_w + 1]
        #right
        pad_img[:,p_w + w:] = pad_img[:,p_w + w - 1:p_w + w]

    else:
        #else is zero padding
        #print('zero padding')
        pass

    return pad_img

if __name__=='__main__':
    src = cv2.imread('../Images/Lena.png', cv2.IMREAD_GRAYSCALE)

    #zero padding
    my_pad_img = my_padding(src, (20, 20))

    #repetition padding
    #my_pad_img = my_padding(src, (20, 20), 'repetition')

    #데이터타입 uint8로 변경
    my_pad_img = (my_pad_img + 0.5).astype(np.uint8)
    cv2.imshow('original', src)
    cv2.imshow('my padding img', my_pad_img)

    cv2.waitKey()
    cv2.destroyAllWindows()


