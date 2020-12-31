import cv2
import numpy as np

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

def my_filtering(src, mask, pad_type='zero'):
    (h, w) = src.shape
    (m_h, m_w) = mask.shape

    pad_img = my_padding(src, (m_h//2, m_w//2), 'repetition')
    dst = np.zeros((h, w))
    for row in range(h):
        for col in range(w):
            dst[row, col] = np.sum(pad_img[row:row + m_h, col:col + m_w] * mask)

    return dst


#low-pass filter를 적용 후 high-pass filter적용
def apply_lowNhigh_pass_filter(src, fsize, sigma=1, pad_type='zero'):

    y, x = np.mgrid[-(fsize //2) : (fsize // 2) +1, -(fsize // 2):(fsize // 2) + 1]

    DoG_x = (-x / sigma**2) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    DoG_y = (-y / sigma**2) * np.exp(-(x**2 + y**2) / (2 * sigma**2))

    Ix = my_filtering(src, DoG_x, pad_type)
    Ix = Ix / 255
    Iy = my_filtering(src, DoG_y, pad_type)
    Iy = Iy / 255

    return Ix, Iy

#Ix와 Iy의 magnitude를 구함
def calcMagnitude(Ix, Iy):
    (h, w) = np.shape(Iy)
    magnitude = np.zeros((h,w))
    for y in range(h):
        for x in range(w):
            magnitude[y,x] = np.sqrt(Ix[y,x]**2 + Iy[y,x]**2)

    return magnitude

#Ix와 Iy의 angle을 구함
def calcAngle(Ix, Iy):
    angle = np.arctan2(Iy,Ix)
    angle = angle * 180. / np.pi
    angle[angle < 0] += 180

    return angle


#non-maximum supression 수행
def non_maximum_supression(magnitude, angle):
    (h, w) = magnitude.shape
    larger_magnitude = np.zeros((h, w))

    for y in range(h):
        for x in range(w):
            try:
                val = 255
                val2 = 255
                sval = abs(np.tan(angle[y, x]))
                tval = 1-sval

                if(angle[y, x] == 0):
                    val = magnitude[y, x + 1]
                    val2 = magnitude[y, x - 1]
                elif (0 < angle[y, x] < 22.5):
                    val = magnitude[y, x + 1] * tval + magnitude[y + 1, x + 1] * sval
                    val2 = magnitude[y, x - 1] * tval + magnitude[y - 1, x - 1] * sval
                elif (22.5 < angle[y, x] < 45):
                    val = magnitude[y, x + 1] * tval + magnitude[y + 1, x + 1] * sval
                    val2 = magnitude[y, x - 1] * tval + magnitude[y - 1, x - 1] * sval
                elif (angle[y, x] == 45):
                    val = magnitude[y + 1, x + 1]
                    val2 = magnitude[y - 1, x - 1]
                elif (45 < angle[y, x] < 67.5):
                    val = magnitude[y + 1, x + 1] * sval + magnitude[y + 1, x] * tval
                    val2 = magnitude[y - 1, x - 1] * sval + magnitude[y - 1, x] * tval
                elif (67.5 < angle[y, x] < 90):
                    val = magnitude[y + 1, x + 1] * sval + magnitude[y + 1, x] * tval
                    val2 = magnitude[y - 1, x - 1] * sval + magnitude[y - 1, x] * tval
                elif (angle[y, x] == 90):
                    val = magnitude[y + 1, x]
                    val2 = magnitude[y - 1, x]
                elif (90 < angle[y, x] < 112.5):
                    val = magnitude[y + 1, x - 1] * sval + magnitude[y + 1, x] * tval
                    val2 = magnitude[y - 1, x + 1] * sval + magnitude[y - 1, x] * tval
                elif (112.5 < angle[y, x] < 135):
                    s = 1 / abs(np.tan(angle[y, x]))
                    val = magnitude[y + 1, x] * tval + magnitude[y + 1, x - 1] * sval
                    val2 = magnitude[y - 1, x] * tval + magnitude[y - 1, x + 1] * sval
                elif (angle[y, x] == 135):
                    val = magnitude[y + 1, x - 1]
                    val2 = magnitude[y - 1, x + 1]
                elif (135 < angle[y, x] < 157.5):
                    s = abs(np.tan(angle[y, x]))
                    val = magnitude[y + 1, x] * tval + magnitude[y + 1, x - 1] * sval
                    val2 = magnitude[y - 1, x] * tval + magnitude[y - 1, x + 1] * sval
                elif (157.5 < angle[y, x] <= 180):
                    val = magnitude[y, x + 1] * tval + magnitude[y - 1, x + 1] * sval
                    val2 = magnitude[y, x - 1] * tval + magnitude[y + 1, x - 1] * sval

                if (magnitude[y,x] >= val) and (magnitude[y,x] >= val2):
                    larger_magnitude[y, x] = magnitude[y, x]
                else:
                    larger_magnitude[y, x] = 0

            except IndexError as e:
                pass

    #larger_magnitude값을 0~255의 uint8로 변환
    larger_magnitude = (larger_magnitude/np.max(larger_magnitude)*255).astype(np.uint8)
    return larger_magnitude

def double_thresholding(src):
    (h, w) = src.shape
    dst = np.zeros((h, w))

    high_value, dst = cv2.threshold(src, 0, 255, cv2.THRESH_OTSU)
    low_value = high_value * 0.4

    for y in range(h):
        for x in range(w):
            # Edge가 확실한 경우
            if(src[y, x] >= high_value):
                dst[y, x] = 255
            # Edge가 확실히 아닌 경우
            elif (src[y, x] < low_value):
                dst[y, x] = 0
            # 회색 선으로 표시한 후 나중에 처리
            else:
                try:
                    if (dst[y - 1, x - 1] == 255) or (dst[y - 1, x] == 255) or (dst[y - 1, x + 1] == 255):
                        dst[y, x] = 255
                    # pixel 좌우가 Edge인 경우
                    elif (dst[y, x - 1] == 255) or (dst[y, x + 1] == 255):
                        dst[y, x] = 255
                    # pixel 인접 하단열이 Edge인 경우
                    elif (dst[y + 1, x - 1] == 255) or (dst[y + 1, x] == 255) or (dst[y + 1, x + 1] == 255):
                        dst[y, x] = 255
                    else:
                        dst[y, x] = 0
                except IndexError as e:
                        pass
    return dst

def my_canny_edge_detection(src, fsize=5, sigma=1, pad_type='zero'):
    #low-pass filter를 이용하여 blur효과
    #high-pass filter를 이용하여 edge 검출
    Ix, Iy = apply_lowNhigh_pass_filter(src, fsize, sigma, pad_type)

    #magnitude와 angle을 구함
    magnitude = calcMagnitude(Ix, Iy)
    angle = calcAngle(Ix, Iy)

    #non-maximum suppression 수행
    larger_magnitude = non_maximum_supression(magnitude, angle)

    #진짜 edge만 남김
    dst = double_thresholding(larger_magnitude)

    return dst


if __name__ =='__main__':
    src = cv2.imread('../Images/Lena.png', cv2.IMREAD_GRAYSCALE)
    dst = my_canny_edge_detection(src)
    cv2.imshow('original', src)
    cv2.imshow('my canny edge detection', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()

