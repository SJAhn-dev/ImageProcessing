import numpy as np
import cv2
import my_padding as my_p
def my_get_Gaussian2D_mask(msize, sigma=1):
    y, x = np.mgrid[-(msize // 2):(msize // 2) + 1, -(msize // 2):(msize // 2) + 1]
    #2차 gaussian mask 생성
    gaus2D =   1 / (2 * np.pi * sigma**2) * np.exp(-(( x**2 + y**2 )/(2 * sigma**2)))
    #mask의 총 합 = 1
    gaus2D /= np.sum(gaus2D)
    print(gaus2D)
    return gaus2D

def my_filtering(src, mask, pad_type='zero'):
    (h, w) = src.shape
    (m_h, m_w) = mask.shape
    pad_img = my_p.my_padding(src, (m_h // 2, m_w // 2), pad_type)
    dst = np.zeros((h, w))
    for row in range(h):
        for col in range(w):
            dst[row, col] = np.sum(pad_img[row:row + m_h, col:col + m_w] * mask)
    return dst

def my_normalize(src):
    dst = src.copy()
    dst = dst - np.min(dst)
    dst = dst / np.max(dst) * 255
    return dst.astype(np.uint8)

def my_bilateral(src, msize, sigma, sigma_r, pad_type='zero'):
    (h, w) = src.shape
    pad_img = my_p.my_padding(src, (msize//2, msize//2))
    dst = np.zeros((h,w))

    y, x = np.mgrid[-(msize // 2):(msize // 2) + 1, -(msize // 2):(msize // 2) + 1]
    gaus_mask = 1 / (2 * np.pi * sigma**2) * np.exp(-((x ** 2 + y ** 2) / (2 * sigma ** 2)))

    for i in range(h):
        for j in range(w):
            slice = pad_img[i:i + msize, j:j + msize]
            centerVal = src[i, j]
            centerVal_array = np.full((msize, msize), centerVal)
            mask = np.exp(-(((centerVal_array - slice) ** 2)/(2 * sigma_r**2)))
            mask = gaus_mask * mask
            mask /= np.sum(mask)
            dst[i, j] = np.sum(pad_img[i:i + msize, j:j + msize] * mask)

            if i == 51 and j == 121:
                print(mask)
                mask_img = cv2.resize(mask, (200,200), interpolation= cv2.INTER_NEAREST)
                mask_img = my_normalize(mask_img)
                cv2.imshow('mask', mask_img)
                img = src[i-msize//2:i+msize//2+1, j-msize//2:j+msize//2+1]
                img = cv2.resize(img, (200,200), interpolation
                = cv2.INTER_NEAREST)
                img = my_normalize(img)
                cv2.imshow('mask img', img)

    dst = my_normalize(dst)

    return dst

if __name__ == '__main__':
    src = cv2.imread('../Images/Penguins_noise.png', cv2.IMREAD_GRAYSCALE)
    dst = my_bilateral(src, 5, 3, 30)
    gaus2D = my_get_Gaussian2D_mask(5 , sigma = 1)
    dst_gaus2D= my_filtering(src, gaus2D)
    dst_gaus2D = my_normalize(dst_gaus2D)

    cv2.imshow('original', src)
    cv2.imshow('my gaussian', dst_gaus2D)
    cv2.imshow('my bilateral', dst)

    cv2.waitKey()
    cv2.destroyAllWindows()

