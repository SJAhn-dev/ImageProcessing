import numpy as np
import cv2
import my_padding as my_p
import time

def my_get_Gaussian2D_mask(msize, sigma=1):
    y, x = np.mgrid[-(msize // 2):(msize // 2) + 1, -(msize // 2):(msize // 2) + 1]
    '''
    y, x = np.mgrid[-1:2, -1:2]
    y = [[-1,-1,-1],
         [ 0, 0, 0],
         [ 1, 1, 1]]
    x = [[-1, 0, 1],
         [-1, 0, 1],
         [-1, 0, 1]]
    '''
    #2차 gaussian mask 생성
    gaus2D =   1 / (2 * np.pi * sigma**2) * np.exp(-(( x**2 + y**2 )/(2 * sigma**2)))

    #mask의 총 합 = 1
    gaus2D /= np.sum(gaus2D)

    return gaus2D

def my_get_Gaussian1D_mask(msize, sigma=1):
    x = np.mgrid[-(msize // 2):(msize // 2) + 1,]
    gaus1D = 1 / (2 * np.pi * sigma**2) * np.exp(-((x**2)/2 * sigma**2))

    gaus1D /= np.sum(gaus1D)

    return gaus1D

def my_filtering(src, mask, pad_type = 'zero'):
    (h, w) = src.shape

    #mask의 크기
    (m_h, m_w) = mask.shape

    #mask 확인
    print('<mask>')
    print(mask)

    # 직접 구현한 my_padding 함수를 이용
    pad_img = my_p.my_padding(src, (m_h//2, m_w//2), pad_type)

    dst = np.zeros((h, w))

    #시간을 확인하려면 4중 for문을 이용해야함
    for row in range(h):
        for col in range(w):
            sum = 0
            for m_row in range(m_h):
                for m_col in range(m_w):
                    sum += pad_img[row + m_row, col+m_col] * mask[m_row, m_col]
            dst[row, col] = sum
    '''
    #4중 for문 시간이 오래걸릴 경우 해당 코드 사용(시간을 확인하려면 해당 코드를 사용하면 안됨)
    for row in range(h):
        for col in range(w):
            dst[row, col] = np.sum(pad_img[row:row + m_h, col:col + m_w] * mask)
    '''
    return dst

if __name__ == '__main__':
    src = cv2.imread('../Images/Lena.png', cv2.IMREAD_GRAYSCALE)
    mask_size = 5
    gaus2D = my_get_Gaussian2D_mask(mask_size, sigma = 1)
    gaus1D = my_get_Gaussian1D_mask(mask_size, sigma = 1)

    #mask size 출력
    print('mask size : ', mask_size)

    #1차 가우시안 필터 적용 및 시간 재기
    print('1D gaussian filter')
    start = time.perf_counter()  # 시간 측정 시작
    dst_gaus1D= my_filtering(src, gaus1D)
    dst_gaus1D= my_filtering(dst_gaus1D, gaus1D.T)
    dst_gaus1D = (dst_gaus1D + 0.5).astype(np.uint8)
    end = time.perf_counter()  # 시간 측정 끝
    print('1D time : ', end-start)

    #2차 가우시안 필터 적용 및 시간 재기
    print('2D gaussian filter')
    start = time.perf_counter()  # 시간 측정 시작
    dst_gaus2D= my_filtering(src, gaus2D)
    dst_gaus2D = (dst_gaus2D + 0.5).astype(np.uint8)
    end = time.perf_counter()  # 시간 측정 끝
    print('2D time : ', end-start)

    #결과 이미지 출력
    cv2.imshow('original', src)
    cv2.imshow('1D gaussian img', dst_gaus1D)
    cv2.imshow('2D gaussian img', dst_gaus2D)

    #1차 가우시안 필터와 2차 가우시안 필터 비교, 차이가 없으면 count = 0
    (h, w) = dst_gaus1D.shape #(h, w) = dst_gaus2D.shape
    count = 0
    for i in range(h):
        for j in range(w):
            if dst_gaus1D[i, j] != dst_gaus2D[i, j]:
                count += 1 #두 값이 다르면 count += 1
    print('count : ', count)

    cv2.waitKey()
    cv2.destroyAllWindows()

