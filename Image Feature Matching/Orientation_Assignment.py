import cv2
import numpy as np
import matplotlib.pyplot as plt


def my_padding(src, filter, type='repetition'):
    (h, w) = src.shape
    if isinstance(filter, tuple):
        (h_pad, w_pad) = filter
    else:
        (h_pad, w_pad) = filter.shape
    h_pad = h_pad // 2
    w_pad = w_pad // 2
    padding_img = np.zeros((h + h_pad * 2, w + w_pad * 2))
    padding_img[h_pad:h + h_pad, w_pad:w + w_pad] = src

    # repetition padding
    # up
    padding_img[:h_pad, w_pad:w_pad + w] = src[0, :]
    # down
    padding_img[h_pad + h:, w_pad:w_pad + w] = src[h - 1, :]
    # left
    padding_img[:, :w_pad] = padding_img[:, w_pad:w_pad + 1]
    # right
    padding_img[:, w_pad + w:] = padding_img[:, w_pad + w - 1:w_pad + w]

    return padding_img


def my_filtering(src, filter):
    (h, w) = src.shape
    (f_h, f_w) = filter.shape

    # filter 확인
    # print('<filter>')
    # print(filter)

    # 직접 구현한 my_padding 함수를 이용
    pad_img = my_padding(src, filter)

    dst = np.zeros((h, w))
    for row in range(h):
        for col in range(w):
            dst[row, col] = np.sum(pad_img[row:row + f_h, col:col + f_w] * filter)

    return dst


def get_my_sobel():
    sobel_x = np.dot(np.array([[1], [2], [1]]), np.array([[-1, 0, 1]]))
    sobel_y = np.dot(np.array([[-1], [0], [1]]), np.array([[1, 2, 1]]))
    return sobel_x, sobel_y


def calc_derivatives(src):
    # calculate Ix, Iy
    sobel_x, sobel_y = get_my_sobel()
    Ix = my_filtering(src, sobel_x)
    Iy = my_filtering(src, sobel_y)
    return Ix, Iy


def calc_angle(Ix, Iy):
    # return np.rad2deg(np.arctan(Iy/(Ix+1E-6)))
    # -180 ~ 180
    angle = np.rad2deg(np.arctan2(Iy, Ix))

    # 0 ~ 360
    return (angle + 360) % 360


def calc_magnitude(Ix, Iy):
    magnitude = np.sqrt(Ix ** 2 + Iy ** 2)
    return magnitude


def calc_patch_hist(patch_ang, patch_mag, angle_range):
    h, w = patch_ang.shape[:2]
    assert h, w == patch_mag.shape[:2]
    vector_size = 360 // angle_range

    vector = np.zeros(vector_size, )
    for row in range(h):
        for col in range(w):
            vector[int(patch_ang[row, col] // angle_range)] += patch_mag[row, col]
    return vector


def get_histogram(angle, magnitude, window_size=16, angle_range=30):
    h, w = angle.shape[:2]
    h = h // window_size
    w = w // window_size

    assert 360 % angle_range == 0
    vector_size = 360 // angle_range
    patches_vector = np.zeros((h, w, vector_size))
    print('calculate histogram...')
    for row in range(h):
        for col in range(w):
            patch_ang = angle[row * window_size:(row + 1) * window_size, col * window_size:(col + 1) * window_size]
            patch_mag = magnitude[row * window_size:(row + 1) * window_size, col * window_size:(col + 1) * window_size]
            patches_vector[row, col] = calc_patch_hist(patch_ang, patch_mag, angle_range)

    return patches_vector

def show_patch_hist(patch_vector):
    index = np.arange(len(patch_vector))
    plt.bar(index, patch_vector)
    plt.title('Image')
    plt.show()

def main():
    src = cv2.imread('../Images/Lena.png')
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    print('get Ix and Iy...')
    Ix, Iy = calc_derivatives(gray)
    print('calculate angle and magnitude')
    angle = calc_angle(Ix, Iy)
    magnitude = calc_magnitude(Ix, Iy)

    patches_vector = get_histogram(angle, magnitude, window_size=16, angle_range=30)

    print('angle')
    print(angle[:4, :4])
    print('magnitude')
    print(magnitude[:4, :4])
    print('vector')
    print(patches_vector[0, 0])

    show_patch_hist(patches_vector[0,0])

if __name__ == '__main__':
    main()