import numpy as np
import cv2

def Quantization_Luminance():
    luminance = np.array(
        [[16, 11, 10, 16, 24, 40, 51, 61],
         [12, 12, 14, 19, 26, 58, 60, 55],
         [14, 13, 16, 24, 40, 57, 69, 56],
         [14, 17, 22, 29, 51, 87, 80, 62],
         [18, 22, 37, 56, 68, 109, 103, 77],
         [24, 35, 55, 64, 81, 104, 113, 92],
         [49, 64, 78, 87, 103, 121, 120, 101],
         [72, 92, 95, 98, 112, 100, 103, 99]])
    return luminance

def my_normalize(src):
    dst = src.copy()
    if np.min(dst) != np.max(dst):
        dst = dst - np.min(dst)
    dst = dst / np.max(dst) * 255
    return dst.astype(np.uint8)


def get_DCT(f, n=8):
    F = np.zeros((n, n))
    for u in range(n):
        for v in range(n):
            x, y = np.mgrid[0:n, 0:n]
            val = np.sum(f * np.cos(((2 * x + 1) * u * np.pi) / (2 * n)) * \
                         np.cos(((2 * y + 1) * v * np.pi) / (2 * n)))

            if(u == 0): C_u = 1/np.sqrt(n)
            else: C_u = np.sqrt(2) / np.sqrt(n)

            if (v == 0): C_v = 1 / np.sqrt(n)
            else: C_v = np.sqrt(2) / np.sqrt(n)

            F[u, v] = C_u * C_v * val

    return F

def get_IDCT(F, n = 8):
    f = np.zeros((n, n))
    u, v = np.mgrid[0:n, 0:n]
    c_u = np.zeros((n,n))
    c_v = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            if u[i,j] == 0: c_u[i,j] = 1 / np.sqrt(n)
            else: c_u[i,j] = np.sqrt(2) / np.sqrt(n)

            if v[i, j] == 0: c_v[i, j] = 1 / np.sqrt(n)
            else: c_v[i, j] = np.sqrt(2) / np.sqrt(n)

    for x in range(n):
        for y in range(n):
            val = np.sum(c_u * c_v * F * np.cos(((2 * x + 1) * u * np.pi) / (2 * n)) * \
                         np.cos(((2 * y + 1) * v * np.pi) / (2 * n)))

            f[x, y] = val

    return f

def my_DCT(src, n=8, type='DCT'):
    (h, w) = src.shape

    h_pad = h + (n - h%n)
    w_pad = w + (n - w%n)

    pad_img = np.zeros((h_pad, w_pad))
    pad_img[:h, :w] = src.copy()
    dst = np.zeros((h_pad, w_pad))
    
    for row in range(h_pad // n):
        for col in range(w_pad // n):
            if type=='DCT':
                dst[row * n: (row + 1) * n, col * n: (col + 1) * n] = \
                    get_DCT(pad_img[row * n: (row + 1) * n, col * n: (col + 1) * n], n)
            elif type=='IDCT':
                dst[row * n: (row + 1) * n, col * n: (col + 1) * n] = \
                    get_IDCT(pad_img[row * n: (row + 1) * n, col * n: (col + 1) * n], n)


    return dst[:h, :w]

def my_JPEG_encoding(src, block_size=8):
    (h, w) = src.shape
    zigzag_dst = np.zeros((h//block_size, w//block_size, block_size**2))
    zigzag_value = src.copy()
    dst = np.zeros((h,w), dtype=np.float)

    # Substract 128
    zigzag_value -= 128

    # DCT
    dct = my_DCT(zigzag_value, block_size,'DCT')

    # Divide Quantization
    n = block_size
    lum = Quantization_Luminance()
    for i in range (h//block_size):
        for j in range (w//block_size):
            div_lum = dct[i * n: (i + 1) * n, j * n: (j + 1) * n]
            dst[i * n: (i + 1) * n, j * n: (j + 1) * n] = np.round(div_lum / lum)

    #ZigZag Scanning
    for row in range(h//block_size):
        for col in range(w//block_size):
            block = dst[row * block_size : (row+1) * block_size, col * block_size : (col+1) * block_size].astype(int)
            zigzag_array = np.zeros(block_size ** 2, dtype=np.float)
            y, x = (0, 0)
            EOB = 0
            direction = 'down'
            for i in range(block_size ** 2):
                if block[y,x] !=0 :
                    zigzag_array[i] = block[y, x]
                    EOB = i

                if y == 0 and x == 0:
                    x += 1
                elif direction=='down':
                    if x == 0 or y+1 == block_size:
                        if x == 0 and y + 1 == block_size:
                            x += 1
                            direction = 'up'
                        elif x == 0 and y + 1 < block_size:
                            y += 1
                            direction = 'up'
                        elif x != 0 and y + 1 < block_size:
                            x += 1
                            direction = 'up'
                        elif x != 0 and y + 1 == block_size:
                            x += 1
                            direction = 'up'
                    else:
                        x -= 1
                        y += 1
                elif direction=='up':
                    if y == 0 or x+1 == block_size:
                        if y == 0 and x + 1 == block_size:
                            y += 1
                            direction = 'down'
                        elif y == 0 and x + 1 < block_size:
                            x += 1
                            direction = 'down'
                        elif y != 0 and x + 1 < block_size:
                            y -= 1
                            direction = 'down'
                        elif y != 0 and x + 1 == block_size:
                            y += 1
                            direction = 'down'
                    else:
                        x += 1
                        y -= 1

            if EOB+1 != block_size**2 : zigzag_array[EOB+1] = np.nan
            zigzag_dst[row, col] = zigzag_array.copy()

    # print(zigzag_dst.shape)
    return zigzag_dst

def my_JPEG_decoding(zigzag_value, block_size=8):
    (row, col, size) = zigzag_value.shape
    (h, w) = (row * block_size, col*block_size)
    zigzag_array = np.zeros((h, w), dtype=np.float)

    # Zigzag Decoding
    for r in range (row):
        for c in range (col):
            y, x = (0, 0)
            block = np.zeros((block_size, block_size), dtype=np.float)
            direction = 'down'
            for i in range (size):
                if np.isnan( zigzag_value[r, c, i] ) : break
                else : block[y, x] = zigzag_value[r, c, i]

                if y == 0 and x == 0:
                    x += 1
                elif direction=='down':
                    if x == 0 or y+1 == block_size:
                        if x == 0 and y + 1 == block_size:
                            x += 1
                            direction = 'up'
                        elif x == 0 and y + 1 < block_size:
                            y += 1
                            direction = 'up'
                        elif x != 0 and y + 1 < block_size:
                            x += 1
                            direction = 'up'
                        elif x != 0 and y + 1 == block_size:
                            x += 1
                            direction = 'up'
                    else:
                        x -= 1
                        y += 1
                elif direction=='up':
                    if y == 0 or x+1 == block_size:
                        if y == 0 and x + 1 == block_size:
                            y += 1
                            direction = 'down'
                        elif y == 0 and x + 1 < block_size:
                            x += 1
                            direction = 'down'
                        elif y != 0 and x + 1 < block_size:
                            y -= 1
                            direction = 'down'
                        elif y != 0 and x + 1 == block_size:
                            y += 1
                            direction = 'down'
                    else:
                        x += 1
                        y -= 1

            zigzag_array[r * block_size : (r+1) * block_size, c * block_size : (c+1) * block_size] = block

    # Multiply Quantization
    n = block_size
    lum = Quantization_Luminance()
    for i in range (h//block_size):
        for j in range (w//block_size):
            mul_lum = zigzag_array[i * n: (i + 1) * n, j * n: (j + 1) * n]
            zigzag_array[i * n: (i + 1) * n, j * n: (j + 1) * n] = np.round(mul_lum * lum)

    # IDCT
    idct = my_DCT(zigzag_array, block_size,'IDCT')

    # Add 128
    dst = idct + 128

    dst = my_normalize(dst)

    return dst


if __name__ == '__main__':
    src = cv2.imread('../Images/Lena.png', cv2.IMREAD_GRAYSCALE)

    src = src.astype(np.float)
    zigzag_value = my_JPEG_encoding(src)
    print(zigzag_value[:10])

    dst = my_JPEG_decoding(zigzag_value)
    src = src.astype(np.uint8)
    cv2.imshow('original', src)
    cv2.imshow('result', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


