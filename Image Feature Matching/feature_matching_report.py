import cv2
import numpy as np
import random


def feature_matching(img1, img2, RANSAC=False, threshold = 300, keypoint_num = None, iter_num = 500, threshold_distance=10):
    sift = cv2.xfeatures2d.SIFT_create(keypoint_num)

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    distance = []
    for idx_1, des_1 in enumerate(des1):
        dist = []
        for idx_2, des_2 in enumerate(des2):
            dist.append(L2_distance(des_1, des_2))

        distance.append(dist)

    distance = np.array(distance)

    min_dist_idx = np.argmin(distance, axis=1)
    min_dist_value = np.min(distance, axis=1)

    points = []
    for idx, point in enumerate(kp1):
        if min_dist_value[idx] >= threshold:
            continue

        x1, y1 = point.pt
        x2, y2 = kp2[min_dist_idx[idx]].pt

        x1 = int(np.round(x1))
        y1 = int(np.round(y1))

        x2 = int(np.round(x2))
        y2 = int(np.round(y2))
        points.append([(x1, y1), (x2, y2)])


    # no RANSAC
    if not RANSAC:

        A = []
        B = []
        for idx, point in enumerate(points):
            A.append([point[0][0], point[0][1], 1, 0, 0, 0])
            A.append([0, 0, 0, point[0][0], point[0][1], 1])
            B.append([point[1][0]])
            B.append([point[1][1]])

        A = np.array(A)
        B = np.array(B)
        X = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, B))

        M = np.zeros((3,3))
        M[0:2, :] = X.reshape(2,3)
        M[2] = [0, 0, 1]


        M = np.array(M)
        M_= np.linalg.inv(M)

        #Backward 방식
        h, w = img1.shape[:2]
        dst = np.zeros((2*h,2*w,3))
        _h, _w = dst.shape[:2]
        for _row in range(_h):
            for _col in range(_w):
                #bilinear
                vec = np.dot(M_, np.array([[_col, _row, 1]]).T)
                c = vec[0,0]
                r = vec[1,0]
                if c < 0 or r < 0 or c > w-1 or r > h-1:
                    continue
                c_left = int(c)
                c_right = min(int(c+1), w-1)
                r_top = int(r)
                r_bottom = min(int(r+1), h-1)
                s = c - c_left
                t = r - r_top
                intensity = (1-s) * (1-t) * img1[r_top, c_left] \
                            + s * (1-t) * img1[r_top, c_right] \
                            + (1-s) * t * img1[r_bottom, c_left] \
                            + s * t * img1[r_bottom, c_right]
                dst[_row, _col] = intensity
        dst = dst.astype(np.uint8)

        top, bottom, left, right = _h, 0, _w, 0
        for row in range(_h):
            for col in range(_w):
                if dst[row, col, 0] != 0:
                    if left > col: left = col
                    if right < col: right = col
                    if top > row: top = row
                    if bottom < row: bottom = row

        new_dst = dst[top:bottom+1, left:right+1, :]
        dst = new_dst

    #use RANSAAC

    else:
        points_shuffle = points.copy()

        inliers = []
        M_list = []
        for i in range(iter_num):
            random.shuffle(points_shuffle)
            three_points = points_shuffle[:3]

            A = []
            B = []
            #3개의 point만 가지고 M 구하기
            for idx, point in enumerate(three_points):
                A.append([point[0][0], point[0][1], 1, 0, 0, 0])
                A.append([0, 0, 0, point[0][0], point[0][1], 1])
                B.append([point[1][0]])
                B.append([point[1][1]])

            A = np.array(A)
            B = np.array(B)
            try:
                X = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, B))

            except:
                print('can\'t calculate np.linalg.inv((np.dot(A.T, A)) !!!!!')
                continue

            M = np.zeros((3, 3))
            M[0:2, :] = X.reshape(2, 3)
            M[2] = [0, 0, 1]

            M_list.append(M)

            count_inliers = 0
            for idx, point in enumerate(points):
                m_point = np.dot(M, np.array([point[0][0], point[0][1], 1]).T).T
                r_point = [point[1][0], point[1][1], 1]
                if L2_distance(m_point, r_point) < threshold_distance:
                    count_inliers += 1

            inliers.append(count_inliers)

        inliers = np.array(inliers)
        max_inliers_idx = np.argmax(inliers)

        best_M = np.array(M_list[max_inliers_idx])

        M = best_M
        M_ = np.linalg.inv(M)

        row, col, layer = img1.shape
        top_left = np.dot(M, np.array([[0, 0, 1]]).T).T[0][:2]
        top_right = np.dot(M, np.array([[col, 0, 1]]).T).T[0][:2]
        bot_left = np.dot(M, np.array([[0, row, 1]]).T).T[0][:2]
        bot_right = np.dot(M, np.array([[col, row, 1]]).T).T[0][:2]

        h_value = abs(max(top_right[1], bot_right[1]) - min(bot_left[1], top_left[0])).astype(np.int)
        w_value = abs(max(top_right[0], top_left[0]) - min(bot_left[0], bot_right[0])).astype(np.int)
        moveValue = [min(bot_left[0], bot_right[0]), min(bot_left[1], top_left[0])]

        dst = np.zeros((h_value, w_value, layer))
        # Backward 방식
        h, w = dst.shape[:2]
        for _row in range(h):
            for _col in range(w):
                vec = np.dot(M_, np.array([[_col + moveValue[0], _row + moveValue[1], 1]]).T)
                c = vec[0, 0]
                r = vec[1, 0]
                if c < 0 or r < 0 or c > col - 1 or r > row - 1:
                    continue
                c_left = int(c)
                c_right = min(int(c + 1), w - 1)
                r_top = int(r)
                r_bottom = min(int(r + 1), h - 1)
                s = c - c_left
                t = r - r_top
                intensity = (1 - s) * (1 - t) * img1[r_top, c_left] \
                            + s * (1 - t) * img1[r_top, c_right] \
                            + (1 - s) * t * img1[r_bottom, c_left] \
                            + s * t * img1[r_bottom, c_right]
                dst[_row, _col] = intensity
        dst = dst.astype(np.uint8)

    return dst


def L2_distance(vector1, vector2):
    distance = np.sqrt(np.sum((vector1 - vector2)**2))
    return distance

def main():
    img = cv2.imread('../Images/building.jpg')
    img_ref = cv2.imread('../Images/building_temp.jpg')

    threshold = 300
    iter_num = 500
    #속도가 너무 느리면 100과 같이 숫자로 입력
    keypoint_num = None
    #keypoint_num = 50
    threshold_distance = 10

    dst_no_ransac = feature_matching(img, img_ref, threshold=threshold, keypoint_num=keypoint_num, iter_num=iter_num, threshold_distance=threshold_distance)
    dst_use_ransac = feature_matching(img, img_ref, RANSAC=True, threshold=threshold, keypoint_num=keypoint_num, iter_num=iter_num, threshold_distance=threshold_distance)

    cv2.imshow('No RANSAC', dst_no_ransac)
    cv2.imshow('Use RANSAC', dst_use_ransac)

    cv2.imshow('original image', img)
    cv2.imshow('reference image', img_ref)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ =='__main__':
    main()


