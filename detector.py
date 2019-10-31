import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import filters

"""
Harris corner detecter
Penghui Wang
2019/10/28
"""

src1 = np.array(Image.open('truck1.jpg').convert('L')).T
src2 = np.array(Image.open('truck2.jpg').convert('L')).T


# def harris_response(img, window_size=3, harris_k=0.05):
#     ix = cv2.convertScaleAbs(cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3))
#     iy = cv2.convertScaleAbs(cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3))
#
#     ixx = ix * ix
#     iyy = iy * ix
#     ixy = ix * iy
#
#     Ixx = cv2.GaussianBlur(ixx, (window_size, window_size), 0)
#     Iyy = cv2.GaussianBlur(iyy, (window_size, window_size), 0)
#     Ixy = cv2.GaussianBlur(ixy, (window_size, window_size), 0)
#
#     det = Ixx * Iyy - Ixy ** 2
#     trace = Ixx + Iyy
#     response = det - harris_k * (trace**2)
#
#     return response

# TODO：why cannot work?
# def harris_response2(img, sigma=3, harris_k=0.05):
#     ix = np.zeros_like(img)
#     iy = np.zeros_like(img)
#     filters.gaussian_filter(img, (sigma, sigma), (0, 1), ix)
#     filters.gaussian_filter(img, (sigma, sigma), (1, 0), iy)
#
#     ixx = filters.gaussian_filter(ix * ix, sigma)
#     iyy = filters.gaussian_filter(iy * iy, sigma)
#     ixy = filters.gaussian_filter(ix * iy, sigma)
#
#     det = ixx * iyy - ixy ** 2
#     tr = ixx + iyy
#     response = det - (tr**2) * harris_k
#
#     return response

def harris_response(im, sigma=3):
    imx = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (0, 1), imx)
    imy = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (1, 0), imy)

    Wxx = filters.gaussian_filter(imx * imx, sigma)
    Wxy = filters.gaussian_filter(imx * imy, sigma)
    Wyy = filters.gaussian_filter(imy * imy, sigma)

    Wdet = Wxx * Wyy - Wxy ** 2
    Wtr = Wxx + Wyy

    return Wdet-0.04*(Wtr**2)

def show_harris_response(response, thresh=0.2):
    # res = np.zeros_like(response)
    # res2 = np.zeros_like(response)
    # cv2.normalize(response, res, 0, 255, cv2.NORM_MINMAX)
    # cv2.threshold(res, 255 * thresh, 255, cv2.THRESH_TOZERO, res2)
    cv2.namedWindow('Harris response', 0)
    cv2.imshow('Harris response', response)
    cv2.waitKey()
    cv2.destroyAllWindows()


def filter_points(response, thresh=0.05):
    threshold = response.max() * thresh
    candidates = (response > threshold) * 1
    coords = np.array(candidates.nonzero()).T

    candidate = [[c[0], c[1], response[c[0], c[1]]] for c in coords]
    return candidate


# TODO: give a tuple, return top_k corners with radius
def non_maxima_suppression(candidates, nms_window=13, top_k=250):
    # 按照响应值排序
    candidates.sort(key=lambda s: -s[2])
    num = len(candidates)
    key_points_count = 0
    for i in range(0, num):
        if key_points_count >= top_k:
            break
        if candidates[i][2] > 0:
            for j in range(i+1, num):
                if (abs(candidates[i][0] - candidates[j][0]) < (nms_window-1)/2) and \
                        (abs(candidates[i][1] - candidates[j][1] < (nms_window-1)/2)):
                    candidates[j][2] = 0
            key_points_count = key_points_count + 1

    key_points = candidates[:i]
    key_points = list(filter(lambda s: s[2] > 0, key_points))

    return key_points


def show_key_points(points, img):
    plt.figure()
    plt.gray()
    plt.imshow(img)
    # plt显示图像坐标轴为右下，画点时要更换横纵坐标。
    plt.plot([s[1] for s in points], [s[0] for s in points],
             'm*', markersize=1)
    plt.axis('off')
    plt.savefig('Harris Points', dpi=600)
    plt.show()


# def get_sift_descriptors(img, points, windows_size):

response = harris_response(src1)
cadidates = filter_points(response)
key_points = non_maxima_suppression(cadidates)
show_key_points(key_points, src1)
show_harris_response(response)
