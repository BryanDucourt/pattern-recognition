import copy
import typing
import cv2 as cv
from skimage import feature as ft
import matplotlib.pyplot as plt
import numpy
import numpy as np


def GenerateFeature(fnames: list):
    features = []
    for file in fnames:
        tmp = []
        img = cv.imread(file, cv.IMREAD_GRAYSCALE)
        upscale = cv.resize(img, (64, 64), interpolation=cv.INTER_CUBIC)
        ret, binary = cv.threshold(upscale, 127, 255, cv.THRESH_BINARY)
        rough = _GenerateRough(binary.copy())
        outerEdge = _GenerateOuterEdge(binary.copy())
        innerEdge = _GenerateInnerEdge(binary.copy())
        hog = ft.hog(binary,orientations=6,pixels_per_cell=[8,8],cells_per_block=[2,2],visualize=True)

        tmp.extend(rough)
        tmp.extend(outerEdge)
        tmp.extend(innerEdge)
        tmp.extend(hog[0])
        features.append(tmp)
    return features


def _GenerateRough(img):
    skeleton = _GenerateSkeleton(img)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    skl = cv.dilate(skeleton,kernel)
    conts, hier = cv.findContours(skl, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    area = []
    for k in range(len(conts)):
        area.append(cv.contourArea(conts[k]))
    # 轮廓索引
    max_idx = np.argsort(np.array(area))
    mask = copy.deepcopy(img)
    # 按轮廓索引填充颜色
    for idx in max_idx:
        # 填充轮廓
        mask = cv.drawContours(mask, conts, idx, (0, 0, 0), cv.FILLED)
    rows = np.hsplit(mask, 4)
    cols = [np.vsplit(x, 4) for x in rows]
    cols_flatten = [np.sum(item == 0) for arr in cols for item in arr]

    return cols_flatten


def _GenerateOuterEdge(img: np.ndarray):
    arr = img
    oe = []
    for i in range(4):
        arr = arr.T[::-1]
        a = arr.copy()
        tmp = []
        for j in range(64):
            cnt = 0
            for k in range(64):
                if arr[j][k] == 0:
                    break
                cnt += 1
                a[j][k]=127
            tmp.append(cnt)
        area = np.array_split(tmp, 4)
        outer = [np.sum(x) for x in area]
        oe.extend(outer)
    return oe


def _GenerateInnerEdge(img: np.ndarray):
    arr = _GenerateSkeleton(img)

    ie = []
    for i in range(4):

        arr = arr.T[::-1]
        a = arr.copy()
        tmp = []
        for j in range(64):
            cnt = 0
            t_arr = a.copy()
            flag = False
            flag_ = False
            for k in range(64):
                if flag:
                    if flag_:
                        if a[j][k] == 0:
                            cnt += 1
                        else:
                            flag = False
                            break
                    else:
                        if a[j][k] == 0:
                            cnt += 1
                            flag_ = True
                else:
                    if a[j][k] == 255:
                        flag = True
            if not flag and flag_:
                tmp.append(cnt)

            else:
                tmp.append(0)
                a = t_arr.copy()
        area = np.array_split(tmp, 4)
        outer = [np.sum(x) for x in area]
        ie.extend(outer)
    return ie


def _GenerateSkeleton(img: np.ndarray):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    ret, binary = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)
    skl = np.full((64, 64), 0, dtype=img.dtype)
    tmp = binary
    while np.sum(tmp == 255) != 0:
        skl = np.bitwise_or(skl, tmp - cv.morphologyEx(tmp, cv.MORPH_OPEN, kernel))
        tmp = cv.erode(tmp, kernel)

    return skl


if __name__ == "__main__":
    feature = GenerateFeature(["./dataset/train_large/L_1.bmp"])
    print(len(feature[0]))
