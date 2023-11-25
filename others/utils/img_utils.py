import re
import cv2
import numpy
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import os
import argparse
import glob
import math


def point2cam(pcd, intrinsic, width, height, x_scale, y_scale):
    # intrinsic = intrinsic.intrinsic_matrix
    # pcd = pcd * tsdf_cubic_size/512
    if intrinsic.shape[1] == 4:
        pcd = np.concatenate([pcd, np.ones([pcd.shape[0], 1], dtype=np.float32)], axis=1)
    cam_pts = np.matmul(intrinsic, pcd.T).T
    pts = cam_pts[:, :2]/np.expand_dims(cam_pts[:, 2], axis=1)    # (n, 2) 2 on behalf of (x, y)
    pts[:, 0] = pts[:, 0] / x_scale
    pts[:, 1] = pts[:, 1] / y_scale
    keep_idx_img_pts = select_points_in_frustum(pts, 0, 0, width, height)
    # img_pts = np.fliplr(pts)    # (n, 2) 2 on behalf of (y, x)
    # img_pts = img_pts[keep_idx_img_pts]
    # img_pts = np.rint(img_pts)

    return pts, keep_idx_img_pts


def select_points_in_frustum(img_pts, x1, y1, x2, y2):
    keep_ind = (img_pts[:, 0] > x1) * \
               (img_pts[:, 1] > y1) * \
               (img_pts[:, 0] < x2) * \
               (img_pts[:, 1] < y2)
    return keep_ind


def read_txt(filepath):
    sents = []
    extrinsic = []
    out = []
    with open(filepath, 'r') as f:
        f = open(filepath, 'r')
        lines = f.readlines()
    for index in range(len(lines)):
        if index > 0:
            sents.append(lines[index])
    for line in sents:
        res = re.split('\t|\n', line)
        extrinsic.append(res)
    for ex in extrinsic:
        for index in range(len(ex)):
            if ex[index] == '':
                continue
            ex[index] = float(ex[index])
            out.append(ex[index])
    s = numpy.asarray(out).reshape(4, 4)
    return s


def list_folders(path):
    folders = []
    for cur in os.listdir(path):
        if os.path.isdir(os.path.join(path, cur)) and not cur.startswith('.'):
            folders.append(cur)
    return folders


def read_intrinsic(filepath, width, height):
    m = np.loadtxt(filepath, dtype=np.float32)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, m[0, 0],
                                                  m[1, 1], m[0, 2], m[1, 2])
    return intrinsic


def read_extrinsic(filepath):
    m = np.loadtxt(filepath, dtype=np.float32)
    if np.isnan(m).any():
        return None
    return m  # (4, 4)


def rotsImg(image, angle, scale):
    """旋转缩放图像"""

    height = image.shape[0]
    width = image.shape[1]

    matRot = cv2.getRotationMatrix2D((height * 0.5, width * 0.5), angle, scale)
    # 中心，角度，缩放系数

    I_rot = cv2.warpAffine(image, matRot, (width, height))

    return I_rot


def img2world(pt):
    intrinsic = read_intrinsic()
    extrinsic = read_extrinsic()


def SIFT_CV(image, imageRe):
    """实现简单的SIFT检测"""
    sift = cv2.xfeatures2d.SIFT_create()

    # imageRot = rotsImg(image, -45, 0.8)  # 旋转缩放
    imageGray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    imageReGray = cv2.cvtColor(imageRe, cv2.COLOR_RGB2GRAY)
    keypoint_1 = []
    kp1, des1 = sift.detectAndCompute(imageGray, None)
    kp2, des2 = sift.detectAndCompute(imageReGray, None)
    for index in range(len(kp1)):
        keypoint = np.asarray(kp1[index].pt)
        keypoint_1.append(keypoint)
    imageSIFT = cv2.drawKeypoints(image, kp1, image, color=(255, 0, 255))  # 画出特征点，并显示为红色圆圈
    imageRotSIFT = cv2.drawKeypoints(imageRe, kp2, imageRe, color=(255, 0, 255))  # 画出特征点，并显示为红色圆圈

    hmerge = np.hstack((imageSIFT, imageRotSIFT))  # 水平拼接

    return hmerge, des1, des2, kp1, kp2, imageRe


def BFmatch(image, des1, des2, kp1, kp2, imageRe, dis=True):
    """使用KNN算法进行匹配，如果dis为false，直接匹配，默认为True"""
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    KP_1, KP_2 = [], []
    if (dis == True):
        good = []
        for m, n in matches:
            if m.distance < 0.35 * n.distance:
                index_1 = m.queryIdx
                index_2 = m.trainIdx
                # keypoint_1 = tuple(map(int, kp1[index_1].pt))
                # keypoint_2 = tuple(map(int, kp2[index_2].pt))
                keypoint_1 = kp1[index_1].pt
                keypoint_2 = kp2[index_2].pt
                KP_1.append(keypoint_1)
                KP_2.append(keypoint_2)
                good.append([m])
                # cv2.circle(image, keypoint_1, radius=5, color=(255, 0, 0), thickness=2)
                # cv2.circle(imageRe, keypoint_2, radius=5, color=(255, 0, 0), thickness=2)
        imgRes = cv2.drawMatchesKnn(image, kp1, imageRe, kp2, good, None, flags=2)
    else:
        imgRes = cv2.drawMatchesKnn(image, kp1, imageRe, kp2, matches, None, flags=2)

    return KP_1, KP_2, imgRes