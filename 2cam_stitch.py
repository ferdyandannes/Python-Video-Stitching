# coding: utf-8

import argparse
import logging
import cv2
import numpy as np
#import matplotlib.pyplot as plt
from random import randrange
import time

flann = cv2.FlannBasedMatcher({'algorithm': 0, 'trees': 5}, {'checks': 50})
sift = cv2.xfeatures2d.SIFT_create()

cap_1 = cv2.VideoCapture('right.mp4')
cap_2 = cv2.VideoCapture('left.mp4')

def Laplacian_Pyramid_Blending_with_mask(A, B, m, num_levels = 6):
    # assume mask is float32 [0,1]

    # generate Gaussian pyramid for A,B and mask
    GA = A.copy()
    GB = B.copy()
    GM = m.copy()
    gpA = [GA]
    gpB = [GB]
    gpM = [GM]
    for i in xrange(num_levels):
        GA = cv2.pyrDown(GA)
        GB = cv2.pyrDown(GB)
        GM = cv2.pyrDown(GM)
        gpA.append(np.float32(GA))
        gpB.append(np.float32(GB))
        gpM.append(np.float32(GM))

    # generate Laplacian Pyramids for A,B and masks
    lpA  = [gpA[num_levels-1]] # the bottom of the Lap-pyr holds the last (smallest) Gauss level
    lpB  = [gpB[num_levels-1]]
    gpMr = [gpM[num_levels-1]]
    for i in xrange(num_levels-1,0,-1):
        # Laplacian: subtarct upscaled version of lower level from current level
        # to get the high frequencies
        LA = np.subtract(gpA[i-1], cv2.pyrUp(gpA[i]))
        LB = np.subtract(gpB[i-1], cv2.pyrUp(gpB[i]))
        lpA.append(LA)
        lpB.append(LB)
        gpMr.append(gpM[i-1]) # also reverse the masks

    # Now blend images according to mask in each level
    LS = []
    for la,lb,gm in zip(lpA,lpB,gpMr):
        ls = la * gm + lb * (1.0 - gm)
        LS.append(ls)

    # now reconstruct
    ls_ = LS[0]
    for i in xrange(1,num_levels):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, LS[i])

    return ls_

def trim(frame):
    if not np.sum(frame[0]):
        return trim(frame[1:])
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    if not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    if not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame

while(cap_1.isOpened()) :
    xrange = range
    start_time = time.time()
    ret_1, seq_1 = cap_1.read()
    seq_1_gray = cv2.cvtColor(seq_1, cv2.COLOR_BGR2GRAY)
    
    ret_2, seq_2 = cap_2.read()
    ee = cv2.copyMakeBorder(seq_2,50,50,150,150, cv2.BORDER_CONSTANT)
    seq_2_gray = cv2.cvtColor(seq_2, cv2.COLOR_BGR2GRAY)

    features0 = sift.detectAndCompute(seq_1_gray, None)
    features1 = sift.detectAndCompute(seq_2_gray, None)

    keypoints0, descriptors0 = features0
    keypoints1, descriptors1 = features1

    #img4 = cv2.drawKeypoints(seq_1, keypoints0, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #img5 = cv2.drawKeypoints(seq_2, keypoints1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    matches = flann.knnMatch(descriptors0, descriptors1, k=2)

    good = []
    for m in matches:
        if m[0].distance < 0.7*m[1].distance:         
            good.append(m)
    matches = np.asarray(good)
    
    #draw_params = dict(matchColor = (0, 255, 0), singlePointColor = (255, 0 ,0), matchesMask = matchesMask, flags = 0)
    #img6 = cv2.drawMatchesKnn(seq_1, keypoints0, seq_2, keypoints1, matchezz, None, **draw_params)

    src = np.float32([ keypoints0[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
    dst = np.float32([ keypoints1[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    dst = cv2.warpPerspective(seq_1, H, (seq_2.shape[1] + seq_1.shape[1], seq_2.shape[0]))
    dst[0:seq_2.shape[0],0:seq_2.shape[1]] = seq_2

    cv2.imshow("original_image_stitched_crop.jpg", trim(dst))

    k = cv2.waitKey(33)
    if k==27:
        break
    #cv2.waitKey(0)
    print("FPS: ", 1.0 / (time.time() - start_time))
        
cap_1.release()
cap_2.release()
cv2.waitKey(0)
