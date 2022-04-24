#! /usr/bin/python3.8

import os
import sys
import cv2
import camera as c_mx

import numpy as np
import cv2
import glob

import json

def calibrate():

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob('calib_radial.jpg')

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7,6),None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
            cv2.imshow('img',img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()
    return objpoints,imgpoints

def load(afile):
    l_data=None
    with open(afile, 'r') as f:
        l_data=json.load(f)
    return l_data

if __name__== "__main__":
    print("launching opencv")
    print(cv2.__version__)
    print("===============================================================")
    #start calibration
    o_pts,i_pts=calibrate()
    print(o_pts)
    #print(i_pts)
    #get camera properties

    img = cv2.imread('calib_radial.jpg')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(o_pts, i_pts, gray.shape[::-1], None, None) 
   
    print(ret)
    print(dist)
    print(rvecs)
    print(tvecs)
    #print(type(mtx))
    #print(mtx)

    data=load('conf.json') 
    cam=c_mx.camera(data["fx"],
                    data["fy"],
                    data["cx"],
                    data["cy"],
                    data["p"][0],
                    data["p"][1],
                    data["k"][0],
                    data["k"][1],
                    data["k"][2])

    print(cam.matrix())
    print(cam.dis_x()) 
    pts_3D=np.array([[1,1,2]],dtype=float)
    print(pts_3D)
    pts_2D=np.array([[1408.569,986.3185]],dtype=float)
    print(pts_2D)

    success, r_vec, t_vec = cv2.solvePnP(pts_3D,pts_2D,cam.matrix(),cam.dis_x(),flags=0)


