---
layout: post
title:  "Camera calibration with OpenCV"
title_img: /assets/post2/title_img.gif
abstract: A simple way to calibrate an optical system with a chessboard pattern by means of the OpenCV to reduce distortion.
date:   2017-05-05 12:00:00 +0300
categories: OpenCV
project: proj1
---
## Introduction

Optical systems are imperfect, that is why any camera prone to produce geometric distortions. There are Barrel and Pincushion distortions. In addiction, a mixture of both distortion types, sometimes referred to as mustache distortion.

Generally speaking, distortion is a deviation from rectilinear projection, hence, raw distorted images could not be used for photogrammetry.

## Camera calibration

In some cases the camera vendor provides the transformation matrix for undistortion. However, if it is not available, one can calculate all necessary coefficients with [OpenCV](http://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html) to be able to perform such undistortional transformation. OpenCV takes into account both the radial and tangential factors. 

To measure distortion of an optical systems it is possible to use photos of real world object with well-known shape. In the case, classical black-white chessboard pattern was used. It can be [downloaded](http://docs.opencv.org/2.4/_downloads/pattern.png) and printed on a well-sized paper. It is really important to attach the test sheet on a flat surface for proper camera calibration. It is enough to have 10-20 images for calibration.

Corners could be automatically found with the following command: {% highlight python %} cv2.findChessboardCorners(img, (Nx_cor,Ny_cor), None) {% endhighlight %}

Resulted code for a bunch of calibration images:

{% highlight python %}
import cv2
import glob
import numpy as np
Nx_cor = 9 # Number of corners to find
Ny_cor = 6
# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((Nx_cor*Ny_cor,3), np.float32)
objp[:,:2] = np.mgrid[0:Nx_cor, 0:Ny_cor].T.reshape(-1,2)
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('path/to/calibration/images') # Make a list of paths to calibration images
# Step through the list and search for chessboard corners
corners_not_found = [] # Calibration images in which OpenCV failed to find corners
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Conver to grayscale
    ret, corners = cv2.findChessboardCorners(gray, (Nx_cor,Ny_cor), None) # Find the corners
    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)
    else:
        corners_not_found.append(fname)
{% endhighlight %}

If it is needed, found corners can be visualized with an OpenCV function like this: {% highlight python %}cv2.drawChessboardCorners(img, (Nx_cor,Ny_cor), corners, ret){% endhighlight %}

![Corners found example](/assets/post2/corners_found.jpg)

Given the object points (an ideal model of desired points on a surface) and the image points (found chessboard corners), it is possible to calculate all calibration coefficients for the undistortion process:

{% highlight python %}
img = cv2.imread('test/image')
img_size = (img.shape[1], img.shape[0])
# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None,None)
{% endhighlight %}

## Image undistortion

OpenCV provides a useful function [undistort](https://en.wikipedia.org/wiki/Distortion_(optics)):
{% highlight python %}
cv2.undistort(img, mtx, dist, None, mtx)
{% endhighlight %}
Here is an example of its use:

![Undistortion example](/assets/post2/undist_img.jpg)

The code was used in the Lane Lines Detection [project][project-gh].

[project-gh]: /proj/proj1

