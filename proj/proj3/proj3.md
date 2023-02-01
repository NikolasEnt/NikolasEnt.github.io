---
layout: project_page
title: Image-Based Localization
permalink: /proj/proj3
project_id: proj3
github_url: https://github.com/udacity/self-driving-car/tree/master/image-localization/community-code/RoboCar
---

The Udacity Self-Driving Car Challenge #3.

My (a.k.a. RoboCar team) results and ideas for the Challenge are presented here.
The solution is based on Deep Learning and realized the "Localization as Classification approach" with GoogLeNet artificial neural networks. The approach took the 3d place on the final [leaderboard](https://github.com/udacity/self-driving-car/tree/master/challenges/challenge-3).

Details on this Challenge can be found [here](https://medium.com/udacity/challenge-3-image-based-localization-5d9cadcff9e7#.cv1xx261f).

### Contents:

1. Data preparation
2. Creation of image classes
3. Model training
4. Post processing


## Task

The task of the challenge was car localization given images from a front-facing camera on the road from Mountain View to San Francisco. Separate rides were provided for training (with known GPS coordinates for each frame) and tests (images only).

## Approach

The problem was solved in two steps:

* Direction classification (SF->MV or MV->SF)
* Localization on the El Camino road

The heart of the approach is a set of three GoogleNets. One is used to determine the direction, two others were dedicated to the localization (classification) of images within SF->MV or MV->SF path. Each of the two "roads" was divided into small "pointers" - areas, created by 50 consequent pictures in a row. The two CNNs were used to classify images by assigning pointers to them. Resulting predictions were interpolated within predicted pointers. As the algorithm does not have any speed estimation, speed from the training dataset was used to calculate the average coordinates shift between frames in each pointer.



### Hardware and Software
#### Hardware
Data processing and training of the NNs were performed on a machine with the following components:

* AMD Phenom(tm) II X4 925
* 16 GB of RAM
* Nvidia GeForce GTX 1070
* SSD storage device

#### Software

* OS OpenSUSE 13.2 (Harlequin) (x86_64), kernel 3.16.7-45-desktop
* Python 2.7.12 with lmdb, cv2, numpy
* Caffe with all Prerequisites and Python interface
* CUDA ver. 8.0.44
* cuDNN 5.1
* Nvidia driver v. 367.57

Average training rate for the setup: 4.23 iter/sec.

Average prediction rate: 45.7 fps.

