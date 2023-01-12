---
layout: post
title:  "Bird's Eye View Transformation"
title_img: /assets/post3/title_img.gif
abstract: Convert a normal image to a Bird's Eye view projection with OpenCV.
date:   2017-05-07 14:00:00 +0300
categories: OpenCV
project: proj1
---
## Introduction

Parallel lines appear to converge on images from a front facing camera due to perspective. In order to keep parallel lines parallel for photogrammetry a bird's eye view transformation should be applied. The post describes how to transform images for lane lines detection.

## What transformation to use

Here it is a sample image to experiment with:

![Test image](/assets/post3/test_img.jpg)

Extract its region of interest:

![Test image](/assets/post3/roi.jpg)

It is possible to transform the image into Bird's Eye View with two different approaches:

_a)_ stretch the top row of pixels while keeping the bottom row unchanged:

![Transformation a](/assets/post3/transform_a.jpg)

_b)_ shrinking the bottom of the image while keeping the top row unchanged;

![Transformation a](/assets/post3/transform_b.jpg)

One may consider the first variant more obvious. However, it increases the spatial resolution (without adding information) for the distant part of the image and it could lead to line bounds erosion, hence, gradient algorithms may have difficulties with its detection. It also reduce viewing angle, therefore it is impossible to track other than central lanes.

The second way of transformation was selected as a better one because it preserves all avalable pixels from the raw image on the top edge where there is lower relative resolution. To find correct transformation, source and destinations points a test image with flat and straight road can be used for perspective measurements. 

## Code implementation with OpenCV

The whole code is quite simple:

{% highlight python %}
import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE_H = 223
IMAGE_W = 1280

src = np.float32([[0, IMAGE_H], [1207, IMAGE_H], [0, 0], [IMAGE_W, 0]])
dst = np.float32([[569, IMAGE_H], [711, IMAGE_H], [0, 0], [IMAGE_W, 0]])
M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
Minv = cv2.getPerspectiveTransform(dst, src) # Inverse transformation

img = cv2.imread('./test_img.jpg') # Read the test img
img = img[450:(450+IMAGE_H), 0:IMAGE_W] # Apply np slicing for ROI crop
warped_img = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H)) # Image warping
plt.imshow(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB)) # Show results
plt.show()
{% endhighlight %}

Inverse transformation can be done with `cv2.warpPerspective` with _Minv_ matrix, which was constracted in the same to M way, but with src and dst point swapped.

{% highlight python %}
img_inv = cv2.warpPerspective(warped_img, Minv, (IMAGE_W, IMAGE_H)) # Inverse transformation
{% endhighlight %}

The code was used in the Lane Lines Detection [project][project-gh].

[project-gh]: /proj/proj1

