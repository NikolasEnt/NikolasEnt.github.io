---
layout: post
title:  "Camera Calibration: Practical OpenCV Refinement Techniques"
title_img: /assets/post22/title.webp
abstract: Practical advice on achieving better calibration accuracy in real-world scenarios while addressing common pitfalls and OpenCV model selection strategies.
date:   2024-12-20 12:00:00 +0000
categories: ComputerVision OpenCV Calibration
article: true
sitemap:
    lastmod: 2024-12-25
---

_This article is part of a series on camera calibration that has accumulated practical tips over more than a decade of experience. The series is not intended to be a comprehensive beginner's guide to camera calibration or provide instructions for a state-of-the-art optical laboratory setup. Rather, it focuses on the practical aspects of camera calibration in real-world environments with limited resources or DIY scenarios, while aiming to achieve the best possible results. This article explores algorithmic aspects of calibration, primarily focusing on OpenCV implementation; however, similar principles apply to other tools._

![Title image](/assets/post22/title.webp)

## Introduction

Despite attention to detail in capturing camera calibration images is a crucial factor for achieving final calibration accuracy, the algorithm itself is equally important. To achieve pixel-level accuracy in applications such as photogrammetry or other optical system measurements, the calibration process must aim for sub-pixel precision. However, there are multiple numerical and computational aspects of this process that can significantly impact the outcome, even for the same set of calibration images.

Understanding the limitations of the acquired  data and the chosen calibration approach is essential to realistically assess which parts of the camera model can be accurately accounted for and which elements might lead to unnecessary complexity in a particular real-world scenario. In many cases, using the most advanced camera models may not only be impractical but could even result in inadequate outcomes.

In this post, we will discuss several techniques for refining the calibration process using OpenCV tools. These include methods for improving keypoint detection, considering data quality and optics, selecting appropriate distortion models, and validating calibration results. This post serves as a continuation of the previous very basic camera calibration tutorial [\[1\]][1] and builds upon the previous discussion of hardware aspects of capturing of the calibration patterns [\[2\]][2].


## Refinement of Keypoints Detection

Once the keypoints are detected, it is possible to refine them to achieve sub-pixel detection accuracy. What is more, the refinement step can compensate for some optical characteristics and issues in the camera image processing pipeline, such as demosaicing or aliasing artifacts.

The following demonstrates the application of [cv2.cornerSubPix](https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga354e0d7c86d0d9da75de9b9701a9a87e) to refine checkerboard pattern corners recognition results. `winSize` and `zeroZone` are used to define the search window around each corner and can be tuned to achieve the best accuracy. `criteria` parameter defines the termination criteria for the iterative refinement process, which can be tuned to achieve the desired level of accuracy.

```python
import cv2

img = ... # Grayscale image, e.g. cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)
N_x, N_y = 9, 6 # Number of columns and rows in the checkerboard pattern
ret, corners = cv2.findChessboardCorners(img, (N_x, N_y), None)
if ret:
    corners_refined = cv2.cornerSubPix(
        image=img,
        corners=corners,
        winSize=(11, 11),
        zeroZone=(-1, -1),
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
```

## Consideration of the data quality and optics

While there are various camera models available for different optical designs, we will use the **pinhole camera model** as an example. The pinhole model forms the foundation for most commonly used calibration processes. It is mathematically defined as follows:

$$ \begin{bmatrix} x \\ y \\ w \end{bmatrix} = \begin{bmatrix} f_x & \alpha f_x & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} r_{11} & r_{12} & r_{13} & t_1 \\ r_{21} & r_{22} & r_{23} & t_2 \\ r_{31} & r_{32} & r_{33} & t_3 \end{bmatrix} \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix} $$

The projection matrix maps real-world point coordinates, $$\begin{bmatrix} X & Y & Z \end{bmatrix}$$, to the sensor's homogeneous coordinates, $$\begin{bmatrix} x & y & w \end{bmatrix}$$. The components of this model are as follows:

* $$(f_x, f_y)$$: **Focal lengths** along the $$x$$- and $$y$$- axis, respectively, expressed in pixel units.
* $$(c_x, c_y)$$: The **principal point**, or the location of intersection of the optical axis with the image plane, expressed in pixel coordinates.
* $$R = \{r_{ij}\}$$: The **rotation matrix**, representing the camera's orientation in 3D space.
* $$t = \begin{bmatrix} t_1 & t_2 & t_3 \end{bmatrix}^T$$: The **translation vector**, representing the camera's position in 3D space.
* $$\alpha$$: The **skew** coefficient, which describes the non-perpendicularity between the $$x$$- and $$y$$- axes in pixel coordinates. In the vast majority of cases, $$\alpha$$ is zero. However, non-zero values may occur in special cases, such as with line-scan image processing.

The general idea is to start with the simplest camera model possible, fixing the majority of degrees of freedom (e.g., assuming no distortion, and equal focal lengths along the x and y axes). This can be achieved by supplying proper flags to the calibration function [cv2.calibrateCamera](https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d), such as `cv2.CALIB_ZERO_TANGENT_DIST` for no radial distortion and `cv2.CALIB_SAME_FOCAL_LENGTH` for equal focal lengths along x and y.

For example, the following set of flags can be used as a starting point.
```python
flags = cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO \
        | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2\
        | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4\
        | cv2.CALIB_FIX_TANGENT_DIST
ret, mtx, dist, rvects, tvects = cv2.calibrateCamera(
                np.array([world_points], dtype=np.float32),
                np.array([corners_refined], dtype=np.float32),
                self.img_size, None, None, flags=flags)
```

Then, when satisfactory results are obtained, and the validation process indicates reliability, the degrees of freedom can be unfrozen one by one, taking into account optical features that are meaningful to optimize during the calibration process.

## Distortion Model

The OpenCV [cv2.calibrateCamera](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d) method supports various distortion models. These range from a simple 4-parameter model with two radial and two tangential distortion coefficients to a more complex models, with up to  14 parameters. The model can incorporate additional effects such as higher-order radial distortions (`cv2.CALIB_RATIONAL_MODEL` flag), thin prism distortions (`cv2.CALIB_THIN_PRISM_MODEL`), and a tilted sensor model, which accounts for cases where the image sensor is not aligned with the camera's optical axis (`cv2.CALIB_TILTED_MODEL`).

When selecting a distortion model, it is crucial to consider:

1. The types of distortions exist and prevalent in the optical system being calibrated.
2. The amount and quality of the calibration data available.

### Distortion Cases:

* **Tangential distortion** arises due to misalignment of optical elements with the optical axis of the lens. This is often a result of imperfect lens elements manufacturing or assembly. If the lens is known to be of high quality and well-aligned, tangential distortion may be negligible in practice.

* **Thin prism distortion** occurs when the lens optical elements are not perfectly aligned to be parallel with the camera sensor plane. Similar to tangential distortion, if the lens is manufactured to high standards, this effect is likely negligible.

* The **tilted sensor model** accounts for cases where the image sensor plane is not orthogonal to the optical axis. High-quality camera systems often employ advanced alignment techniques (for example, [Active Sensor Alignment](Active Sensor Alignment) by LUCID) to mitigate sensor tilt. In these cases, including the tilted sensor model may lead to overfitting and unnecessary complexity in the calibration process.
  However, there are situations where a tilted sensor model is essential. For instance, in [Scheimpflug principle](https://en.wikipedia.org/wiki/Scheimpflug_principle) applications (e.g., for lenses called 'tilt-shift' or 'perspective control' lenses), the optical axis is deliberately tilted relative to the sensor plane. In such cases, using the tilted sensor model is necessary for accurate calibration. A comprehensive review of Scheimpflug camera calibration methods, including advanced techniques, is available in [3].

It may happen that a particular optical system requires more complex models than those available in OpenCV.

### Practical Considerations

* **Overfitting**: Using a distortion model with unnecessary parameters can lead to overfitting, particularly if the calibration dataset lacks sufficient coverage or accuracy. It is generally advisable to start with simpler models and progressively add complexity only if needed.

* **Calibration Data**: Ensure that the calibration dataset includes a diverse set of calibration images with sufficient coverage of the whole image area by the calibration patterns to adequately constrain all distortion parameters being estimated.


## Refinement of Camera Parameters

* When the intrinsic camera parameters are already known, and only the extrinsic parameters are required (e.g., to find the position of the camera in the place of installation for a pre-calibrated camera), it is more reliable to use [cv2.solvePnP](https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga549c2075fac14829ff4a58bc931c033d) or other methods of the family, selecting the most optimal [method](https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga357634492a94efe8858d0ce1509da869). If there is an initial guess of the camera pose, such as a known installation location, it is worth considering running the camera pose refinement with virtual visual servoing ([cv2.solvePnPRefineVVS](https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga17491c0286a96d992f82c3e6dfa525fa)) or a general Levenberg-Marquardt iterative algorithm ([cv2.solvePnPRefineLM](https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga650ba4d286a96d992f82c3e6dfa525fa)).
* The camera matrix itself may also need refinement in some cases, especially when initial parameters are derived roughly from the camera parameters, like the sensor size and focal length. There are several ways to do it:
    * **Using Initial Estimates**: Provide the initial estimates as input to the standard calibration method, `cv2.calibrateCamera`. This allows OpenCV to iteratively refine the intrinsic parameters.
    * **Refinement with Fixed Parameters**: If some parameters such as focal length are known with high confidence, use flags like `cv2.CALIB_FIX_FOCAL_LENGTH` to fix these values during calibration, enabling more accurate refinement of the remaining parameters.
* In the case of inaccurate keypoints calibration data, roughly planar calibration targets and similar imperfect situations, it is beneficial to use a more advanced camera calibration approach ([cv2.calibrateCameraRO](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga11eeb16e5a458e1ed382fb27f585b753)), described in [4]. 

## Results validation

To evaluate the calibration images, one can use various techniques to access the image quality. For example, sharpness of the checkerboard pattern images can be assessed with [cv2.estimateChessboardSharpness ](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga1b976b476cd2083edd4323a34e9e1ffa). It makes no sense to use the data until the quality of the images is sufficient. OpenCV manual suggests that the sharpness value, which is a measure of a black-white transition edge width, should be below 3 px.

Validate the calibration by splitting the calibration images into two subsets: one used for calibration itself and another one for validation to ensure that your model is robust.

Another useful technique, which helps to verify if the derived calibration parameters are actually informative and accurate is to perform the calibration process several times (from data collection to calibration and results validation) and compare the results. The standard deviation of the calibration parameters can shed some light on whether a particular calibration parameter is reliable in the experimental setup. Out of experience, the principal point position often shows significant variation between experiments, indicating that the values should be fixed to the sensor center or the calibration process revised to improve accuracy.

Similarly, it is worth checking the calibration parameters standard deviations estimates, as produced by [cv2.calibrateCameraExtended](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d). 

To quantify calibration accuracy, various metrics can be used, such as reprojection error and the [cv2.norm](https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga55a581f0accd8d990af775d378e7e46c) function for $$L_2$$ metric computation.

```python
sum_error = 0
for i in range(len(obj_points)):
    img_points_reproj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(img_points[i], img_points_reproj, cv.NORM_L2) / len(img_points_reproj)
    sum_error += error

total_error = sum_error / len(objpoints)
print("Reprojection Error: ", total_error)
```

## Conclusion

Accurate camera calibration is a foundational step in many Optical and Computer Vision applications. By starting with a simple model, refining keypoint detection, and iteratively increasing model complexity, one can achieve robust and reliable results. OpenCV offers a comprehensive set of tools for camera calibration, enabling users to handle various optical configurations and real-world constraints effectively.

Ultimately, successful camera calibration is a combination of mathematical rigor, attention to detail, and iterative refinement.


## References:


[1]: {% post_url 2017-05-05-Camera-calibration-with-OpenCV %}
\[1\]: [Camera calibration with OpenCV]({% post_url 2017-05-05-Camera-calibration-with-OpenCV %})

[2]: {% post_url 2024-12-10-Tips-for-better-camera-calibration %}
\[2\]: [Camera Calibration: What to perfect before touching the code]({% post_url 2024-12-10-Tips-for-better-camera-calibration %})

\[3\]: C. Sun, H. Liu, M. Jia and Sh. Chen. Review of Calibration Methods for Scheimpflug Camera // Journal of Sensors, 2018, DOI: [10.1155/2018/3901431](https://doi.org/10.1155/2018/3901431).

\[4\]: K. H. Strobl and G. Hirzinger. More accurate pinhole camera calibration with imperfect planar target. // In 2011 IEEE International Conference on Computer Vision (ICCV), PP. 1068–1075, 2011. [pdf](https://elib.dlr.de/71888/1/strobl_2011iccv.pdf).