---
layout: post
title:  "Camera calibration: Homography vs OpenCV"
title_img: /assets/post20/title.webp
abstract: Comparison of the direct OpenCV camera calibration method with an approach based on a homography matrix.
date:   2024-11-12 12:00:00 +0000
categories: ComputerVision OpenCV Calibration
article: true
sitemap:
    lastmod: 2025-06-19
---

*The idea for this post arose as a response to a common misconception I frequently hear: that camera calibration is equivalent to homography computation. This post will demonstrate the difference between OpenCV-based camera calibration and the pure homography-based calibration method.*

![Title image](/assets/post20/title.webp)

## Two calibration approaches

Camera calibration is the process of obtaining camera extrinsic and intrinsic parameters. These parameters are essential for performing accurate photogrammetry tasks, such as measuring the sizes or other characteristics, like speed, of physical objects based on camera imagery data. The process is also applicable when it is necessary to determine the camera's pose relative to a known world origin point.

The key step in calibrating a camera involves obtaining one or several images of a well-known object and a list of corresponding points in both the camera and world coordinate systems. There are many ways to find these correspondences, as demonstrated in previous posts: checkboard patterns [\[1\]][1] and football pitch lines as the calibration pattern [\[2\]][2]. The general rule is that the more points used and the better their uniform coverage of the entire frame by keypoints, the more accurate the calibration will be.
In addition to acquiring point pairs, there is a subsequent step of processing these points to obtain the camera parameters, which can be done in several different ways.

One option is to use OpenCV's [calibrateCamera](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d) method, which is based on Zhengyou Zhang's algorithm [3]. Internally, this method computes the initial intrinsic camera parameters, then estimates the camera pose by solving Perspective-n-Points problem (see [solvePnP](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga549c2075fac14829ff4a58bc931c033d)), and finally refines all parameters through Levenberg-Marquardt optimisation to minimise reprojection error.

```python
import cv2

_, mtx, dist, rvect, tvect = cv2.calibrateCamera(
    world_points, camera_points, img_size,
    cameraMatrix=None, distCoeffs=None)
```

The second option is to use a homography-based method, which is based on the idea of finding a transformation between two images that maps points from one image to another. From this homography matrix, it is possible to derive the intrinsic and extrinsic parameters of the camera. The computation of intrinsic parameters leverages multiple constraints, such as the square pixel assumption, to make the problem solvable. This algorithm is described in detail in the classical book "Multiple View Geometry in Computer Vision" by Richard Hartley and Andrew Zisserman [4], specifically algorithm 8.2 on page 225.

1. Find homography matrix, filtering points by RANSAC method. See details and other parameters of the function in [cv2.findHomography](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga4abc2ece9fab9398f2e560d53c8c9780) documentation.

    ```python
    import cv2

    # ransacReprojThreshol - reprojection error in px to consider the point pair
    homography, _ = cv2.findHomography(
        world_points, camera_points, cv2.RANSAC, ransacReprojThreshol=10)
    ```

2. Compute the intrinsic camera parameters from the homography matrix.

    ```python
    import numpy as np
    # Based on implementation from
    # https://github.com/SoccerNet/sn-calibration/blob/main/src/camera.py

    def calibration_matrix_from_homography(
            homography: np.ndarray,
            img_size: tuple[int, int] = (1280, 720)):

        image_width, image_height = img_size
        H = np.reshape(homography, (9,))
        A = np.zeros((5, 6))  # Constraint matrix
        A[0, 1] = 1.0
        A[1, 0] = 1.0
        A[1, 2] = -1.0
        A[2, 3] = image_height / image_width
        A[2, 4] = -1.0
        A[3, 0] = H[0] * H[1]
        A[3, 1] = H[0] * H[4] + H[1] * H[3]
        A[3, 2] = H[3] * H[4]
        A[3, 3] = H[0] * H[7] + H[1] * H[6]
        A[3, 4] = H[3] * H[7] + H[4] * H[6]
        A[3, 5] = H[6] * H[7]
        A[4, 0] = H[0] * H[0] - H[1] * H[1]
        A[4, 1] = 2 * H[0] * H[3] - 2 * H[1] * H[4]
        A[4, 2] = H[3] * H[3] - H[4] * H[4]
        A[4, 3] = 2 * H[0] * H[6] - 2 * H[1] * H[7]
        A[4, 4] = 2 * H[3] * H[6] - 2 * H[4] * H[7]
        A[4, 5] = H[6] * H[6] - H[7] * H[7]

        _, _, vh = np.linalg.svd(A)  # Note: SVD may not converge
        w = vh[-1]
        W = np.zeros((3, 3))
        W[0, 0] = w[0] / w[5]
        W[0, 1] = w[1] / w[5]
        W[0, 2] = w[3] / w[5]
        W[1, 0] = w[1] / w[5]
        W[1, 1] = w[2] / w[5]
        W[1, 2] = w[4] / w[5]
        W[2, 0] = w[3] / w[5]
        W[2, 1] = w[4] / w[5]
        W[2, 2] = w[5] / w[5]

        # Note: Cholesky decomposition may fail if the matrix is not positive definite
        Ktinv = np.linalg.cholesky(W)
        # pinv instead of inv to be more robust to numerical issues
        K = np.linalg.pinv(np.transpose(Ktinv))
        # Normalize K so K[2, 2] element is 1
        K /= K[2, 2]
        fx = K[0, 0]
        fy = K[1, 1]
        cx = image_width / 2.0
        cy = image_height / 2.0

        return np.ndarray([[fx, 0, cx],
                            [0, fy, cy],
                            [0, 0, 1]])

    ```

3. Estimate the camera pose from the homography matrix.

    ```python
    # homography from cv2.findHomography
    # camera_matrix from calibration_matrix_from_homography

    def extrinsic_from_homography(
        homography: np.ndarray,
        camera_matrix: np.ndarray):

        hprim = np.linalg.pinv(camera_matrix) @ homography
        lambda1 = 1 / np.linalg.norm(hprim[:, 0])
        lambda2 = 1 / np.linalg.norm(hprim[:, 1])
        lambda3 = np.sqrt(lambda1 * lambda2)

        r0 = hprim[:, 0] * lambda1
        r1 = hprim[:, 1] * lambda2
        r2 = np.cross(r0, r1)

        R = np.column_stack((r0, r1, r2))
        u, s, vh = np.linalg.svd(R)
        R = u @ vh
        if np.linalg.det(R) < 0:
            u[:, 2] *= -1
            R = u @ vh
        rotation = R
        t = hprim[:, 2] * lambda3
        position = - np.transpose(R) @ t
        return R, position
    ```

## Pros and cons of the approaches

OpenCV `calibrateCamera` is an easy to use function, incorporated optimization algorithm refines parameters iteratively, potentially leading to a more accurate calibration results. However, the algorithm requires at least 6 points (internally, DLT algorithm needs at least 6 points for pose estimation from 3D-2D point correspondences). The approach can be flexible for extra parameters estimation, such as distortion coefficients, however it is beyond the scope of this post.

Homography-based camera calibration, on the other hand, has simplicity in its concept, relies on establishing a relationship between corresponding points in two planes (e.g., the calibration pattern and its image). A homography matrix can be derived with as few as 4 point correspondences, making it less demanding than `calibrateCamera` for initial estimation. In addition, the crucial aspect is that all the experimental data is compressed into a homography matrix, which has only 8 degrees of freedom. This means that the calibration results are susceptible to noise in the homography matrix computation and the following calibration process. Although the computation of the homography matrix can analyse more suitable points via the RANSAC process, it does not benefit significantly from additional keypoints due to the compression of the data into the homography matrix. In addition, refinement of calibration parameters requires an additional step. But comparing computational demands, the homography-based approach can be more efficient in case of a lot of points used for calibration, because all steps, apart from the homography computation itself, do not scale with the number of points.

* **Flexibility**: The `calibrateCamera` function is more flexible, can be used for handling non-planar calibration patterns and supports additional parameters like distortion coefficients, whereas homography-based calibration is inherently limited to planar patterns.

* **Accuracy and Robustness**: The iterative optimization in `calibrateCamera` generally results in more robust and accurate calibration, especially in the presence of noise. Homography-based calibration is more sensitive to noise due to its reliance on a single transformation matrix.

* **Computational Demand**: Homography-based methods can be computationally lighter for initial parameter estimation, but require further processing and refinement to achieve comparable accuracy. Numerical stability issues can arise, especially with noisy data for the homography-based calibration method.

It is worth mentioning that the results of extrinsic parameters estimations with the homography-based approach can be refined using an additional steps, such as application of OpenCV [solvePnPRefineLM](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga650ba4d286a96d992f82c3e6dfa525fa) algorithm, which uses the available points to refine the camera pose. An example code implementing the approach is available [here](https://github.com/NikolasEnt/soccernet-calibration-sportlight/blob/9ac3ad2adc2458af9af331ea82e1a0e8097c7e38/baseline/camera.py#L105).

## Comparison on SoccerNet Camera Calibration dataset

As a real-world example, we will use validation part of the Soccernet Camera Calibration dataset, previously used in CVPR CVSports 2023 entry [\[2\]][2]. The dataset contains a set of images captured by broadcast cameras at soccer games. The goal is to calibrate these cameras and obtain their intrinsic and extrinsic parameters as accurately as possible from a single view image. Unlike the solution of the challenge, for the experiments below we will use keypoints detected by a neural network [\[2\]][2] on the groundplane only for the sake of simplicity.

In all the following experiments, we assume no geometric distortions, astigmatism, and keep the principal point to be the center of the image since the accuracy of the keypoints detection is not enough for reliably taking into account faint second-order effects. Parameters of the keypoint-extraction algorithms were kept the same for both experiments.

For the OpenCV-based calibration, the experimental setup is equivalent to the following:

```python
flags = cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO\
        | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2\
        | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4\
        | cv2.CALIB_FIX_TANGENT_DIST
# Note: calibrateCamera requires at least 6 points
# No initial guess on the camera matrix or distorion coefficients
_, mtx, dist, rvect, tvect = cv2.calibrateCamera(
    world_points, camera_points, img_size,
    cameraMatrix=None, distCoeffs=None, flags=flags)
```

For the homography-based approach the code was discribed in the previous section. Refinement for the homography-based method was performed using OpenCV `solvePnPRefineLM` function.

The applied metrics are the same as those used in the CVPR Camera Calibration challenge 2023, as detailed in [\[2\]][2].

### Results

| **Metric**                    | **OpenCV Calibration** | **Homography Calibration** | **Homography Calibration + Refinement** | **CVPR entry** |
|-------------------------------|------------------------|-----------------------------|---------------------------------------|--------------|
| **$$L_2$$ Reprojection Error, px**  | 13.85               | 28.24                    | 13.70                              | 9.95     |
| **Accuracy@5**                    | 0.7142              | 0.1430                   | 0.5942                             | 0.7418    |
| **Completeness**                | 0.7273              | 0.7531                   | 0.7531                             | 0.7462     |
| **Final metric** | 0.5194              | 0.1077                    | 0.4475                             | 0.5536    |


**$$L_2$$​ Reprojection Error** : OpenCV Calibration has an error of 13.8 px, while Homography Calibration alone shows a much higher error at 28.2 px. With refinement steps added to the method, the error drops to a value comparable with the OpenCV method: 13.7 px.

**Accuracy@5**: OpenCV Calibration achieves an accuracy of 0.714, which is notably higher than the basic Homography method's low accuracy of 0.143, aligning well with the differences in reprojection error. The refinement step helps to improve the Homography calibration accuracy to 0.594, which is still notably lower than the OpenCV method. This may be explained by the fact that the homography-based method uses only a homography matrix for intrinsic parameter estimation, which does not allow it to fully benefit from more than the minimal number of keypoints available for calibration and joint parameters refinement.

**Completeness**: Both Homography-based methods have identical completeness of 0.753, unsurprising given that both are based on the same homography matrix and keypoint sets. OpenCV has lower completeness of 0.727 because its calibration process requires six points, while the homography-based method can use a minimum of four keypoints.

**Final metric**: The final metric demonstrates that the slight outperformance of the Homography-based methods in terms of completeness is not enough to compensate for the reduced accuracy.

The last column of the table `CVPR entry` shows the results of the CVPR entry algorithm [2, 5], demonstrating the impact of tricks and heuristics on the keypoints. Among other tricks, it includes a combination of OpenCV calibration with the homography-based calibration when there are not enough keypoints available for OpenCV calibration.

## Conclusion

Despite the simplicity of the homography-based approach concept and its mathematical correctness, in reality, the approach is less accurate in real-world scenarios as data compression to a single homography matrix can lead to significant loss of information. The homography-based approach can be useful when only 4-5 points are available for calibration, but in real-world cases, the OpenCV-based method or Zhang's algorithm in general is more suitable and adds a lot of flexibility regarding additional value computations, such as distortion estimation. On the other hand, homography-based methods are widely used in deep-learning-based camera calibration methods due to their simplicity of representation and ease of application for many other relevant tasks, such as camera-to-camera calibration in multiview systems or rough estimation of an object's position on the ground when precision is less important. For further details on the deep-learning-based methods, there is a great survey [6].

More complicated pipelines may use a combination of the approaches to get the best from both methods. Extra heuristics and data processing techniques, especially for data cleaning and refinement, can be used to achieve the best possible result.

## References:

[1]: {% post_url 2017-05-05-Camera-calibration-with-OpenCV %}.
\[1\]: [Camera calibration with OpenCV]({% post_url 2017-05-05-Camera-calibration-with-OpenCV %})

[2]: {% post_url 2023-06-20-SoccerNet-Camera-Calibration-2023 %}
\[2\]: [Top-1 solution of SoccerNet Camera Calibration Challenge 2023]({% post_url 2023-06-20-SoccerNet-Camera-Calibration-2023 %})

\[3\]: Zhengyou Zhang. _A flexible new technique for camera calibration_. // IEEE Transactions on Pattern Analysis and Machine Intelligence, 2000, V. 22, I. 11, pp. 1330-1334.

\[4\]: Richard Hartley and Andrew Zisserman. Multiple View Geometry in Computer Vision. Second Edition. Cambridge University Press, 2004.

\[5\]: Falaleev N., Chen R. _Enhancing Soccer Camera Calibration Through Keypoint Exploitation_. //
MMSports'24: Proceedings of the 7th ACM International Workshop on Multimedia Content Analysis in Sports, 2024, pp. 65-73, DOI: [10.1145/3689061.3689074](https://dl.acm.org/doi/abs/10.1145/3689061.3689074). [GitHub repo](10.1145/3689061.3689074).

\[6\]: Deep Learning for Camera Calibration and Beyond: A Survey // [arXiv:2303.10559](https://arxiv.org/abs/2303.10559v2).