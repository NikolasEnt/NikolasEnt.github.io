---
layout: post
title:  "Multicamera systems: Calibration and beyond"
title_img: /assets/post23/title.webp
abstract: Classical strategies and learning-based approaches for achieving accurate multicamera calibration in real-world.
date:   2025-01-27 6:00:00 +0000
categories: ComputerVision OpenCV Calibration DeepLearning
article: true
sitemap:
    lastmod: 2025-01-28
---

_This article is part of a series on camera calibration that has accumulated practical tips over more than a decade of experience. The series is not intended to be a comprehensive beginner's guide to camera calibration or provide instructions for a state-of-the-art optical laboratory setup. This article explores the complexities of multicamera calibration, addressing key considerations and strategies for enhancing accuracy. Additionally, we review a variety of learning-based approaches, offering an overview of potential techniques. While this article does not provide step-by-step recipes, it can serve as a reference for exploring solutions to challenges in multicamera calibration.._

![Title image](/assets/post23/title.webp)
[\[2\]][2]

## Applications

Multicamera calibration is essential in many real-world Computer Vision projects, including robotics, augmented reality, and complex perception systems such as autonomous vehicles. Here are some use cases:

* Data Translation Between Views: One camera may be optimised for high-quality data extraction through deep learning models, while the predictions need to be overlaid onto a video stream from another camera, such as a live broadcast camera.
* Extending Field of View: Multicamera systems can stitch together multiple views to create a panoramic view, effectively extending what can be seen with a single camera.
* Stereoscopic Depth Perception: This is particularly useful in applications like 3D ball trajectory reconstruction in sports, where depth information is critical for accurate tracking and photogrammetry analysis.
* 3D Reconstruction: By combining data from multiple cameras, detailed 3D models of the environment or objects can be created. These models have a wide range of applications, including digital animation and creating digital twins, as well as mixed reality projects such as augmented reality mobile apps and games. 

## General approaches

It is worth mentioning that, at its core, the multicamera calibration process consists of several general steps: 

1. Calibration of intrinsic parameters for individual cameras.
2. Identification of a system of markers or keypoints to use as references for camera calibration. Detection of these keypoints.
3. Determination of the relative positions of the cameras with respect to each other and/or the world origin.
4. Refinement of the camera poses.

Each step can be performed in different ways depending on requirements and available resources, and some algorithms combine these steps.

In the following, we will primarily discuss cases where the cameras have some overlap in their fields of view. For systems with many cameras that do not share a common coverage area, system calibration can be viewed as determining the pose of each camera relative to a global origin. The key difference between scenarios with overlapping fields of view and those without is that in the latter case, it is not possible to apply general techniques that leverage multiple cameras to enhance individual calibrations. Therefore, this scenario can be treated as independent camera calibration for each camera in the system.

## Calibration process

### Stereo systems

First of all, it makes sense to discuss two cameras systems, as the most common use case. These systems consist of two cameras positioned apart from each other, mimicking living creature binocular vision. It is possible to use direct stereo calibration for a system of two cameras, for example using OpenCV's [`cv2.stereoCalibrate`](https://docs.opencv.org/4.11.0/d9/d0c/group__calib3d.html#ga9d2539c1ebcda647487a616bdf0fc716). This function can operates in two modes. The first one means end-to-end calibration, including finding intrinsic parameters for each of the two cameras and the extrinsic parameters between the cameras. The second options consumes pre-calibrated intrinsics of the cameras, obtained during calibration of the individual cameras (this was discussed in detail in previous posts ([\[1\]][1] and [\[2\]][2]), and outputs relative rotations and translations from the first camera's coordinates system to the second camera's coordinates system, as well as essential and fundamental matrices, meaning it performs extrinsic calibration only.

In reality, in the absolute majority of cases, it can be recommended to perform individual cameras intrinsics calibration first, and then use the camera parameters as fixed to find multicamera calibration values only. This approach usually result in more accurate results and allows performing individual cameras calibration more accurately if performed in controlled environment rather than as part of an already installed multicamera system (like a set of cameras covering a grocery store).

One of the main disadvantages of using this function is that it requires identical sets of keypoints for both cameras, which can be challenging to achieve in practice. The keypoints or markers used for calibration may not be visible to both cameras simultaneously due to differences in their perspectives. Furthermore, if the cameras or calibration target are moved during the calibration process, the keypoints must be captured simultaneously, meaning  both cameras should be synchronised. This synchronization is particularly challenging if the cameras lack hardware triggers synchronisation features or if they are fundamentally different with different specifications and features, such as an industrial camera and a TV broadcast camera.

A more practical and often simpler approach involves detecting well-defined markers in the real world that appear in the views of multiple cameras. These markers can then be used to derive the position of each camera relative to a common reference point in world coordinates. A good example of such markers is ArUco codes, which can be easily detected (with 6D pose estimation) using OpenCV. ArUco markers are widely used in Computer Vision due to their robustness, wide adoption and ease of integration. There is an official comprehensive OpenCV [tutorial](https://docs.opencv.org/4.11.0/d5/dae/tutorial_aruco_detection.html) covering the generation and detection of these markers.

<div style="text-align: center;">
    <img src="/assets/post23/markers.jpg" alt="ArUco markers" style="max-width: 40%; height: auto; display: inline-block;">
    <br><i>An example of ArUco markers used for calibration. The image is from <a href="https://docs.opencv.org/4.11.0/d5/dae/tutorial_aruco_detection.html">OpenCV tutorial</a>.</i>
</div>

If the markers positions are known in the world coordinates, once the markers are detected in any way, they can be used to find the camera pose with respect to them.

```python
# aruco_dict - a dict used during the ArUco codes generation
aruco_params = cv2.aruco.DetectorParameters_create()  # Default values, modify if required
corners, ids, _ = cv2.aruco.detectMarkers(image, aruco_dict, parameters=aruco_params)

# Known 3D coordinates of markers in world frame.
# Make sure all marker IDs are present in the dictionary
marker_points_3d = {
    0: np.array([(0, 0, 0), (0.1, 0, 0), (0.1, 0.1, 0), (0, 0.1, 0)]),
    # Add more marker coordinates as needed
}

object_points = []
image_points = []
for marker_id, marker_corners in zip(ids, corners):
    object_points.extend(marker_points_3d[marker_id[0]])
    image_points.extend(marker_corners[0])

object_points = np.array(object_points, dtype=np.float32)
image_points = np.array(image_points, dtype=np.float32)

# rvect, tvect - rotation and translation of the camera
# camera_matrix, dist_coeffs are from intrinsic camera calibration
success, rvect, tvect = cv2.solvePnP(
    object_points, image_points, camera_matrix, dist_coeffs
)
```

Note that the camera pose will be calculated using the same coordinate system and the same units as those defined for the marker positions. In the case there are multiple markers visible and it is expected some of the markers are not as reliable as others it may be useful to use [cv2.solvePnPRansac](https://docs.opencv.org/4.11.0/d9/d0c/group__calib3d.html#ga50620f0e26e02caa2e9adc07b5fbf24e) to deal with the potential outliers.

The following code allows visualising the reprojection error:
```python
reprojected_points, _ = cv2.projectPoints(object_points, rvect, tvect, camera_matrix, dist_coeffs)
reprojected_points = reprojected_points.reshape(-1, 2)
for img_pt, reproj_pt in zip(image_points, reprojected_points):
    cv2.circle(image, tuple(img_pt.astype(int)), 3, (0, 255, 0), 1)
    cv2.circle(image, tuple(reproj_pt.astype(int)), 3, (0, 0, 255), 1)

error = np.mean(np.abs(image_points - reprojected_points))
print(f"Reprojection Error: {error:.3f} px")
```

### Multicamera systems

For systems with more than two cameras, there are additional challenges and opportunities. If reusable markers are available for relative position calibration among multiple cameras (e.g., cameras A, B, and C), it is possible to derive not only direct coordinate transformations such as A -> B but also multistage transformations like A -> C -> B. In the ideal case, these multistage transformations should result in the projection of a point from camera A's coordinates onto camera B, which should match the direct transformation.

Given that multiple experimentally derived transformations are available, statistical methods for data fusion can be leveraged to derive even more accurate calibration results. While simple averaging can be used in the simplest cases, more sophisticated approaches that account for uncertainties or measurement error levels of the experimental values are generally preferable.

### Dynamic systems

Kalman filtering can also be employed to fuse multiple transformation or smooth the results dynamically over time. This method is particularly useful when dealing with dynamic scenes where camera positions may change from frame to frame, for example in the case of moving broadcast cameras used for sports events coverage, allowing for accurate real-time adjustments based on new measurements, while taking into account uncertainties of the observations and smoothing the camera's pose.

## Learning-based techniques

As mentioned earlier, a key aspect of multicamera calibration involves identifying reference keypoints for camera calibration. In cases where cameras have overlapping fields of view, finding corresponding keypoints in images from different cameras can be addressed using commonly used feature detection and matching approaches, such as traditional algorithms like SIFT or SURF. Deep learning techniques can extract more robust features compared to traditional methods, leading to better alignment between images.

One possible solution for robust keypoint extraction is the recognition of well-known shapes within the camera's field of view by a deep learning model. This could include industrial equipment or parts, sports playground lines, or other objects that are easily identifiable. These shapes, with their known sizes, can be used as reference keypoints for calibration. For example, a previous [solution]({% post_url  2023-06-20-SoccerNet-Camera-Calibration-2023 %}) of the SoccerNet Camera Calibration challenge demonstrated the use of various visual features of a football pitch for precise camera calibration.

<div style="text-align: center;">
    <img src="/assets/post23/pseudopillars.png" alt="PseudoPillars model architecture" style="max-width: 90%; height: auto; display: inline-block;">
    <br><i>PseudoPillars model, which regresses relative sensor poses transformation via an intermediate depth-map representation [3].</i>
</div>

In addition to direct methods, various deep learning models propose intermediate representations that enhance the calibration process. For example, a recent paper [3] suggests using a deep learning model to estimate depth from a camera view, which can be used to find corresponding transformations with LiDAR point clouds. This approach can also be applied to regress the transformation between different cameras, instead of the originally proposed LiDAR-camera calibration. 

<div style="text-align: center;">
    <img src="/assets/post23/shiftnet.png" alt="ShiftNet model architecture" style="max-width: 90%; height: auto; display: inline-block;">
    <br><i>ShiftNet model architecture, which regresses relative offsets of panorama patches with subpixel accuracy, enabling precise refinement for panorama stitching [4].</i>
</div>

It is often important to consider the specific final objective rather than solving the problem in its most general form. For instance, ShiftNet model [4] is a lightweight convolutional neural network that regresses relative offsets of panorama patches with subpixel accuracy, enabling precise refinement for panorama stitching and the results are better than in case of more traditional methods. Similar regression approach can also be applied to other types of panoramas, not only the considered in the paper cylindrical panorama.

<div style="text-align: center;">
    <img src="/assets/post23/dunhuangstitch.png" alt="DunHuangStitch model architecture" style="max-width: 90%; height: auto; display: inline-block;">
    <br><i>DunHuangStitch model architecture, which improves panorama stitching for low-texture images from multiview data [5].</i>
</div>

Learning-based algorithms can be especially beneficial in markerless multiview calibration or when it is difficult to define keypoints due to low texture in image content. For example, DunHuangStitch [5] focuses on applying deep learning techniques to improve image stitching for low-texture panoramas from multiview data. The deep learning-based approach uses a progressive regression image alignment network and a feature differential reconstruction soft-coded seam stitching network in an unsupervised learning setup. This enhances the accuracy of stitching wide panoramas compared to conventional methods, particularly for challenging low-texture scenarios. 

For moving cameras, learning models are particularly beneficial as they can model camera movement based on observed visual changes in images, such as recent Transformer-based models adopted for motion estimation. Additionally, end-to-end approaches can be extremely helpful in many cases. For example, in a multicamera perception system, the raw values of camera calibration may not be required, but accurate environment reconstruction is crucial. There are multiple suitable end-to-end models for such use cases. 

For instance, [6] proposes an end-to-end framework that simultaneously solves camera and subject registration by leveraging their mutual dependence, thereby eliminating the need for explicit multi-camera calibration. The pipeline consists of a view-transform module that projects each detected object to a virtual bird's eye view (BEV). A spatial alignment module then estimates relative camera poses in a unified BEV using multi-view geometry. Finally, the framework selects and refines camera poses and fuses object information in the unified BEV space.

## Conclusion

In summary, multicamera calibration presents both challenges and exciting opportunities that can be addressed through various approaches. Classical methods require mastering single-camera calibration and leveraging multiview data for accurate results. However, in many cases, it is beneficial to tailor solutions specifically to the given problem using learning-based end-to-end approaches rather than solving a general calibration problem. 

Classical techniques provide a solid foundation and can be extended with multiview constraints while maintaining interpretability and controllable accuracy. Modern learning-based methods offer advantages in handling complex scenarios where traditional approaches may struggle, particularly for environment reconstruction or imagery data stitching. 

By using either classical calibration techniques, advanced machine learning models, or combining them as appropriate, multicamera systems can achieve robust and precise performance in a wide range of applications, from static setups to dynamic applications.

_If you have any questions or need help in incorporating this into your projects, feel free to contact me! Happy coding!_

## References:


[1]: {% post_url 2017-05-05-Camera-calibration-with-OpenCV %}
\[1\]: [Camera calibration with OpenCV.]({% post_url 2017-05-05-Camera-calibration-with-OpenCV %})

[2]: {% post_url 2024-12-20-Practical-OpenCV-Refinement-Techniques %}
\[2\]: [Camera Calibration: Practical OpenCV Refinement Techniques.]({% post_url 2024-12-20-Practical-OpenCV-Refinement-Techniques %})

\[3\]: M. Cocheteux, J. Moreau and F. Davoine. PseudoCal: Towards Initialisation-Free Deep Learning-Based Camera-LiDAR Self-Calibration. // The 34th British Machine Vision Conference, 2023, (BMVC 2023). [pdf](https://papers.bmvc2023.org/0829.pdf).

\[4\]: L. Kang, Y. Wei, J. Jiang, and Y. Xie. Robust Cylindrical Panorama Stitching for Low-Texture Scenes Based on Image Alignment Using Deep Learning and Iterative Optimization. // Journal of Sensors, 2019, DOI: [10.3390/s19235310](https://doi.org/10.3390/s19235310).

\[5\]: Y. Mei, L. Yang, M. Wang, T. Yu and K. Wu. DunHuangStitch: Unsupervised Deep Image Stitching of Dunhuang Murals. // IEEE Transactions on Visualization and Computer Graphics, 2024, DOI: [10.1109/TVCG.2024.3398289](https://doi.org/10.1109/TVCG.2024.3398289), [GitHub code](https://github.com/MmelodYy/DunHuangStitch).

\[6\]: Z. Qian, R. Han, W. Feng and S. Wang. From a Bird's Eye View to See: Joint Camera and Subject Registration without the Camera Calibration. // In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 863-873. [pdf](https://openaccess.thecvf.com/content/CVPR2024/papers/Qian_From_a_Birds_Eye_View_to_See_Joint_Camera_and_CVPR_2024_paper.pdf), [GitHub code](https://github.com/zekunqian/bevsee).