---
layout: project_page
title: Vehicle Detection and Tracking Project
permalink: /proj/proj2
project_id: proj2
---

This Project is based on the fifth task of the Udacity Self-Driving Car Nanodegree program. The main goal of the project is to create a software pipeline to identify vehicles in a video from a front-facing camera on a car.

Additionally, a lane line finding algorithm was added. See **[Lane Lines Detection Project][proj1]** for details.

It was implemented in Python with OpenCV and Scikit-learn libraries. Linear SVM was used as a classifier for HOG, binned color and color histogram features. The project repo is availuble on [Github][projectRepo].


### Contents:

1. [Image classification using SVM][postSVM]

## Final project video:

<iframe width="560" height="315" src="https://www.youtube.com/embed/waYJjmkRZfw" frameborder="0" allowfullscreen></iframe>

[proj1]: /proj/proj1
[projectRepo]: https://github.com/NikolasEnt/Vehicle-Detection-and-Tracking
[postSVM]: {% post_url 2017-09-21-Neural-network-for-multiclass-image-segmentation %}
