---
layout: post
title:  "About Kaggle TGS Salt Identification Challenge"
title_img: /assets/logos/tgs.png
abstract: Kaggle Challenge to segment salt deposits beneath the Earth's surface on seismic images.
date:   2018-10-23 19:00:00 +0300
categories: DeepLearning Competitions
project: comp3
---

## The goal

The overall goal of the competition on [Kaggle](https://www.kaggle.com/c/tgs-salt-identification-challenge) platform was to build an algorithm that automatically and accurately identifies if a subsurface target is salt or not on seismic images. The task is crucial for oil and gas company drillers.
The object of the competition is seismic data collected using reflection seismology. In a nutshell, the problem can be formulated as a semantic segmentation computer vision task.

![Salt dome](/assets/post13/salt_dome.png){: .center-image }

_A salt body depicted on a seismic reflection image (the sample created out of the Challenge data)._

## Data overview

There were 4000 grayscale images 101x101 px with labeled masks for training. Another set with 18000 images was provided for testing. In addition to the images, the depth of the imaged location was supplied for each image.

![Sample image and mask](/assets/post13/train_sample.png){: .center-image }

_A sample image with its binary mask. Green area corresponds to a salt region, whereas blue - no salt._

## Evaluation
This competition was evaluated on the mean average precision at different intersection over union (IoU) thresholds. The threshold values range from 0.5 to 0.95 with a step size of 0.05.

__Next__: [Out 14/3434 place solution][Post]

[Post]: {% post_url 2018-10-24-Semantic-Segmentation-of-Seismic-Reflection-Images %}


