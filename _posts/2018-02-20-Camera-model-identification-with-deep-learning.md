---
layout: post
title:  "Camera model identification with deep learning"
title_img: /static/projects/comp1/title_img.jpg
abstract: Ideas and approach to the Kaggle IEEE's Signal Processing Society - Camera Model Identification challenge.
date:   2018-02-20 12:00:00 +0300
categories: DeepLearning Competitions
project: comp1
---

## About the task

The goal of the competition on [Kaggle][Kaggle] platform was to build an algorithm that identifies which camera model captured a given image by using traces intrinsically left in the raw image.
The discussed solution is based on convolutional neural networks as classifiers. 

The list of camera models for the challenge:

* Sony NEX-7
* Motorola Moto X
* Motorola Nexus 6
* Motorola DROID MAXX
* LG Nexus 5x
* Apple iPhone 6
* Apple iPhone 4s
* HTC One M7
* Samsung Galaxy S4
* Samsung Galaxy Note 3

One may notice that there is a Sony NEX-7 camera, which is a mirrorless interchangeable lens camera, so, different lenses could cause some extra difficulties as lens type is not provided as a metadata parameter for the test dataset. 
Fortunately, it was not a big problem as the camera's images have superior visual quality and it barely was the only class which it was possible to distinguish at a glance.

This challenge was evaluated on the weighted categorization accuracy of predictions. Weights were 0.7 for unaltered images and 0.3 for altered images.

## Dataset

Originally, the dataset contained 275 random images for each camera for training and 2640 images in the test dataset.

The train data consisted of full images, while the test dataset was presented only by single 512x512 pixel patches cropped from the center of images taken with the device.
What is more, half of the images in the test imageset was manipulated and labeled accordingly *_unalt* and *_manip*. The test images possible transformations are listed below:

* JPEG compression (quality = 70, 90)
* Resizing by bicubic interpolation (scale factors: 0.5, 0.8, 1.5, 2.0)
* Gamma correction (0.8, 1.2)

Interestingly, manipulation levels were limited by the listed discrete values.

## Dataset enriching

One of the keys to the competition was to download as many photos taken by appropriate cameras as possible.
For the purpose, the [Flickr API][Flickr] was used. Unfortunately, it does not provide any tools for simple images sorting by camera model from EXIF without necessity to download images. That is why images were found by tags and description and filtered locally.
It was very important to filter out all irrelevant images (e.q. edited) as it may lead to incorrect features extraction in a CNN.

Filtering criteria:
* Correct image size according to the camera specification. An image could be in horizontal or vertical orientation
* Jpeg quality > 93 for filtering out compressed and edited images
* Correct camera model in EXIF information
* No comments on editing in photo editing software, such as Photoshop, Lightroom, GIMP, etc.

These parameters can be checked in a batch manner with a very useful [identify][identify] cli utility. For example, ```identify -format '%[EXIF:Model]' /path/to/file```
returns the camera model from EXIF; ```identify -format '%Q' /path/to/file``` - jpeg quality. So, a bash script was created for image filtering.

Finally, a dataset of 25000 images was utilized for training and 500 - for validation.

## Let's train CNNs

The Kaggle competitor Andres Torrubia shared an excellent starter [code][AndresTorrubia] written with Keras and TensorFlow backend, which was really helpful for participants. Many thanks to Andres!
It allows training different CNN architectures, such as ResNet, DenseNet, Inception V3, VGG, MobileNet, etc., set patch size and dropout rate.
What is interesting about the implementation: there is an additional parameter 'is_manipulated' which actually is a flag indicating whether a given image is a raw one or manipulated.
The flag was added to the final stage of CNN's architectures in the fully connected layers of the image classifier and it looks like it helps.

For me, the competition was an interesting personal experience with Nvidia Tesla V100 GPU as provided by AWS EC2 p3.2xlarge instances and it definitely was a pleasant impression.  
Finally, ResNet50, DenseNet201, InceptionResNetV2 and MobileNet was used for the final ensemble. All models were trained on 512x512 px patches. Adam optimizer with initial learning rate = 1e-4 and reduce learning rate on plateau policy with patience 5 and factor 0.5 was involved.
Predictions were blended by mean square and post-processed. Do not forget to renormalize the raw predicted by CNNs probabilities to sum up 1!

## Postprocessing - the key to the competition

Test time augmentation was applied for predictions on the test dataset. A symmetry of a dihedral group D<sub>4</sub> (4 pi/2 rotation and all flips) was used.
The resulted predictions were averaged by mean square. Arithmetic mean was examined as well, but performed worse for the competition on the public leaderboard.

It was observed that the predicted class distribution is near to equal. Hence, one may assume that there is equal number of images from each class. The altered images were considered as separate classes.
So, there are 20 classes for images assignment and 132 images are expected for each class (given 2640 images in the test set). Given the assumption, predictions where balanced for the final submission. See the [post][ClassBalancer] for details.

## Results

As it was mentioned, my final submission is in top-3% on the [leaderboard][Leaderboard] which is the best personal result on Kaggle so far and, what is more important, a lot of useful lessons were learned during the challenge. Such as training models on AWS, competitions data science tricks as class balancing, etc.

## Further thoughts

* It may be quite interesting to visualize a CNN, trained for the task, in order to find out what features are attractive and important for the model.
* One should distinguish the real-world task and a Kaggle challenge. Here, several tricks, such as predicted class balancing may work. And, well, it probably is the major drawback of such competitions.

[Kaggle]: https://www.kaggle.com/c/sp-society-camera-model-identification
[Flickr]: https://www.flickr.com/services/api/
[identify]: https://www.imagemagick.org/script/identify.php
[AndresTorrubia]: https://github.com/antorsae/sp-society-camera-model-identification
[ClassBalancer]: {% post_url 2018-02-21-An-approach-to-predicted-class-balancing %}
[Leaderboard]: https://www.kaggle.com/c/sp-society-camera-model-identification/leaderboard
