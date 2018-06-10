---
layout: post
title:  "Discussion of the Lyft Perception Challenge"
title_img: /assets/post12/title_img.jpg
abstract: How to increase inference speed on a semantic segmentation task and further ideas.
date:   2018-06-05 12:00:00 +0300
categories: DeepLearning Competitions
project: comp2
---

## Examined ideas
Besides the [described][Post] pipeline, several other approaches were tested, but did not lead to better results:

* Usage of an extra class “windows”. It was quite challenging for the neural network to correctly label pixels of transparent parts of vehicles. It seemed rather promising to predict the extra class “windows” and exclude the predictions from the “car” class. Unfortunately, the approach forces to use sigmoid as the output layer to allow classes overlap which ultimately led to overall score reduction.

* Usage of any weights to force the CNN to pay more attention to the “car” class and deal with class imbalance. Usage of BCE element in the loss function also was not a good idea.

* Application of gaussian noise augmentation. The simulator produces clear images without any image-capture process related noises. Such kind of augmentation may help for real application, but only slows down the training process on the simulated data.

* Other encoders: ResNet18 was not able to achieve comparable to ResNet34 score, whereas  ResNet50 heavily overfit on the train data.

* Train separate models for “road” and “car” classes. Since the prediction rate is more than two times higher than required, we can use two models. In fact, it does not work well because the classes starts overlap. One network for both target classes works better, and in this case application of more classes may work as a kind of regularization which forces the network to understand the scene better.

## Further ideas

Some further ideas for prediction results improvement:

* The CARLA simulator provides a limited variety of cars and landscapes, so, it is possible to collect more data and virtually “overfit” to all possible scenes.

* Use more augmentations. One may consider applying some extra augmentations to mimic lens flares and weather conditions.

* Add some techniques to prevent overfitting: weights regularization, dropout, etc.

* Find the best encoder architecture, experiment with the decoder blocks. Or experiment with other architectures, such as 
* Use extra input channels with semantic masks from the previous frame. This may make masks predictions more sustainable and smooth between frames, however, it should decrease the neural network prediction speed.


## Results:

<iframe width="560" height="315" src="https://www.youtube.com/embed/15vnXdaoo8Q?rel=0" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>

![Leaderoard](/assets/post12/lb.jpg)

[Post]: {% post_url 2018-06-01-Multiclass-semantic-segmentation-with-LinkNet34 %}
