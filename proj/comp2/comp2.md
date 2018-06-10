---
layout: competition_page
title: Lyft Perception Challenge
permalink: /proj/comp2
project_id: comp2
---

[Lyft Perception Challenge][Lyft] was organized by Lyft and Udacity.

The goal of the challenge is pixel-wise semantic segmentation of images from a front facing camera mounted on a vehicle. Actually, the camera data for this challenge comes from an open-source [CARLA][Carla] simulator.

Two classes were included in the final scoring: roads and cars. The competition is noteworthy due to the fact that participants performance evaluation based on F-beta scores and the prediction frame rate (FPS) on a target machine was an essential part of the metric.


The final result of participation: the __4th__ place out of 155 participants (__top-3%__). The submitted pipeline was also the __fastest__ one.

### Contents:

1. [About Lyft Perception Challenge][About]
2. [Multiclass semantic segmentation with LinkNet34][Linknet]
4. [Discussion of the Lyft Perception Challenge][Discussion]

## Final results

<iframe width="560" height="315" src="https://www.youtube.com/embed/15vnXdaoo8Q?rel=0" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>

The project code is available on [Github][Github].

[Lyft]: https://www.udacity.com/lyft-challenge
[Carla]: http://carla.org/
[About]: {% post_url 2018-05-31-About-Lyft-Perception-Challenge %}
[Linknet]: {% post_url 2018-06-01-Multiclass-semantic-segmentation-with-LinkNet34 %}
[Discussion]: {% post_url 2018-06-05-Discussion-of-the-Lyft-Perception-Challenge %}
[Github]: https://github.com/NikolasEnt/Lyft-Perception-Challenge
