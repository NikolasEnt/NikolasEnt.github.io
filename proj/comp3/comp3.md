---
layout: competition_page
title: Kaggle TGS Salt Identification Challenge
permalink: /proj/comp3.html
project_id: comp3
---

The goal of the [challenge][TGS-Salt] on Kaggle platform is pixel-wise semantic segmentation of salt bodies depicted on a seismic reflection images.

[Here][Solution] you can find a description of the 14th place solution by Argus team ([Ruslan Baikulov][Ruslan_li], [Nikolay Falaleev][Nikolay_li]).

The final result of participation: the __14th__ place out of 3234 teams (__top-0.5%__, Kaggle __gold__ megal).

### Contents:

1. [About Kaggle TGS Salt Identification Challenge][About]
2. [Semantic Segmentation of Seismic Reflection Images][Solution]

## Sample results

![Sample predictions](/assets/post14/postprocess.png){: .center-image }

_Example of the whole mosaic post-processing. Green/blue - salt/empty regions from the train dataset; red - predicted mask; yellow - inpainted by the post-processing (used in the final submission)._

The project code is available on [Github][Github].

_The title image is from [here][title_link]._

[TGS-Salt]: https://www.kaggle.com/c/tgs-salt-identification-challenge
[Ruslan_li]: https://www.kaggle.com/romul0212
[Nikolay_li]: https://www.kaggle.com/nikolasent

[About]: {% post_url 2018-10-23-About-Kaggle-TGS-Salt-Identification-Challenge %}
[Solution]: {% post_url 2018-10-24-Semantic-Segmentation-of-Seismic-Reflection-Images %}
[Github]: https://github.com/lRomul/argus-tgs-salt
[title_link]: https://www.domeenergy.com/understanding-salt-domes/
