---
layout: post
title:  "An approach to predicted class balancing"
title_img: /assets/logos/kaggle.png
abstract: An approach how to increase your position on a leaderpoard in a classification datascience competition by balancing predictions.
date:   2018-02-21 12:00:00 +0300
categories: DeepLearning Competitions
project: comp1
---

## Postprocessing of predictions

It was observed that the predicted class distribution in the [Kaggle: IEEE's Signal Processing Society - Camera Model Identification Challenge][comp1] is near to equal (see the [post][Post] for details on the challenge). Hence, one may assume that there is equal number of images from each class. The altered images were considered as separate classes.
So, there are 20 classes for images assignment and, given 2640 test samples, one may expect 132 images in each class.

Given the assumption, predictions were balanced.

We have predictions, which could be represented as a pandas dataset like so:

<table border="1" class="dataframe">  <thead>    <tr style="text-align: right;">      <th></th>      <th>frame</th>      <th>HTC-1-M7</th>      <th>iPhone-6</th>      <th>...</th>      <th>Motorola-Nexus-6</th>      <th>Samsung-Galaxy-Note3</th>      <th>Sony-NEX-7</th>    </tr>  </thead>  <tbody>    <tr>      <th>0</th>      <td>img_0002a04_manip.tif</td>      <td>5.083291e-04</td>      <td>1.730842e-03</td>      <td>...</td>      <td>6.565361e-02</td>      <td>7.312844e-04</td>      <td>5.617269e-04</td>    </tr>    <tr>      <th>1</th>      <td>img_001e31c_unalt.tif</td>      <td>1.333470e-03</td>      <td>1.378153e-03</td>      <td>...</td>      <td>1.011069e-03</td>      <td>1.311640e-04</td>      <td>5.842330e-04</td>    </tr>    <tr>      <th>...</th>      <td>...</td>      <td>...</td>      <td>...</td>      <td>...</td>      <td>...</td>      <td>...</td>      <td>...</td>    </tr>    <tr>      <th>2638</th>      <td>img_ff56ac2_unalt.tif</td>      <td>1.707251e-04</td>      <td>2.834913e-04</td>      <td>...</td>      <td>4.762203e-04</td>      <td>8.150683e-05</td>      <td>9.958064e-01</td>    </tr>    <tr>      <th>2639</th>      <td>img_ffaeda7_unalt.tif</td>      <td>6.653962e-11</td>      <td>3.320259e-11</td>      <td>...</td>      <td>4.800383e-11</td>      <td>8.671302e-12</td>      <td>4.138669e-13</td>    </tr>  </tbody></table>

A sample .csv file with predicted probabilities is available [here][SampleCsv]. 

A two-stage algorithm was introduced for class balancing:
1. For each class, one by one, sort samples by predicted probability and examine them in descending order. Assign samples while there are free slots for the desired class and the predicted probability is higher than the given threshold. The threshold was selected to be 0.5. It means that images with one predicted class probability higher than the value could not have any other class with higher probability. Exclude assigned samples from further consideration.
2. Find the highest probability in the remaining subset and try to assign a sample with it to the corresponding class. If the class already has enough samples, set the examined probability to zero. Exclude the sample from further consideration in case of successful class assignment.

The following function implements the proposed algorithm and it can be applied to manipulated and unaltered images separately:

{% highlight python linenos%}
import numpy as np
import pandas as pd

def class_balancer(df):
    n_per_cl = len(df) // (len(names_list)-1)  # Images per class
    names = df.columns.tolist()[1:]
    res = {}  # A dict with the results
    for i, nam in enumerate(names):  # For each class
        res[nam] = []
        df = df.sort_values(by=[nam], ascending=False).reset_index(drop=True)
        k = 0
        while float(df.iloc[0, (i+1)]) > THRES and len(res[nam]) < n_per_cl:
            # If the prediction is confident enought and we have slots
            # in the class, add the sample to it
            res[nam].append(df.iloc[0, 0])
            df = df.drop([0]).reset_index(drop=True)
            k += 1
    df = df.reset_index(drop=True)
    while len(df) > 0:  # While we have any images for classses filling
        df_l = df.loc[:, df.columns != 'frame']
        row = df.max(axis=1).idxmax()  # A sample with the max probability in the remaining set
        nam = df_l.max(axis=0).idxmax()  # Class of the max probability
        if len(res[nam]) < n_per_cl:
            # If it is posseble to fill the most probable class
            res[nam].append(df.iloc[row, 0])
            df = df.drop([row]).reset_index(drop=True)
        else:  # Set probobility to 0
            df.at[row, nam] = 0.0
    return res
{% endhighlight %}


The function was used along with the following code for data reading, blending predictions by mean square, preparing pandas dataframes for unaltered and manipulated images, processing, and generation of the final submission file.

{% highlight python linenos%}
import os

THRES = 0.5  # Minimal probaility to consider as a confident prediction
results_path = './results/'  # Path to the directory with results in .csv format
names_list = ['frame', 'HTC-1-M7', 'iPhone-6', 'Motorola-Droid-Maxx', 'Motorola-X',
              'Samsung-Galaxy-S4', 'iPhone-4s', 'LG-Nexus-5x', 'Motorola-Nexus-6',
              'Samsung-Galaxy-Note3', 'Sony-NEX-7']

def make_sub(name, dic):  # Make a file for submission
    with open(name, 'w') as f:
        f.write('fname,camera\n')
        for r in dic.keys():
            for p in dic[r]:
                f.write(p+','+r+'\n')

# Read all files with predictions in the results_path directory
dfs = []  # A list of dataframes with predicted probabilities
for fname in os.listdir(results_path):
    dfs.append(pd.read_csv(fname, skiprows=1, names=names_list))

# Blending by mean square
df_fin = None
for df in dfs:
    if df_fin is None:
        df_fin = df
        df_fin.loc[:, df_fin.columns != 'frame'] = df_fin.loc[:, df_fin.columns != 'frame'].pow(2)
    else:
        df2 = df.loc[:, df.columns != 'frame'].pow(2)
        df2['frame'] = df['frame']
        df_fin.loc[:, df_fin.columns != 'frame'] = df2.loc[:, df2.columns != 'frame']\
                                                   .add(df_fin.loc[:, df_fin.columns != 'frame'])
df_fin.loc[:, df_fin.columns != 'frame'] = np.sqrt(df_fin.loc[:, df_fin.columns != 'frame'])

df_manip = df.loc[df.frame.str.contains("_manip")==True]
df_unalt = df.loc[df.frame.str.contains("_unalt")==True]

res_manip = class_balancer(df_manip)
res_unalt = class_balancer(df_unalt)

for i in res_unalt.keys():  # Append res_unalt to res_manip
    for j in res_unalt[i]:
        res_manip[i].append(j)

make_sub("results.csv", res_manip)
{% endhighlight %}

_It is the real competition code, written in the last two hours of the competition. Sorry, pep8 :)_

An increase of accuracy on the leaderboard by ~1.2-1.6% was observed as the result of this approach application, which makes a huge difference in the final standing.

[comp1]: /proj/comp1/
[Post]: {% post_url 2018-02-20-Camera-model-identification-with-deep-learning %}
[SampleCsv]: /assets/post9/sample_predictions.csv

