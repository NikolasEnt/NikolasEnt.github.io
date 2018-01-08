---
layout: post
title:  "Image classification using SVM"
title_img: /assets/post5/hog_image.jpg
abstract: Application of a linear SVM for image classification with HOG, binned color and color histogram features.
date:   2017-08-01 12:00:00 +0300
categories: Classifier
project: proj2
---
## Introduction

The main goal of the project is to create a software pipeline to identify vehicles in a video from a front-facing camera on a car. It is implemented as an image classifier which scans an input image with a sliding window. A linear SVM was used as a classifier for HOG, binned color and color histogram features, extracted from the input image. The classifier is described here.

## Feature extraction
To address the task three types of features were used: HOG (Histogram of Oriented Gradients) (shape features), binned color (color and shape features) and color histogram features (color only features). This combination of features can provide enough information for image classification.

It is possible to use built-in functions from different libraries for features extraction.

* **HOG** is implemented in [skimage][skimage]. All that you need is to provide parameters. A detailed description of what is the HOG features available [here][HOG]. It can be used as follows:

{% highlight python %}
from skimage.feature import hog

features = hog(input_image, orientations = number_of_orientations, 
                       pixels_per_cell = (pix_per_cell, pix_per_cell),
                       cells_per_block = (cell_per_block, cell_per_block), 
                       transform_sqrt = True, 
                       visualise = False, feature_vector = True)
{% endhighlight %}

The function is very powerful and can even produce the HOG features images just by setting `visualise = True` and adding an extra output. Here it is an example:


| ![Original image](/assets/post5/hog_image.jpg) | ![HOG image](/assets/post5/hog_example.jpg) |


* **Binned color** features is just a downscaled image. Scaling can be done with the OpenCV function `resize`:
{% highlight python %}
import cv2

def bin_spatial(input_image, size=(16, 16)):
    return cv2.resize(input_image, size).ravel() 
{% endhighlight %}


* **Color histogram** features can be computed with the NumPy `histogram` function. One may use one channel grayscale image or use all colors stacked:
{% highlight python %}
import numpy as np

def color_hist(input_image, nbins=32):
    ch1 = np.histogram(input_image[:,:,0], bins = nbins, range = (0, 256))[0] # [0] is because we need only the histogram, not bins edges
    ch2 = np.histogram(input_image[:,:,1], bins = nbins, range = (0, 256))[0]
    ch3 = np.histogram(input_image[:,:,2], bins = nbins, range = (0, 256))[0]
    return np.hstack((ch1, ch2, ch3))
{% endhighlight %}

## Classifier

### Data loading

First of all, we need to load the data.

{% highlight python %}
import glob

images = glob.glob('*vehicles/*/*')
cars = []
notcars = []
for image in images:
    if 'non' in image:
        notcars.append(image)
    else:
        cars.append(image)
{% endhighlight %}

### Hyperparameters
We are going to use the linear SVM classifier from [sklearn][sklearn]. It can work with default parameters (but they may be fine-tuned later). There are a lot of hyperparameters of the feature extraction process. So, one may consider using automated way of the parameters tuning.

{% highlight python %}
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from skopt import gp_minimize

space  = [(8, 64),                  # Number of bins for color histogram
          (6, 12),                  # HOG number of orientations
          (4, 16),                  # HOG pixels per cell
          (1, 2)]                   # HOG cells per block
i = 0

def obj(params):
    global i
    nbins, orient, pix_per_cell, cell_per_block = params
    # Use only every 10th images to speed things up.
    car_features = extract_features(cars[0:len(cars):10], nbins, orient, pix_per_cell, cell_per_block)
    notcar_features = extract_features(notcars[0:len(notcars):10], nbins, orient, pix_per_cell, cell_per_block)
    y = np.hstack((np.ones(len(cars[0:len(cars):10])), np.zeros(len(notcars[0:len(notcars):10]))))
    X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=22)
    svc = LinearSVC()
    svc.fit(X_train, y_train)
    test_acc = svc.score(X_test, y_test)
    print i, params, test_acc
    i += 1
    return 1.0 - test_acc
    
res = gp_minimize(obj, space, n_calls = 20, random_state = 22)
"Best score=%.4f" % res.fun
{% endhighlight %}

The `extract_features` function extract features from the provided list of images with desired parameters.

The hyperparameters may be fine tuned manually as well, for instance, to decrease the feature vector length and, consequently, increase the speed of classification. Final settings:

{% highlight python %}
spatial_size = (16, 16) # Spatial binning dimensions
nbins = 32 # Number of bins for color histogram
orient = 8  # HOG number of orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
{% endhighlight %}


### Test the classifier

{% highlight python %}
import time

X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
X_scaler = StandardScaler().fit(X) # Fit a per-column scaler
scaled_X = X_scaler.transform(X) # Apply the scaler to X
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features)))) # Define the labels vector
# Split up data into randomized training and test sets
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=22)
print('Feature vector length:', len(X_train[0]))
svc = LinearSVC(loss='hinge') # Use a linear SVC 
t=time.time() # Check the training time for the SVC
svc.fit(X_train, y_train) # Train the classifier
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train the SVC...')
print('Test Accuracy of the SVC = ', round(svc.score(X_test, y_test), 4))
{% endhighlight %}

_('Feature vector length:', 2432)_

_(13.97, 'Seconds to train the SVC...')_

_('Test Accuracy of the SVC = ', 0.9873)_

As a result, **98.7%** accuracy was obtained.

For details of the implementation see the [project repo][projectRepo].

[skimage]: http://scikit-image.org/docs/dev/api/skimage.html
[HOG]: https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients
[sklearn]: http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
[projectRepo]: https://github.com/NikolasEnt/Vehicle-Detection-and-Tracking
