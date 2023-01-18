---
layout: post
title:  "How to run TensorFlow with CUDA 9 and cuDNN 7 in openSUSE on Ryzen"
title_img: /assets/post7/title_img.jpg
abstract: A set of instruction to run a modern version of the deep learning framework TensorFlow on AMD Ryzen.
date:   2017-10-20 12:00:00 +0300
categories: Hardware DeepLearning
article: true
sitemap:
    lastmod: 2022-01-18
---

## Introduction

I was faced with the necessity of computational power increase in order to meet my needs in computer vision research. As for me, AMD Ryzen CPU's provide the best value/cost ratio on the market at the given time. That is why I built my new PC on the platform. However, the path of software installation for deep learning is not straightforward, so, here it is a set of instructions I followed to start TensorFlow 1.4 on Nvidia GPUs with the latest CUDA 9 and cuDNN 7.0.

Generally speaking, every step is quite simple, however, it is challenging to select appropriate versions of software to make the whole system run. The versions compatibility is the main focus of the post.

_I cannot guarantee your results, so, do it voluntarily, with deep understanding what you do and at your own risk._

UPDATE: Some of the provided instructions are redundant since TensorFlow 1.5 with CUDA 9 support can be simply installed by `pip install tensorflow-gpu`.
However, one may use the article as a reference for TensorFlow build from source for obtaining the most recent version or processor-specific optimization.

## Hardware

* AMD Ryzen 7 1700X
* ASUS X370-PRO Motherboard
* Nvidia GPU

## Right after assembly

My motherboard was shipped with outdated BIOS. I had to install the latest one from the official site of the motherboard vendor.
It is preferred to install a version with AGESA 1.0.0.6b as it can [resolve][agesa] some Ryzen issues.

The motherboard was unable to correctly set default CPU and RAM voltage (it was too high for the processor and too low for memory in my case which causes some issues with the system boot). So, I had to check and set them manually to the values recommended by the parts manufacturer.

## Install openSUSE

There are a great variety of operating system distros, so everyone can choose one to meet own tastes. As for me, openSUSE is a good option because of its stability.

It is possible to simply install the OS with an .iso image from [openSUSE][opensuse] official site. We have to use Leap 42.2 version as it the only version supported by proprietary precompiled CUDA 9.
One can follow GUI instructions during the installation process. In the partitioner, I prefer to use LVM for future flexibility.

After the OS installation, one has to upgrade the Linux kernel version because AMD Ryzen's are supported since kernel v. 4.10, while default kernel in openSUSE Leap 42.2 is 4.4.
It can be performed with [Zypper][kernel]. Finally, I use Linux kernel 4.13.9 from the [repo][kernel_repo]. Unfortunately, 4.14 is under development and unstable, so, I did not manage to make it work properly.

After that, it is possible to overclock the setup, if you want. The system has a great overclocking potential. I, personally, achieved 3.8 GHz CPU rate and 2933 MHz RAM clock rate on near to stock voltage. However, it all depends on your ambitions, luck and efficiency of the cooling system. 
[Prime95][prime95] can be used for system stability testing.

## Install Nvidia

The latest available Nvidia driver can be installed with [Zypper][driver]. [Cuda 9][cuda] can be downloaded from the official Nvidia site and installed without any problems as well as [cuDNN 7][cudnn]. One may need to add the installation location into the `$LD_LIBRARY_PATH`.

```
LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
LD_LIBRARY_PATH=/usr/local/cuda/include:$LD_LIBRARY_PATH
LD_LIBRARY_PATH=/usr/local/cuda:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH
```
You may add the lines to your .bashrc file

## TensorFlow

To build Tensorflow from source (as it is the only option to make it runnable with CUDA 9) we need [Bazel][bazel]. It should be compiled from source as well.
However, I was not able to compile it with default java OpenJDK, so, I had to install original [Java JDK 8][java]. `/usr/local/` is my place to install it. Do not forget to add its location as `$JAVA_HOME`

```
export JAVA_HOME=/usr/local/jdk
export PATH=$JAVA_HOME/bin:$PATH
```

The Bazel installation [instruction][bazel_install] is quite simple, however, do not forget to install the Bazel dependencies (`zip` is the only missed package in case of described here installation of openSUSE 42.2).
After compilation, I placed it in the   `/usr/local/` as well and add it to the $PATH.

```
export PATH=/usr/local/bazel/bin:$PATH
```

Finally, it is possible to compile the [TensorFlow][tensorflow]!

We need a Python 3.5 environment as it is the only version supported by TensorFlow. I prefer to use [miniconda][miniconda] virtual environments. So, I installed miniconda3 and created a Python 3.5.4 environment.

```
conda create -n tensorflow python=3.5.4
```

After `source activate tensorflow` command you are able to install all TensorFlow dependencies in the virtual environment with `pip install`.

Let's compile the TensorFlow 1.4.

```
git clone https://github.com/tensorflow/tensorflow 
cd tensorflow
git checkout r1.4
./configure
```
At this step, you have to specify the path to installed packages and versions you'd like to use. After configure, it can be built and installed.

```
bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
sudo pip install /tmp/tensorflow_pkg/tensorflow-YOUR-VERSION.whl
```

UPDATE: You may consider building Tensorflow with the processor-specific optimization, such as AVX2 or SSE4.2 instructions. See [How to compile Tensorflow with SSE4.2 and AVX instructions][TensorflowAvx].
It really can increase the system performance for about 10-20% according to my personal experience.

And it is done!

[agesa]: https://www.phoronix.com/scan.php?page=news_item&px=AGESA-1.0.0.6b-Update
[opensuse]: https://software.opensuse.org/
[kernel]: https://doc.opensuse.org/documentation/leap/reference/html/book.opensuse.reference/cha.tuning.multikernel.html#cha.tuning.multikernel.zypper
[kernel_repo]: http://download.opensuse.org/repositories/Kernel:/stable/standard/x86_64/
[prime95]: https://www.mersenne.org/download/
[driver]: https://en.opensuse.org/SDB:NVIDIA_drivers#Recommended_Procedure
[cuda]: https://developer.nvidia.com/cuda-downloads
[cudnn]: https://developer.nvidia.com/cudnn
[bazel]: https://docs.bazel.build/versions/master/install.html
[bazel_install]: https://docs.bazel.build/versions/master/install-compile-source.html
[java]: http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html
[tensorflow]: https://www.tensorflow.org/install/install_sources
[miniconda]: https://conda.io/miniconda.html
[TensorflowAvx]: https://stackoverflow.com/questions/41293077/how-to-compile-tensorflow-with-sse4-2-and-avx-instructions
