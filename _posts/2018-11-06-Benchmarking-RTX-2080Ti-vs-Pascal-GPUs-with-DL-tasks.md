---
layout: post
title:  "Benchmarking RTX 2080Ti vs Pascal GPUs with DL tasks"
title_img: /assets/post15/title.jpg
abstract: Comparation of Nvidia RTX 2080 Ti with GTX 1080 Ti and 1070.
date:   2018-11-06 12:00:00 +0300
categories: Hardware, DeepLearning
article: true
---

<style>
table, th, td {
    border: 1px solid black;
    border-collapse: collapse;
}
th, td {
    padding: 5px;
    text-align: left;    
}
</style>

## Testing
The post presents results of Turing and Pascal GPUs benchmarking with a popular [Deep Learning Benchmark][Benchmark].
PyTorch based tests with both floating point precisions (FP32 and FP16) were chosen for the comparison.

All tests were performed with nvcr.io/nvidia/pytorch:18.10-py3 nvidia-docker (Pytorch 1.0a0, CUDA 10, cuDNN 7400) for reproducibility. Proprietary Nvidia driver version: 410.73.
It can be obtained from the [Nvidia NGC Registry][NGC]. Other test parameters were set the same to the original [tests][Benchmark], that allows a direct comparison between the results.

An AMD Ryzen 7 1700X CPU powered the testing machine with 64 GB of RAM. All pieces of hardware were on stock frequencies without overclocking.

## Results
The tables contains time for a forward (```eval```) or forward and backward (```train```) passes for different models.
The relative results as compared with GTX 1080 Ti as a reference are given in brakets.

<table style="width:100%">
  <caption><b>FP32 results</b></caption>
  <tr>
    <th rowspan="2">GPU</th>
    <th colspan="2">VGG-16</th>
    <th colspan="2">ResNet-152</th>
    <th colspan="2">DenseNet-161</th>
  </tr>
  <tr>
    <td>eval</td>
    <td>train</td>
    <td>eval</td>
    <td>train</td>
    <td>eval</td>
    <td>train</td>
  </tr>
  <tr>
  	<th>RTX 2080 Ti</th>
    <td>28.4 ms <font color="green">(-28.3%)</font></td>
    <td>97.5 ms <font color="green">(-22.9%)</font></td>
    <td>42.7 ms <font color="green">(-27.6%)</font></td>
    <td>151.2 ms <font color="green">(-24.2%)</font></td>
    <td>46.6 ms <font color="green">(-27.2%)</font></td>
    <td>155.9 ms <font color="green">(-25.9%)</font></td>
  </tr>
  <tr>
  	<th>GTX 1080 Ti</th>
    <td>39.6 ms</td>
    <td>126.4 ms</td>
    <td>59.0 ms</td>
    <td>199.5 ms</td>
    <td>64.0 ms</td>
    <td>210.4 ms</td>
  </tr>
  <tr>
  	<th>GTX 1070</th>
    <td>65.9 ms <font color="red">(+66.4%)</font></td>
    <td>205.6 ms <font color="red">(+62.7%)</font></td>
    <td>102.4 ms <font color="red">(+73.6%)</font></td>
    <td>333.9 ms <font color="red">(+67.4%)</font></td>
    <td>109.0 ms <font color="red">(+70.3%)</font></td>
    <td>348.7 ms <font color="red">(+65.7%)</font></td>
  </tr>
</table>



<table style="width:100%">
  <caption><b>FP16 results</b></caption>
  <tr>
    <th rowspan="2">GPU</th>
    <th colspan="2">VGG-16</th>
    <th colspan="2">ResNet-152</th>
    <th colspan="2">DenseNet-161</th>
  </tr>
  <tr>
    <td>eval</td>
    <td>train</td>
    <td>eval</td>
    <td>train</td>
    <td>eval</td>
    <td>train</td>
  </tr>
  <tr>
  	<th>RTX 2080 Ti</th>
    <td>19.3 ms <font color="green">(-40.0%)</font></td>
    <td>70.7 ms <font color="green">(-39.1%)</font></td>
    <td>25.0 ms <font color="green">(-49.4%)</font></td>
    <td>101.8 ms <font color="green">(-48.1%)</font></td>
    <td>30.7 ms <font color="green">(-41.7%)</font></td>
    <td>116.4 ms <font color="green">(-39.6%)</font></td>
  </tr>
  <tr>
  	<th>GTX 1080 Ti</th>
    <td>36.4 ms</td>
    <td>116.1 ms</td>
    <td>49.4 ms</td>
    <td>196.2 ms</td>
    <td>52.7 ms</td>
    <td>192.6 ms</td>
  </tr>
  <tr>
  	<th>GTX 1070</th>
    <td>61.2 ms <font color="red">(+68.1%)</font></td>
    <td>190.9 ms <font color="red">(+64.4%)</font></td>
    <td>86.1 ms <font color="red">(+74.3%)</font></td>
    <td>309.3 ms <font color="red">(+57.6%)</font></td>
    <td>88.2 ms <font color="red">(+67.4%)</font></td>
    <td>306.2 ms <font color="red">(+59.0%)</font></td>
  </tr>
</table>


To sum up, the new generation of GPU's has a bit less than 30% increase in computational power as compared with the Pascal 1080 Ti. However, one should note up to 50% increase in FP16 performance which is archived by the hardware support of the half precision calculations. Such an increase might produce a huge difference for practical application, especially for inference speed-up.

[Benchmark]: https://github.com/u39kun/deep-learning-benchmark
[NGC]: https://ngc.nvidia.com
