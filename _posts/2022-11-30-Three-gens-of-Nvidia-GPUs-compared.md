---
layout: post
title:  "Four generations of Nvidia GPUs compared"
title_img: /assets/post17/title.png
abstract: Benchmark results of GTX 1080 TI, RTX 2080Ti, 3090 and 4090 on DL tasks.
date:   2022-11-30 12:00:00 +0100
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

The post continues the series of benchmark published [previously][Benchmark] with results for GPUS of the latest two generations of GPS. The four generations of Nvidia GPUs constitutes progress of the high-end consumer grade graphics adapter, which I have being using in my Deep Learning activities over the past five years.

### Method

The GPUs were evaluated by a benchmark script from [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) repo by [Ross Wightman](https://github.com/rwightman). The test code was selected because it seems to represent real workload of Computer Vision Deep learning tasks quite accuratly. All tests were performed in the same docker environment, based on _nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04_ docker image with PyTorch 1.13.0.

In all evaluations, the models were given the same input image size of 224x224 pixels and a batch size of 256 for all inference experiments. The number of samples per batch varied depending on the available VRAM on the GPU.

All GPUs were tested with stock clock speed, no overclocking applied.

<table style="width:100%">
  <caption><b>GPU specs</b></caption>
  <tr>
    <th >GPU</th>
    <th  style="text-align:center">Number of CUDA cores</th>
    <th  style="text-align:center">Base Clock, MHz</th>
    <th  style="text-align:center">Number of Tensor Cores</th>
    <th  style="text-align:center">VRAM, GB</th>
    <th  style="text-align:center">TDP, W</th>
    <th  style="text-align:center">Release date</th>
  </tr>
  
  <tr>
  	<th>GTX 1080 Ti</th>
    <td style="text-align:center"> 3584 </td>
    <td style="text-align:center"> 1480 </td>
    <td style="text-align:center"> - </td>
    <td style="text-align:center"> 11 </td>
    <td style="text-align:center"> 250 </td>
    <td style="text-align:center"> Mar 2017 </td>
  </tr>
  <tr>
  	<th>RTX 2080 Ti</th>
    <td style="text-align:center"> 4352 </td>
    <td style="text-align:center"> 1350 </td>
    <td style="text-align:center"> 544 </td>
    <td style="text-align:center"> 11 </td>
    <td style="text-align:center"> 250 </td>
    <td style="text-align:center"> Sep 2018 </td>
  </tr>
  <tr>
  	<th>RTX 3090</th>
    <td style="text-align:center"> 10496 </td>
    <td style="text-align:center"> 1395 </td>
    <td style="text-align:center"> 328 </td>
    <td style="text-align:center"> 24 </td>
    <td style="text-align:center"> 350 </td>
    <td style="text-align:center"> Sep 2020 </td>
  </tr>
  <tr>
  	<th>RTX 4090</th>
    <td style="text-align:center"> 16384 </td>
    <td style="text-align:center"> 2230 </td>
    <td style="text-align:center"> 512 </td>
    <td style="text-align:center"> 24 </td>
    <td style="text-align:center"> 450 </td>
    <td style="text-align:center"> Sep 2022 </td>
  </tr>
</table>

It is important to note that Tensor Core technology has been updated with each generation of GPUs, particularly with the inclusion of a wider range of precisions and improved throughput, which has made the use of Tensor Cores more convenient, easier and rewarding.

## Results
The tables contains inference (```eval```) and training (```train```) rate in samples per second. Nvidia GTX 1080 Ti is used as the reference.

<table style="width:100%">
  <caption><b>FP32 results</b></caption>
  <tr>
    <th rowspan="2">GPU</th>
    <th colspan="2" style="text-align:center">vgg16</th>
    <th colspan="2" style="text-align:center">resnet50</th>
    <th colspan="2" style="text-align:center">tf_efficientnetv2_b0</th>
    <th colspan="2" style="text-align:center">swin_base_patch4 _window7_224</th>
    <th rowspan="2">Average</th>
  </tr>
  <tr>
    <td style="text-align:center">eval</td>
    <td style="text-align:center">train</td>
    <td style="text-align:center">eval</td>
    <td style="text-align:center">train</td>
    <td style="text-align:center">eval</td>
    <td style="text-align:center">train</td>
    <td style="text-align:center">eval</td>
    <td style="text-align:center">train</td>
  </tr>
  <tr>
  	<th>GTX 1080 Ti</th>
    <td style="text-align:center">405</td>
    <td style="text-align:center" >104</td>
    <td style="text-align:center">685</td>
    <td style="text-align:center">185</td>
    <td style="text-align:center">1730</td>
    <td style="text-align:center">418</td>
    <td style="text-align:center">129</td>
    <td style="text-align:center">50</td>
    <td style="text-align:center">0%</td>
  </tr>
  <tr>
   	<th>RTX 2080 Ti</th>
    <td style="text-align:center">513 <font color="green">(+26.7%)</font></td>
    <td style="text-align:center">132 <font color="green">(+26.9%)</font></td>
    <td style="text-align:center">912 <font color="green">(+33.1%)</font></td>
    <td style="text-align:center">252 <font color="green">(+36.2%)</font></td>
    <td style="text-align:center">2456 <font color="green">(+42.0%)</font></td>
    <td style="text-align:center">609 <font color="green">(+45.7%)</font></td>
    <td style="text-align:center">234 <font color="green">(+81.4%)</font></td>
    <td style="text-align:center">76 <font color="green">(+52.0%)</font></td>
    <td style="text-align:center"><font color="green">+43.0%</font></td>
  </tr>
  <tr>
 	 	<th>RTX 3090</th>
    <td style="text-align:center">997 <font color="green">(+146.2%)</font></td>
    <td style="text-align:center">285 <font color="green">(+174.0%)</font></td>
    <td style="text-align:center">1708 <font color="green">(+149.3%)</font></td>
    <td style="text-align:center">535 <font color="green">(+189.2%)</font></td>
    <td style="text-align:center">4211 <font color="green">(+143.4%)</font></td>
    <td style="text-align:center">1118 <font color="green">(+167.5%)</font></td>
    <td style="text-align:center">370 <font color="green">(+186.8%)</font></td>
    <td style="text-align:center">129 <font color="green">(+158.0%)</font></td>
    <td style="text-align:center"><font color="green">+164.3%</font></td>
  </tr>
  <tr>
    <th>RTX 4090</th>
  	<td style="text-align:center">1388	<font color="green">(+242.7%)</font></td>
    <td style="text-align:center">457 <font color="green">(+339.4%)</font></td>
    <td style="text-align:center">2310	<font color="green">(+237.2%)</font></td>
    <td style="text-align:center">721 <font color="green">(+289.7%)</font></td>
    <td style="text-align:center">6027	<font color="green">(+248.4%)</font></td>
    <td style="text-align:center">1543 <font color="green">(+269.1%)</font></td>
    <td style="text-align:center">674	<font color="green">(+422.5%)</font></td>
    <td style="text-align:center">404 <font color="green">(+708.0%)</font></td>
    <td style="text-align:center"><font color="green">+344.6%</font></td>
  </tr>
</table>


<table style="width:100%">
  <caption><b>FP16 results</b></caption>
  <tr>
    <th rowspan="2">GPU</th>
    <th colspan="2" style="text-align:center">vgg16</th>
    <th colspan="2" style="text-align:center">resnet50</th>
    <th colspan="2" style="text-align:center">tf_efficientnetv2_b0</th>
    <th colspan="2" style="text-align:center">swin_base_patch4 _window7_224</th>
    <th rowspan="2">Average</th>
  </tr>
  <tr>
    <td style="text-align:center">eval</td>
    <td style="text-align:center">train</td>
    <td style="text-align:center">eval</td>
    <td style="text-align:center">train</td>
    <td style="text-align:center">eval</td>
    <td style="text-align:center">train</td>
    <td style="text-align:center">eval</td>
    <td style="text-align:center">train</td>
  </tr>
  <tr>
  	<th>GTX 1080 Ti</th>
    <td style="text-align:center">417</td>
    <td style="text-align:center">94</td>
    <td style="text-align:center">887</td>
    <td style="text-align:center">235</td>
    <td style="text-align:center">2136</td>
    <td style="text-align:center">499</td>
    <td style="text-align:center">152</td>
    <td style="text-align:center">57</td>
    <td style="text-align:center">0%</td>
  </tr>
  <tr>
   	<th>RTX 2080 Ti</th>
    <td style="text-align:center">966	<font color="green">(+131.7%)</font></td>
    <td style="text-align:center">309 <font color="green">(+228.7%)</font></td>
    <td style="text-align:center">1995	<font color="green">(+124.9%)</font></td>
    <td style="text-align:center">554 <font color="green">(+135.7%)</font></td>
    <td style="text-align:center">4617	<font color="green">(+116.2%)</font></td>
    <td style="text-align:center">1124 <font color="green">(+125.3%)</font></td>
    <td style="text-align:center">680	<font color="green">(+347.4%)</font></td>
    <td style="text-align:center">225 <font color="green">(+294.7%)</font></td>
    <td style="text-align:center"><font color="green">+229.6%</font></td>
  </tr>
  <tr>
 	 	<th>RTX 3090</th>
    <td style="text-align:center">1394	<font color="green">(+234.3%)</font></td>
    <td style="text-align:center">442 <font color="green">(+370.2%)</font></td>
    <td style="text-align:center">3017	<font color="green">(+240.1%)</font></td>
    <td style="text-align:center">890 <font color="green">(+278.7%)</font></td>
    <td style="text-align:center">7059	<font color="green">(+230.5%)</font></td>
    <td style="text-align:center">1706 <font color="green">(+241.9%)</font></td>
    <td style="text-align:center">1026	<font color="green">(+575.0%)</font></td>
    <td style="text-align:center">341 <font color="green">(+500.0%)</font></td>
    <td style="text-align:center"><font color="green">+333.8%</font></td>
  </tr>
  <tr>
    <th>RTX 4090</th>
  	<td style="text-align:center">2359	<font color="green">(+465.7%)</font></td>
    <td style="text-align:center">729 <font color="green">(+675.5%)</font></td>
    <td style="text-align:center">4495	<font color="green">(+406.8%)</font></td>
    <td style="text-align:center">1285 <font color="green">(+446.8%)</font></td>
    <td style="text-align:center">11856	<font color="green">(+455.1%)</font></td>
    <td style="text-align:center">2598 <font color="green">(+420.6%)</font></td>
    <td style="text-align:center">1692	<font color="green">(+1013.2%)</font></td>
    <td style="text-align:center">563 <font color="green">(+887.7%)</font></td>
    <td style="text-align:center"><font color="green">+596.4%</font></td>
  </tr>
</table>

## Observations

* The latest architecture benefits more from recent hardware advances. 
* It appears that Tensor Cores play a significant role in the performance increase. The most prominent boost in float32 performance occurred when moving from the 20th series to the 30th series, which coincides with the support of TF32 by Tensor Cores. The most significant jump was seen with float16 precision with the first introduction of Tensor Cores in the RTX 2080 Ti.
* Interestingly, the performance of older GPUs has also been improved through software updates. In previous [tests][Benchmark] with PyTorch 1.0 and CUDA 10.0, the GTX 1080 Ti performed just slightly better in float16 mode compared to float32. However, the current test results show an average improvement of about 15%. Additionally, the performance gain of the RTX 2080 Ti in float16 mode was previously below 2x, but the current results show a significantly higher performance benefit. This suggests that the software updates (both by Nvidia and PyTorch) are also improving performance and older GPUs benefit from updates.
* During the training step, the performance increase with newer hardware is slightly better than in inference mode. This may be because the training mode is less dependent on data transfer as GPU computations take longer.
* While the average performance increase is impressive, the increased power consumption of the RTX 4090 compared to the GTX 1080 Ti (an increase of 80%) may make the results less appealing to some users.
* If we consider the Moor's law as doubling of computational performance every two years, the time between the oldest and the newest GPU from this study should result in about 6.75x perforamnce increase. The demonstrated average improvement on Deep Learning tasks is approximately 6x, which is in good agreement, given imperfections of the measurements and empirical nature of the relationship. Therefore, it can be concluded that Moore's law is still not dead, at least for float16 GPU computations :).



## Acknowledgments

Many thanks to [Ruslan Baikulov](https://github.com/lRomul/), who contributed some of the test results.

[Benchmark]: {% post_url 2018-11-06-Benchmarking-RTX-2080-Ti-vs-Pascal-GPUs-with-DL-tasks %}
[NGC]: https://ngc.nvidia.com
