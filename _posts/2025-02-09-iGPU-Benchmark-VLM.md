---
layout: post
title: Performance Analysis of Intel iGPUs in VLM and LLM applications
title_img: /assets/post24/title.webp
abstract: >
    Analysis of cost-effective use of Intel iGPUs with ipex-llm to accelerate VLMs and LLMs with detailed benchmarks and practical setup instructions.
date: 2025-02-09 21:00:00 +0000
categories: Hardware DeepLearning
article: true
sitemap:
    lastmod: 2025-03-23
---

<style>
table{
    border: none;
    border-collapse: collapse;
    text-align: center;
}
</style>
_The topic of the blog has always been and remains Computer Vision. Sometimes, when working on computer vision tasks, it can be handy to use language models or vision-language models. For example, these models can serve as components in agent-based systems for research on specific topics, or in the case of vision-language models, they can act as simple prototypes for custom computer vision models or a source of annotation. Typically, capable models often require substantial hardware, powerful dedicated GPUs with a large amount of VRAM. However, this post explores a less widely known budget-friendly option for running these models._

![Benchmark results](/assets/post24/title.png)

Complementary GitHub repository: [https://github.com/NikolasEnt/ollama-webui-intel](https://github.com/NikolasEnt/ollama-webui-intel)

## Ollama on Intel iGPUs

Hardware acceleration on Intel hardware can be achieved using [ipex-llm](https://ipex-llm-latest.readthedocs.io/en/latest/index.html), which can be used directly with PyTorch or integrated with other tools, for example, a provided [Ollama](https://github.com/ollama/ollama) version [patched by Intel](https://github.com/intel/ipex-llm/blob/main/docs/mddocs/Quickstart/ollama_quickstart.md) to support the relevant hardware.

## Hardware Acceleration Using IPEX-LLM

It is an optimised library that enhances performance for transformer-based models on Intel processors and other hardware (GPUs, iGPUs, NPUs). `ipex-llm` is designed to provide efficient execution by using Intel's oneAPI Deep Neural Network Library (oneDNN). It supports various levels of low-bit inference: INT2/INT4/INT8, FP4/FP8/FP16 and BF16.

For convenience and reproducibility, an all-in-one Docker compose solution is provided in the GitHub repository for this post: [https://github.com/NikolasEnt/ollama-webui-intel](https://github.com/NikolasEnt/ollama-webui-intel). It includes Ollama with IPEX-LLM as an accelerated backend, compatible with both Intel iGPUs and dedicated GPUs (such as Arc, Flex, and Max), along with the required parameters and settings, as well as the [Open WebUI](https://github.com/open-webui/open-webui) interface.

Using Intel GPUs for Deep Learning tasks computations requires having Intel firmware installed. For example, on Debian-like systems:

```bash
sudo apt-get install firmware-misc-nonfree firmware-intel-graphics
sudo update-initramfs -u -k all 
```

All experiments were conducted using an Intel Ultra 5 125H (equipped with an Intel Arc Xe-LPG 112EU GPU) [1] with 64GB of DDR5 RAM at 5600 MHz in a dual-channel configuration. Note that the full power of the Intel Arc iGPU is only available with at least 16GB of system memory in a dual-channel configuration [2]. The CPU supports AVX2 but does not support AVX-512, so it cannot benefit from Ollama 0.5.8 optimisations, which include AVX512 acceleration. The entire setup was obtained as parts from the Chinese marketplace for under $500, making it an attractive option for running larger models without significant investments. 

Models’ inference speeds were compared with Nvidia RTX 3090 results, which is de facto the most cost-efficient GPU solution for single and dual-GPU workstations when running LLMs locally.

### Benchmarking

For convenience and reproducibility, a benchmarking [script](https://github.com/NikolasEnt/ollama-webui-intel/blob/master/scripts/benchmark.py) is provided. The benchmarking results include uncertainty measures as standard deviations from multiple runs. These measurements were obtained by executing different prompts for text-based models and describing several images for visual models. The standard deviation across these runs gives an estimate of the variability in inference times due to differences in task complexity or types of visual content.

Although tokens do not directly correspond to words, we can use the following rough estimations for actual speed: 2 tokens/s is about the speed of human typing; 5-10 tokens/s is a comfortable model output rate for reading results in chat-based interactions; ~30 tokens/s is required inference speed for comfortable real-time code completion in code-assisting scenarios, such as using Ollama as an inference backend for [continue.dev](https://github.com/continuedev/continue) autocomplete.

All tests were performed using Ollama 0.5.1, which is the latest version for now supported by `ipex-llm` as an accelerated backend for iGPUs or dedicated Intel GPUs like Arc, Flex, and Max. Experiments were conducted on an Intel Ultra 5 125H (Meteor Lake) CPU with 64GB of RAM. 12 threads were used for CPU inference. The value was tuned empirically by optimising throughput on some models; however, it is worth noting that the parameter may require further optimisation for each individual model and context length to achieve optimal performance. All tests were performed with an 8k context length where appropriate for the model and using Q4_K_M quantisation, which is the default recommended quantisation level for Ollama. Power mode was set to 'Performance'. Note that the SoC was not further overclocked during the experiments, so it could theoretically demonstrate a higher generation rate.

| Model              | Ultra 5 CPU tokens/s | Ultra 5 iGPU tokens/s | RTX 3090 tokens/s |
|--------------------|----------------------|-----------------------|-------------------|
| deepseek-r1:70b    | 1.12 ± 0.07          | 1.65 ± 0.08           | NA                |
| llama3.3:70b       | 1.16 ± 0.01          | 1.58 ± 0.00           | NA                |
| llama3.1:70b       | 1.17 ± 0.00          | 1.57 ± 0.00           | NA                |
| llama3.1:8b        | 9.76 ± 0.18          | 12.69 ± 0.20          | 104.31 ± 2.06     |
| qwen2.5:72b        | 1.11 ± 0.01          | 1.24 ± 0.00           | NA                |
| qwen2.5:32b        | 2.46 ± 0.01          | 3.44 ± 0.02           | 31.91 ± 0.34      |
| qwen2.5:7b         | 10.26 ± 0.18         | 13.06 ± 0.09          | 101.03 ± 1.01     |
| qwq                | 2.29 ± 0.08          | 3.01 ± 0.04           | 30.53 ± 0.75      |
| mistral-small:24b  | 3.37 ± 0.03          | 4.87 ± 0.02           | 45.31 ± 0.25      |
| phi4:14b           | 5.27 ± 0.08          | 7.11 ± 0.06           | 64.09 ± 0.95      |
| phi3.5:3.8b        | 19.07 ± 0.86         | 19.60 ± 2.42          | 171.51 ± 1.15     |
| llama3.2:3b        | 20.63 ± 0.44         | 23.20 ± 0.26          | 161.96 ± 3.01     |
| smallthinker:3b    | 13.83 ± 0.63         | 14.66 ± 0.42          | 105.53 ± 1.84     |
| smollm2:1.7b       | 27.41 ± 0.66         | 27.84 ± 0.65          | 209.49 ± 1.78     |
| smollm2:360m       | 57.56 ± 2.63         | 35.13 ± 0.32          | 250.60 ± 8.13     |
| starcoder2:3b      | 19.47 ± 1.51         | 22.30 ± 2.38          | 177.34 ± 3.42     |
| qwen2.5-coder:1.5b | 27.19 ± 0.26         | 36.74 ± 0.23          | 170.02 ± 4.20     |
| opencoder:1.5b     | 32.88 ± 1.60         | 17.67 ± 0.90          | 207.72 ± 3.92     |

An interesting observation is that the iGPU outperforms CPU inference despite being more energy efficient. Excluding very small models, iGPU offers up to 30% faster inference on Intel Ultra 5 125H than the CPU. At the same time, the iGPU consumes about three times less power than the CPU itself.

`Deepseek-r1:32b` performance is not included in the test results above, but it performs similarly to qwen2.5:32b. 

Of course, inference on Nvidia RTX 3090 is much faster, outperforming the Intel iGPU by 7.5-10x. At the same time, the results are achieved with a 25-30x increase in energy consumption of the GPU subsystem (partially due to the much older 8 nm process versus TSMC's N5 (5nm) used for the graphics tile of Intel’s SoC and, secondarily, due to differences in model computation optimisations applied for different hardware architectures). 

Running 70B models on a single GPU with 24GB of space at a reasonable quantisation level is not possible. Offloading some layers to the CPU will make overall inference much slower, diminishing the overall benefits of having a dedicated GPU. This means that CPUs with integrated GPUs and unified memory offer unique opportunities for prototyping or work that does not require instant responses from larger models.

### Quantization Levels

The following table presents the impact of different quantization levels on the model inference speed on the Intel iGPU.

| Model                     | Mean ± STD Tokens/s |
|---------------------------|--------------------|
| qwen2.5:32b-instruct-q8_0 | 2.02 ± 0.01        |
| qwen2.5:32b-q4_K_M        | 3.22 ± 0.01        |
| qwen2.5:7b-instruct-fp16  | 4.74 ± 0.02        |
| qwen2.5:7b-instruct-q8_0  | 8.26 ± 0.05        |
| qwen2.5:7b-q4_K_M         | 12.08 ± 0.08       |

The larger RAM capacity allows for the execution of higher precision models, which can be useful in applications where inference accuracy is preferable over inference time. This capability enables the inference of quantization levels that are not possible on consumer-grade GPUs with smaller VRAM sizes, including full precision (e.g., FP16). From the experiments, we may suggest that Q8 quantisation inference is about 35% slower than Q4 quantisation, while FP16 is 60% slower but offers higher quality of results compared to lower-precision models.

## VLMs

<div style="text-align: center;">
    <img src="/assets/post24/offload.png" alt="Ollama terminal stdout" style="max-width: 60%; height: auto; display: inline-block;">
    <br><i>An example of llama3.2-vision:90b model being offladed to iGPU inference. The model requires more memory than available in two 24GB consumer-grade GPUs, making the SoC with enough memory an interesting platform for prototyping with such models</i>
</div>

These models are useful for auto data extraction from images or for image annotation for training traditional Computer Vision models, as well as quick prototyping of Computer Vision systems.

| Model               | Ultra 5 iGPU tokens/s | RTX 3090 tokens/s |
|---------------------|-----------------------|-------------------|
| llama3.2-vision:90b | 0.92 ± 0.01           | NA                |
| llama3.2-vision:11b | 5.73 ± 0.03           | 61.90 ± 0.20      |
| minicpm-v:8b        | 14.94 ± 0.41          | 98.69 ± 0.18      |
| llava-phi3:3.8b     | 18.93 ± 0.12          | 154.73 ± 1.62     |
| moondream:1.8b      | 35.53 ± 1.48          | 280.98 ± 45.34    |


Note that some computations during the inference of VLM architectures may still be performed on the CPU, but this may change in the future.

Unfortunately, Ollama does not yet support Qwen2.5-VL, a family of very useful models for computer vision applications, including automatic data extraction from images and auto data annotation. However, these models can be inferred directly using other Intel tools, which may be covered in one of the posts in the future.

## Comments on application

This setup, based on a modern Intel CPU with a capable iGPU that can utilize unified RAM, provides unique capabilities for running relatively large models locally. The iGPUs are surprisingly efficient, offering better performance than the CPU while maintaining energy efficiency, even when compared to dedicated GPUs. Offloading model inference onto the iGPU allows the CPU to be utilized for other tasks, or enabling parallel execution of multiple models. For example, the setup can run llama3.3:70b (`1.05` tokens/s on iGPU) and `qwen2.5:32b` (1.77 tokens/s on CPU) in parallel with a combined throughput higher than sequential execution of the two models. With up to 96GB of RAM, the system could potentially run two 70B models simultaneously.

Solutions developed with prototyping stage on such an iGPU platform can be ported to other, more powerful Intel inference solutions without significant changes, which is yet another attractive point for some cases.

This versatile platform is well-suited for agentic or general language/vision applications where immediate results are not required. For instance, the author frequently runs 70B models overnight for agent-based research projects and automatic data annotation.

For direct usage, the iGPU provides good speed for chat-based interactions with models up to around 7-8B parameters. Larger models are slower but still usable in real-time chat scenarios with higher quality answers. Smaller models (~2B) can serve as local code completion assistants.

Another potential application is running smart home assistants using the low-power device for 24/7 operations. The hardware can support `qwen2.5:7b` as the language model, Whisper for speech-to-text (STT), and a text-to-speech (TTS) model for private local inference in voice-controlled smart homes (for example, as part of Home Assistant integration) with performance comparable to commercial online solutions.

Another interesting component of the Intel SoC is the NPU, which is also [supported](https://github.com/intel/ipex-llm/tree/main/python/llm/example/NPU/HF-Transformers-AutoModels/LLM) by ipex-llm, will be benchmarked in future posts.

As a final remark, it is worth noting that if fast inference is required and cost is not a limiting factor, Nvidia-based solutions can undeniably offer much higher performance and throughput for big models. However, Intel SoCs may be suitable for specific applications where energy efficiency and cost are important.


References:

1. Intel Core Ultra Series 1 [product brief](https://www.intel.com/content/www/us/en/products/docs/processors/core-ultra/core-ultra-series-1-product-brief.html#:~:text=Intel%C2%AE%20Arc%E2%84%A2%20GPUs%20only%20available%20on%20select%20H%2Dseries).
2. Intel Ultra 5 125H [specs](https://www.intel.com/content/www/us/en/products/sku/236848/intel-core-ultra-5-processor-125h-18m-cache-up-to-4-50-ghz/specifications.html).








