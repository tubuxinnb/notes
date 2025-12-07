---
date: 
  created: 2025-12-07
categories:
  - NOTE 
  - CUDA
links:
  - posts/cuda/guide-1.md
  - posts/cuda/guide-3.md
draft: true
---

# CUDA Programming Guide-2
Now it comes to the second part of the guide, which is about programming GPUs in CUDA and introduces some basic concepts in the CUDA programming model.

<!-- more -->

## 2.1. Intro to CUDA C++
1. Focus on CUDA runtime API.
> [CUDA Runtime API and CUDA Driver API](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html#cuda-platform-driver-and-runtime) discusses the difference between the APIs and CUDA driver API discusses writing code that mixes the APIs.
2. [The CUDA Quickstart](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html) Guide for basic installation.

### 2.1.1. Compilation with NVCC
NVCC is the CUDA compiler.
> The nvcc chapter of this guide covers common use cases of nvcc, and complete documentation is provided by the [nvcc user manual](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html).
### 2.1.2. Kernels
kernels are functions executed by the GPU and launched by the CPU.
#### 2.1.2.1. Specifying Kernels
`__global__` is a keyword that specifies a function as a kernel.
```
// Kernel definition
__global__ void vecAdd(float* A, float* B, float* C)
{
...
}
```
#### 2.1.2.2. Launching Kernels
The num of threads executing the kernel is specified as part of the execution configuration.

2 ways to launch kernels:
1. **triple chevron notation**
2. `cudaLaunchKernel`
