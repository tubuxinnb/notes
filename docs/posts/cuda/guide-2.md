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
2. `cudaLaunchKernelEx`, which will be talked later.

##### 2.1.2.2.1 Triple Chevron Notation

Now there are some concrete grammars and definitions, referred to the source of guide. Here list some keywords: 
`dim3`: CUDA type for description of grid and block

`<<dim3 grid, dim3 block>>`

`threadIdx` gives the index of a thread within its thread block. Each thread in a thread block will have a different index.

`blockDim` gives the dimensions of the thread block, which was specified in the execution configuration of the kernel launch.

`blockIdx` gives the index of a thread block within the grid. Each thread block will have a different index.

`gridDim` gives the dimensions of the grid, which was specified in the execution configuration when the kernel was launched.

`cuda::ceil_div` does the ceiling divide to calculate the number of blocks needed for a kernel launch.
```
int blocks = cuda::ceil_div(vectorLength, threads);
```

### 2.1.3. Memory in GPU Computing
A, B, C in `vecadd` should be accessible for threads. The ways are various. Here are 2, more in the [Unified Memory]().

#### 2.1.3.1 Unified Memory
Usage: Memory is allocated using the cudaMallocManaged API or by declaring a variable with the `__managed__` specifier. 
Function: the allocated memory is accessible to the GPU or CPU whenever either tries to access it.
Unified memory can be released using cudaFree.

!!! quote

    On some Linux systems, (e.g. those with address translation services or heterogeneous memory management) all system memory is automatically unified memory, and there is no need to use cudaMallocManaged or the __managed__ specifier.

__syncthreads()