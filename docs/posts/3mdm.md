---
date: 2025-12-07
categories:
  - CUDA
  - PhD Application
---

# 3 Matrix Hadamard Product
Dot product is element-wise: 
$A \cdot B = C$
where $C_{i,j} = A_{i,j} \cdot B_{i,j}$
because every element in $A$ and $B$ is only calculated once. the Arithmetic Intensity (AI) is a constant.
FLOPS = $N^2$
Bytes = $2 \times N^2 \times 3$ (read A, B, write C)
$\text{AI} = \text{FLOPS} / \text{Bytes} = \frac{1}{6}$
The Arithmetic Intensity is a fixed constant.
So our goal is to improve the memory throughput of the dot product.
We choose 2 ways to improve the memory throughput:
1. **kernel fusion**
2. **memory coalescing**
for 3 matrix dot multiple, we should access 3 matrixes in memory.
However, in Pytorch, this multiplication is implemented in **2 individual kernels** of element-wise multiplication.
## Kernel Fusion
A intuitive way is to fuse these two kernels into one kernel:
```
__global__ void mul_tri_naive_kernel(const float* a, const float* b, const float* c, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * b[idx] * c[idx];
    }
}
```
where the elements in this kernel is not vectorized. 

## Memory Coalescing
Considering that the memory access width of a single request is 128 bytes while the float's width is 32 bytes, there are 75% of the memory accesses are wasted.
In my evaluation, as the size of matrix increases, the performance degrades severely. 
However, we can use vectorized multiplication to improve the performance.  So we can coalesce 4 `float` into a single memory access using `float4` type:

```
__global__ void mul_tri_vec_kernel(const float* a, const float* b, const float* c, float* out, int n_vec) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_vec) {
        float4 va = reinterpret_cast<const float4*>(a)[idx];
        float4 vb = reinterpret_cast<const float4*>(b)[idx];
        float4 vc = reinterpret_cast<const float4*>(c)[idx];
        float4 vout;
        vout.x = va.x * vb.x * vc.x;
        vout.y = va.y * vb.y * vc.y;
        vout.z = va.z * vb.z * vc.z;
        vout.w = va.w * vb.w * vc.w;
        reinterpret_cast<float4*>(out)[idx] = vout;
    }
}
```
Comparison between different kernels: at Thread/Block Level:

| Thread/Block | kernel_num | Block_num/Kernel | Thread_num/Block | Mul/Thread | mem/Thread |
| :--- | :---: | :---: | :---: | :---: | :---: |
| fuse+vectorize | 1 | $N/4/256$ | 256 | 8 | 4 |
| vectorize | 2 | $N/4/256$ | 256 | 4 | 3 |
| fuse | 1 | $N/256$ | 256 | 2 | 4 |
| base | 2 | $N/256$ | 256 | 1 | 3 |
| Pytorch | 2 | Unknown | 256 | 4 | Unknown |


整体性能的fundamental因素：计算带宽和访存带宽
### Block/Thread Level
关于计算，考虑以下几个因素：
计算数据类型：都是float
计算方式：都是乘法
计算次数：fused kernel 2次，base kernel 1次；vectorize kernel 4次，base kernel 1次
计算开销：
- Fused & Vectorized: $T_{compute} = 8~T_{\times}$
- Vectorized: $T_{compute} = 4~T_{\times}$
- Fused: $T_{compute} = 2~T_{\times}$
- Base: $T_{compute} = T_{\times}$
  
关于访存：影响因素颇多。
1. 不考虑 L2 Cache (L2 Cache all miss)，只考虑从 global memory 读写：

   - Fused: $RQ_{mem} = 3 \times RQ_{read} + RQ_{write} = 4 ~RQ_{r/w}$
   - Base: $RQ_{mem} = 2 \times RQ_{read} + RQ_{write}=3~RQ_{r/w}$

2. 经过warm up后，考虑 L2 Cache (L2 Cache all hit)，如果size足够小，只考虑从 L2 Cache 读写：**(todo)**

以上我们的分析针对的是单个kernel中的单个thread上，现在往上分析一层，看看单个kernel的整体的计算和访存开销
### Kernel Level
| For Kernel| kernel num | Block num | Thread_num | Mul | memory access |
| :--- | :---: | :---: | :---: | :---: | :---: |
| fuse+vectorize | 1 | $N/1024$ | $N/4$ | $2N$ | $N$ |
| vectorize | 2 | $N/1024$ | $N/4$ | $N$ | $0.75N$ |
| fuse | 1 | $N/256$ | $N$ | $2N$ | $4N$ |
| base | 2 | $N/256$ | $N$ | $N$ | $3N$ |
| Pytorch | 2 | Unknown | Unknown | Unknown | Unknown |

**关于计算**：~~同一kernel中的thread的计算是并行的，不存在数据依赖，我们考虑单个kernel启动的thread和block数量。参考上表，vectorization的kernel的block和对应的thread的数量减少到原来的1/4。如果GPU的SM能够并行执行这些blocks，那么速度应该不会受到影响，否则可能会有warp的排队延迟**。但单就这一层来说，减少block的数量总是好的。~~
在GPU中计算延迟(Computing latency)通常不再考虑，主要考虑计算吞吐(Computing Throughput), 用CQ来表示计算吞吐：
$CQ = \frac{num(thread)}{num(SM)} \times T_{compute}$
抽象出来简单的倍数关系：
 - Fused & Vectorized: $CQ = 2N~T_{\times}$
 - Vectorized: $CQ = N~T_{\times}$
 - Fused:  $CQ = 2N~T_{\times}$
 - Based: $CQ = N~T_{\times}$

**关于访存**：单个kernel执行的访存请求次数是有区别的，参考上表，fusion会将访存请求增长1/3，vectorized kernel的访存请求减少3/4，是带宽敏感的metrics：如果带宽够，这个数据将不会影响kernel的执行速度，否则这会影响性能，但减少访存请求次数总是好的。
 - Fused & Vectorized: $RQ_{mem} = N \times RQ_{r/w}$
 - Vectorized: $RQ_{mem} = 0.75N \times RQ_{r/w}$
 - Fused: $RQ_{mem} = 4N \times RQ_{r/w}$
 - Based: $RQ_{mem} = 3N \times RQ_{r/w}$

### Consider All together
Notably, the unfused approach introduces a **data dependency** between the successive kernels： 
 - Fused: $T_{total} = T_{kernel}$
 - Non-fused: $T_{total} = 2~T_{kernel}$

考虑 $T_{kernel}$, 我们**假设计算能够完全并行（大概率是），且所有访存请求皆发送到 global memory**，那么：
$$
T_{kernel} = T_{compute} + T_{mem},\\
$$
$$
T_{compute} =\frac{ CQ}{CQ(\times)}\\
$$

$$
T_{mem} = \frac{RQ_{mem} \times 4\text{Bytes}}{BW(global)}
$$
where $T_{sm}$ is the SM instruction transmission cycle, and $BW_g$ is the global memory bandwidth.
translate to the formula using $CQ(\times), RQ_{global},N$: 

 - Fused & Vectorized: $T_{total} = (\frac{2}{CQ(\times)} + \frac{4}{BW(global)})~N$
 - Vectorized: $T_{total} = 2 \times (\frac{1}{CQ(\times)} + \frac{3}{BW(global)})~N$
 - Fused: $T_{total} = (\frac{2}{CQ(\times)} + \frac{16}{BW(global)}) ~N$
 - Based: $T_{total} =2 \times (\frac{1}{CQ(\times)} + \frac{12}{BW(global)})~N$

很显然，计算开销并不会随着kernel的融合与否而变化，同时，kernel的切换也存在一定的开销，此外内存体系的影响也不可忽略，在这里我们默认了从global memory读取，所以决定做一个N不断scale的speedup的实验图:
### Configuration
GPU: 1x RTX4090, L2 Cache Size = 72MB
$N$: matrix size, $N\times N$
![图片alt](speedup.png "图片title")

$N < 1600$: 

Q: 为什么Fused的优化低于1.5？
A: cuda-extension存在launch开销

Q: 为什么Vectorization没有优化效果？
A: 内存带宽没有打满

$ 1600 < N < 2200$:

Q: 为什么性能有急剧提升？
A: L2 Cache cover了所有的数据，$2200 \times 2200 \times 4 \times 4B \approx 72MB $

$ 2200 < N < 3200$:
Q: 性能坠落 & 重新爬起？
A: TODO

$N > 3200$:
数据量达到了我们的分析假设的情况，所有Memory Access同时访问Global Memory，同时计算 & 启动开销忽略不计，是一个promising的结果
Overall: 
Q: 为什么在整个过程中，vectorization的效果不明显？
A: Gemini: 因为GPU硬件会进行访存融合 $(memory~coalescing)$