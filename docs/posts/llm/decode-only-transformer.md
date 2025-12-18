---
categories:
  - mlsys
  - NOTE
date:
  created: 2025-11-15
---



# Some Notes about Transformer in LLM

最近想入门一下大模型的serving和inference。可能Transformer的架构是一个好的切入点。
<!-- more -->
第一步，transformer需要将输入的prompt变成input embeddings

## 1. Tokenizer

word embedding uses various methods (tiktoken in Llama), while the position emdedding is different.

in original Transformer, 

$$
PE{(t, 2i)} = \sin\left(\frac{t}{10000^{2i/d{\text{model}}}}\right)
$$

$$
PE{(t, 2i+1)} = \cos\left(\frac{t}{10000^{2i/d{\text{model}}}}\right)
$$

, where $i$ is the dimension position in a token, $t$ is the token's absolute position in the query.

transformer takes the sum of $\mathbf{p}_t$ (Word Embedding) and  $\mathbf{e}_t$ (Word Embedding) as the input $\mathbf{z}_t$: 

$$
\mathbf{z}_t = \mathbf{e}_t + \mathbf{p}_t
$$

### Optimizations
#### RoPE
**Llama** uses another kind of Position Embedding, **RoPE** to describe the postion, and it works on the $Q \& K$ caculation by **rotation** not the input generation by addition.

For a token at the absolute position $m$, the rotation matrix $R_m$, it will rotate every pairs $(q_1, q_2)$ in $Q_m$. the number of the pairs is $d_{model}/2$, 

$$
Q'_m = R_mQ_m
$$


where every pair $(q_1, q_2)$ is rotated by $\theta_i$, 

$$
\begin{pmatrix} q'_1 \\ q'_2 \end{pmatrix} = 
\begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix} \begin{pmatrix} q_1 \\ q_2 \end{pmatrix}
$$


Where the $\theta$ is calculated by $i$ and $d_{model}$, $\theta_i = \frac{1}{b^{2i / d}}, b = 10000$


Notice that there is no additional computation compared with original implementation (easy to prove).

Now look into the implementation in Llama: 

```python
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

```

## 2. Q, K, V Calculation

However we have got the input $X$ (shown as vector $z$ in Sec 1).
The shape of $X$ can be defined by 2 parameters:

- $L$: the length of the sequence that $X$ represents.
- $D$: the dimension of each token in the sequence.

We define $X$ as: 
$$
X \in \mathbb{R}^{L \times d_{\text{model}}}
$$

Now we compute the Queries, Keys and Values:
In the self-attention mechanism, we compute the Queries, Keys and Values using 3 weight matrixes:

$$
W^Q \in \mathbb{R}^{L \times d_{\text{k}}},\\
W^K \in \mathbb{R}^{L \times d_{\text{k}}},\\
W^V \in \mathbb{R}^{L \times d_{\text{v}}}
$$
X will be projected into 3 subspaces to ge Q, K and V:

$$
Q = XW^Q,\\
K = XW^K,\\
V = XW^V
$$

note: Q, K, V always have the same dimension $d$.

### Quantization
#### Basic Principle
The usage of these "$W$" occupies much memory bandwidth due to their data type being FP16 or FP32. So one optimization is to quantize them into integer(INT8 or INT4).

The quantization is done by:
$$
W_{quant} = \text{clip}\left(\text{round}\left(\frac{W_{fp}}{S} + Z\right), min, max\right)
$$
$W_{fp}$: The original high-precision weight (e.g., 16-bit float).

$S$ (Scale): A floating-point factor that determines the step size of the quantization buckets.

$Z$ (Zero Point): An integer that ensures the real value zero is exactly representable (crucial for sparsity).

$W_{quant}$: The resulting integer index (e.g., a 4-bit integer ranging from 0 to 15, or -8 to 7).

During inference, we will dequantize the quantized weight $W_{quant}$ back to the original floating-point weight $W_{fp}$ using the following formula:

$$
W'_{fp} = S \times (W_{quant} - Z)
$$
#### Floating Point Family
Used primarily for Training and High-Precision Inference.
| Data Type | Bit Width | Structure | Primary Use Case | Key Characteristics |
| :---: | :---: | :---: | :---: | :---: |
| **FP32** (Single Precision) | 32-bit | 1 Sign, 8 Exp, 23 Mantissa | **Master Weights** | The "Gold Standard" for accuracy. Used to store master weights during training to prevent underflow. |
| **FP16** (Half Precision) | 16-bit | 1 Sign, 5 Exp, 10 Mantissa | **Legacy Inference** | Standard on older GPUs (Volta/Turing). **Risk:** Prone to overflow (values > 65,504 turn to NaN) during training. |
| **BF16** (Brain Float 16) | 16-bit | 1 Sign, 8 Exp, 7 Mantissa | **Standard Training** | Keeps the **same dynamic range (Exponent) as FP32** but truncates precision. Much more stable than FP16 for training. |
| **FP8** (E4M3) | 8-bit | 1 Sign, 4 Exp, 3 Mantissa | **Inference/Weights** | Higher precision relative to range. Optimized for storing weights and activations in H100+ GPUs. |
| **FP8** (E5M2) | 8-bit | 1 Sign, 5 Exp, 2 Mantissa | **Training Gradients** | Higher dynamic range (equivalent to FP16). Optimized for gradients which vary wildly in magnitude. |

#### Integer Family
Used primarily for Quantized Inference to save memory and bandwidth.

| Data Type | Bit Width | Range | Primary Use Case | Key Characteristics |
| :---: | :---: | :---: | :---: | :---: |
| **INT8** | 8-bit | -128 to 127 | **Production Inference** | The industry standard for "lossless-feeling" quantization. **Supported natively by Tensor Cores.** |
| **INT4** | 4-bit | -8 to 7 | **Weight Storage** | The current "sweet spot" for LLMs. Weights are stored in INT4 and dequantized on-the-fly for computation. |
| **INT1** | 1-bit | 0 or 1 | **Experimental** | Used in "1-bit LLMs" (e.g., BitNet). Weights are effectively ternary $\{-1, 0, 1\}$. Extremely efficient but requires specialized training. |

The accuracy loss seems apparently great. But the memory savings is huge ( $2-4\times$)


## 3. Scaled Dot-Product Attention
### 3.1.0. Reshape for Multi-head Attention
The shape of Q and K is $L \times d_k$.
Now we reshape the Q and k to $L \times h \times d_h$, where $ d_k = d_h \times h$.

Note that $d_h$ is the hidden dimension within every token in the $L$ sequence.

We want the computation of $QK^T$ to be more parallelizable and efficient. 

So we reshape the Q and K to $h \times L \times d_h$ and $h \times L \times d_h$, respectively.
No the shape of Q and K is `[h, L, d_h]` (no considering Batch Size).

Now we can ensure the $d_h$ at fixed size: $128 /256$. and parallelize the computation of $QK^T$ between the $h$ heads.

### 3.1.1. Attention Calculation
The score is calculated as follows:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_h}} + M\right)V$$
In every head, the shape of $QK^T$ is `[L, L]`, and the shape of $V$ is `[L, d_h]`. 
Notably the $d_v$ is always equal to $d_k$. So now we use $d$ to unify $d_v$ and $d_k$.

**For MHA:**

The V will be divided to `[h, L, d_h]` align with the Q and K. And the Attention Weights will be multiplied with a corresponding V in the same head.

**For GQA:**

To optimize the memory consumption of KV cache. GQA choose to store less heads of V and distribute them to different groups of attention weights

For example, if we have 8 heads, and we want to distribute them to 2 groups. Then we will have 4 heads in each group. Instead of storing 8 heads of V, we will store 2 heads of V for 2 groups and 4 heads of attention weights in each group.

### 3.1.2. Head Concatenation

Now we have $H$ heads, each head $i$ has an attention matrix $Z_i$ of shape: `[L, d_h]`

We concatenate all heads along the last dimension $d_h$ into a single matrix $Z$ of shape: `[L, d]`, where $d = d_h \times H$:

$$
Z = \text{Concat}(Z_1, Z_2, \dots, Z_h)
$$
Now then apply a output linear projection to $Z$:

$$
\text{Attention} = Z_{concat} \times W^O
$$


## 4. Add & Normalization

### 4.1. Residual Connection
Execute Residual Connection **(Add)** and Layer Normalization: 

$$X = X_{input} + \text{Attention}(X_{input})$$

### 4.2. Normalization
Here are 2 ways to normalize:

#### 4.2.1 Layer Normalization

For $x \in X$, $x$ of shape`[d, 1]` is normalized by:

$$
\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$
where $\mu$ is the mean of $x$ and $\sigma^2$ is the variance of $x$.

#### 4.2.2 Root Mean Square (RMS) Normalization
$$\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}$$

!!! note 

    Originally, Normalization was done after Attention Calculation like $Attention \rightarrow Add \rightarrow Norm$. However, Nowadays, Normalization(RMS Norm) is done before Attention like $Norm \rightarrow Attention \rightarrow Add$.
    The same thing will happen in FFN blocks.

## 5. FFN / MLP
The defining characteristic of this block is that it is applied independently to every single token position.

 - The same matrix weights ($W_{up/down}, b$) are used for every token.
 - Token $i$ does not interact with Token $j$ inside this block.

Mathematically, if the input is a matrix $X \in \mathbb{R}^{L \times d}$, the FFN operates on each row $x_i$ identically.


The fundamental structure involves projecting the input vector into a higher-dimensional ($d_{ff}$) space (Expansion), applying a non-linearity, and projecting it back (Contraction).
### 5.1. Standard FFN
$$
\text{FFN}(x) = \text{Activation}(x W_{up} + b_1) W_{down} + b_2
$$
where the $d_{ff}$ is usually the $4 \times d$
### 5.2. Gated FFN

$$
\text{FFN}_{\text{SwiGLU}}(x) = (\text{SiLU}(x W_{gate}) \odot (x W_{up})) W_{down}
$$
where the $d_{ff}$ is usually the $2.6 \times d$, because there are **1 more projection weight matrixes**, engineers want to keep the number of parameters  align with the standard FFN.
### 5.3. Activation
#### ReLU
Rectified Linear Unit:
$$
f(x) = max(0,x)
$$
#### GELU
Gaussian Error Linear Unit:
$$
f(x) = x \cdot \Phi(x)
$$
Where $\Phi(x)$ is the Cumulative Distribution Function (CDF) of the Standard Normal Distribution:
$$
\Phi(x) = \frac{1}{2} \left[ 1 + \text{erf}\left( \frac{x}{\sqrt{2}} \right) \right]
$$
Approximation (Used in practice for speed):
$$
f(x) \approx 0.5x \left( 1 + \tanh \left[ \sqrt{\frac{2}{\pi}} \left( x + 0.044715 x^3 \right) \right] \right)
$$

#### SiLU
Swish-Gated Linear Unit:
$$
f(x) = x \cdot \sigma(x)
$$
Where $\sigma(x)$ is the Sigmoid function:
$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$
#### Why SiLU in Gated FFN?
 It is the **"Goldilocks"** function—smooth, non-linear, and computationally efficient enough to be multiplied across billions of parameters.


After FFN, we apply another Add and then enter **the next iteration** of Attention Block:

$$
x_{output} = x_{mid} + \text{FFN}( \text{RMSNorm}(x_{mid}) )
$$

!!! note

    the normalization is applied **before** the FFN in modern transformers, while it is applied after in the original transformer paper, which is aligned with the Attention Block.

## Summary
For decode-only transformer, here are 3 key blocks:

  1. token embedding
  2. attention block
  3. feed forward network block

To be more specific, it consists of many kernels with variety of weight matrixes. 
They are full of `gemm`  and `norm & add` operations, where much optimization can be done.
