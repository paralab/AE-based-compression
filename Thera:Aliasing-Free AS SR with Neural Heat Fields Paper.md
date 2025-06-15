# Thera: Aliasing-Free Arbitrary-Scale Super-Resolution with Neural Heat Fields

Alexander Becker1,*     Rodrigo Caye Daudt1,∗     Dominik Narnhofer¹     Torben Peters¹

Nando Metzger¹     Jan Dirk Wegner²     Konrad Schindler¹

1ETH Zurich     2University of Zurich

| Learned 2D frequency bank | Neural heat field | Anti-aliased signal for various pixel grids |
| ------------------------- | ----------------- | ------------------------------------------- |
|                           | Φθ(x, y, t)       |                                             |
| Input image               | Phase shifts      | t                                           |
|                           | +++ +++++         | t = 0.01                                    |
|                           | +++ +++++         | t = 0.5                                     |
|                           | +++ +++++         |                                             |
| shift                     | ξ(·, t)           |                                             |
|                           | +++ +++++         |                                             |
|                           | +++ +++++         |                                             |
| Backbone                  | scale             | thermal                                     |
| Hypernetwork              | activation        |                                             |
|                           | + + + + ++        | t = f(x)                                    |
|                           | t = 0.9           |                                             |
|                           | ++++              |                                             |
|                           | + + + + ++        |                                             |
|                           | ++++              |                                             |
| Amplitudes                |                   |                                             |
|                           | ++++              |                                             |
|                           | pixel grid        |                                             |

Figure 1. We present Thera, the first method for arbitrary-scale super-resolution with a built-in physical observation model. Given an input image, a hypernetwork predicts the parameters of a specially designed neural heat field, inherently decomposing the image into sinusoidal components. The field’s architecture automatically attenuates frequencies as a function of the scaling factor so as to match the output resolution at which the signal is re-sampled.

# Abstract

Recent approaches to arbitrary-scale single image super-resolution (ASR) use neural fields to represent continuous signals that can be sampled at arbitrary resolutions. However, point-wise queries of neural fields do not naturally match the point spread function (PSF) of pixels, which may cause aliasing in the super-resolved image. Existing methods attempt to mitigate this by approximating an integral version of the field at each scaling factor, compromising both fidelity and generalization. In this work, we introduce neural heat fields, a novel neural field formulation that inherently models a physically exact PSF. Our formulation enables analytically correct anti-aliasing at any desired output resolution, and – unlike supersampling – at no additional cost. Building on this foundation, we propose Thera, an end-to-end ASR method that substantially outperforms existing approaches, while being more parameter-efficient and offering strong theoretical guarantees. The project page is at https://therasr.github.io.

# 1. Introduction

Over the years, learning-based image super-resolution (SR) methods have achieved increasingly better results. However, unlike interpolation techniques that can resample images at any resolution, these methods typically require retraining for each scaling factor. Recently, arbitrary-scale SR (ASR) approaches have emerged, which allow users to specify any desired scaling factor without retraining, significantly increasing flexibility. Notably, with LIIF, Chen et al. pioneered the use of neural fields for single-image SR, exploiting their continuous representation to enable SR at arbitrary scaling factors. LIIF has since inspired several follow-ups which build upon the idea of using per-pixel neural fields. This is not surprising: Neural fields are in many ways a natural match for variable-resolution computer vision and graphics. By implicitly parameterizing a target signal as a neural network that maps coordinates to signal value, they offer a compact representation, defined over a continuous input domain, and are analytically differentiable.

While neural fields naturally model continuous functions, they do not easily allow for observations of such functions other than point-wise evaluations. For many tasks, however, integral observation models such as point spread...
---
# 32.1 Thera Pro

# 32

| 31.9      |        | MSIT       |        |        |
| --------- | ------ | ---------- | ------ | ------ |
| 31.8      | (Ours) |            |        |        |
| Thera Air |        | Thera Plus |        |        |
| PSNR (dB) |        |            |        |        |
| 31.7      | CUF    | SRNO       | CiaoSR | CLIT   |
| 31.6      |        |            |        | LIIF   |
| 31.5      |        |            |        | MetaSR |

10−2 10−1 100 101

Figure 2. Comparison of recent ASR methods, averaged over ×{2,3,4} scales. We generally achieve higher performance at lower parameter counts. Our best model, Thera Pro, achieves highest overall performance by a large margin.

Functions (PSFs) are desirable. This is particularly true for neural fields-based ASR methods, which by nature do not commit to a fixed upscaling factor a priori but regress continuous representations with unbounded spectra that can be observed at various sampling rates. If the Nyquist frequency corresponding to the desired sampling rate is lower than the highest frequency represented by the field, the sampling operation is prone to aliasing. This explains the initially counterintuitive relevance of anti-aliasing for super-resolution: When using neural fields, signals are first upsampled to infinite (continuous) resolution and then resampled at the desired resolution, and this latter operation must be done carefully. Incorporating a physically plausible observation model is not trivial [2–4, 17, 27, 51], but has the potential to avoid aliasing. For this reason, Chen et al. [10] and successor works [7, 9, 24, 56] have already taken a first step towards learning multi-scale representations, via cell encoding. Fundamentally, these “learning-based anti-aliasing” approaches try to learn an integrated version of the field for any scaling factor, wasting field capacity for approximating a relation that can be described exactly through Fourier theory.

In this work, we combine recent advances in implicit neural representations with ideas from classical signal theory to introduce neural heat fields, a novel type of neural field that guarantees anti-aliasing by construction. The key insight is that sinusoidal activation functions [41] enable selective attenuation of individual components depending on their spatial frequency, following Fourier theory. This allows for the exact computation of Gaussian-blurred versions of the field for any desired (isotropic) blur radius. When rasterizing an image, the field can therefore be queried with a Gaussian PSF that matches the target resolution, effectively preventing aliasing. Notably, filtering with neural heat fields incurs no computational overhead: The querying cost is the same for any width of the anti-aliasing filter kernel, including infinite and zero widths.

Building on this, we then propose Thera, an end-to-end ASR method that combines a hypernetwork [16] with a grid of local neural heat fields, offering theoretical guarantees with respect to multi-scale representation (see Fig. 1). Empirically, Thera outperforms all competing ASR methods, often by a substantial margin, and is more parameter-efficient (see Fig. 2). To the best of our knowledge, Thera is also the first neural field method to allow bandwidth control at test time.

# In summary, our main contributions are:

1. We introduce neural heat fields, which represent a signal with a built-in, principled Gaussian observation model, and therefore allow anti-aliasing with minimal overhead.
2. We use neural heat fields to build Thera, a novel method for ASR that offers theoretically guaranteed multi-scale capabilities, delivers state-of-the-art performance and is more parameter efficient than prior art.

# 2. Related Work

# 2.1. Neural Fields

A neural field, also called an implicit neural representation, is a neural network trained to map coordinates onto values of some physical quantity. Recently, neural fields have been used for parameterizing various types of visual data, including images [10, 12, 22, 24, 41, 42, 47], 3D scenes (e.g., represented as signed distance fields [36, 40, 41, 46, 47], occupancy fields [33, 37], LiDAR fields [20], view-dependent radiance fields [2–4, 34, 47]), or digital humans [8, 15, 49, 52, 55]. Frequently, it is desirable to impose some prior over the space of learnable implicit representations. A common approach for such conditioning is encoder-based inference [48], where a parametric encoder maps input observations to a set of latent codes z, which are often local [7, 9, 10, 24, 44]. The encoded latent variables z are then used to condition the neural field, for instance by concatenating z to the coordinate inputs or through a more expressive hypernetwork [16], mapping latent codes z to neural field parameters θ. An early example of this approach, which is gaining popularity [48], was proposed in [41].

# 2.2. Arbitrary-Scale Super-Resolution

ASR is the sub-field of single-image SR in which the desired SR scaling factor can be chosen at inference time to be (theoretically) any positive number, allowing maximum flexibility, such as that of interpolation methods. The first work along this line is MetaSR [18], which infers the parameters of a convolutional upsampling layer using a hypernetwork [16] conditioned on the desired scaling factor. An influential successor work is LIIF [10], in which the high-resolution image is implicitly described by local neural fields. These fields are conditioned via concatenation,
---
with features extracted from the low-resolution input image. The continuous nature of the neural fields allows for sampling target pixels at arbitrary locations and thus also arbitrary resolution.

Most subsequent work has since been built upon the LIIF framework. For example, UltraSR [50] improves the modeling of high-frequency textures with periodic positional encodings of the coordinate space, as is common practice for e.g., neural radiance fields [2–4, 34]. LTE [24] makes learning higher frequencies more explicit by effectively implementing a learnable coordinate transformation into 2D Fourier space, prior to a forward pass through an MLP. Vasconcelos et al. [44] use neural fields in CUF to parameterize continuous upsampling filters, which enables arbitrary-scale upsampling. More recently, methods like CiaoSR [7], CLIT [9], and most recently MSIT [56] have integrated (multi-scale) attention mechanisms, improving reconstruction quality. In a parallel line of research, Wei et al. [45] propose SRNO, an attention-based neural operator that learns a continuous mapping between low- and high-resolution function spaces.

# 2.3. Anti-Aliasing in Neural Fields

Early in the recent development of implicit neural representations, concerns regarding aliasing were raised. Barron et al. [2] proposed integrating a positional encoding with Gaussian weights, which reduced aliasing in NeRF [34]. Improvements were later proposed for unbounded scenes [3] and to improve efficiency [17]. In [4], Barron et al. tackle anti-aliasing within the Instant-NGP [35] approach. Recent work has succeeded in limiting the bandwidth using multiplicative filter networks [27], polynomial neural fields [51] or cascaded training [39], although these works are restricted to discrete, pre-defined band limits (and thus resolutions) and have not tackled super-resolution tasks. These methods are not a good fit for ASR because they do not allow for continuous anti-aliasing, nor bandwidth control at test time. To perform scale-dependent filtering, most fields-based ASR methods instead explicitly provide the scale as input to the field, attempting to learn an appropriate observation model from data. While this approach may work reasonably well in in-distribution settings, it seeks to learn a model from data that can be described exactly with a differential equation, ultimately sacrificing fidelity and generalization.

In contrast, in this paper we explore a way to directly integrate a physics-informed observation model into the neural field representation.

# 3. Method

In this section we introduce Thera, a novel neural fields-based ASR method that guarantees analytical anti-aliasing at any desired output resolution at no additional cost. First, we present neural heat fields, a special type of neural field that inherently achieves anti-aliasing by implicitly attenuating high-frequency components as a function of a time coordinate. Next, we propose a mechanism for learning a prior over a grid of neural heat fields, enabling them to represent a multi-scale output image conditioned on a lower-resolution input image. Finally, we show that our formulation allows us to impose a regularizer on the underlying, continuous signal itself – something that, to the best of our knowledge is not possible in previous methods.

# 3.1. Neural Heat Fields for Analytical Anti-Aliasing

Let x ∈ R2 denote the spatial coordinates of a continuous image function f(x). Aliasing occurs when this continuous signal is sampled at a rate that does not adequately capture its highest frequency components, resulting in overlapping spectral replicas in the Fourier domain. One must therefore apply a low-pass filter g(x) whose cut-off frequency is aligned with the Nyquist frequency of the sampling rate, then sample the band-limited signal f ⊛ g(x). The key of our method is that, if a signal is decomposed into sinusoidal components, such filtering can be done simply by re-scaling each component by a factor that depends on their frequency and a time coordinate t. This behavior is naturally accomplished by parameterizing the field Φ as a two-layer perceptron,

Φ(x, t) = W2 · ξ (W1x + b1, ν(W1), κ, t) + b2,

with parameters θ := {W1, W2, b1, b2}. Intuitively, W1 serves as a frequency bank, with its components acting as the basis functions that compose the signal Φ(x, 0), and phase shifts encoded by b1. The matrix W2, with one row per output channel, contains initial magnitudes of these components, and b2 is the global bias of Φ per channel. Finally, we introduce the thermal activation function ξ(·), which models the aforementioned decay of sinusoidal components (implied by W1) over time:

ξ(z, ν, κ, t) = sin(z) · exp(− |ν|2 κt).

Here, |ν| = |ν(W1)| denotes the row-wise Euclidean norm of W1, representing the magnitudes of the implied wave numbers (frequencies). Interestingly, Eq. (1) constitutes the solution of the isotropic heat equation ∂Φ/∂t = κ · ∇2 Φ, as derived in Appendix A. We therefore refer to this MLP as a neural heat field.

There is an ideal bijection between the desired sampling rate fs and t. At t = 0 no filtering takes place, implying a continuous signal (fs → ∞). A low-pass filtered version of the signal is observed for t > 0. To obtain a desired level of anti-aliasing, we only need to compute the corresponding value of t. The relationship between the cut-off frequency of the filter and t is controlled by the diffusivity.
---
# Figure 3. Overview of Thera.

A hypernetwork estimates parameters {b₁,W₂}(i,j) of pixel-wise, local neural heat fields. The phase shifts b₁ operate on globally learned components, before thermal activations scale each component depending on their frequency and the desired scaling factor. The components are then linearly combined using coefficients W₂, resulting in an appropriately-blurred, continuous local neural field. This field is then rasterized at the appropriate sampling rate (resolution) to yield a part of the final output image (red square).

constant κ, which we can freely set to any positive number. During training, the local fields are supervised with values of high-resolution target pixels at the appropriate spatial coordinates x and time index t (ensuring that the signal is correctly blurred for the target resolution), and the entire architecture is optimized end-to-end. In practice, we directly optimize a single global frequency bank W1, rather than having the hypernetwork predict a separate W1 for each low-resolution pixel. Not only does this better fit the idea to represent the signal with a single, consistent basis, it also reduces the total parameter count.

To subsample the signal D by a factor S, the field Φ should be sampled at t = S².

The described scheme, which we call Thera, is depicted in Fig. 3. It allows for arbitrary-scale super-resolution, combining the multi-scale signal representation within neural heat fields with the expressivity of proven feature extraction backbones for SR and image restoration. As the entire network is trained end-to-end, the feature extractor can learn super-resolution priors for a whole range of resolutions covered by the training data. E.g., a network trained with scaling factors up to ×4 will encode priors that enable us to observe the field at t > 1.

By training on multiple resolutions, we can also make κ a trainable parameter that allows the network to adapt to different downsampling operators. Finally, we set the bias terms for the three color channels of every local field Φ (i.e., b2) to the RGB values of the associated low-resolution pixel. Thus, the hypernetwork only predicts field-wise phase shifts b1 and amplitudes W2.

# 3.3. Total Variation at t = 0

To allow Thera to better generalize to higher, out-of-domain scaling factors, we can place an unsupervised regularizer at t = 0. Note that this is a prior on the continuous signal itself – something that, to the best of our knowledge, sets Thera apart from all previous methods. In our implementation it takes the form of a total variation (TV) loss term, well.
---
known to promote piece-wise constant signals that describe natural images well [11]. We use an ℓ1 variant of TV,

LTV(Φ(x,0)) = E[|∇Φ(x,0)|]. (5)

Given our continuous signal representation, ∇Φ(x,0) can be computed analytically by automatic differentiation, rather than falling back to a neighborhood approximation as in most previous work [38]. We further motivate this approach in Fig. 5, which demonstrates that our method faithfully recovers the gradients of super-resolved images.

# 3.4. Implementation and Training

Thera is implemented in JAX [6]. Similar to prior work [7, 10, 24, 56], we randomly sample a scaling factor r ∼ U(1.2,4) for each image during training, then randomly crop an area of size (48r)2 pixels as the target patch, from which the source is generated by bicubic downsampling to size 482. As corresponding targets, 482 random pixels are sampled from the target patch. We train with standard augmentations (random flipping, rotation, and resizing), using the Adam optimizer [23] with a batch size of 16 for 5 × 106 iterations, with initial learning rate 10−4, β1 = 0.9, β2 = 0.999 and ϵ = 10−8. The learning rate is decayed to zero according to a cosine annealing schedule [30]. We use MAE as reconstruction loss, to which the TV loss from Eq. 5 is added with a weight of 10−4. Like previous work [26, 43, 44], we employ geometric self-ensembling (GSE) instead of the local self-ensembling introduced in LIIF [10]. In GSE, the results for four rotated versions of the input are averaged at test time. Including reflections did not improve performance.

# 4. Results

Throughout this section we evaluate three variants of our method, which differ solely in the size of the hypernetwork and the number of field parameters:

- Thera Air: A tiny version with the number of globally shared components in W1 set to 32, and the hypernetwork being a single 1 × 1 convolution that maps features to field parameters. This version adds only 8,256 parameters on top of the backbone.
- Thera Plus: A balanced version that employs an efficient ConvNeXt-based [29] hypernetwork. Its parameter count of ≈1.41 M matches that of recent medium-sized competitors like [7].
- Thera Pro: The strongest version uses a high-capacity, transformer-based [25, 28] hypernetwork. Its added parameter count is ≈4.63 M, still less than the most recent competitor [56] and much smaller than [9].

# Datasets and metrics.

Following previous work, our models are trained with the DIV2K [1] training set, consisting of 800 high-resolution RGB images of diverse scenes. We report evaluation metrics on the official DIV2K validation split as well as on standard benchmark datasets: Set5 [5], Set14 [53], BSDS100 [31], Urban100 [19], and Manga109 [32]. Following prior work, we use peak signal-to-noise ratio (PSNR, in decibels) as the main evaluation metric and compute it in RGB space for DIV2K and on the luminance (Y) channel of the YCbCr representation for benchmark datasets. Additional quantitative results are given in Appendix C. Not all numbers could be computed for competing methods for which code or checkpoints were not publicly shared (see Appendix F).

# Backbones.

Like previous work, we combine our method with two standard backbones for super-resolution and image restoration: (i) EDSR-baseline [26] (1.22 M parameters) and (ii) RDN [54] (22.0 M parameters).

# 4.1. Super-Resolution Performance

# Qualitative results.

Upon visual inspection – see Fig. 4 for examples – we observe that Thera produces results that are both perceptually convincing and more correct, particularly in the presence of repeating structures. Neural heat fields enable Thera to reproduce a high level of detail without suffering from aliasing, no matter the sampling scale (see also Fig. 9 in Appendix B).

# Quantitative results.

We first evaluate the three variants of our method on the held-out DIV2K validation set, following the setup described above. Table 1 shows PSNR values for all tested methods, for both in-distribution (×2 to ×4) and out-of-distribution (×6 to ×30) scaling factors. Thera Pro outperforms all competing methods at all scaling factors, often by a substantial margin (e.g., 29.51 vs. 29.22 on EDSR ×4), even though its parameter overhead on top of the backbone is lower compared to the second-best method MSIT [56], and less than a third of CLIT [9]. Interestingly, even our minimal variant Thera Air – with only about 8000 parameters on top of the backbone – performs on par with or better than methods of much higher parameter count. This supports our claim that hard-wiring a theoretically principled sampling model, which rules out signal aliasing, enables better generalization and higher-fidelity reconstruction. For comparison with conventional interpolation, we also report numbers obtained with Lanczos (sinc) resampling on top of the respective ×4 backbone. This baseline is consistently outperformed by dedicated ASR methods, indicating that the latter do learn scale-specific priors. Like earlier work, we further report the performance of Thera on five popular benchmark datasets with an RDN backbone in Tab. 2. Our method again outperforms all competing methods in all settings, often substantially (e.g., 29.58 vs. 29.14 on Urban100 ×3). We hypothesize that Thera’s hard-wired PSF is also beneficial when generalizing to unseen datasets. Once again we observe that the perfor-
---
# Figure 4

Qualitative examples for a representative ×6 scale factor, with an RDN [54] backbone for all methods. Best viewed zoomed in.

| Low-res input | LIIF \[10] | SRNO \[45] | MSIT \[56] | Thera Pro (ours) | GT    |
| ------------- | ---------- | ---------- | ---------- | ---------------- | ----- |
| DIV2K         | DIV2K      | Set14      | Urban100   | OMICS            | OMIcS |
| OMIS          | OmicS      | OMICS      | Manga109   | Manga109         |       |

mance of Thera Air is often comparable to that of methods with orders of magnitude higher parameter counts. In fact, due to the use of thermal activations – and unlike all prior work based on multi-layer ReLU-activated fields – Thera is infinitely differentiable.

# 4.2. Ablation Studies

Fidelity of the Signal and its Derivatives. Neural fields with periodic activation functions have been shown to be superior when it comes to fitting high-resolution, natural signals, and to correctly recovering their derivatives [41]. We observe similar effects for Thera, whose thermal activations at t = 0 can be seen as a special case of periodic activations.

In Table 3, we ablate individual components and design choices of our method to understand their contributions to overall performance. The comparisons use Thera Plus with EDSR backbone, and are representative of all variants.
---
Backbone

|                               |                      |        |                 |                     |       |       |       |       |       |       |
| ----------------------------- | -------------------- | ------ | --------------- | ------------------- | ----- | ----- | ----- | ----- | ----- | ----- |
| Method                        | Num. of add. params. |        | In-distribution | Out-of-distribution |       |       |       |       |       |       |
|                               |                      | ×2     |                 | ×3                  | ×4    | ×6    | ×12   | ×18   | ×24   | ×30   |
| EDSR- baseline \[26] (1.22 M) | Bicubic              | —      | 31.01           | 28.22               | 26.66 | 24.82 | 22.27 | 21.00 | 20.19 |       |
|                               | + Sinc interpol.     | —      | 34.52           | 30.89               | 28.98 | 26.75 | 23.76 | 22.25 | 21.29 | 20.62 |
|                               | MetaSR \[18]         | 0.45 M | 34.64           | 30.93               | 28.92 | 26.61 | 23.55 | 22.03 | 21.06 | 20.37 |
|                               | LIIF \[10]           | 0.35 M | 34.67           | 30.96               | 29.00 | 26.75 | 23.71 | 22.17 | 21.18 | 20.48 |
|                               | LTE \[24]            | 0.49 M | 34.72           | 31.02               | 29.04 | 26.81 | 23.78 | 22.23 | 21.24 | 20.53 |
|                               | CUF \[44]            | 0.30 M | 34.79           | 31.07               | 29.09 | 26.82 | 23.78 | 22.24 | —     | —     |
|                               | CiaoSR \[7]          | 1.43 M | 34.88           | 31.12               | 29.19 | 26.92 | 23.85 | 22.30 | 21.29 | 20.44 |
|                               | CLIT \[9]            | 15.7 M | 34.81           | 31.12               | 29.15 | 26.92 | 23.83 | 22.29 | 21.26 | 20.53 |
|                               | SRNO \[45]           | 0.80 M | 34.85           | 31.11               | 29.16 | 26.90 | 23.84 | 22.29 | 21.27 | 20.56 |
|                               | MSIT \[56]           | 4.83 M | 34.95           | 31.23               | 29.22 | 26.94 | 23.83 | 22.27 | 21.26 | 20.54 |
|                               | Thera Air (ours)     | .008 M | 34.75           | 31.09               | 29.10 | 26.84 | 23.80 | 22.26 | 21.26 | 20.56 |
|                               | Thera Plus (ours)    | 1.41 M | 34.89           | 31.22               | 29.24 | 26.96 | 23.89 | 22.34 | 21.32 | 20.61 |
|                               | Thera Pro (ours)     | 4.63 M | 35.19           | 31.50               | 29.51 | 27.19 | 24.09 | 22.51 | 21.48 | 20.73 |
|                               | + Sinc interpol.     | —      | 34.59           | 31.03               | 29.12 | 26.89 | 23.87 | 22.34 | 21.36 | 20.68 |
|                               | MetaSR \[18]         | 0.45 M | 35.00           | 31.27               | 29.25 | 26.88 | 23.73 | 22.18 | 21.17 | 20.47 |
|                               | LIIF \[10]           | 0.35 M | 34.99           | 31.26               | 29.27 | 26.99 | 23.89 | 22.34 | 21.31 | 20.59 |
|                               | LTE \[24]            | 0.49 M | 35.04           | 31.32               | 29.33 | 27.04 | 23.95 | 22.40 | 21.36 | 20.64 |
|                               | CUF \[44]            | 0.30 M | 35.11           | 31.39               | 29.39 | 27.09 | 23.99 | 22.42 | —     | —     |
| RDN \[54] (22.0 M)            | CiaoSR \[7]          | 1.43 M | 35.13           | 31.39               | 29.43 | 27.13 | 24.03 | 22.45 | 21.41 | 20.55 |
|                               | CLIT \[9]            | 15.7 M | 35.10           | 31.39               | 29.39 | 27.12 | 24.01 | 22.45 | 31.38 | 20.64 |
|                               | SRNO \[45]           | 0.80 M | 35.16           | 31.42               | 29.42 | 27.12 | 24.03 | 22.46 | 21.41 | 20.68 |
|                               | MSIT \[56]           | 4.83 M | 35.16           | 31.42               | 29.42 | 27.11 | 23.99 | 22.42 | 21.37 | 20.65 |
|                               | Thera Air            | .008 M | 35.06           | 31.42               | 29.43 | 27.13 | 24.04 | 22.48 | 21.44 | 20.71 |
|                               | Thera Plus           | 1.41 M | 35.00           | 31.40               | 29.44 | 27.16 | 24.06 | 22.49 | 21.45 | 20.71 |
|                               | Thera Pro            | 4.63 M | 35.25           | 31.56               | 29.57 | 27.25 | 24.14 | 22.56 | 21.52 | 20.77 |

Table 1. Quantitative comparison of peak signal-to-noise ratio (PSNR, in dB) obtained by various methods on the held-out DIV2K validation set. The highest PSNR value per backbone and scaling factor is bold and the second highest is underlined.

Method

|            |       |       |       |          |          |       |       |       |       |       |       |       |       |       |       |
| ---------- | ----- | ----- | ----- | -------- | -------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| ×2         | Set5  | Set14 | B100  | Urban100 | Manga109 |       |       |       |       |       |       |       |       |       |       |
|            |       |       |       |          | ×3       | ×4    | ×2    | ×3    | ×4    | ×2    | ×3    | ×4    | ×2    | ×3    | ×4    |
| MetaSR     | 38.22 | 34.63 | 32.38 | 33.98    | 30.54    | 28.78 | 32.33 | 29.26 | 27.71 | 32.92 | 28.82 | 26.55 | —     | —     | —     |
| LIIF       | 38.17 | 34.68 | 32.50 | 33.97    | 30.53    | 28.80 | 32.32 | 29.26 | 27.74 | 32.87 | 28.82 | 26.68 | 39.26 | 34.21 | 31.20 |
| LTE        | 38.23 | 34.72 | 32.61 | 34.09    | 30.58    | 28.88 | 32.36 | 29.30 | 27.77 | 33.04 | 28.97 | 26.81 | 39.28 | 34.32 | 31.30 |
| CUF        | 38.28 | 34.80 | 32.63 | 34.08    | 30.65    | 28.92 | 32.39 | 29.33 | 27.80 | 33.16 | 29.05 | 26.87 | —     | —     | —     |
| CiaoSR     | 38.29 | 34.85 | 32.66 | 34.22    | 30.65    | 28.93 | 32.41 | 29.34 | 27.83 | 33.30 | 29.17 | 27.11 | 39.51 | 34.57 | 31.57 |
| CLIT       | 38.26 | 34.80 | 32.69 | 34.21    | 30.66    | 28.98 | 32.39 | 29.34 | 27.82 | 33.13 | 29.04 | 26.91 | —     | —     | —     |
| SRNO       | 38.32 | 34.84 | 32.69 | 34.27    | 30.71    | 28.97 | 32.43 | 29.37 | 27.83 | 33.33 | 29.14 | 26.98 | 39.52 | 34.67 | 31.61 |
| MSIT       | 38.31 | 34.85 | 32.72 | 34.26    | 30.70    | 28.97 | 32.42 | 29.35 | 27.81 | 33.27 | 29.14 | 26.93 | 39.44 | 34.62 | 31.58 |
| Thera Air  | 38.18 | 34.75 | 32.60 | 34.13    | 30.70    | 28.95 | 32.33 | 29.29 | 27.80 | 33.16 | 29.14 | 26.91 | 39.03 | 34.57 | 31.66 |
| Thera Plus | 38.11 | 34.67 | 32.56 | 34.20    | 30.67    | 28.97 | 32.26 | 29.28 | 27.81 | 33.14 | 29.15 | 26.97 | 38.69 | 34.42 | 31.65 |
| Thera Pro  | 38.36 | 34.88 | 32.79 | 34.43    | 30.85    | 29.08 | 32.46 | 29.39 | 27.87 | 33.63 | 29.58 | 27.26 | 39.62 | 34.98 | 31.98 |

Table 2. Results on common benchmark datasets for in-distribution scale factors with an RDN [54] backbone. The numbers represent PSNR in dB, calculated on the luminance (Y) channel of the YCbCr representation following previous work.

Single scale training. We run three experiments using a prior performance of single-scale training when tested at the single scale (×2, ×3, ×4) to test how this affects scale generalization. κ was fixed at the theoretically derived value for these experiments, as multi-scale training is required to optimize it. As expected, we observe equal or even superior performance compared to the default multi-scale version when generalizing to other scaling factors.
---
ReLU-based Thera (ours) GT grades significantly without it for out-of-distribution scales.

Thermal activations. We replace thermal activations (Eq. 2) with standard ReLU activations. What remains is only the hypernetwork controlling the parameters of the local fields. A consistent loss in performance shows the impact of the proposed thermal activation underlying our multi-scale representation.

Shared components. Predicting W1 along with b1 and W2 leads to negligible gains. This comes at the cost of doubling the amount of field parameters that the hypernetwork predicts. Thus, Thera uses a shared, global frequency bank.

# 4.3. Limitations and Future Work

Neural heat fields as introduced in this paper, and by extension Thera, come with relatively strict architectural requirements that currently only allow for a single hidden layer in the neural field. While this can be beneficial from a computational standpoint, it limits hierarchical feature learning and potentially makes modeling of complex non-linear relations harder than necessary. Nonetheless, as our experiments show, the current neural heat field architecture does easily have enough capacity to model local, subpixel information for the scaling factor range discussed in this paper. We have compensated for the relatively less expressive fields with a higher-capacity hypernetwork, and we speculate that there may be ways to extend the signal-theoretic guarantees of Thera to multi-layer architectures in future work. This could result in even higher parameter efficiency, and potentially better generalization. We also expect that more advanced priors than TV could be even more effective at regularizing Φ. Priors at t = 0, made possible by Thera, have the potential to regularize the continuous signal itself, and therefore improve SR quality for all scaling factors.

| Experiment       | In-distribution | Out-of-distribution |       |       |       |       |
| ---------------- | --------------- | ------------------- | ----- | ----- | ----- | ----- |
| Thera Plus       | 34.89           | 31.22               | 29.24 | 26.96 | 22.34 | 20.61 |
| ×2 only, fixed κ | 35.01           | 31.00               | 28.84 | 26.47 | 21.97 | 20.33 |
| ×3 only, fixed κ | 34.74           | 31.25               | 29.25 | 26.91 | 22.25 | 20.54 |
| ×4 only, fixed κ | 34.47           | 31.17               | 29.24 | 26.97 | 22.32 | 20.59 |
| Fixed κ          | 34.86           | 31.22               | 29.24 | 26.95 | 22.31 | 20.58 |
| No GSE           | 34.81           | 31.15               | 29.18 | 26.91 | 22.29 | 20.57 |
| No TV prior      | 34.89           | 31.23               | 29.24 | 26.87 | 20.42 | 18.68 |
| ReLU instead ξ   | 34.80           | 31.11               | 29.13 | 26.87 | 22.28 | 20.57 |
| Predicted comps. | 34.90           | 31.23               | 29.25 | 26.97 | 22.34 | 20.61 |

Trainable κ. Fixing κ at the theoretically derived value (Eq. 3) leads to a small drop in performance. This suggests that there remain effects that are not accounted for by our proposed observation model, albeit very minor.

Geometric self-ensemble. In line with previous work [26, 43, 44] we see a notable performance boost with geometric self-ensembling. Note, though, if an application prioritizes inference speed over quality this add-on can be disabled at test-time without re-training the network.

Total variation prior. The regularizer has a negligible effect for in-domain scaling factors, but performance de-

# 5. Conclusion

We have developed a novel paradigm for arbitrary-scale super-resolution by combining traditional signals theory with modern implicit neural representations. Our proposed neural heat fields implicitly describe an image as a combination of sinusoidal components, which can be selectively modulated according to their frequency to perform Gaussian filtering (anti-aliasing) between scales analytically, with negligible overhead. Our experimental evaluation shows that Thera, our ASR method based on neural heat fields, consistently outperforms competing methods. At the same time, it is more parameter-efficient and offers theoretical guarantees w.r.t. aliasing. We believe that Thera-style representations could benefit other computer vision tasks and hope to inspire further research into neural methods that integrate physically meaningful and theoretically grounded observation models.
---
# A. Theory

# A.1. Preliminaries

As was described in Section 3.1, the idea underlying our neural heat field with thermal activations is to formulate a neural field Φ(x, t), with x being the 2-dimensional spatial coordinates (x1, x2), such that Φ follows the heat equation:

∂Φ/∂t = κ · ∇2Φ = κ · (∂2Φ/∂x12 + ∂2Φ/∂x22)

The reason for this is that the analytical solution to the (isotropic) heat equation can be modeled as a convolution of the initial state Φ(x,0) with a Gaussian kernel:

g(x, t) = (1 / (4πκt)) · exp(- (x12 + x22) / (4κt))

By fitting the data (image I) at Φ(x,1), we are assuming a Gaussian point spread function (PSF) with the shape:

PSF(x) = (1 / (4πκ)) · exp(- (x12 + x22) / (4κ))

In this formulation, we attempt to recover a “pure” signal at t = 0 or higher sampling rates 0 < t < 1 given an observation at t = 1. Note that:

Φ(x, t) is
meaningless, if t < 0
pure signal, if t = 0
ill-posed, if 0 < t < 1
I, if t = 1
well-posed, if t > 1

The ill-posed problem for 0 < t < 1 is the interesting case, where this formulation relates to super-resolution. The super-resolution algorithm should somehow condition the solution space to find the appropriate solution in this domain.

For all the formulations here, we define the image I to correspond to the coordinates x1, x2 ∈ [−0.5, 0.5].

# A.2. Thermal Diffusivity Coefficient

To use the above formulations, we need to compute the thermal diffusivity coefficient κ. One way to do so is to match the cut-off frequency of the filter in Eq. 7 at t = 1 to the well-known Nyquist frequency given by the image’s sampling rate. We take the cut-off frequency of the Gaussian filter defined in Eq. 7 to be the frequency whose amplitude is halved, which is:

fc = (p / ln(4)) · σ = (√(pln(4)) / (2π2kt))

For the signal compressed into the domain [−0.5,0.5], we can compute the Nyquist frequency to be:

fNyquist = N / 2

where N is the number of samples along a given dimension. This formulation assumes even sampling over x1 and x2. To extend this formulation to non-square images, it would be necessary to change the shape of the signal’s domain in order to maintain even sampling in all spatial dimensions.

If we solve for fc = fNyquist at t = 1, we get:

κ = (pln(4)) / (2π2N2)

For our proposed Thera formulation, we want Φ to contain a single pixel at t = 1. This is the pixel from the low-resolution input which will become SR2 pixels for super-resolution with a scaling factor of SR. Therefore we initialize κ with:

κ = (pln(4)) / (2π2)

Note that the exact value of κ will depend on the characteristics of the system that is being modeled and the anti-aliasing filter that was used (or is assumed). Lower values of κ allow for sharper signals to be represented at any given value of t, but are also more prone to aliasing.

Finally, we would like to highlight that Eq. 10 is specific to the case where x is 2-dimensional. The theoretically ideal value of κ is the only part of our formulation that does not directly apply when using our field’s formulation in spaces with numbers of spatial dimensions other than 2.

Nonetheless, computing κ for other cases would be a simple matter of repeating the steps above with the formulas for a Gaussian filter with the appropriate number of dimensions.

# A.3. Relationship Between t and s

Assuming that the field is learned appropriately, we still need to know at what time t we should sample from to obtain the correct (aliasing-free) signal for a different sampling rate. If we define S to be the subsampling rate (i.e., if our base image has N = 128 and we want to subsample it down to N = 64, we have S = 2) we need to find t such that fNyquist scales by 1/S. Using Eq. 10 and Eq. 11, we can easily find the quadratic relationship:

t = S2

For instance, if we want to upsample the image by a factor of 2, we should use t = 0.52 = 0.25. Thus, 0 < t < 1 refers to super-resolution, while t > 1 refers to downsampling. This is intuitive: As t grows, the image becomes blurrier (the Gaussian kernel gets wider), which corresponds to stronger low-pass filters and therefore lower sampling rates.
---
shown in the figure, it would simply mean that sampling the center location at t = 0 would not be representative of the pixel’s footprint. Figure 8 further illustrates how such blur is equivalent to a scale-appropriate anti-aliasing filter.

# A.4. Other Filters

In theory, the formulation presented in Section 3.1 allows us to use any low-pass filter we want, since we can modulate different components freely. Gaussian filters are an obvious choice since they are often used as anti-aliasing filters, and since they are fully defined by a single parameter, the standard deviation. Initial explorations of a sharp low-pass filter that completely removes components above fNyquist led to a performance reduction, likely due to the associated effect on gradients during training. It remains an open question whether more complex filters (e.g., Butterworth) would improve the current formulation in any way. For quantitative evaluations, this is unlikely, since the downsampling operations use Gaussian anti-aliasing, but in real-world applications or other scenarios, this may be desirable.

# A.5. Initialization of Components

We have noticed during our experiments that the initialization of the components, W1 in Eq. 1, is important. Sitzmann et al. [41] made similar observations when periodic activation functions were first used for neural fields. The final distribution of frequencies |ν(W1)| did not change much during training. Thus, we choose to initialize W1 such that

p(|ν(w1)|) ∝ |ν(w1)|

up to a given maximum frequency, allotting more components to higher frequencies. See code for more details.

# B. Continuous Upsampling Example

In Fig. 9, we provide a practical showcase of the continuous upsampling capabilities of our method.

# C. Additional Quantitative Results

# C.1. Further Metrics

Table 4 shows SSIM metrics on the DIV2K validation set, which complement the PSNR values reported in Table 1. We use the SSIM implementation from torchmetrics [13]. We observe strong performance of Thera, although overall there is relatively little variance across all methods using this metric.

Furthermore, in Table 5 we show quantitative evaluations which are out of distribution both in terms of data (benchmark datasets) and in terms of scaling factors (above ×4).

Figure 6 shows an example where we fit a neural heat field at t = 1 to the image. After training, any low-pass filtered version of the image can be generated by setting t according to Eq. 4. We emphasize that: (i) Computing these filtered images requires no over-sampling or convolutions; (ii) The computational cost does not depend on the size of the blur kernel or on t; (iii) Given Φ(x, t0), the filtered versions Φ(x, t) are known for any t ≥ t0.

Figure 7. Example situations where aliasing would occur without the suppression of high frequencies as modeled by neural heat fields. Sampling the center location at t = 0 would not be representative of the pixel’s footprint.

Figure 8. Blurring an image before a point-wise sample is equivalent to observing with PSF equivalent to the blur kernel.
---
# Figure 9. Showcase of multiscale upsampling using Thera Pro with a RDN [54] backbone, shown with non-integer scaling factors.

|             | x2   | x3   | x4   | x6   | x8   |
| ----------- | ---- | ---- | ---- | ---- | ---- |
| LIIF \[10]  | ×2   | ×3   | ×4   | ×6   | ×8   |
| LTE \[24]   | .934 | .866 | .804 | .704 | .636 |
| CiaoSR \[7] | .936 | .868 | .807 | .707 | .639 |
| SRNO \[45]  | .938 | .871 | .814 | .718 | .651 |
|             | .937 | .873 | .819 | .738 | .685 |
| MSIT \[56]  | .942 | .882 | .829 | .751 | .700 |
| Thera Pro   | .943 | .884 | .833 | .757 | .706 |

Table 4. SSIM scores (higher is better) for several methods and for the larger variants Thera Plus and Thera Pro. These scaling factors evaluated on the (hold-out) DIV2K validation set. We use the RDN backbone for all models. Some methods did not provide checkpoints at the time of writing, see Appendix F.

| Method     | ×6    | Set5  | ×8    | Set14 | B100  | Urban100 | Manga109 |
| ---------- | ----- | ----- | ----- | ----- | ----- | -------- | -------- |
| MetaSR     | 29.04 | 29.96 | 26.51 | 24.97 | 25.90 | 24.83    | 23.99    |
| LIIF       | 29.15 | 27.14 | 26.64 | 25.15 | 25.98 | 24.91    | 24.20    |
| LTE        | 29.32 | 27.26 | 26.71 | 25.16 | 26.01 | 24.95    | 24.28    |
| CUF        | 29.27 | —     | 26.74 | —     | 26.03 | —        | 24.32    |
| CiaoSR     | 29.46 | 27.36 | 26.79 | 25.28 | 26.07 | 25.00    | 24.58    |
| CLIT       | 29.39 | 27.34 | 26.83 | 25.35 | 26.07 | 25.00    | 24.43    |
| SRNO       | 29.38 | 27.28 | 26.76 | 25.26 | 26.04 | 24.99    | 24.43    |
| MSIT       | 29.34 | 27.29 | 26.75 | 25.26 | 26.05 | 24.98    | 24.43    |
| Thera Air  | 29.31 | 27.25 | 26.76 | 25.27 | 26.05 | 24.99    | 24.39    |
| Thera Plus | 29.31 | 27.29 | 26.80 | 25.32 | 26.07 | 25.00    | 24.45    |
| Thera Pro  | 29.51 | 27.34 | 26.90 | 25.38 | 26.12 | 25.04    | 24.70    |

Table 5. PSNR (Y channel) on common benchmark datasets for out-of-distribution scale factors, with an RDN [54] backbone. For some methods, code and/or checkpoints were not publicly available, see Appendix F.

# C.2. Parameter Efficiency

In Fig. 10 we compare the number of additional parameters and PSNR values for various methods and individual upsampling factors on the DIV2K validation set.

# D. Hypernetwork Architecture

In our implementation, Thera Air uses no feature refinement blocks on top of the backbone except a 1 × 1 convolution mapping pixel-wise features into field parameters, as done in SIREN [41]. Thera Plus uses 6 ConvNeXt [29] blocks with d = 64 followed by 7 ConvNeXt blocks with d = 128, prior to the final mapping layer. Projection blocks are added between blocks with different d, which consist of a layer normalization and a 1 × 1 convolution operation. For Thera Pro, two SwinIR [25] blocks are used with depth 7 and 6, respectively, and 6 attention heads per layer.

The STN component follows a standard architecture, consisting of a simple convolutional localization network with adaptive pooling to handle variable input sizes. The network processes the input image along with the scale factor and outputs six parameters of a 2D affine transformation.
---
|      |                 | ×2 Scaling Factor |             |           | ×3 Scaling Factor |      | ×4 Scaling Factor         |           |     |    |          |
| ---- | --------------- | ----------------- | ----------- | --------- | ----------------- | ---- | ------------------------- | --------- | --- | -- | -------- |
| 35.2 |                 |                   | Thera Pro   | 31.5      | Thera Pro         | 29.5 | Thera Pro                 |           |     |    |          |
| 35.1 |                 |                   |             | 31.4      |                   | 29.4 |                           |           |     |    |          |
| 35   | Thera Plus MSIT |                   |             | 31.3      | Thera Plus MSIT   | 29.3 | Thera Plus MSIT           |           |     |    |          |
| 34.9 |                 | SRNO              | 31.2 (Ours) | SRNO      | 29.2 (Ours)       | SRNO |                           |           |     |    |          |
|      |                 |                   |             | PSNR (dB) | (Ours)            |      | CUF                       | PSNR (dB) | CUF |    |          |
| 34.8 | Thera Air       | CiaoSR CLIT       |             | 31.1      | LIIF CiaoSR CLIT  | 29.1 | Thera Air CUF CiaoSR CLIT |           |     |    |          |
| 34.7 |                 |                   |             |           |                   |      | LIIF LTE                  | 31        | LTE | 29 | LIIF LTE |

10−2 10−1 100 101 10−2 10−1 100 101 10−2 10−1 100 101

Model Size (M parameters)

Figure 10. Comparison of ASR methods for different scaling factors (×2, ×3, and ×4) on DIV2K. Thera consistently achieves better performance at lower parameter counts across all scaling factors.

(xy-translation, anisotropic xy-scale, rotation and shear). values with “—” in the tables. Furthermore, we could not create any qualitative samples using CUF.

CLIT [9]. For CLIT, at the time of writing, there is a public code, but no checkpoints. We have made a bona fide attempt to reproduce the models, but due to the cascaded training schedule and the large model size, the training process would require excessive amounts of compute: over a month using 8× Nvidia GeForce RTX 3090 GPUs. Unfortunately, the authors did not respond to our requests for the trained checkpoints used in their paper.

Thera++ addresses an important limitation of the COZ dataset. When looking at the data it becomes obvious that COZ samples are not perfectly aligned, resulting in xy-jitter between images. Additionally, dynamic objects like people, dust, leaves, and moving shadows appear inconsistently across images of the same scene, significantly increasing the noise level. These challenges make super-resolution particularly difficult for this dataset. However, Thera++ outperforms previous state-of-the-art methods, including LMI, across all scaling factors, highlighting its applicability under real-world imaging conditions, see Tab. 6.

| Method       | In-distribution | Out-of-distribution | ×2    | ×3    | ×4    | ×5 | ×6 |
| ------------ | --------------- | ------------------- | ----- | ----- | ----- | -- | -- |
| MetaSR \[18] | 28.70           | 26.55               | 25.17 | 24.31 | 23.25 |    |    |
| LIIF \[10]   | 28.72           | 26.61               | 25.16 | 24.32 | 23.23 |    |    |
| LTE \[24]    | 28.67           | 26.55               | 25.15 | 24.37 | 23.26 |    |    |
| SRNO \[45]   | 28.73           | 26.59               | 25.15 | 24.31 | 23.25 |    |    |
| LIT \[9]     | 28.74           | 26.58               | 25.15 | 24.35 | 23.19 |    |    |
| LMI \[14]    | 28.86           | 26.66               | 25.22 | 24.39 | 23.29 |    |    |
| Thera++      | 29.06           | 26.84               | 25.45 | 24.49 | 23.46 |    |    |

Table 6. Results (PSNR in dB) on the COZ test set.

F. Reproducibility of Existing Methods

We encountered challenges attempting to recreate the results reported for some of the competing methods, which explains why some are missing or differ from the originally reported numbers. Details are provided below.

CUF [44]. At the time of writing, there were no public code or checkpoints available for CUF. Therefore, we could not generate numbers for datasets and scaling factors not reported in the original paper. We have denoted those missing values with “—” in the tables.
