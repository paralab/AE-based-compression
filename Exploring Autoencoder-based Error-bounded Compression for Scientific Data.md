# 2021 IEEE International Conference on Cluster Computing (CLUSTER)

# Exploring Autoencoder-based Error-bounded Compression for Scientific Data

Jinyang Liu,∗ Sheng Di,† Kai Zhao,∗ Sian Jin,‡ Dingwen Tao,‡ Xin Liang,§ Zizhong Chen,∗ Franck Cappello†¶

∗University of California, Riverside, CA, USA

†Argonne National Laboratory, Lemont, IL, USA

‡Washington State University, Pullman, WA, USA

§ Missouri University of Science and Technology, Rolla, MO, USA

¶ University of Illinois at Urbana-Champaign, Urbana, IL, USA

jliu447@ucr.edu, sdi@anl.gov, kzhao016@ucr.edu, sian.jin@wsu.edu,

dingwen.tao@wsu.edu, xliang@mst.edu, zizhong.chen@ucr.edu, cappello@mcs.anl.gov

# Abstract

Error-bounded lossy compression is becoming an indispensable technique for the success of today’s scientific projects with vast volumes of data produced during the simulations or instrument data acquisitions. Not only can it significantly reduce data size, but it also can control the compression errors based on user-specified error bounds. Autoencoder (AE) models have been widely used in image compression, but few AE-based compression approaches support error-bounding features, which are highly required by scientific applications. To address this issue, we explore using convolutional autoencoders to improve error-bounded lossy compression for scientific data, with the following three key contributions. (1) We provide an in-depth investigation of the characteristics of various autoencoder models and develop an error-bounded autoencoder-based framework in terms of the SZ model. (2) We optimize the compression quality for main stages in our designed AE-based error-bounded compression framework, fine-tuning the block sizes and latent sizes and also optimizing the compression efficiency of latent vectors. (3) We evaluate our proposed solution using five real-world scientific datasets and comparing them with six other related works. Experiments show that our solution exhibits a very competitive compression quality from among all the compressors in our tests. In absolute terms, it can obtain a much better compression quality (100%∼800% improvement in compression ratio with the same data distortion) compared with SZ2.1 and ZFP in cases with a high compression ratio.

# I. INTRODUCTION

Today’s scientific applications are producing extremely large amounts of data during simulation or instrument data acquisition. Advanced instruments such as the Linac Coherent Light Source (LCLS) [1] and Advanced Photon Source [2], for example, may produce vast amounts of data with a very high data acquisition rate (250 GB/s [3]). Consequently, reducing the data volumes with user-tolerable data distortion is critical to the efficient data storage and transfer. Error-bounded lossy compression is arguably the most efficient way to significantly reduce the data volumes for scientific applications with big data issues. Unlike lossless compressors [4]–[7] that suffer from very low compression ratios (generally ∼2:1) on floating-point datasets, error-bounded lossy compressors can obtain fairly high compression ratios (10+ or even several hundreds [3], [8], [9]). Moreover, error-bounded lossy compressors are able to keep a high fidelity of the reconstructed data for the user’s post hoc analysis based on the user’s required bounds on data distortion, as verified by many recent studies [10]–[12].

As a classic type of deep learning model, the Autoencoder (AE) has been gaining more and more attention. Such a deep neural network (DNN) architecture is composed of both an encoder (encoding the input data) and a decoder (decoding the encoded data) and is trained to minimize the error between the reconstructed data and the initial data. In general, because the trained encoder and decoder can be used separately, AE can be used to learn efficient data representation (or coding), typically for dimensionality reduction. The corresponding DNN will be trained to reconstruct the main patterns in the dataset effectively based on the reduced information generated from the original data. Recently, several variations of the AE have been developed with different model frameworks and training paradigms for improving the effectiveness of data reconstruction and for handling more tasks such as data generation. Nevertheless, although AE has been widely used in the image compression domain, few studies explored the possibility of leveraging it for error-bounded compression.
---
# II. RELATED WORK

In this paper we explore the possibility of leveraging the AE model to improve the error-bounded lossy compression significantly. Such a study faces several challenges. First, many types of autoencoders exist, each with different architectures or training methods, so that determining the most effective AE model is challenging. Second, applying AE in the error-bounded model with a proper configuration setting is nontrivial. Third, latent vectors from AE need to be stored in the compressed data, so minimizing the latent vector overhead while maintaining a high reconstruction quality is challenging and critical to getting a good rate distortion in the high-compression cases.

In this work we propose a novel error-bounded lossy compressor, AE-SZ, which combines the classic prediction-based error-bounded compression framework SZ and the Sliced-Wasserstein Autoencoder (SWAE) model with convolutional neural network implementations. The key contributions of the paper are summarized as follows:

- Our autoencoder-based error-bounded compression framework is designed on the basis of a blockwise model, which can adapt to diverse data changes in a dataset well. To the best of our knowledge, AE-SZ is the first AE-based error-bounded lossy compressor that exhibits a better rate distortion than the three state-of-the-art models SZauto, SZ, and ZFP.
- We investigate various autoencoder models and identify the most effective one for the error-bounded lossy compression model and also carefully optimize the related configurations, such as block sizes and strategies of compressing latent vectors.
- We evaluate the proposed AE-SZ by using the scientific datasets generated by five different real-world high-performance computing (HPC) applications across different domains. We identify the effectiveness of AE-SZ by comparing it with two other AE-based lossy compression methods and four other state-of-the-art error-bounded lossy compressors. Our experiments show that AE-SZ is the best compression method in the category of AE-based compressors. AE-SZ also exhibits competitive rate distortion compared with existing state-of-the-art error-bounded lossy compressors. Specifically, when the compression ratio is greater than 100, AE-SZ can get 100%∼800% higher compression ratios than can SZ2.1 and ZFP, with the same peak signal-to-noise ratio (PSNR).

The rest of this paper is organized as follows. In Section II we discuss related work. In Section III we formulate the research problem. In Section IV we present the overall design of AE-SZ as well as the detailed optimization strategies. In Section V we evaluate our solution using multiple real-world scientific simulation datasets. In Section VI we conclude the paper with a vision of the future work.
---
Total of seven layers including the latent vector), and the size λ, a number of random projections L, and a predefined latent distribution qZ, SWAE optimizes the following loss function:

L(φ, ψ) = 1 ∑M c (xm, ψ (φ (xm)))

∑       ∑      (              (     ))

~~M~~  m=1                                (1)

λ          L    M  c θ · z

+~~LM~~  l=1  m=1  l  ˜i[m], θl · φ xj[m],

in which

{x1, . . . , xM } is sampled from training set (i.e. pX),

{z ˜ , . . . , z 1  ˜M } is sampled from qZ,

{θ1, . . . , θL} is sampled from Sd−1 (K-dimensional unit sphere),

i[m] and j[m] are the indices of sorted θ · z l  ˜m s and θl · φ (xm), respectively, and

c(x, y) = ||x − y||2.            (2)

Kolouri et al. [44] proved that optimizing this loss function is equal to optimizing

argminφ,ψ Wc (pX , pY ) + λSWc (pZ , qZ ),  (3)

in which Wc (pX , pY ) is the Wasserstein distance from pX (distribution of input data X) to pY (distribution of decoded data Y) and SWc (pZ , qZ ) is the sliced-Wasserstein distance from pZ (distribution of encoded latent Z) to qZ. Kolouri et al. [44] also show the efficiency of computing Eq. 1.

# III. BACKGROUND AND PROBLEM FORMULATION

# A. Research Background – Autoencoder

We describe autoencoder briefly as follows. A stereotype autoencoder model is composed of an encoder network and decoder network. The former encodes the input data to a latent vector in reduced size, and the latter decodes the latent vector to an approximate reconstruction of data. The latent vector stands as a compressed representation of the input data, and different autoencoders have different technical details for computation of the latent vector.

The nature of the autoencoders grants them the potential for being leveraged for data reduction, in that the reconstructed data based on the latent vector can approximate the original data to a certain extent. Figure 1 shows visualization of the reconstructed data versus the original data with the autoencoder [40] (reduction ratio = 64×) on a turbulence dataset.

# 1) Leveraging Autoencoders in Error-bounded Scientific Data Compression

The autoencoder itself cannot bound the compression errors, which is a significant gap to scientific user’s demand for error controls. As shown in Figure 1, the maximum point-wise compression error is up to 1.2, which is about 20% of the original data value range (−3.06, 2.64]. In comparison, scientists often need to control the point-wise errors to a much smaller bound such as 1% of the original value range [9], [15].

In this work we aim to develop a deep learning based error bounded lossy compressor. Specifically, for some scientific applications, we train neural networks based on a certain amount of training data, and then apply the trained networks to compress the testing data generated by the same applications. We separate the training data and testing data because we expect that the pre-trained networks can be used to compress new data for the same applications, such that the training time and model size can be excluded from the compression time and size. In our experiments, the training and test data are from different time steps or the simulation running with different configuration settings in the same application.

# 2) Math Formulations for Error-bounded Lossy Data Compression

The compression ratio (denoted by ρ) is defined as ||D| , where |D| and |D′| denote the original data size and compressed data size (both in bytes), respectively.

Error-bounded lossy compression has one important constraint, namely, that the reconstructed data respect a user-specified error bound (denoted by e) strictly. Under this constraint, the rate distortion often serves as a criterion to assess the compression quality, which involves two critical.

Fig. 1. Reconstructed data of AE (64×) on a turbulence dataset (original value range: [−3.06 , 2.64], max pointwise absolute error = 1.2)
---
terms: bit rate and data distortion. The bit rate is defined as the average number of bits used to represent one data point after the compression; hence, the lower the bit rate, the higher the compression ratio. In the lossy compression community rate distortion is often evaluated by the PSNR, defined as shown below:

PSNR = 20 log10(vrange(D)− 10log10(mse(D,D′)))

where D′ is the reconstructed dataset after decompression (i.e., decompressed dataset), vrange(D) represents the value range of the original dataset D (i.e., the difference between its highest value and lowest value), and mse refers to mean squared error. The higher the PSNR value is, the smaller the mean squared error, which means higher precision of the decompressed data.

Our objective is to obtain higher compression ratios than other related works obtain (including other deep-learning-based compressor and traditional error-bounded lossy compressors) with the same PSNR value, while also strictly respecting the user’s error bound, especially aiming at optimizing the use cases with high compression ratio. We can write the research problem formulations as follows:

maximize ρ

s.t. PSNR(D, D′) = λ,

|di − d′| ≤ e

where λ is a particular PSNR value representing a specific data distortion level and di and d′ refer to any data point in the original dataset D and decompressed dataset D′, respectively.

# IV. AE-SZ: AUTOENCODER-BASED ERROR-BOUNDED LOSSY COMPRESSION FRAMEWORK

In this section we present the design overview of AE-SZ and describe the detailed optimization strategies for AE-SZ.

# A. Design Overview of AE-SZ

We present the overall framework of our designed autoencoder-based error-bounded lossy compression framework AE-SZ as shown in Figure 2. The overall compression involves two stages: offline training and online compression.

During the offline training, we split the training data snapshots into multiple small fixed-size blocks (such as 32×32 for a 2D data field or 8×8×8 for a 3D data field) and train the network with numerous small blocks. The advantage of such a design is twofold: (1) the AE model works more efficiently on the divided data blocks, which can catch fine-grained data features; (2) such a data-splitting design creates numerous training samples (i.e., data blocks), so that the AE model is tractable.

During the online compression, AE-SZ executes four steps as shown in Figure 2: (1) splitting the input data to be compressed into many small blocks (with the same block size as during the training stage), (2) prediction, (3) linear-scale quantization, and (4) entropy/dictionary encoding. Specifically, in each block, the data are predicted by a predictor (either autoencoder or Lorenzo), and the prediction errors will be selected out for this block.

The pseudo code of the AE-SZ compression procedure is presented in Algorithm 1. As mentioned before, AE-SZ compresses the input data block by block, and the compression of each block follows the same routine. Thus, in the following, we describe the compression procedure mainly on a single data block (i.e., line 2∼16), without loss of generality. For any block, AE-SZ first generates predicted data based on two predictors (Lorenzo and autoencoder) for this block respectively (line 3∼8). Then, the predictor with lower element-wise l1-loss is selected out for this block (line 9∼13).
---
The reason is that the smaller the prediction errors are, the more uneven the distribution of quantization bins in general, and hence the higher compression ratio of quantization bins. Then, AE-SZ uses linear-scale quantization to quantize the prediction errors based on the user-specified error bound e (line 14). Similar to SZ2.1 [9], [15], we need to set a maximum number of quantization bins (65,536 by default) for the linear quantization, in order to keep high performance. The total quantization range may not cover all predicted values as the prediction errors may be large. The corresponding data points, called unpredictable data, will be saved separately (denoted as U in line 14). For more details about linear-scale quantization, we refer readers to read our paper [9].

# Algorithm 1 AE-SZ Compression Algorithm

Input: Input data D, block size S, error-bound e, latent error-bound el.

Output: Compressed data D′={z ˆ}

1: Split D into blocks of Size ˆ, Z, U.
2: for (each block B in S(1D), S×S(2D), or S×S×S(3D). do
3:   z ← Eec(B). /*Encode B with the encoder network Eec.*/
4:   z ′← f(z,el). /*Get decompressed latent vector z′ based on el*/
5:   B = Dec(z′). /*Get Decoded B′ using decoder network Dec*/
6:   loss1 = ||B − B′||1. /*Compute l1 loss of B′ vs. B.*/
7:   B′′ = Lorenzo(B). /*Predict B with Lorenzo.*/
8:   loss2 = ||B − B′||1. /*Compute l1 loss of Lorenzo predictor.*/
9:   if loss2 ≤ loss1 then
10:      Bp = B′′. /*Select Lorenzo-predicted values*/
11:    else
12:      Bp = B′. /*Select Autoencoder-predicted values*/
13:    end if
14:    Q, U = Quantize(B, Bp, e). /*linear-scale quantization with e, to get quantization codes Q and unpredictable data U.*/
15: end for
16: Compress all saved coefficients from AE and Lorenzo.
17: H ← Huffman Encode(Q). /*Huffman encoding*/
18: ˆ Z ← Zstd(H). /*Zstd compression*/

In the following subsections, we present several critical optimization strategies for AE-SZ, which are developed in terms of fundamental takeaways we summarized from our in-depth analysis or comprehensive experimental evaluation.

# B. Design Detail: AE network structure in AE-SZ

The structure of our designed autoencoder network used in AE-SZ is illustrated in Figure 3. Like most of the autoencoders, it consists of an encoder network to generate the latent vectors as the compressed representation of input original data and a decoder network to reconstruct the data from latent vectors. The input of the network are (batchs of) data blocks, which will be linearly normalized to the range of [-1, 1] based on the global maximum and minimum of data before being put in the network, and the output of the network needs to be denormalized to generate the final prediction values.

The encoder and decoder networks are both formed with several convolutional/deconvolutional blocks and a fully connected layer for resizing latents, and their structures are mirror-symmetric except for an additional final output layer-set in the decoder network. To adapt to different datasets, the number of Convolutional blocks and the number of channels in each block may vary, but the overall structure remains the same. For example, the main difference between autoencoders used for 2D/3D datasets is just the dimension (2D or 3D) of convolutional/deconvolutional operation in the network layers.

# Fig. 3. Our Designed Blockwise Convolutional AE network for Compression

# Fig. 4. (a) The Convolutional blocks used in AE-SZ encoder network. (b) The Deconvolutional blocks used in AE-SZ decoder network.

| Input HxWxC      | Input XxYxZxC        | Input HxWxC    | Input XxYxZxC     |
| ---------------- | -------------------- | -------------- | ----------------- |
| Conv2D           | Conv3D               | DeConv2D       | DeConv3D          |
| Stride:          | Stride:              | Stride:        | Stride:           |
| Kernel: 3x3xC    | Kernel: 3x3x3xC      | Kernel: 3x3xC  | Kernel: 3x3x3xC   |
| Padding:         | Padding: 1           | Padding: 1     | Padding: 1        |
| Conv2D           | Conv3D               | DeConv2D       | DeConv3D          |
| Stride:          | Stride:              | Stride:        | Stride:           |
| Kernel: 3x3xC    | Kernel: 3x3x3xC      | Kernel: 3x3xC  | Kernel: 3x3x3xC   |
| Padding:         | Padding: 1           | Padding: 1     | Padding: 1        |
| GDN              | GDN                  | iGDN           | iGDN              |
| Output H/ZxWIZxC | Output XIZxYIZxZIZxC | Output-ZHxZWxC | Output-2XxZYxZZxC |

---
# D. Design Detail: Optimizing AE Configurations

Takeaway 2: The performance of AE may differ a lot with different configurations under the same model structure. Optimizing the AE configurations, especially the input block size and the latent vector size, is critical to the final performance of AE-SZ.

# C. Design Details: Choosing the Autoencoder Type

Takeaway 1: Sliced-Wasserstein Autoencoder is particularly suitable for data prediction in scientific data compression compared with other AE models.

A key point in designing AE-SZ is that we need to select the most appropriate model for scientific data prediction from multiple variations of autoencoder models. In AE-SZ, we select sliced-Wasserstein autoencoder (SWAE) for the AE compressor and predictor. The advantages of SWAE in data compression are as follows:

- Compared with the other tested autoencoders, SWAE shows less reconstruction loss on scientific data.
- Different from traditional variational autoencoders (VAEs), the encoding and decoding computation in SWAE are both determinant. VAEs such as [53]–[57] actually compute means and variances with input data and sample latent vectors with the means and variances from the prior distribution. Therefore, in multiple runs with the same input, the latent vector as the output of encoder in a VAE will differ, which makes the VAE being unstable for data compression tasks.
- Compared with Wasserstein autoencoders (WAE), the computation of training loss in SWAE is more numerically efficient. Similar to SWAE, WAE computes Wasserstein distances for training losses, and its computation cost is higher than the computation of sliced-Wasserstein distances. With both n samples from the training set and prior distribution, the computational cost of the Wasserstein distance is O(n2) whereas the computational cost of the sliced-Wasserstein distance is O(n log n).

Table II shows the average prediction PSNR and AE-SZ compression ratio (error bound = 1E-2) of different input block sizes under the same latent ratio (input block size divides latent vector size, 64 for CESM-CLDHGH and 32 for NYX-baryon density). We conclude that optimizing the input block size is of great importance because autoencoders can achieve apparently different performances under the same latent overhead but various input block sizes. In our work we optimize the input block size of the autoencoder in AE-SZ separately for each field, and we find that 32×32 input block fits most of the 2D data fields tested and that 8×8×8 input block fits most of the 3D data fields tested.

|       | Blocksize | CESM-CLDHGH   | Blocksize     | NYX-baryon density |      |      |
| ----- | --------- | ------------- | ------------- | ------------------ | ---- | ---- |
| 16×16 | PSNR 42.5 | CR(1e-2) 55.5 | 8×8×8         | 46.6               | 71.1 |      |
|       | 32×32     | PSNR 43.9     | CR(1e-2) 60.9 | 16×16×16           | 35.7 | 23   |
|       | 64×64     | PSNR 41.7     | CR(1e-2) 50.1 | 32×32×32           | 28.9 | 23.9 |

Table I presents the reconstruction quality (PSNR) on different types of autoencoders that we explored. We trained 8 types of autoencoders on a split of snapshots of the CESM-CLDHGH data field: a vanilla autoencoder, vanilla variational autoencoder [53], β-VAE [54], DIP-VAE [55], Info-VAE [56], LogCosh-VAE [57], WAE [45], and SWAE [44]. After training, the AEs were tested by using another split of data snapshots. From this table, we observe that SWAE has the best prediction accuracy with highest PSNR, which motivates us to use it as our final predictor in AE-SZ.

| AE type     | PSNR |
| ----------- | ---- |
| AE          | 42.2 |
| VAE         | 36.2 |
| β-VAE       | 40.1 |
| DIP-VAE     | 32.2 |
| Info-VAE    | 26.5 |
| LogCosh-VAE | 39.0 |
| WAE         | 42.4 |
| SWAE        | 43.9 |

Table III presents the final compression ratio of AE-SZ under the error bound of 1E-2 with AEs of different latent sizes on the Hurricane-U data field. The input block size is 8×8×8, and the rest part of the network remains the same for different latent sizes. We can see that different latent sizes bring a 40%+ difference in final compression ratios, which motivates us to choose an appropriate latent size in our design.

|   |   |   | Latent size | Latent ratio | CR (1E-2) |
| - | - | - | ----------- | ------------ | --------- |
|   |   |   | 4           | 128          | 123.4     |
|   |   | 6 |             | 85.3         | 137.4     |
|   |   |   | 8           | 64           | 149.1     |
|   |   |   | 12          | 45.7         | 127.7     |
|   |   |   | 16          | 32           | 106       |

---
# E. Design Detail: Lossy compression of AE latent vectors

Takeaway 3: Predicting the data with error-bounded lossy decompressed latent vectors can maintain a very small loss of prediction accuracy, while greatly reducing the latent vector size (i.e., latent overhead).

One main disadvantage of autoencoders is the overhead of storing latent vectors, which can be reduced but cannot be eliminated. To maximize the compression ratio with autoencoders, instead of using the original encoder output latent vectors for compression, AE-SZ compresses the latent vectors with a built-in customized compressor and then uses the decompressed latent vectors for decoding. In this approach, the compressed latents are to be stored. The computation of compressed latents and autoencoder predictions in AE-SZ is shown in Figure 5. For the original latent vector z as the encoder network, a lossy compressor generates the compressed latent zc in reduced size and the decompressed zd (which can also be directly computed from zc); then the decoder network computes the prediction with zd as its input.

|   |   | RTM  |     | NYX darkmatterdensity | EXAFEL |     |     |     |
| - | - | ---- | --- | --------------------- | ------ | --- | --- | --- |
|   |   | 1E−2 | 6.9 | 5.9                   | 7.1    | 6.2 | 6.6 | 5.7 |
|   |   | 1E−3 | 3.9 | 3.4                   | 4.1    | 3.6 | 3.6 | 2.9 |
|   |   | 1E−4 | 2.5 | 2.0                   | 3.2    | 2.5 | 1.9 | 1.4 |

We customize an efficient method for compressing AE-SZ latent vectors (called customized or custo. for short) with two steps: (1) quantize the original value using error bound of 0.1e, where e is the user-specified error bound (the error bounds are value-range based) for the dataset; and (2) use Huffman + Zstd to compress the quantization codes. The advantage of such a design is twofold. First, this can get better compression ratios than SZ2.1, as shown in Table IV. The key reason is that the latent vector data are not quite smooth across adjacent elements, based on our observation, while SZ2.1 strongly relies on the spatial smoothness. Second, the custo. design is consistent with an important constraint required in the AE-SZ: the compression of each data block must be independent of other data blocks, which is explained as follows. Note that we select the better prediction method between AE and Lorenzo based on their prediction accuracy for each block. After this step, all the blocks (either AE-predicted blocks or Lorenzo-predicted blocks) have corresponding predicted data, which can be applied with the quantization directly. Obviously, in order to minimize the latent overhead, we should not store the AE latents for the Lorenzo-predicted blocks, but that requires that the compression of latents be independent across data blocks. SZ2.1 has data dependency across blocks, which makes it unsuitable for the latent vector compression here.

Through masses of experiments using different datasets, we note that choosing a reasonable error bound can achieve a relatively high compression ratio of latents with small loss of prediction accuracy. Figure 6 presents two rate-distortion plots of the AE prediction values with different compression ratios of latent vectors. The prediction accuracy (w.r.t. PSNR) does not degrade at all when the latent vectors are compressed with a ratio such as 4 (corresponding to bit-rate 0.25 in the figure as the original latent size is 1 of the input size). That is, compressing latent vectors with a relatively high compression ratio (under a certain error bound) does not affect the compression of quantization bins much.

# F. Design Detail: Combination of AE and Lorenzo

Takeaway 4: The autoencoder model has a high ability to represent the data roughly with a high reduction ratio, but it is not as effective as Lorenzo in high-precision use cases. Therefore, a combination of AE and Lorenzo can effectively mitigate their own particular limitations in data prediction. Although AE has a great ability in learning the distribution of data, it still has two critical drawbacks that prevent it from being directly used as a data predictor especially for high-precision error-bounded compression use-cases. The first drawback is that, similar to linear regression, the latent vectors generated by AE for decompression sometimes bring cost due to redundancies of space. Specifically, we observe that quite a few data blocks may have constant or approximately constant values in scientific data. For these blocks, applying a simple and low-cost predictor is accurate enough, while being able to reduce the storage size as much as possible. Second, to maintain the learning effectiveness and efficiencies, the reconstructed data blocks from the autoencoder always suffer from certain noises, making it inadequate for extremely high-precision compression. By comparison, we note that the Lorenzo predictor outperforms the autoencoder especially when a relatively small error bound is used.

We use Figure 7 to illustrate the pros and cons of the autoencoder and Lorenzo predictor under different error bounds.
---
# Probability Density Function (PDF)

# Hurricane [62]:

A simulation of a hurricane from the National Center for Atmospheric Research in the United States.

# EXAFEL [63]:

An Exascale Computing Project for analyzing molecular structure X-ray diffraction data generated by the LCLS [1]. The data contains groups of 32 2D arrays of size 185×388. We discard the groups with nearly uniform data points; and following [64], we concatenate the 2D arrays in each group to form a single 5920×388 2D array for each group.

More detailed information of the datasets (all in single precision) is shown in Table V. The fields of NYX are transformed to their logarithmic value before compression for better visualization, as suggested by domain scientists.

# Fig. 7. Distribution of prediction errors on CESM-FREQSH data field

This figure demonstrates the prediction error distributions of Lorenzo predictor, linear regression predictor, and our trained autoencoder under an error bound of 1E-2 and 1E-4, respectively (the input data is a snapshot of CESM-FREQSH data field). One can clearly observe that under the large error bound 1E-2, the autoencoder has a better (sharper) prediction error distribution. In contrast, the prediction accuracy of the Lorenzo predictor grows rapidly as the error bound decreases to a small value 1E-4.

During the online compression, AE-SZ selects a predictor between autoencoder and Lorenzo for each data block. The selection criterion is checking which predictor has lower prediction errors (i.e., loss) for the given block. The details can be found in Algorithm 1 (see line 6∼13).

# V. PERFORMANCE EVALUATION

In this section we present the experimental setup and then discuss the results.

# A. Experimental Setup

# 1) Experiment Environment:

We perform the experiments on the gpu v100 smx2 nodes of the Argonne National Laboratory Joint Laboratory for System Evaluation computation cluster. Each node is driven by two Intel Xeon GOLD 6152 processors with 188 GB of DRAM and NVIDIA TESLA V100 GPUs.

# 2) Data Used in Experiments:

We perform the evaluation using five real-world application datasets in different domains that are commonly used in testing lossy compressors. Most of the datasets such as CESM, NYX, Hurricane can be downloaded from SDRBench [58].

- CESM [59]: A well-known climate simulation package. We use its atmosphere model [58] in our experiments. These datasets are 2D, although some fields exhibit three dimensions in their metadata. For the CLOUD field (26×1800×3600), for instance, SZ2.1 has a better compression ratio (31.1 vs. 22.6) if we compress it with the range-based error bound 1E-3 in 2D mode instead of 3D mode.
- RTM: Reverse time migration (RTM) code for seismic imaging in areas with complex geological structures [60].
- NYX [61]: An adaptive mesh, hydrodynamics code designed to model astrophysical reacting flows on HPC systems. Two separate simulations are performed for generation of training and test data.

# TABLE V

# BASIC INFORMATION ABOUT APPLICATION DATASETS

| App.      | # Files and fields  | Dimensions  | Fields used    | Domain          |
| --------- | ------------------- | ----------- | -------------- | --------------- |
| RTM       | 1 field, 3600 files | 449×449×235 | snapshot       | Seismic Wave    |
| CESM      | 26 fields, 62 files | 1800×3600   | CLDHGH, FREQSH | Weather         |
| EXAFEL    | 1 field, 352 files  | 5920×388    | raw data       | Crystallography |
| NYX       | 6 fields, 5 files   | 512×512×512 | bd, t, dmd     | Cosmology       |
| Hurricane | 13 fields, 48 files | 100×500×500 | U, QVAPOR      | Weather         |

# 3) Comparison Lossy Compressors in Our Evaluation:

In our experiment, we compare AE-SZ with six other lossy compressors. The first four are classic error-bounded compressors: SZ2.1 [8], [9], [15] and ZFP0.5.5 [13], which have been widely used in the community, and two recent works based on the SZ framework and developed from SZ2.1: SZauto [14] and SZinterp [31]. The fifth one is a recent work of an autoencoder-based scientific data compressor [43], called AE-A in our evaluation. The sixth one is a pure convolutional autoencoder model [40], called AE-B, proposed for compressing turbulence data, which is not error bounded.

# 4) Experimental Configurations:

For SZ2.1, ZFP0.5.5, SZauto, and SZinterp we adopt value-range-based error bounds and use default configurations for other parameters. For the training phase of the autoencoders in AE-SZ, we train different autoencoders for different data fields on selected parts of the data, then test and compare all compressors on the remaining parts. Table VI shows the input block size, length of latent vectors, number of the convolutional blocks in encoder network, and number of channels in convolutional blocks of encoder network. The number of deconvolutional blocks is the same as the encoder’s, and the channel numbers of the decoder network are symmetric with those in the encoder network.

# TABLE VI

# CONFIGURATIONS FOR EACH DATA FIELD

| Data field       | Input block | Latent size | Block num. | Channels         |
| ---------------- | ----------- | ----------- | ---------- | ---------------- |
| CESM-CLDHGH      | 32x32       | 16          | 4          | \[32,64,128,256] |
| CESM-FREQSH      | 32x32       | 32          | 4          | \[32,64,128,256] |
| EXAFEL           | 32x32       | 16          | 4          | \[32,64,128,256] |
| RTM              | 16x16x16    | 16          | 4          | \[32,64,128,256] |
| NYX (all fields) | 8x8x8       | 16          | 3          | \[32,64,128]     |
| Hurricane-U      | 8x8x8       | 8           | 3          | \[32,64,128]     |
| Hurricane-QVAPOR | 8x8x8       | 16          | 3          | \[32,64,128]     |

---
# TRAIN-TEST TABLE VII

| Dataset      | Train split  | Test split                        |
| ------------ | ------------ | --------------------------------- |
| CESM         | \[0,49]      | \[50,62]                          |
| EXAFEL       | \[0,299]     | \[300,351]                        |
| RTM          | \[1400,1499] | 1510 to 1600 step 10              |
| NYX redshift | \[54,42]     | another simulation at redshift 42 |
| Hurricane    | \[1,40]      | \[41,48]                          |

We observe that AE-SZ is significantly better than the other two AE-based lossy compressors (AE-A and AE-B) in terms of rate distortion. That is, our developed AE-SZ compression method is arguably the best AE-based lossy compressor to date. We also compare the most competitive error-bounded lossy compressors (to the best of our knowledge): SZinterp [31], SZauto [14], SZ2.1 [15], and ZFP [13]. Generally speaking, AE-SZ obtains much better rate distortions than SZauto, SZ2.1, and ZFP do under low bit rates (i.e., in high-compression-ratio cases) and have a comparable quality with SZ2.1 for high bit rates. We observe that AE-SZ generally has 100%∼800% higher compression ratios than SZ2.1 has in the high-compression-ratio cases on both 2D and 3D datasets. In the 2D datasets, for example, AE-SZ exhibits the best rate distortion (240% higher compression ratio than the second best at the same PSNR around 44) for the CESM-FREQSH data field. On the EXAFEL dataset, AE-SZ has a 200% higher compression ratio than the second best (SZ2.1) in the high-compression cases. On the 3D datasets, AE-SZ also exhibits very competitive rate distortions from among all the seven compressors. Its compression quality is close to that of SZinterp in the low-bit-rate range (e.g., [0,1]).

For AE-A, we download their code from GitHub, which supports only double-precision floating data originally. We improve the code by enabling it to compress single-precision floating data, in that most of datasets in our test are stored in single-precision. We trained its model using the same training data split for 100 epochs. The .dvalue files generated by the model are compressed by SZ2.1 following the instruction of [43]. After fine-tuning, we applied the same value-range-based relative error bound to compress the .dvalue file.

For AE-B, since Glaws et al. [40] does not provide enough details for training from scratch, following the paper’s recommendation, we fine-tuned a pretrained autoencoder (from GitHub indicated by [40]) on different data fields for 5 epochs each.

# 5) Evaluation Metrics:

We evaluate the seven lossy compressors based on the critical metrics described below.

- Rate distortion: Rate distortion is the most commonly used metric by the lossy compression community to assess compression quality. Rate distortion involves and plots with two critical metrics: peak signal-to-noise ratio (PSNR) and bit rate. The definition of PSNR is introduced in section III, and bit rate is defined as the average number of bits used per data point in the compressed data. Generally speaking, Bit rate equals Sizeof (datatype)/cr, in which Sizeof (datatype) is the byte size of input data (32 for single-precision data for example), and cr is the compression ratio. Therefore, smaller bit rate means better compression ratio, and vice versa.
- Visualization with the same compression ratio (CR): Compare the visual quality of the reconstructed data based on the same CR.
- Compression speed and decompression speed: original size (MB/s) and reconstructed size (MB/s).

In the following experimental results, when it comes to error bound values, without loss of generality, we adopt value-range-based error bounds (denoted as ), which takes the same effect with absolute error bound (denoted e) because e = · (max(D) − min(D)).

# B. Evaluation Results and Analysis

1) Rate distortions of different lossy compressors: We present the rate distortion results of all seven lossy compressors on all tested data fields, illustrating the PSNR of final decompression results with bit rates. Figure 8 shows the rate distortion plots for each lossy compressor on eight data fields. Only four compressors are shown in Figure 8 (a), (b), and (c) because the other three compressors (SZauto, SZ2.1, and ZFP) are not included.

2) Decompression data visualizations of different lossy compressors: We present data visualizations in Figure 9 on the NYX-baryon density field to verify the effectiveness of the reconstructed data of AE-SZ at high compression ratio use cases. We clearly observe that the reconstructed data at the PSNR of 46.8 under AE-SZ has a very good visual quality. Other prior works [15], [65] show that PSNR in the range of [30,60] is good enough to have a high visual quality for different scientific applications. Moreover, Figure 9 demonstrates that with the same compression ratio of 180, AE-SZ has a much better visual quality compared with that of the three state-of-the-art lossy compressors, SZauto, SZ2.1, and ZFP0.5, and is also better than SZinterp.

3) Performances of AE-SZ predictors under different error bounds: To better understand how the autoencoder and Lorenzo predictor in AE-SZ cooperatively contribute to the compression ratios, we record the percentage of data blocks predicted by AE-SZ autoencoders on three different data fields, as shown in Figure 10. For better vision of the plots, the x-axis is logged error bounds. The plots show that autoencoders in AE-SZ achieve advantages over Lorenzo under a range of medium error bounds (about 5E-3 to 2E-2), under which most of the data blocks can be better predicted by autoencoders. As the error bound decreases, the Lorenzo predictor becomes better than autoencoders on more data blocks. When the error bound becomes very high, the latents need to be compressed with a high error bound, so the prediction error of autoencoders may drop rapidly, and Lorenzo may also turn better.
---
# PSNR (dB)

| 85 | sz2.1 | 90               | sz2.1   |   |   |   |   |
| -- | ----- | ---------------- | ------- | - | - | - | - |
| 80 | zfp   | 80               | zfp     |   |   |   |   |
| 75 | AESZ  | 110% Improvement | AESZ    |   |   |   |   |
| 70 | AE−A  | 70               | A       |   |   |   |   |
| 65 | (dB)  | improved         | PSNR    |   |   |   |   |
| 60 | 46    | 60               | by 240% |   |   |   |   |
| 44 | 420   | 0.2              | 0.4     |   |   |   |   |
| 50 |       | 40               |         |   |   |   |   |
| 45 |       |                  |         |   |   |   |   |
| 40 | 0     | 1                | 2       | 3 | 4 | 5 | 6 |

# (a) Bit−Rate

| 90    | CESM-CLDHGH (2D)            | 90   |           | (b) CESM-FREQSH (2D) |    |   |   |
| ----- | --------------------------- | ---- | --------- | -------------------- | -- | - | - |
| sz2.1 | zfp                         | 80   |           | AESZ                 |    |   |   |
|       |                             | AE−A | 70        | AE−A                 |    |   |   |
|       | 200% improvement over SZ2.1 |      | PSNR (dB) | AE−B                 | 50 |   |   |
|       |                             | 43   | 60        | 48                   |    |   |   |
| 60    |                             |      | 50        | 46                   | 44 |   |   |
| 50    |                             |      | 40        | 42                   |    |   |   |
| 40    | 0                           | 1    | 2         | 3                    | 4  | 5 | 6 |

# (c) Bit−Rate

|     |   | EXAFEL (2D) |        |      | (d) NYX-baryon density (3D) |    |          |    |
| --- | - | ----------- | ------ | ---- | --------------------------- | -- | -------- | -- |
| 70  |   | AE−         | 70     |      |                             |    |          |    |
| 65  |   | sz2.1       | B      | 65   |                             |    |          |    |
| zfp |   | AESZ        | 60     | AESZ |                             |    |          |    |
|     |   |             |        |      | AE−                         | A  | (dB)     |    |
| 55  |   |             | SZauto | 55   | SZauto                      |    |          |    |
|     |   |             |        |      | 50                          | 44 | SZinterp | 49 |
| 45  |   | 48          | 45     |      | 43                          |    |          |    |
| 40  |   | 46          | 40     |      | 42                          |    |          |    |
| 35  |   | 44          | 35     |      | 40                          |    |          |    |
|     |   |             |        | 43   |                             |    |          |    |
|     |   | 0.1         | 0.2    | 0.3  | 0.4                         |    |          |    |

# (e) NYX-temperature (3D)

| 90                  | AE−       | 90     | sz2.1 |
| ------------------- | --------- | ------ | ----- |
| zfp                 | SZauto    | 80     | AESZ  |
| AESZ                | SZinterp  | AE−A   | 70    |
| by 800% over SZauto | PSNR (dB) | 60     | AE−B  |
| 60                  | 58        | SZauto |       |
| 56                  | SZinterp  | 54     |       |
| 50                  | 52        | 1      |       |
| 48                  | 40        | 500    | 0.1   |
| 0.2                 | 0.3       | 0.4    | 0.5   |

# (g) Hurricane-U (3D)

# (h) RTM (3D)

Fig. 8. Rate distortion of different compressors

and combining both, as shown in Figure 11. The figure shows that AE+Lorenzo achieves the best quality at all bit rates since it can take advantage of both predictors adaptively.

# 4) Compression speeds and autoencoder training speeds:

The average compression speed of each error-bounded lossy compressor on all the tested datasets under the error bound of 1E-3 are shown in Table VIII in units of Mb/s (SZauto, SZinterp, and AE-B have speeds only on 3D data because they currently do not support 2D data). Because of the relatively high computation cost of neural networks, AE-SZ cannot achieve comparable compression throughput with traditional lossy compressors (its speed is about 10%-40% as fast as that of SZ2.1 and SZinterp). In fact, the current version of AE-SZ code is in the experimental stage, so it is not as optimized as the off-the-shelf compressors such as SZ2.1 and ZFP. We believe that with further optimization AE-SZ can be much accelerated. In fact, the throughput of AE-SZ outperforms AE-A [43] with similar.

Authorized licensed use limited to: TUFTS UNIV. Downloaded on June 22, 2025 at 18:23:20 UTC from IEEE Xplore. Restrictions apply.
---
# AE-SZ Rate Distortion Comparison

The best rate distortion results in 2D datasets. On 3D datasets, it obtains a much better rate distortion than SZ2.1 and ZFP do (about 100%∼800% improvement with the same PSNR). AE-SZ also exhibits very close rate distortions with those of SZinterp in high compression cases, demonstrating its great potential in error-bounded lossy compression.

- AE-SZ has a higher visual quality at the same compression ratio compared with SZauto, SZ2.1, and ZFP.
- AE-SZ is slower than SZ2.1 and ZFP, but is 30×∼200× faster than other autoencoder-based error-bounded lossy compressors.

In the future, we plan to improve AE-SZ in several ways, including (1) optimizing the network structure and the hyper-parameters of autoencoders in AE-SZ and (2) speeding up the compression and decompression speeds for AE-SZ.

# ACKNOWLEDGMENTS

This research was supported by the Exascale Computing Project (ECP), Project Number: 17-SC-20-SC, a collaborative effort of two DOE organizations – the Office of Science and the National Nuclear Security Administration, responsible for the planning and preparation of a capable exascale ecosystem, including software, applications, hardware, advanced system engineering and early testbed platforms, to support the nation’s exascale computing imperative. The material was supported by the U.S. Department of Energy, Office of Science, under contract DE-AC02-06CH11357, and supported by the National Science Foundation under Grant OAC-2003709 and OAC-2003624/2042084. We acknowledge the computing resources provided on Bebop (operated by Laboratory Computing Resource Center at Argonne) and on Theta and JLSE (operated by Argonne Leadership Computing Facility).

# TABLE VIII

# COMPRESSION/DECOMPRESSION SPEEDS (MB/S): ERROR BOUND=1E-3

| Type      | Dataset | SZ    | ZFP   | SZ    | SZ    | AE-SZ | AE-A \[43] | AE-B \[40] |
| --------- | ------- | ----- | ----- | ----- | ----- | ----- | ---------- | ---------- |
| Comp      | CESM    | 145.0 | 174.7 | N/A   | N/A   | 26.7  | 0.4        | N/A        |
| RTM       | 196.4   | 432.4 | 231.4 | 183.2 | 68.2  | 0.4   | 15.6       |            |
| Hurricane | 166.2   | 91.8  | 198.7 | 168.6 | 22.0  | 0.4   | 15.5       |            |
| NYX       | 142.9   | 171.7 | 152.9 | 99.6  | 15.7  | 0.5   | 16.0       |            |
| EXAFEL    | 162.3   | 213.7 | N/A   | N/A   | 12.2  | 0.4   | N/A        |            |
| Decomp    | CESM    | 249.7 | 222.7 | N/A   | N/A   | 58.0  | 0.6        | N/A        |
| RTM       | 430.3   | 941.3 | 516.1 | 452.6 | 135.3 | 0.7   | 14.1       |            |
| Hurricane | 264.9   | 243.1 | 328.6 | 357.4 | 48.2  | 0.6   | 14.7       |            |
| NYX       | 217.7   | 310.4 | 197.9 | 115.6 | 34.6  | 0.7   | 14.4       |            |
| EXAFEL    | 219.1   | 224.7 | N/A   | N/A   | 26.1  | 0.6   | N/A        |            |

# TABLE IX

# AUTOENCODER TRAINING TIME (IN HOURS)

| Dataset   | AE-SZ | AE-A \[43] |
| --------- | ----- | ---------- |
| CESM      | 1.0   | 1.5        |
| RTM       | 3.4   | 21.4       |
| NYX       | 5.5   | 4.7        |
| Hurricane | 2.4   | 2.5        |
| EXAFEL    | 2.2   | 3.5        |

# VI. CONCLUSION AND FUTURE WORK

In this paper we explored leveraging convolutional autoencoders to improve error-bounded lossy compression. To this end, we developed an efficient method called AE-SZ, by integrating autoencoders in the SZ compression model with a series of optimizations. We comprehensively evaluated AE-SZ by comparing it with six related works on five real-world simulation datasets, with the following key findings.

- AE-SZ is competitive in the low-bit-rate range (i.e., high-compression-ratio cases). Specifically, it exhibits.

# REFERENCES

1. SLAC National Accelerator Laboratory, “Linac coherent light source (lcls-ii),” https://lcls.slac.stanford.edu/, 2017, online.
2. T. E. Fornek, “Advanced Photon Source Upgrade Project preliminary design report,” 9 2017.
3. F. Cappello, S. Di, S. Li, X. Liang, G. M. Ali, D. Tao, C. Yoon Hong, X.-c. Wu, Y. Alexeev, and T. F. Chong, “Use cases of lossy compression for floating-point data in scientific datasets,” International Journal of High Performance Computing Applications (IJHPCA), vol. 33, pp. 1201–1220, 2019.
4. L. P. Deutsch, “GZIP file format specification version 4.3,” 1996.
5. Y. Collet, “Zstandard – real-time data compression algorithm,” http://facebook.github.io/zstd/, 2015.
6. Zlib, https://www.zlib.net/, online.
7. M. Burtscher and P. Ratanaworabhan, “FPC: A high-speed compressor for double-precision floating-point data,” IEEE Transactions on Computers, vol. 58, no. 1, pp. 18–31, Jan 2009.
8. S. Di and F. Cappello, “Fast error-bounded lossy HPC data compression with SZ,” in IEEE International Parallel and Distributed Processing Symposium, 2016, pp. 730–739.
9. D. Tao, S. Di, Z. Chen, and F. Cappello, “Significantly improving lossy compression for scientific data sets based on multidimensional prediction and error-controlled quantization,” in 2017 IEEE International Parallel and Distributed Processing Symposium. IEEE, 2017, pp. 1129–1139.
10. A. H. Baker, H. Xu, J. M. Dennis, M. N. Levy, D. Nychka, S. A. Mickelson, J. Edwards, M. Vertenstein, and A. Wegener, “A methodology for evaluating the impact of data compression on climate simulation data,” in Proceedings of the 23rd International Symposium on High-performance Parallel and Distributed Computing, ser. HPDC ’14. NY, USA: ACM, 2014, pp. 203–214.
---
# References

[11] N. Sasaki, K. Sato, T. Endo, and S. Matsuoka, “Exploration of lossy bounded lossy compression,” in Proceedings of the 28th International Symposium on High-Performance Parallel and Distributed Computing, ser. HPDC ’19. New York, NY, USA: ACM, 2019, pp. 159–170.

[12] A. H. Baker, H. Xu, D. M. Hammerling, S. Li, and J. P. Clyne, “Toward a multi-method approach: Lossy data compression for climate simulation data,” in High Performance Computing. Springer International Publishing, 2017, pp. 30–42.

[13] P. Lindstrom, “Fixed-rate compressed floating-point arrays,” IEEE transactions on visualization and computer graphics, vol. 20, no. 12, pp. 2674–2683, 2014.

[14] K. Zhao et al., “Significantly improving lossy compression for HPC datasets with second-order prediction and parameter optimization,” in Proceedings of the 29th International Symposium on High-Performance Parallel and Distributed Computing, ser. HPDC ’20, 2020, pp. 89–100.

[15] X. Liang, S. Di, D. Tao, S. Li, S. Li, H. Guo, Z. Chen, and F. Cappello, “Error-controlled lossy compression optimized for high compression ratios of scientific datasets,” in 2018 IEEE International Conference on Big Data. IEEE, 2018.

[16] S. W. Son, Z. Chen, W. Hendrix, A. Agrawal, W.-k. Liao, and A. Choudhary, “Data compression for the exascale computing era- survey,” Supercomputing frontiers and innovations, vol. 1, no. 2, pp. 76–88, 2014.

[17] P. Lindstrom and M. Isenburg, “Fast and efficient compression of floating-point data,” IEEE transactions on visualization and computer graphics, vol. 12, no. 5, pp. 1245–1250, 2006.

[18] J. Tian, S. Di, C. Zhang, X. Liang, S. Jin, D. Cheng, D. Tao, and F. Cappello, “Wavesz: A hardware-algorithm co-design of efficient lossy compression for scientific data,” in Proceedings of the 25th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming, ser. PPoPP ’20. New York, NY, USA: Association for Computing Machinery, 2020, p. 74–88.

[19] M. Ainsworth, O. Tugluk, B. Whitney, and S. Klasky, “Multilevel techniques for compression and reduction of scientific data—the multivariate case,” SIAM Journal on Scientific Computing, vol. 41, no. 2, pp. A1278–A1303, 2019.

[20] S. Li, N. Marsaglia, V. Chen, C. M. Sewell, J. P. Clyne, and H. Childs, “Achieving portable performance for wavelet compression using data parallel primitives,” in EGPGV, 2017, pp. 73–81.

[21] X. Delaunay, A. Courtois, and F. Gouillon, “Evaluation of lossless and lossy algorithms for the compression of scientific datasets in netCDF-4 or HDF5 files,” Geoscientific Model Development, vol. 12, no. 9, pp. 4099–4113, 2019.

[22] C. S. Zender, “Bit grooming: statistically accurate precision-preserving quantization with compression, evaluated in the netCDF Operators (NCO, v4. 4.8+),” Geoscientific Model Development, vol. 9, no. 9, pp. 3199–3211, 2016.

[23] S. Lakshminarasimhan, N. Shah, S. Ethier, S.-H. Ku, C.-S. Chang, S. Klasky, R. Latham, R. Ross, and N. F. Samatova, “ISABELA for effective in situ compression of scientific data,” Concurrency and Computation: Practice and Experience, vol. 25, no. 4, pp. 524–540, 2013.

[24] X.-C. Wu, S. Di, E. M. Dasgupta, F. Cappello, H. Finkel, Y. Alexeev, and F. T. Chong, “Full-state quantum circuit simulation by using data compression,” in Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, ser. SC ’19. New York, NY, USA: Association for Computing Machinery, 2019.

[25] A. M. Gok, S. Di, Y. Alexeev, D. Tao, V. Mironov, X. Liang, and F. Cappello, “PaSTRI: Error-bounded lossy compression for two-electron integrals in quantum chemistry,” in 2018 IEEE International Conference on Cluster Computing (CLUSTER), Sep. 2018, pp. 1–11.

[26] D. Tao, S. Di, X. Liang, Z. Chen, and F. Cappello, “Fixed-psnr lossy compression for scientific data,” in 2018 IEEE International Conference on Cluster Computing (CLUSTER), 2018, pp. 314–318.

[27] J. Tian et al., “CuSZ: An efficient gpu-based error-bounded lossy compression framework for scientific data,” in Proceedings of the ACM International Conference on Parallel Architectures and Compilation Techniques, ser. PACT ’20, 2020, p. 3–15.

[28] D. Tao, S. Di, Z. Chen, and F. Cappello, “In-depth exploration of single-snapshot lossy compression techniques for n-body simulations,” in 2017 IEEE International Conference on Big Data (Big Data), 2017, pp. 486–493.

[29] S. Jin, S. Di, X. Liang, J. Tian, D. Tao, and F. Cappello, “Deepsz: A novel framework to compress deep neural networks by using error-controlled lossy compression,” 2019.
---
# References

1. Association Annual Summit and Conference (APSIPA ASC). IEEE, 2019, pp. 53–57.
2. D. P. Kingma and M. Welling, “Auto-encoding variational bayes,” arXiv preprint arXiv:1312.6114, 2013.
3. I. Higgins, L. Matthey, A. Pal, C. Burgess, X. Glorot, M. Botvinick, S. Mohamed, and A. Lerchner, “beta-VAE: Learning basic visual concepts with a constrained variational framework,” 2016.
4. A. Kumar, P. Sattigeri, and A. Balakrishnan, “Variational inference of disentangled latent concepts from unlabeled observations,” 2018.
5. S. Zhao, J. Song, and S. Ermon, “InfoVAE: Information maximizing variational autoencoders,” 2018.
6. P. Chen, G. Chen, and S. Zhang, “Log hyperbolic cosine loss improves variational auto-encoder,” 2018.
7. K. Zhao, S. Di, X. Lian, S. Li, D. Tao, J. Bessac, Z. Chen, and F. Cappello, “SDRBench: Scientific data reduction benchmark for lossy compressors,” in 2020 IEEE International Conference on Big Data (Big Data), 2020, pp. 2716–2724.
8. J. E. Kay and et al., “The Community Earth System Model (CESM) large ensemble project: A community resource for studying climate change in the presence of internal climate variability,” Bulletin of the American Meteorological Society, vol. 96, no. 8, pp. 1333–1349, 2015.
9. S. Kayum et al., “GeoDRIVE – a high performance computing flexible platform for seismic applications,” First Break, vol. 38, no. 2, pp. 97–100, 2020.
10. NYX simulation, https://amrex-astro.github.io/Nyx, 2019, online.
11. Hurricane ISABEL simulation data, http://vis.computer.org/vis2004contest/data.html, 2004, online.
12. E. project, https://www.exascaleproject.org/project/exafel-data-analytics-exascale-free-electron-lasers/, 2019, online.
13. F. Cappello, S. Di, S. Li, X. Liang, A. M. Gok, D. Tao, C. H. Yoon, X.-C. Wu, Y. Alexeev, and F. T. Chong, “Use cases of lossy compression for floating-point data in scientific data sets,” The International Journal of High Performance Computing Applications, vol. 33, no. 6, pp. 1201–1220, 2019.
14. X. Liang, S. Di, S. Li, D. Tao, B. Nicolae, Z. Chen, and F. Cappello, “Significantly improving lossy compression quality based on an optimized hybrid prediction model,” in Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, 2019, pp. 1–26.

Authorized licensed use limited to: TUFTS UNIV. Downloaded on June 22, 2025 at 18:23:20 UTC from IEEE Xplore. Restrictions apply.