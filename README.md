
# CentralFormer
## [CentralFormer: Centralized Spectral-Spatial Transformer for Hyperspectral Image Classification with Adaptive Relevance Estimation and Circular Pooling](https://ieeexplore.ieee.org/document/10772042)  
Authors: Ningyang Li, Zhaohui Wang, Faouzi Alaya Cheikh, Lei Wang  
Journal: IEEE Transactions on Geoscience and Remote Sensing (TGRS)  
Environment: Python 3.9.16, Tensorflow 2.6.0, Keras 2.6.0.  


- To train and test the CentralFormer model, you need to download the prepared samples ahead and run "train_CentralFormer.py" file.
- Samples can be acquired from **[Google Drive](https://drive.google.com/drive/folders/1Htr4jgtJyRT24VSbVbg2jED7kXAYUGqV?usp=drive_link)** and **[Baidu Disk](https://pan.baidu.com/s/1G6ktXC-EKGUVdvJJ-2xWkw)** code: ntnu.
- Model weights are provided in "weights" folder.
- Classification results have been recorded in "results" folder.
- Be sure to set your own "project folder" before training.
- An independent Circular Pooling (CP) class file has been built, you can call and integrate it to other baselines flexibly.



## Abstract
Classification of hyperspectral image (HSI) is a hotspot in the field of remote sensing. Recent deep learning-based approaches, especially for the transformer architectures, have been investigated to extract the deep spectral-spatial features. However, the ability of these approaches to efficiently represent the crucial attention patterns and distinguishing features suffers from the neglect of the relevant areas, including the center pixel, and high computational complexity, thereby their classification performances still need to be improved. This article proposes a centralized spectral-spatial transformer (CentralFormer), which contains the central encoder, adaptive relevance estimation (ARE) module, and cross-encoder relevance fusion (CERF) module, for HSI classification. To recognize the relevant areas, the ARE modules access both spectral and spatial associations between the center pixel and its neighborhoods flexibly. By focusing on these areas and emphasizing them during attention inference, the central encoders can extract the key attention modes and discriminating features effectively. Moreover, the CERF modules are deployed to prevent the reliability of relevance map from being harmed by the feature deviation between encoders. To handle the high computational occupancy, a novel circular pooling strategy reduces the circles and bands of features. Unlike regular pooling methods, it can well improve the relevant characteristics for subsequent encoders. By integrating these techniques, the CentralFormer model can represent the discriminating spectralspatial features efficiently for HSI classification. Experimental results on four classic HSI data sets reveal the proposed CentralFormer model outperforms the state-of-the-arts in terms of both classification accuracy and computational efficiency. 

## Contributions
1. A central encoder, which contains a spectral-spatial local representation (SSLR) module, a centralized multi-head attention (CMHA) module, a feature fusion part (FFP), and a CP layer, is designed to deduce the relevant attention modes and acquire more effective spectral-spatial features. The SSLR module aims to represent the local spectral dependency and spatial structures. The CMHA module processes the relevant areas as the key set and emphasizes the dot-product similarity of them via 1D Gaussian function, thereby providing the important attention modes and global features. After the feature fusion, a novel CP strategy, which regards an HSI cube as the center pixel and several circles, reduces the circles and bands of features. Unlike regular pooling ways, it protects the characteristics of relevant areas saliently, which is the core to improve computational complexity and subsequent features.
2. To promote the involvement of relevant areas in attention deduction and feature extraction, an ARE module easy to be optimized is developed to discover the pixels relevant to the center pixel. Both spectral angle distances and discrete Gaussian distances between the center pixel and its neighborhoods are integrated adaptively during this process.
3. A CERF module is built to relieve the negative influence of network depth on the reliability of relevance maps. It can fuse current relevance map and previous one adaptively to take
into full account their relative importance. Before fusing, the CP layer is also utilized to adjust the size of previous relevance map and maintain its relevant peculiarity.
4. An end-to-end CentralFormer model is proposed for HSI classification. Compared with other methods, especially for the transformer architectures, it excels at inferring the critical
attention modes in connection with the relevant areas and extracting the discriminating spectral-spatial features efficiently.

<img src="https://github.com/ningyang-li/CentralFormer/blob/914b3f1a8f5bfb3c633aefb88afabeb01fc5830f/pic/overview.png" width="1000" />

<img src="https://github.com/ningyang-li/CentralFormer/blob/914b3f1a8f5bfb3c633aefb88afabeb01fc5830f/pic/CMHA.png" width="1000" />

<img src="https://github.com/ningyang-li/CentralFormer/blob/914b3f1a8f5bfb3c633aefb88afabeb01fc5830f/pic/CP.png" width="500" />

<img src="https://github.com/ningyang-li/CentralFormer/blob/914b3f1a8f5bfb3c633aefb88afabeb01fc5830f/pic/ARE.png" width="500" />

<img src="https://github.com/ningyang-li/CentralFormer/blob/914b3f1a8f5bfb3c633aefb88afabeb01fc5830f/pic/CERF.png" width="500" />


## Citation
```
N. Li, Z. Wang, F. A. Cheikh and L. Wang, "CentralFormer: Centralized Spectral-Spatial Transformer for Hyperspectral Image Classification with Adaptive Relevance Estimation and Circular Pooling," in IEEE Transactions on Geoscience and Remote Sensing, doi: 10.1109/TGRS.2024.3509455.
```


```
@ARTICLE{10772042,
author={Li, Ningyang and Wang, Zhaohui and Cheikh, Faouzi Alaya and Wang, Lei},
journal={IEEE Transactions on Geoscience and Remote Sensing}, 
title={CentralFormer: Centralized Spectral-Spatial Transformer for Hyperspectral Image Classification with Adaptive Relevance Estimation and Circular Pooling}, 
year={2024},
volume={},
number={},
pages={1-1},
keywords={Feature extraction;Transformers;Computer architecture;Hyperspectral imaging;Correlation;Computational modeling;Computational complexity;Accuracy;Kernel;Image classification;Hyperspectral image classification;transformer;center pixel;relevant area;circular pooling;attention mechanism},
doi={10.1109/TGRS.2024.3509455}}
```

<h4>Here we are so grateful to all reviewers and editors for their constructive comments.</h4>

<h4>If you have some problems when implement our code, please feel free to contact us :relaxed:</h4>

<h4>Thanks for your attention.</h4>
