# Learning Spatial-Temporal Regularized Correlation Filters for Visual Tracking (CVPR 2018)

Matlab implementation of our Spatial-Temporal Regularized Correlation Filters (STRCF) tracker.

# Abstract 
Discriminative Correlation Filters (DCF) are efficient in visual tracking but suffer from unwanted boundary effects.
Spatially Regularized DCF (SRDCF) has been suggested to resolve this issue by enforcing spatial penalty on DCF coefficients, which, inevitably, improves the tracking performance at the price of increasing complexity.
To tackle online updating, SRDCF formulates its model on multiple training images, further adding difficulties in improving efficiency.
In this work, by introducing temporal regularization to SRDCF with single sample, we present our spatial-temporal regularized correlation filters (STRCF).
Motivated by online Passive-Agressive (PA) algorithm, we introduce the temporal regularization to SRDCF with single sample, thus resulting in our spatial-temporal regularized correlation filters (STRCF).
The STRCF formulation can not only serve as a reasonable approximation to SRDCF with multiple training samples, but also provide a more robust appearance model than SRDCF in the case of large appearance variations.Besides, it can be efficiently solved via the alternating direction method of multipliers (ADMM).
By incorporating both temporal and spatial regularization, our STRCF can handle boundary effects without much loss in efficiency and achieve superior performance over SRDCF in terms of accuracy and speed.
Experiments are conducted on three benchmark datasets: OTB-2015, Temple-Color, and VOT-2016.
Compared with SRDCF, STRCF with hand-crafted features provides a ~5 times speedup and achieves a gain of 5.4% and 3.6% AUC score on OTB-2015 and Temple-Color, respectively. Moreover, STRCF combined with CNN features also performs favorably against state-of-the-art CNN-based trackers and achieves an AUC score of 68.3% on OTB-2015.

# Publication

Details about the STRCF tracker can be found in our CVPR 2018 paper:

Feng Li, Cheng Tian, Wangmeng Zuo, Lei Zhang and Ming-Hsuan Yang.  
Learning Spatial-Temporal Regularized Correlation Filters for Visual Tracking.</br>
In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018. 

The paper link is: https://arxiv.org/abs/1803.08679

Please cite the above publication if you find STRCF useful in your research. The bibtex entry is:

@Inproceedings{Li2018STRCF,  
  title={Learning Spatial-Temporal Regularized Correlation Filters for Visual Tracking},  
  author={Li, Feng and Tian, Cheng and Zuo, Wangmeng and Zhang, Lei and Yang, Ming Hsuan},  
  booktitle={CVPR},  
  year={2018},  
}

# Contact

Feng Li

Email: fengli_hit@hotmail.com

# Installation

### Using git clone

1. Clone the GIT repository:

   $ git clone https://github.com/lifeng9472/STRCF.git

2. Clone the submodules.  
   In the repository directory, run the commands:

   $ git submodule init  
   $ git submodule update

3. Start Matlab and navigate to the repository.  
   Run the install script:

   |>> install

4. Run the demo script to test the tracker:

   |>> demo_STRCF


**Note**:
This package requires matconvnet [1], if you want to use CNN features, and PDollar Toolbox [2], if you want to use HOG features. Both these externals are included as git submodules and should be installed by following step 2. above.

## Description and Instructions

### How to run

The files in root directory are used to run the tracker in OTB and Temple-Color datasets.

These files are included:

* run_STRCF.m  -  runfile for the STRCF tracker with hand-crafted features (i.e., HOG+CN).

* run_DeepSTRCF.m  -  runfile for the DeepSTRCF tracker with CNN features.

Tracking performance on the OTB-2015, Temple-Color is given as follows,

<div align="center">
    <img src="https://github.com/lifeng9472/STRCF/blob/master/results/OTB2015-HF.jpg" width="270px" alt="Hand-crafted features on OTB-2015"><img src="https://github.com/lifeng9472/STRCF/blob/master/results/OTB2015-DF.jpg" width="270px" alt="CNN features on OTB-2015"><img src="https://github.com/lifeng9472/STRCF/blob/master/results/Temple-Color.jpg" width="270px" alt="all features on Temple-Color">
 </div>   


Results on the VOT-2016 dataset are also provided.

* tracker_DeepSTRCF.m -  this file integrates the tracker into the VOT-2016 toolkit.

**Note**:

To run the tracker on VOT-2016 dataset, two things need to be taken:

1. Change the location of the pre-trained CNN with `absolute path` rather than the `relative path` in feature_extraction/load_CNN.m. 

2. Change the location of the STRCF tracker in tracker_DeepSTRCF.m.

|               | ECO  | SRDCF| SRDCFDecon| BACF | DeepSRDCF | ECO-HC | STRCF | DeepSTRCF|
| :-----------: |:----:|:----:|:---------:|:----:|:---------:|:------:| -----:|:--------:|
|      EAO      | 0.375| 0.247|   0.262   | 0.223|  0.276    |  0.322 | 0.279 |  0.313   |
|   Accuracy    | 0.53 | 0.52 |   0.53    | 0.56 |  0.51     |  0.54  | 0.53  |  0.55    |
|  Robustness   | 0.73 | 1.5  |   1.42    | 1.88 |  1.17     |  1.08  | 1.32  |  0.92    |

### Features

1. Deep CNN features. It uses matconvnet [1], which is included as a git submodule in external_libs/matconvnet/. The `imagenet-vgg-m-2048` network available at http://www.vlfeat.org/matconvnet/pretrained/ was used. You can try other networks, by placing them in the feature_extraction/networks/ folder.

2. HOG features. It uses the PDollar Toolbox [2], which is included as a git submodule in external_libs/pdollar_toolbox/.

3. Lookup table features. These are implemented as a lookup table that directly maps an RGB or grayscale value to a feature vector.

4. Colorspace features. Currently grayscale and RGB are implemented.

## Acknowledgements

We thank for Dr. `Martin Danelljan` and  `Hamed Kiani` for their valuable help on our work. In this work,
we have borrowed the feature extraction modules from the ECO tracker (https://github.com/martin-danelljan/ECO) and the parameter settings from BACF (www.hamedkiani.com/bacf.html).

## References

[1] Webpage: http://www.vlfeat.org/matconvnet/  
    GitHub: https://github.com/vlfeat/matconvnet

[2] Piotr Dollár.  
    "Piotr’s Image and Video Matlab Toolbox (PMT)."  
    Webpage: https://pdollar.github.io/toolbox/  
    GitHub: https://github.com/pdollar/toolbox  
