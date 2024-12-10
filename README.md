# Deblur4DGS
PyTorch implementation of [**Deblur4DGS: 4D Gaussian Splatting from Blurry Monocular Video**](https://arxiv.org/pdf/2412.06424)


[![arXiv](https://img.shields.io/badge/arXiv-2404.04908-b10.svg)](https://arxiv.org/pdf/2412.06424)
[![Project](https://img.shields.io/badge/Project-Website-orange)](https://deblur4dgs.github.io/)
![visitors](https://visitor-badge.laobi.icu/badge?page_id=ZcsrenlongZ.Deblur4DGS)

## TODO
-  Code is comming soon (About two weeks).



## 1. Abstract
Recent 4D reconstruction methods have yielded impressive results but rely on sharp videos as supervision. However, motion blur often occurs in videos due to camera shake and object movement, while existing methods render blurry results when using such videos for reconstructing 4D models.
Although a few NeRF-based approaches attempted to address the problem, they struggled to produce high-quality results, due to the inaccuracy in estimating continuous dynamic representations within the exposure time. 
Encouraged by recent works in 3D motion trajectory modeling using 3D Gaussian Splatting (3DGS), we suggest taking 3DGS as the scene representation manner, and propose the first 4D Gaussian Splatting framework to reconstruct a high-quality 4D model from blurry monocular video, named Deblur4DGS. 
Specifically, we transform continuous dynamic representations estimation within an exposure time into the exposure time estimation. 
Moreover, we introduce exposure regularization to avoid trivial solutions, as well as multi-frame and multi-resolution consistency ones to alleviate artifacts. Furthermore, to better represent objects with large motion, we suggest blur-aware variable canonical Gaussians. 
Beyond novel-view synthesis, Deblur4DGS can be applied to improve blurry video from multiple perspectives, including deblurring, frame interpolation, and video stabilization. 
Extensive experiments on the above four tasks show that Deblur4DGS outperforms state-of-the-art 4D reconstruction methods.


## 2. Method

<p align="center"><img src="./figures/deblur4dgs.png" width="95%"></p>

<p>  Training and Rendering of Deblur4DGS. Deblur4DGS produces the sharp image with user-provided timestamp and camera pose.</p>

## 3. Novel-View Synthesis Results
<img src="./figures/novel_vis.png" alt="Example 1" style="width:100%;">

Compared with state-of-the-art 4D reconstruction methods, Deblur4DGS produces more visually pleasant results in both static and dynamic areas, as marked with yellow and red boxes respectively.

## Acknowledgement

Special thanks to the following awesome projects!

- [Shape-of-Motion](https://github.com/vye16/shape-of-motion)
- [BAD-Gaussian](https://github.com/WU-CVGL/BAD-Gaussians)

## Citation
If you make use of our work, please cite our paper.
```bibtex
@article{Deblur4DGS,
  title={Deblur4DGS: 4D Gaussian Splatting from Blurry Monocular Video},
  author={Wu, Renlong and Zhang, Zhilu and Chen, Mingyang and Fan, Xiaopeng and Yan, Zifei, and Zuo, Wangmeng},
  journal={arXiv preprint arXiv:2412.06424},
  year={2024}
}
```