## SPIGA: Shape Preserving Facial Landmarks with Graph Attention Networks. (Pytorch Train Code Implementation)

[![Project Page](https://badgen.net/badge/color/Project%20Page/purple?icon=atom&label)](https://bmvc2022.mpi-inf.mpg.de/155/)
[![arXiv](https://img.shields.io/badge/arXiv-2210.07233-b31b1b.svg)](https://arxiv.org/abs/2210.07233)
[![PyPI version](https://badge.fury.io/py/spiga.svg)](https://badge.fury.io/py/spiga)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)

This repository contains the train code of **SPIGA, a face alignment and headpose estimator**.

***Note:*** The original repository of this project is from [SPIGA](https://github.com/andresprados/SPIGA). I add the simple train section to train my own model. I followed the original text and proceeded with the training process, but it could be very different from the actual way the author did. Please check the original repository and paper for more information.

## Training Process

The training process was conducted into three stpes according to paper.
Step 1. Train the backbone (HG network).
Step 2. Train the pose estimation networks and the backbone.
Step 3. Freeze the backbone and the pose estimation. Train the GAT model.
***Note:*** The training code can be found in 'build/' folder.

## Setup
I tested this repository on Windows 11 with CUDA 11.7, the latest version of cuDNN, python 3.9. It is noted that the original repository has been tested on Ubuntu 20.04 with CUDA 11.4, the latest version of cuDNN, Python 3.8 and Pytorch 1.12.1.

**Models:** By default, model weights are automatically downloaded on demand and stored at ```./spiga/models/weights/```.
You can also download them from [Google Drive](https://drive.google.com/drive/folders/1olrkoiDNK_NUCscaG9BbO3qsussbDi7I?usp=sharing). 

***Note:*** All the callable files provide a detailed parser that describes the behaviour of the program and their inputs.



<div align="center">


</div>


<p align="center">

<img src="https://raw.githubusercontent.com/andresprados/SPIGA/main/assets/demo.gif" width=250px height=250px>
&nbsp;&nbsp;&nbsp;

</p>

## Acknowledgements

This repository borrows code from [face alignment](https://github.com/1adrianb/face-alignment) and [SPIGA](https://github.com/andresprados/SPIGA)

## BibTeX Citation
```
@inproceedings{Prados-Torreblanca_2022_BMVC,
  author    = {Andrés  Prados-Torreblanca and José M Buenaposada and Luis Baumela},
  title     = {Shape Preserving Facial Landmarks with Graph Attention Networks},
  booktitle = {33rd British Machine Vision Conference 2022, {BMVC} 2022, London, UK, November 21-24, 2022},
  publisher = {{BMVA} Press},
  year      = {2022},
  url       = {https://bmvc2022.mpi-inf.mpg.de/0155.pdf}
}
```


