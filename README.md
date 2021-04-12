# DADIP_code
## Introduction

In the problem of image deblurring, the restoration of details in severely blurred images has always been difficult. In this paper, we focus on effectively eliminating the ringing artifact and wrinkles that appear after deburring, and propose a novel blind debluring method based on dual attentional deep image prior (DADIP) network and 2-dimensional (2D) blur kernel estimation with convolutional neural network (CNN). In the DADIP network, the dual attention mechanism is firstly com- bined with squeeze and excitation network (SENet), which greatly improves the restoration effect of image details. More importantly, the 2D blur kernel estimation approach via CNN is developed to suppress the ringing artifact of the image, which significantly outperforms previous fully connected net- work based methods. Experiments show that our deblurring approach achieves superior performance compared with most existing methods.

## Requirments

- Python 3.6, PyTorch >= 0.4

- Requirements: opencv-python, tqdm

- GPU: 12GB at least for color images

  ​			3GB at least for greyscale images

- MATLAB

## Dataset

Url：https://share.weiyun.com/6mY1JRdv 

pw：mqw39a



## Demo

![cmp1](/Users/sheldon-t/Desktop/DADIP_code/images/cmp1.gif)

![cmp2](/Users/sheldon-t/Desktop/DADIP_code/images/cmp2.gif)

