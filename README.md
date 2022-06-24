# Low-light-Image-Enhancement

A Python implementation for the following papers:

 * Dual Illumination Estimation for Robust Exposure Correction [[link](https://arxiv.org/pdf/1910.13688.pdf)]
 * LIME: Low-light Image Enhancement via Illumination Map Estimation [[link](https://ieeexplore.ieee.org/document/7782813)]

## Installation
This implementation runs on python >= 3.7, use pip to install dependencies:
```shell
pip install -r requirements.txt
```

You can also use conda to install dependencies:

```shell
conda install -file requirements.txt
```

## Usage

Go to ```enhance.py```,adjust your filepath and type the following command in your shell:

```shell
python enhance.py
```

## Result

Here are two examples for LIME:

| Low light image  | Enhanced image with LIME  |
|:----------------:|:-------------------------:|
| ![](pics/2.jpg)  | ![](pics/LIME_2.jpg)  |
| ![](pics/wx.jpg) | ![](pics/LIME_wx.jpg) |

**DUAL implementation remains to be solved!**

## Thanks

Special thanks for the writers of the papers.

Also thanks for the explanations given here:

 * Low-light Image Enhancement (Bhavya Vasudeva, Puneesh Deora) [[link](https://drive.google.com/file/d/1aph-GUsr_Br2dMLTR3e0kYqAM5aThmj1/view)]
