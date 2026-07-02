<h2 align="center"> <a href="https://arxiv.org/abs/2410.16707">DI-MaskDINO: A Joint Object Detection and Instance Segmentation Model</a></h2>
<h4 align="center" color="A0A0A0"> Zhixiong Nan, Xianghong Li, Tao Xiang*, Jifeng Dai</h4>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>

<div align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2410.16707-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2410.16707)
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/CQU-ADHRI-Lab/DI-MaskDINO)
[![License](https://img.shields.io/badge/Code%20License-Apache2.0-yellow)](https://github.com/CQU-ADHRI-Lab/DI-MaskDINO/blob/main/LICENSE)

</div>

<div align=center>
<img src="figures/results.gif" width="960px">
</div>



# DI-MaskDINO
This is the official implementation of the paper "DI-MaskDINO: A Joint Object Detection and Instance Segmentation Model".

<div align="center">
  <img src="figures/framework.png"/>
</div><br/>

Motivated by an interesting phenomenon that _the performance of object detection lags behind that of instance segmentation at the beignning decoder layer_, **DI-MaskDINO** is proposed to alleviate the detection-segmentation imbalance issue.
**DI-MaskDINO** is implemented by configuring proposed **_De-Imbalance (DI)_** module and **_Balance-Aware Tokens Optimization (BATO)_** module to MaskDINO.
**_DI_** module contributes to strengthen the detection task at the beginning decoder layer to balance the performance of the two tasks, and the core of **_DI_** module is _residual double-selection mechanism_.
**DI-MaskDINO** outperforms existing SOTA joint object detection and instance segmentation model MaskDINO (**+1.2** _AP<sup> box</sup>_ and **+0.9** _AP<sup> mask</sup>_ on COCO, using ResNet50 backbone with 12 training epochs), SOTA object detection model DINO (**+1.0** _AP<sup> box</sup>_ on COCO), and SOTA segmentation model Mask2Former(**+3.0** _AP<sup> mask</sup>_ on COCO).

## Update
[2024/12] Code for [DI-MaskDINO](https://github.com/CQU-ADHRI-Lab/DI-MaskDINO) is available here!

[2024/9] DI-MaskDINO has been accepted at NeurIPS 2024 as a poster!

## Installation

Please install PyTorch and [detectron2](https://github.com/facebookresearch/detectron2) first — see detectron2's [installation instructions](https://detectron2.readthedocs.io/en/latest/tutorials/install.html). A CUDA toolkit matching your GPU (`CUDA_HOME` set, or `FORCE_CUDA=1`) must be available at install time, since installing this package compiles the `MultiScaleDeformableAttention` CUDA extension.

Once PyTorch and detectron2 are installed:

```bash
pip install git+https://github.com/Enodo-Robotics/DI-MaskDINO.git
```

This builds and installs the `dimaskdino` package together with its CUDA extension in one step. To install from a local checkout instead (e.g. for development):

```bash
git clone https://github.com/Enodo-Robotics/DI-MaskDINO.git
cd DI-MaskDINO
pip install .
# or `pip install -e .` for an editable install
```

## Models

<table style="width: 100%; border-collapse: collapse;"  id="model-table">
    <tr style="border: 1px solid black; background-color: #f2f2f2; text-align: center; padding: 8px;">
        <th align="center">Name</th>
        <th align="center">Backbone</th>
        <th align="center">Epochs</th>
        <th align="center"><i>AP<sup style="font-size: smaller;">box</sup></i></th>
        <th align="center"><i>AP<sup style="font-size: smaller;">mask</sup></i></th>
        <th align="center">Download</th>
    </tr>
    <tr align="center">
        <td align="center"><a href="./configs/dimaskdino_r50_4scale_bs16_12ep.yaml" style="text-decoration: none; color: black;">DI-MaskDINO</a></td>
        <td align="center">ResNet50</td>
        <td align="center">12</td>
        <td align="center">46.9</td>
        <td align="center">42.3</td>
        <td align="center"><a href="https://drive.google.com/file/d/1oQociMQSt_jmQtH3pm92i6zat8Gt83Oh/view?usp=drive_link" style="text-decoration: none; color: blue;">model</a></td>
    </tr>
    <tr align="center">
        <td align="center"><a href="./configs/dimaskdino_r50_4scale_bs16_24ep.yaml" style="text-decoration: none; color: black;">DI-MaskDINO</a></td>
        <td align="center">ResNet50</td>
        <td align="center">24</td>
        <td align="center">49.6</td>
        <td align="center">44.8</td>
        <td align="center"><a href="https://drive.google.com/file/d/1FjoYiwGnrk_bk0nUM7GPeIMGENVLdx7n/view?usp=drive_link" style="text-decoration: none; color: blue;">model</a></td>
    </tr>
    <tr align="center">
        <td align="center"><a href="./configs/dimaskdino_r50_4scale_bs16_50ep.yaml" style="text-decoration: none; color: black;">DI-MaskDINO</a></td>
        <td align="center">ResNet50</td>
        <td align="center">50</td>
        <td align="center">51.9</td>
        <td align="center">46.7</td>
        <td align="center"><a href="https://drive.google.com/file/d/17_SErWYBWWCdYfR6i7OJtcBIlCDES10z/view?usp=drive_link" style="text-decoration: none; color: blue;">model</a></td>
    </tr>
</table>

## Run

### Training

Train DI-MaskDINO with 8 GPUs:

```sh
python train_net.py --num-gpus 8 --config-file configs/dimaskdino_r50_4scale_bs16_12ep.yaml OUTPUT_DIR /path/to/output
```

### Evaluation

You can download our pretrained models and evaluate them with the following commands. 
```sh
python train_net.py --eval-only --num-gpus 8 --config-file /path/to/config_file MODEL.WEIGHTS /path/to/checkpoint_file
```
For example, to reproduce our result, you can copy the config path from the model table, download the pretrained checkpoint into `/path/to/checkpoint_file`, and run 
```sh
python train_net.py --eval-only --num-gpus 8 --config-file configs/dimaskdino_r50_4scale_bs16_12ep.yaml MODEL.WEIGHTS /path/to/checkpoint_file
```


## <a name="CitingDIMaskDINO"></a>Citing DI-MaskDINO

If you find our work helpful for your research, please consider citing the following BibTeX entry.

```BibTeX
@inproceedings{nan2024di,
  title={DI-MaskDINO: A Joint Object Detection and Instance Segmentation Model}, 
  author={Zhixiong Nan and Xianghong Li and Tao Xiang and Jifeng Dai},
  booktitle={Proceedings of the Neural Information Processing Systems},
  year={2024}
}
```

## Acknowledgement

Many thanks to these excellent opensource projects: 
* [MaskDINO](https://github.com/IDEA-Research/MaskDINO)
* [Mask2Former](https://github.com/facebookresearch/Mask2Former) 
* [DINO](https://github.com/IDEA-Research/DINO)
