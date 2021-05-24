<!-- <img src='imgs/teaser_720.gif' align="right" width=360> -->

<!-- <br><br><br><br> -->

# Robust pose transfer with dynamic details using neural video rendering framwork
### [Video](https://www.bilibili.com/video/BV1y64y1C7ge/) | [Paper(comming soon)]()<br>
Pytorch implementation of our method for high-fidelity human video generation from 2D/2D+3D pose consequence. <br><br>

## pose transfer with dynamic details
- rendering results compared with static texture rendering
<p align='center'>  
  <img src='imgs/output.gif' width='600'/>
</p>

- refine the background during the training
<p align='center'>  
  <img src='imgs/background.gif' width='600'/>
</p>

- pose transfer result
<p align='center'>  
  <img src='imgs/result1.gif' width='600'/>
</p>

<p align='center'>  
  <img src='imgs/result2.gif' width='600'/>
</p>

- background replacing result
<p align='center'>  
  <img src='imgs/result3.gif' width='600'/>
</p>

## Prerequisites
- Linux or macOS
- Python 3
- NVIDIA GPU (11G memory or larger) + CUDA cuDNN

## Getting Started
### Installation
```bash
pip install -r requirement.txt
```

### Testing
- We also provide some keypoint jsons and a pre-trained model for test. We provide the checkpoint `dance15_18Feature_Temporal` and driving poses in `keypoints`.
```bash
bash test_start/start.sh
```
Note that the `pose_tgt_path`(keypoint json of the target person) is also needed to align the keypoints of different persons.

### Dataset preparation
The data used in this code can be any single-person video with 2K~4K frames. It should be organized like this:
```
dataName
├── bg.jpg
├── dataName (this is video frames folder)
├── densepose
├── flow
├── flow_inv
├── LaplaceProj (this is the 3D pose label)
├── mask
├── openpose_json
└── texture.jpg
```
The bg.jpg is the background image extracted with [inpainted apporach](https://github.com/JiahuiYu/generative_inpainting) from the video (or your can even use a common video frame). It will update during the training process.

The densepose is extracted by the [Densepose](https://github.com/facebookresearch/detectron2/tree/master/projects/DensePose).

The flow and flow_inv is the forward and backward flow calclulated by [FlowNet2](https://github.com/NVIDIA/flownet2-pytorch).

The LaplaceProj contains the 3D pose labels explained in this [paper](https://arxiv.org/abs/2003.13510). It's optional in our approach.

The mask is extracted by [Human-Segmentation-PyTorch](https://github.com/thuyngch/Human-Segmentation-PyTorch). The predicted mask will also refine automatically and becomes more accuarte than this "ground truth".

The openpose_json is extracted by Openpose[https://github.com/CMU-Perceptual-Computing-Lab/openpose].

The texture.jpg is the initial texture calculated from densepose results and video frames by executing `python3 unfold_texture.py $video_frame_dir $densepose_dir`.


### Training
Our approach needs to pre-train the UV generator at start.  This training is person-agnostic. One can use our pre-trained checkpoint `uvGenerator_pretrain_new` or train on any speficied dataset.

- Pre-train of UV generator (optional):
```bash
bash pretrainTrans.sh
```
The data required for pretrain includeds `pose_path`(2d keypoints json / 2d+3d pose labels), `densepose_path` and  `mask_path`.

- Train the whole model end-to-end:
```bash
bash train_start/pretrain_start.sh
```
To train successfully, one need to specificy the aforementioned prepared dataset.

- To view training results, please checkout intermediate results in `$checkpoints_dir/$name/web/index.html`.
If you have tensorflow installed, you can see tensorboard logs in `$checkpoints_dir/$name/logs`.


## Acknowledgments
This code borrows heavily from [pix2pixHD](https://github.com/NVIDIA/pix2pixHD).
